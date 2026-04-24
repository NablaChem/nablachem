import numpy as np
import pytest

from nablachem.krr import kernels as kernels_module


# Grid and ncheby used by ExponentialToChebychev.
GRID = 1.5 ** np.linspace(-15, 15, 100)
NCHEBY = 21


def _reference_build_power_moments(power_moments, Ds, atoms_per_mol, ncheby, grid):
    """Pre-change sort+cumsum reference for `_build_power_moments`.

    Mirrors the in-class loop that lived in `ExponentialToChebychev.__init__`
    before it was replaced by the numba-jitted `_build_power_moments`. `Ds`
    holds the per-atom distance-like values the old code consumed: distances
    for power=1, squared distances for power=2.
    """
    nmols = len(atoms_per_mol)
    atoms_per_mol_cumsum = np.concatenate([[0], np.cumsum(atoms_per_mol)])
    pair_idx = 0
    for i in range(nmols):
        for j in range(i, nmols):
            x = Ds[
                atoms_per_mol_cumsum[i] : atoms_per_mol_cumsum[i + 1],
                atoms_per_mol_cumsum[j] : atoms_per_mol_cumsum[j + 1],
            ].flatten()
            x = np.sort(x).astype(np.float64)
            x = x[np.isfinite(x)]

            if len(x) == 0:
                pair_idx += 1
                continue

            cum_moments = np.zeros((ncheby, len(x) + 1))
            cum_moments[0, 1:] = np.cumsum(np.ones_like(x))
            for k in range(1, ncheby):
                cum_moments[k, 1:] = np.cumsum(x**k)

            select_indices = np.minimum(
                np.searchsorted(x, grid, side="right"), len(x)
            )

            power_moments[pair_idx, :, :] = cum_moments[:, select_indices]
            pair_idx += 1


def _distance_matrix(X, power):
    """Gram-based distance matrix matching the new impl's clamp convention."""
    norms = np.sum(X * X, axis=1)
    G = X @ X.T
    d = norms[:, None] + norms[None, :] - 2.0 * G
    d = np.maximum(d, 0.0)
    return np.sqrt(d) if power == 1 else d


def _nn_tracking_args(atoms_per_mol):
    """Dummy NN-tracking args for `_build_power_moments`.

    These tests only check the moments output; the NN buffer is written but
    not inspected. Shapes must still be valid for the JIT function.
    """
    atoms_per_mol = np.asarray(atoms_per_mol, dtype=np.int64)
    nmols = len(atoms_per_mol)
    total = int(atoms_per_mol.sum())
    anchors_list = []
    k = 4
    while k < nmols:
        anchors_list.append(k)
        k *= 2
    anchors_list.append(nmols)
    anchors = np.asarray(anchors_list, dtype=np.int64)
    anchor_bucket_of_mol = np.searchsorted(
        anchors, np.arange(nmols), side="right"
    ).astype(np.int64)
    nn_per_anchor = np.full((total, len(anchors)), np.inf, dtype=np.float64)
    return anchor_bucket_of_mol, nn_per_anchor


def _run_comparison(X, atoms_per_mol, power):
    Ds = _distance_matrix(X, power)
    nmols = len(atoms_per_mol)
    npairs = nmols * (nmols + 1) // 2

    ref = np.zeros((npairs, NCHEBY, len(GRID)), dtype=np.float64)
    _reference_build_power_moments(ref, Ds, atoms_per_mol, NCHEBY, GRID)

    got = np.zeros_like(ref)
    anchor_bucket_of_mol, nn_per_anchor = _nn_tracking_args(atoms_per_mol)
    kernels_module._build_power_moments(
        got,
        X,
        atoms_per_mol,
        power,
        NCHEBY,
        GRID,
        np.zeros(0, dtype=np.float64),
        False,
        anchor_bucket_of_mol,
        nn_per_anchor,
    )

    # atol covers self-pair diagonal FP residuals: the new impl computes
    # d = ||X||^2 + ||X||^2 - 2<X,X> (catastrophic cancellation gives ~1e-16,
    # sqrt'd to ~1e-8 for power=1), while the ref sees the same residual but
    # summed in a different order. Artifact stays well below 1e-6.
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-6)


class TestBuildPowerMoments:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123])
    @pytest.mark.parametrize("power", [1, 2])
    @pytest.mark.parametrize(
        "nmols,atom_lo,atom_hi,nfeatures",
        [
            (5, 2, 6, 8),
            (15, 3, 10, 12),
            (25, 2, 8, 6),
        ],
    )
    def test_matches_reference(
        self, seed, power, nmols, atom_lo, atom_hi, nfeatures
    ):
        rng = np.random.default_rng(seed)
        atoms_per_mol = rng.integers(
            atom_lo, atom_hi + 1, size=nmols
        ).astype(np.int64)
        total_atoms = int(atoms_per_mol.sum())
        X = rng.uniform(size=(total_atoms, nfeatures))

        _run_comparison(X, atoms_per_mol, power)

    @pytest.mark.parametrize("power", [1, 2])
    def test_batching_boundary(self, power):
        # BATCH_TARGET = 1000 inside `_build_power_moments`; aim for ~2500
        # total atoms so batching produces 3 chunks and both intra- and
        # inter-batch pair loops are exercised.
        rng = np.random.default_rng(7)
        nmols = 300
        atoms_per_mol = rng.integers(6, 12, size=nmols).astype(np.int64)
        deficit = 2500 - int(atoms_per_mol.sum())
        if deficit > 0:
            idx = rng.choice(nmols, size=deficit, replace=True)
            for k in idx:
                atoms_per_mol[k] += 1
        total_atoms = int(atoms_per_mol.sum())
        assert total_atoms >= 2500
        X = rng.uniform(size=(total_atoms, 6))

        _run_comparison(X, atoms_per_mol, power)

    @pytest.mark.parametrize("seed", [0, 5, 42])
    @pytest.mark.parametrize("power", [1, 2])
    def test_single_atom_molecules(self, seed, power):
        rng = np.random.default_rng(seed)
        atoms_per_mol = np.array([1, 3, 1, 5, 2, 1, 4], dtype=np.int64)
        total_atoms = int(atoms_per_mol.sum())
        X = rng.uniform(size=(total_atoms, 8))

        _run_comparison(X, atoms_per_mol, power)

    @pytest.mark.parametrize("seed", [0, 11, 99])
    def test_power2_large_magnitudes(self, seed):
        # power=2 uses x_val = squared distance (no sqrt), so x^ncheby can
        # reach ~nfeatures^(ncheby/2). Wider features + higher atom counts
        # stress the high-chebyshev-power accumulation for this path.
        rng = np.random.default_rng(seed)
        atoms_per_mol = rng.integers(5, 11, size=12).astype(np.int64)
        total_atoms = int(atoms_per_mol.sum())
        X = rng.uniform(size=(total_atoms, 25))

        _run_comparison(X, atoms_per_mol, 2)

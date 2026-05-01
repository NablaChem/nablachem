import pathlib

import numpy as np
import pytest

from nablachem.krr.dataset import DataSet
from nablachem.krr.features import SLATMGlobal, SLATMLocal
from nablachem.krr import kernels
from nablachem.krr.matrix import (
    GlobalKernelMatrix,
    LocalKernelMatrix,
    ElementalKernelMatrix,
    _LocalKernelMatrix,
)

DATA_FILE = pathlib.Path(__file__).parent / "data" / "molecules.jsonl"


@pytest.fixture(scope="module")
def slatm_global_dataset():
    ds = DataSet(str(DATA_FILE), "A")
    SLATMGlobal().build(ds)
    return ds


@pytest.fixture(scope="module")
def slatm_local_dataset():
    ds = DataSet(str(DATA_FILE), "A")
    SLATMLocal().build(ds)
    return ds


def test_global_kernel_matrices_differ_by_kernel_func(slatm_global_dataset):
    """GlobalKernelMatrix produces different results for different kernel functions."""
    reps = slatm_global_dataset.representations
    X = np.stack(reps, axis=0)
    ntrain = len(reps)
    sigma = 10.0

    K_gaussian = GlobalKernelMatrix(X, kernels.Gaussian()).compute_train_kernel_matrix(
        sigma, ntrain
    )
    K_exponential = GlobalKernelMatrix(
        X, kernels.Exponential()
    ).compute_train_kernel_matrix(sigma, ntrain)

    assert not np.allclose(K_gaussian, K_exponential)


def test_local_kernel_matrices_differ_by_kernel_func(slatm_local_dataset):
    """LocalKernelMatrix should produce different results for different kernel functions, but doesn't."""
    reps = slatm_local_dataset.representations
    train_counts = np.array([rep.shape[0] for rep in reps])
    X = np.concatenate(reps, axis=0)
    ntrain = len(reps)
    sigma = 10.0

    K_gaussian = LocalKernelMatrix(
        X,
        train_counts,
        kernels.Gaussian(),
    ).compute_train_kernel_matrix_exact(sigma, ntrain)
    K_exponential = LocalKernelMatrix(
        X, train_counts, kernels.Exponential()
    ).compute_train_kernel_matrix_exact(sigma, ntrain)

    assert not np.allclose(K_gaussian, K_exponential)


def test_local_kernel_matrix_manual(slatm_local_dataset):
    """Manually compute local kernel matrix and compare to LocalKernelMatrix output."""
    reps = slatm_local_dataset.representations
    train_counts = np.array([rep.shape[0] for rep in reps])

    nmols = 2
    k_gaussian = np.zeros((nmols, nmols))
    k_exponential = np.zeros((nmols, nmols))

    # Manually compute pairwise distances and kernel values
    for mol1 in range(nmols):
        for mol2 in range(nmols):
            for atom1 in range(train_counts[mol1]):
                for atom2 in range(train_counts[mol2]):
                    dr = np.linalg.norm(reps[mol1][atom1] - reps[mol2][atom2])
                    dr /= 10
                    k_gaussian[mol1, mol2] += np.exp(-(dr**2))
                    k_exponential[mol1, mol2] += np.exp(-dr)

    # normalize
    diag = np.sqrt(np.diag(k_gaussian))
    k_gaussian /= np.outer(diag, diag)
    diag = np.sqrt(np.diag(k_exponential))
    k_exponential /= np.outer(diag, diag)

    # Compare to LocalKernelMatrix output
    K_gaussian = LocalKernelMatrix(
        np.concatenate(reps, axis=0), train_counts, kernels.Gaussian()
    ).compute_train_kernel_matrix_exact(10.0, 2)
    K_exponential = LocalKernelMatrix(
        np.concatenate(reps, axis=0),
        train_counts,
        kernels.Exponential(),
    ).compute_train_kernel_matrix_exact(10.0, 2)

    assert np.allclose(K_gaussian, k_gaussian)
    assert np.allclose(K_exponential, k_exponential)


def test_global_kernel_matrix_manual(slatm_global_dataset):
    """Manually compute global kernel matrix and compare to GlobalKernelMatrix output."""
    reps = slatm_global_dataset.representations
    X = np.stack(reps, axis=0)
    ntrain = len(reps)
    sigma = 10.0

    # Manually compute pairwise distances and kernel values
    nmols = len(reps)
    k_gaussian = np.zeros((nmols, nmols))
    k_exponential = np.zeros((nmols, nmols))

    for mol1 in range(nmols):
        for mol2 in range(nmols):
            dr = np.linalg.norm(X[mol1] - X[mol2]) / 10
            k_gaussian[mol1, mol2] = np.exp(-(dr**2))
            k_exponential[mol1, mol2] = np.exp(-dr)

    # normalize
    diag = np.sqrt(np.diag(k_gaussian))
    k_gaussian /= np.outer(diag, diag)
    diag = np.sqrt(np.diag(k_exponential))
    k_exponential /= np.outer(diag, diag)

    # Compare to GlobalKernelMatrix output
    K_gaussian = GlobalKernelMatrix(X, kernels.Gaussian()).compute_train_kernel_matrix(
        sigma, ntrain
    )
    K_exponential = GlobalKernelMatrix(
        X, kernels.Exponential()
    ).compute_train_kernel_matrix(sigma, ntrain)

    assert np.allclose(K_gaussian, k_gaussian)
    assert np.allclose(K_exponential, k_exponential)


def test_global_test_batched(slatm_global_dataset):
    reps = slatm_global_dataset.representations
    X_train = np.stack(reps, axis=0)

    # patterned holdout data
    X_holdout = np.stack(reps[:1] * 1024, axis=0)
    powers = 2 ** np.arange(1, 11) - 1
    for power in powers:
        X_holdout[power] = reps[1]

    kernel = kernels.Gaussian()
    kmat = GlobalKernelMatrix(X_train, kernel)
    batch_size = kmat._batch_size
    batch_0 = kmat.compute_test_kernel_matrix(10.0, len(reps), X_holdout[:batch_size])
    batch_1 = kmat.compute_test_kernel_matrix(10.0, len(reps), X_holdout[batch_size:])

    K_train = np.concatenate([batch_0, batch_1], axis=0)
    actual = np.where(K_train[:, 1] > 0.9)[0]
    assert np.array_equal(actual, powers)
    actual = np.where(K_train[:, 0] < 0.9)[0]
    assert np.array_equal(actual, powers)


def test_local_test_batched(slatm_local_dataset):
    reps = slatm_local_dataset.representations
    train_counts = np.array([rep.shape[0] for rep in reps])
    X_train = np.concatenate(reps, axis=0)

    # patterned holdout data
    holdout = reps[:1] * 1024
    powers = 2 ** np.arange(1, 11) - 1
    for power in powers:
        holdout[power] = reps[1]

    kmat = LocalKernelMatrix(X_train, train_counts, kernels.Gaussian())
    batch_size = kmat._batch_size

    mols_0 = holdout[:batch_size]
    counts_0 = np.array([rep.shape[0] for rep in mols_0])
    batch_0 = kmat.compute_test_kernel_matrix(
        10.0, len(reps), np.concatenate(mols_0, axis=0), counts_0
    )
    mols_1 = holdout[batch_size:]
    counts_1 = np.array([rep.shape[0] for rep in mols_1])
    batch_1 = kmat.compute_test_kernel_matrix(
        10.0, len(reps), np.concatenate(mols_1, axis=0), counts_1
    )

    K_train = np.concatenate([batch_0, batch_1], axis=0)
    actual = np.where(K_train[:, 1] > 0.9)[0]
    assert np.array_equal(actual, powers)
    actual = np.where(K_train[:, 0] < 0.9)[0]
    assert np.array_equal(actual, powers)



def test_evaluate_models_uses_per_model_sigma():
    from nablachem.krr.krr import AutoKRR

    n_train_max = 8
    n_holdout = 4
    n_all = n_train_max + n_holdout
    X = np.arange(n_all, dtype=float).reshape(-1, 1)
    y = X[:, 0].copy()

    class _MockRepresenter:
        def __init__(self, X):
            self._molecules = list(range(len(X)))
            self._X = X

        def __getitem__(self, key):
            if isinstance(key, slice):
                return self.compute(self._molecules[key])
            return [self._X[key]]

        def compute(self, mol_indices):
            return [self._X[i] for i in mol_indices]

    class _MockDataset:
        def __init__(self):
            self.representations = _MockRepresenter(X)
            self.labels = y.copy()

        def __len__(self):
            return n_all

    sigma_map = {4: 0.1, 8: 10.0}
    lam = 1e-6

    class _ForcedSigmaKRR(AutoKRR):
        def _optimize_hyperparameters(self, ntrain, _length_heuristic):
            return {"sigma": sigma_map[ntrain], "lambda": lam}, 1.0, 1.0

    krr = _ForcedSigmaKRR(
        _MockDataset(),
        mincount=4,
        maxcount=8,
        kernel_func=kernels.Gaussian(),
        detrend_atomic=False,
    )

    X_train = X[:n_train_max]
    X_holdout = X[n_train_max:]
    kmat = GlobalKernelMatrix(X_train, kernels.Gaussian())

    shift = np.mean(y[:4])
    y_train_4 = y[:4] - shift
    K_train_4 = kmat.compute_train_kernel_matrix(sigma_map[4], 4)
    alpha_4 = np.linalg.solve(K_train_4 + lam * np.eye(4), y_train_4)
    y_test_4 = y[n_train_max:] - shift
    K_test_4 = kmat.compute_test_kernel_matrix(sigma_map[4], 4, X_holdout)
    expected_residuals = y_test_4 - K_test_4 @ alpha_4

    assert np.allclose(krr.holdout_residuals[4], expected_residuals, atol=1e-6)


def test_elemental_kernel_masks_cross_element_pairs():
    """ElementalKernelMatrix should zero atom-atom contributions between different elements.

    Setup: 2 molecules, each with one H (Z=1) and one C (Z=6) atom.
    Features are chosen so H atoms are close and C atoms are close.

      mol0: H @ [1,0], C @ [0,1]
      mol1: H @ [0.9,0.1], C @ [0.1,0.9]

    With sigma=1, the elemental kernel K[0,1] must equal
      ( k(H0,H1) + k(C0,C1) ) / sqrt( k(H0,H0)+k(C0,C0) ) / sqrt( k(H1,H1)+k(C1,C1) )
    i.e. only same-element pairs contribute.
    """
    # atom features: rows are [H_mol0, C_mol0, H_mol1, C_mol1]
    X = np.array([
        [1.0, 0.0],   # H in mol 0
        [0.0, 1.0],   # C in mol 0
        [0.9, 0.1],   # H in mol 1
        [0.1, 0.9],   # C in mol 1
    ], dtype=float)
    nuclear_charges = np.array([1, 6, 1, 6])
    counts = np.array([2, 2])
    sigma = 1.0
    kernel = kernels.Gaussian()

    # manually compute expected elemental kernel
    def k(a, b):
        return np.exp(-np.sum((a - b) ** 2))

    # same-element pairs only
    k_HH = k(X[0], X[2])
    k_CC = k(X[1], X[3])
    k_self0 = k(X[0], X[0]) + k(X[1], X[1])  # = 2.0
    k_self1 = k(X[2], X[2]) + k(X[3], X[3])  # = 2.0
    expected_K01 = (k_HH + k_CC) / (np.sqrt(k_self0) * np.sqrt(k_self1))

    kmat = ElementalKernelMatrix(X, counts, kernel, nuclear_charges=nuclear_charges)
    K = kmat.compute_train_kernel_matrix_exact(sigma, ntrain=2)

    assert np.isclose(K[0, 1], expected_K01), (
        f"K[0,1]={K[0,1]:.6f} but expected {expected_K01:.6f}; "
        "cross-element pairs are likely not being masked"
    )
    assert np.isclose(K[0, 0], 1.0)
    assert np.isclose(K[1, 1], 1.0)


def test_elemental_kernel_masks_cross_element_pairs_holdout():
    """ElementalKernelMatrix should mask cross-element pairs in the test/holdout path.

    Setup: 2 train molecules and 1 holdout molecule, each with one H (Z=1) and one C (Z=6).

      mol0 (train): H @ [1.0, 0.0], C @ [0.0, 1.0]
      mol1 (train): H @ [0.9, 0.1], C @ [0.1, 0.9]
      mol_h (holdout): H @ [0.8, 0.2], C @ [0.2, 0.8]

    K_test[0, i] must equal ( k(H_h, H_i) + k(C_h, C_i) ) / sqrt(self_h) / sqrt(self_i),
    i.e. only same-element pairs contribute.
    """
    X_train = np.array([
        [1.0, 0.0],  # H in mol0
        [0.0, 1.0],  # C in mol0
        [0.9, 0.1],  # H in mol1
        [0.1, 0.9],  # C in mol1
    ], dtype=float)
    X_holdout = np.array([
        [0.8, 0.2],  # H in holdout mol
        [0.2, 0.8],  # C in holdout mol
    ], dtype=float)
    train_charges = np.array([1, 6, 1, 6])
    holdout_charges = np.array([1, 6])
    train_counts = np.array([2, 2])
    holdout_counts = np.array([2])
    sigma = 1.0
    kernel = kernels.Gaussian()

    def k(a, b):
        return np.exp(-np.sum((a - b) ** 2))

    # self-kernels: each molecule has one H and one C, k(x,x)=1 for both → sum=2
    self_0 = k(X_train[0], X_train[0]) + k(X_train[1], X_train[1])  # 2.0
    self_1 = k(X_train[2], X_train[2]) + k(X_train[3], X_train[3])  # 2.0
    self_h = k(X_holdout[0], X_holdout[0]) + k(X_holdout[1], X_holdout[1])  # 2.0

    expected_K_test_0 = (k(X_holdout[0], X_train[0]) + k(X_holdout[1], X_train[1])) / (
        np.sqrt(self_h) * np.sqrt(self_0)
    )
    expected_K_test_1 = (k(X_holdout[0], X_train[2]) + k(X_holdout[1], X_train[3])) / (
        np.sqrt(self_h) * np.sqrt(self_1)
    )

    kmat = ElementalKernelMatrix(X_train, train_counts, kernel, nuclear_charges=train_charges)
    K_test = kmat.compute_test_kernel_matrix(sigma, ntrain=2, X_batch=X_holdout, counts_batch=holdout_counts, nc_batch=holdout_charges)

    assert K_test.shape == (1, 2)
    assert np.isclose(K_test[0, 0], expected_K_test_0), (
        f"K_test[0,0]={K_test[0,0]:.6f} but expected {expected_K_test_0:.6f}"
    )
    assert np.isclose(K_test[0, 1], expected_K_test_1), (
        f"K_test[0,1]={K_test[0,1]:.6f} but expected {expected_K_test_1:.6f}"
    )


# --- Blockwise local kernel matrix reference + tests --------------------------
#
# Memory-heavy reference implementations mirror the pre-block code path: build
# the full atom-atom distance matrix, apply the kernel, aggregate to molecular
# kernel, and normalize. The block-based production code must produce the same
# result up to floating-point reassociation.


def _reference_local_train_exact(X, counts, kernel_func, sigma, nuclear_charges=None):
    counts = np.asarray(counts)
    natoms = int(counts.sum())
    X = np.asarray(X[:natoms])

    D2 = _LocalKernelMatrix._dist_squared(X)
    if nuclear_charges is not None:
        z = np.asarray(nuclear_charges[:natoms])
        D2[z[:, None] != z[None, :]] = np.inf
    K_atom = kernel_func.exact(np.sqrt(D2) / sigma)
    K_mol = _LocalKernelMatrix.aggregate_atomic_kernel(K_atom, counts, counts)
    d_sqrt = np.sqrt(np.diag(K_mol))
    return K_mol / np.outer(d_sqrt, d_sqrt)


def _reference_local_test(
    X_train, train_counts, X_batch, counts_batch, kernel_func, sigma,
    train_charges=None, batch_charges=None,
):
    train_counts = np.asarray(train_counts)
    counts_batch = np.asarray(counts_batch)
    natoms_train = int(train_counts.sum())
    X_train = np.asarray(X_train[:natoms_train])
    X_batch = np.asarray(X_batch)

    D2 = _LocalKernelMatrix._dist_squared(X_train, X_batch)
    if train_charges is not None:
        z_train = np.asarray(train_charges[:natoms_train])
        z_batch = np.asarray(batch_charges)
        D2[z_batch[:, None] != z_train[None, :]] = np.inf
    K_atom = kernel_func.exact(np.sqrt(D2) / sigma)
    K_test = _LocalKernelMatrix.aggregate_atomic_kernel(K_atom, counts_batch, train_counts)

    D2_tt = _LocalKernelMatrix._dist_squared(X_train)
    if train_charges is not None:
        D2_tt[z_train[:, None] != z_train[None, :]] = np.inf
    K_atom_tt = kernel_func.exact(np.sqrt(D2_tt) / sigma)
    K_train_unnorm = _LocalKernelMatrix.aggregate_atomic_kernel(
        K_atom_tt, train_counts, train_counts
    )
    d_train_sqrt = np.sqrt(np.diag(K_train_unnorm))

    d_test = np.empty(len(counts_batch))
    start = 0
    for i, c in enumerate(counts_batch):
        c = int(c)
        atoms_i = X_batch[start : start + c]
        d2_ii = _LocalKernelMatrix._dist_squared(atoms_i)
        if train_charges is not None:
            z_i = z_batch[start : start + c]
            d2_ii[z_i[:, None] != z_i[None, :]] = np.inf
        d_test[i] = kernel_func.exact(np.sqrt(d2_ii) / sigma).sum()
        start += c
    d_test_sqrt = np.sqrt(d_test)

    return K_test / np.outer(d_test_sqrt, d_train_sqrt)


def _pad_to_total_atoms(counts, target, rng):
    deficit = target - int(counts.sum())
    if deficit > 0:
        idx = rng.choice(len(counts), size=deficit, replace=True)
        for k in idx:
            counts[k] += 1
    return counts


class TestLocalKernelMatrixBlockwise:
    FEATURES = 50
    KERNELS = [kernels.Gaussian, kernels.Exponential]
    TOL = dict(rtol=1e-10, atol=1e-8)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    @pytest.mark.parametrize("kernel_cls", KERNELS)
    @pytest.mark.parametrize("sigma", [0.5, 5.0])
    @pytest.mark.parametrize(
        "nmols,atom_lo,atom_hi",
        [(5, 2, 6), (15, 3, 10), (25, 2, 8)],
    )
    def test_train_matches_reference(
        self, seed, kernel_cls, sigma, nmols, atom_lo, atom_hi
    ):
        rng = np.random.default_rng(seed)
        counts = rng.integers(atom_lo, atom_hi + 1, size=nmols).astype(np.int64)
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))

        kf = kernel_cls()
        K_got = LocalKernelMatrix(X, counts, kf).compute_train_kernel_matrix_exact(
            sigma, nmols
        )
        K_ref = _reference_local_train_exact(X, counts, kf, sigma)

        np.testing.assert_allclose(K_got, K_ref, **self.TOL)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    @pytest.mark.parametrize("kernel_cls", KERNELS)
    @pytest.mark.parametrize("sigma", [0.5, 5.0])
    @pytest.mark.parametrize(
        "nmols,atom_lo,atom_hi,n_test",
        [(5, 2, 6, 3), (15, 3, 10, 7), (25, 2, 8, 10)],
    )
    def test_test_matches_reference(
        self, seed, kernel_cls, sigma, nmols, atom_lo, atom_hi, n_test
    ):
        rng = np.random.default_rng(seed)
        counts = rng.integers(atom_lo, atom_hi + 1, size=nmols).astype(np.int64)
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))
        counts_test = rng.integers(atom_lo, atom_hi + 1, size=n_test).astype(np.int64)
        X_test = rng.uniform(size=(int(counts_test.sum()), self.FEATURES))

        kf = kernel_cls()
        kmat = LocalKernelMatrix(X, counts, kf)
        K_got = kmat.compute_test_kernel_matrix(sigma, nmols, X_test, counts_test)
        K_ref = _reference_local_test(X, counts, X_test, counts_test, kf, sigma)

        np.testing.assert_allclose(K_got, K_ref, **self.TOL)

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_train_batching_boundary(self, kernel_cls):
        # ~2500 train atoms across ~300 molecules → multiple batches of 1000.
        rng = np.random.default_rng(7)
        nmols = 300
        counts = rng.integers(6, 12, size=nmols).astype(np.int64)
        counts = _pad_to_total_atoms(counts, 2500, rng)
        assert int(counts.sum()) >= 2500
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))

        kf = kernel_cls()
        sigma = 5.0
        K_got = LocalKernelMatrix(X, counts, kf).compute_train_kernel_matrix_exact(
            sigma, nmols
        )
        K_ref = _reference_local_train_exact(X, counts, kf, sigma)

        np.testing.assert_allclose(K_got, K_ref, **self.TOL)

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_test_batching_boundary(self, kernel_cls):
        # ~2500 train atoms × ~1500 test atoms → multi-batch × multi-batch sweep.
        rng = np.random.default_rng(11)
        counts = rng.integers(6, 12, size=300).astype(np.int64)
        counts = _pad_to_total_atoms(counts, 2500, rng)
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))
        counts_test = rng.integers(6, 12, size=180).astype(np.int64)
        counts_test = _pad_to_total_atoms(counts_test, 1500, rng)
        assert int(counts_test.sum()) >= 1500
        X_test = rng.uniform(size=(int(counts_test.sum()), self.FEATURES))

        kf = kernel_cls()
        sigma = 5.0
        kmat = LocalKernelMatrix(X, counts, kf)
        K_got = kmat.compute_test_kernel_matrix(sigma, len(counts), X_test, counts_test)
        K_ref = _reference_local_test(X, counts, X_test, counts_test, kf, sigma)

        np.testing.assert_allclose(K_got, K_ref, **self.TOL)

    @pytest.mark.parametrize("seed", [0, 5, 42])
    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_single_atom_molecules(self, seed, kernel_cls):
        rng = np.random.default_rng(seed)
        counts = np.array([1, 3, 1, 5, 2, 1, 4], dtype=np.int64)
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))
        X_test_counts = np.array([1, 2, 1], dtype=np.int64)
        X_test = rng.uniform(size=(int(X_test_counts.sum()), self.FEATURES))

        kf = kernel_cls()
        sigma = 1.5
        kmat = LocalKernelMatrix(X, counts, kf)
        K_train = kmat.compute_train_kernel_matrix_exact(sigma, len(counts))
        K_train_ref = _reference_local_train_exact(X, counts, kf, sigma)
        np.testing.assert_allclose(K_train, K_train_ref, **self.TOL)

        # Fresh instance so the cache doesn't hide bugs in the test-path diag.
        kmat2 = LocalKernelMatrix(X, counts, kf)
        K_test = kmat2.compute_test_kernel_matrix(
            sigma, len(counts), X_test, X_test_counts
        )
        K_test_ref = _reference_local_test(X, counts, X_test, X_test_counts, kf, sigma)
        np.testing.assert_allclose(K_test, K_test_ref, **self.TOL)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    @pytest.mark.parametrize("kernel_cls", KERNELS)
    @pytest.mark.parametrize("sigma", [0.5, 5.0])
    def test_elemental_train_matches_reference(self, seed, kernel_cls, sigma):
        rng = np.random.default_rng(seed)
        counts = rng.integers(3, 10, size=20).astype(np.int64)
        natoms = int(counts.sum())
        X = rng.uniform(size=(natoms, self.FEATURES))
        charges = rng.choice([1, 6, 7, 8, 9], size=natoms).astype(float)

        kf = kernel_cls()
        K_got = ElementalKernelMatrix(
            X, counts, kf, nuclear_charges=charges
        ).compute_train_kernel_matrix_exact(sigma, len(counts))
        K_ref = _reference_local_train_exact(X, counts, kf, sigma, nuclear_charges=charges)

        np.testing.assert_allclose(K_got, K_ref, **self.TOL)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    @pytest.mark.parametrize("kernel_cls", KERNELS)
    @pytest.mark.parametrize("sigma", [0.5, 5.0])
    def test_elemental_test_matches_reference(self, seed, kernel_cls, sigma):
        rng = np.random.default_rng(seed)
        counts = rng.integers(3, 10, size=20).astype(np.int64)
        natoms = int(counts.sum())
        X = rng.uniform(size=(natoms, self.FEATURES))
        charges = rng.choice([1, 6, 7, 8, 9], size=natoms).astype(float)

        counts_test = rng.integers(3, 10, size=8).astype(np.int64)
        natoms_test = int(counts_test.sum())
        X_test = rng.uniform(size=(natoms_test, self.FEATURES))
        charges_test = rng.choice([1, 6, 7, 8, 9], size=natoms_test).astype(float)

        kf = kernel_cls()
        kmat = ElementalKernelMatrix(X, counts, kf, nuclear_charges=charges)
        K_got = kmat.compute_test_kernel_matrix(
            sigma, len(counts), X_test, counts_test, nc_batch=charges_test
        )
        K_ref = _reference_local_test(
            X, counts, X_test, counts_test, kf, sigma,
            train_charges=charges, batch_charges=charges_test,
        )

        np.testing.assert_allclose(K_got, K_ref, **self.TOL)

    @pytest.mark.parametrize("kernel_cls", KERNELS)
    def test_elemental_batching_boundary(self, kernel_cls):
        rng = np.random.default_rng(13)
        counts = rng.integers(6, 12, size=300).astype(np.int64)
        counts = _pad_to_total_atoms(counts, 2500, rng)
        natoms = int(counts.sum())
        X = rng.uniform(size=(natoms, self.FEATURES))
        charges = rng.choice([1, 6, 7, 8, 9], size=natoms).astype(float)

        counts_test = rng.integers(6, 12, size=180).astype(np.int64)
        counts_test = _pad_to_total_atoms(counts_test, 1500, rng)
        natoms_test = int(counts_test.sum())
        X_test = rng.uniform(size=(natoms_test, self.FEATURES))
        charges_test = rng.choice([1, 6, 7, 8, 9], size=natoms_test).astype(float)

        kf = kernel_cls()
        sigma = 5.0
        kmat = ElementalKernelMatrix(X, counts, kf, nuclear_charges=charges)
        K_train_got = kmat.compute_train_kernel_matrix_exact(sigma, len(counts))
        K_train_ref = _reference_local_train_exact(
            X, counts, kf, sigma, nuclear_charges=charges
        )
        np.testing.assert_allclose(K_train_got, K_train_ref, **self.TOL)

        # Use a second instance so the test-path diag is recomputed from scratch.
        kmat2 = ElementalKernelMatrix(X, counts, kf, nuclear_charges=charges)
        K_test_got = kmat2.compute_test_kernel_matrix(
            sigma, len(counts), X_test, counts_test, nc_batch=charges_test
        )
        K_test_ref = _reference_local_test(
            X, counts, X_test, counts_test, kf, sigma,
            train_charges=charges, batch_charges=charges_test,
        )
        np.testing.assert_allclose(K_test_got, K_test_ref, **self.TOL)

    def test_train_ntrain_subset(self):
        rng = np.random.default_rng(0)
        counts = rng.integers(2, 8, size=30).astype(np.int64)
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))

        kf = kernels.Gaussian()
        sigma = 3.0
        kmat = LocalKernelMatrix(X, counts, kf)
        K_got = kmat.compute_train_kernel_matrix_exact(sigma, 15)
        K_ref = _reference_local_train_exact(X, counts[:15], kf, sigma)

        np.testing.assert_allclose(K_got, K_ref, **self.TOL)

    def test_d_train_cache_populated_by_train_path(self):
        rng = np.random.default_rng(1)
        counts = rng.integers(3, 8, size=12).astype(np.int64)
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))

        kf = kernels.Gaussian()
        sigma = 2.0
        kmat = LocalKernelMatrix(X, counts, kf)
        kmat.compute_train_kernel_matrix_exact(sigma, len(counts))
        assert (sigma, len(counts)) in kmat._d_train_cache

    def test_d_train_cache_populated_by_test_path(self):
        rng = np.random.default_rng(2)
        counts = rng.integers(3, 8, size=12).astype(np.int64)
        X = rng.uniform(size=(int(counts.sum()), self.FEATURES))
        counts_test = rng.integers(3, 8, size=4).astype(np.int64)
        X_test = rng.uniform(size=(int(counts_test.sum()), self.FEATURES))

        kf = kernels.Gaussian()
        sigma = 2.0
        kmat = LocalKernelMatrix(X, counts, kf)
        kmat.compute_test_kernel_matrix(sigma, len(counts), X_test, counts_test)

        key = (sigma, len(counts))
        assert key in kmat._d_train_cache

        # Cached diag equals the reference train self-kernel diag.
        D2_tt = _LocalKernelMatrix._dist_squared(X)
        K_atom_tt = kf.exact(np.sqrt(D2_tt) / sigma)
        K_train_unnorm = _LocalKernelMatrix.aggregate_atomic_kernel(
            K_atom_tt, counts, counts
        )
        expected = np.sqrt(np.diag(K_train_unnorm))
        np.testing.assert_allclose(kmat._d_train_cache[key], expected, **self.TOL)


def test_elemental_approx_masks_disjoint_element_pair():
    # Two molecules with no shared elements: N2 and H2. Every atom-atom pair
    # across them is cross-element, so under elemental masking the Chebyshev
    # power moments for pair (0, 1) must be identically zero.
    kf = kernels.Gaussian()
    counts = np.array([2, 2], dtype=np.int64)
    charges = np.array([7.0, 7.0, 1.0, 1.0])  # N2, H2
    X = np.random.default_rng(0).uniform(size=(4, 5))

    ElementalKernelMatrix(X, counts, kf, nuclear_charges=charges)

    # pair_idx layout for 2 mols: 0→(0,0), 1→(0,1), 2→(1,1).
    cross = kf._chebytrick._local_power_moments[1]
    assert np.all(cross == 0.0)


def _reference_length_scale(X, counts, ntrain, charges=None):
    """Median atomic nearest-neighbour distance over atoms of first ``ntrain`` mols.

    Mirrors the pre-block implementation on main: full atom-atom squared-distance
    matrix (diag=0 is the self-pair, the 2nd-smallest per column is the NN).
    """
    counts = np.asarray(counts)
    natoms = int(counts[:ntrain].sum())
    X_sub = np.asarray(X[:natoms])
    D2 = _LocalKernelMatrix._dist_squared(X_sub)
    if charges is not None:
        z = np.asarray(charges[:natoms])
        D2[z[:, None] != z[None, :]] = np.inf
    nnvals = np.partition(D2, 1, axis=0)[1]
    return float(np.sqrt(np.median(nnvals)))


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("kernel_cls", [kernels.Gaussian, kernels.Exponential])
def test_length_scale_matches_reference(seed, kernel_cls):
    """length_scale at anchor ntrain values matches a simple np.partition reference."""
    rng = np.random.default_rng(seed)
    nmols = 16
    counts = rng.integers(4, 8, size=nmols).astype(np.int64)
    natoms = int(counts.sum())
    X = rng.uniform(size=(natoms, 6))

    kmat = LocalKernelMatrix(X, counts, kernel_cls())
    for anchor in [4, 8, 16]:
        got = kmat.length_scale(anchor)
        expected = _reference_length_scale(X, counts, anchor)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("kernel_cls", [kernels.Gaussian, kernels.Exponential])
def test_length_scale_elemental_matches_reference(seed, kernel_cls):
    """Elemental length_scale restricts NN to same-element pairs; matches reference."""
    rng = np.random.default_rng(seed)
    nmols = 16
    # Enough atoms per mol so that even at anchor 4 every element has neighbours.
    counts = rng.integers(8, 14, size=nmols).astype(np.int64)
    natoms = int(counts.sum())
    X = rng.uniform(size=(natoms, 6))
    # Only two elements → robust same-element sampling at every anchor.
    charges = rng.choice([1.0, 6.0], size=natoms)

    kmat = ElementalKernelMatrix(
        X, counts, kernel_cls(), nuclear_charges=charges
    )
    for anchor in [4, 8, 16]:
        got = kmat.length_scale(anchor)
        expected = _reference_length_scale(X, counts, anchor, charges=charges)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("kernel_cls", [kernels.Gaussian, kernels.Exponential])
def test_approx_matches_exact_local(seed, kernel_cls):
    """Chebyshev-trick approx kernel matches the exact kernel within 1e-8."""
    rng = np.random.default_rng(seed)
    nmols = 20
    counts = rng.integers(3, 8, size=nmols).astype(np.int64)
    natoms = int(counts.sum())
    X = rng.uniform(size=(natoms, 4))

    kf = kernel_cls()
    kmat = LocalKernelMatrix(X, counts, kf)
    sigma = 0.4 if kernel_cls is kernels.Gaussian else 0.2
    K_approx = kmat.compute_train_kernel_matrix_approx(sigma, nmols)
    K_exact = kmat.compute_train_kernel_matrix_exact(sigma, nmols)
    assert K_approx is not None, "approx rejected by validity check"
    np.testing.assert_allclose(K_approx, K_exact, atol=1e-8, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("kernel_cls", [kernels.Gaussian, kernels.Exponential])
def test_approx_matches_exact_elemental(seed, kernel_cls):
    """Elemental Chebyshev-trick approx matches elemental exact within 1e-8."""
    rng = np.random.default_rng(seed)
    nmols = 20
    counts = rng.integers(3, 8, size=nmols).astype(np.int64)
    natoms = int(counts.sum())
    X = rng.uniform(size=(natoms, 4))
    charges = rng.choice([1.0, 6.0, 7.0, 8.0], size=natoms)

    kf = kernel_cls()
    kmat = ElementalKernelMatrix(X, counts, kf, nuclear_charges=charges)
    sigma = 0.4 if kernel_cls is kernels.Gaussian else 0.2
    K_approx = kmat.compute_train_kernel_matrix_approx(sigma, nmols)
    K_exact = kmat.compute_train_kernel_matrix_exact(sigma, nmols)
    assert K_approx is not None, "approx rejected by validity check"
    np.testing.assert_allclose(K_approx, K_exact, atol=1e-8, rtol=0)

import numpy as np
import pytest
import ase
import click
from scipy import linalg

from nablachem.krr.cli import _parse_detrending
from nablachem.krr.dataset import DataSet
from nablachem.krr.krr import DETRENDING_TERMS, AutoKRR


def make_dataset(molecules):
    """Bypass DataSet.__init__ to inject ASE Atoms objects directly."""
    ds = DataSet.__new__(DataSet)
    ds.molecules = molecules
    return ds


def hh_dimer(d):
    return ase.Atoms("HH", positions=[[0, 0, 0], [0, 0, d]])


def co_dimer(d):
    return ase.Atoms("CO", positions=[[0, 0, 0], [0, 0, d]])


# ---------------------------------------------------------------------------
# Feature calculation tests
# ---------------------------------------------------------------------------


def test_gcp_homonuclear_dimer():
    """H-H dimer: one pair, Z=1 each → exp(-3*d)."""
    d = 2.0
    ds = make_dataset([hh_dimer(d)])
    features, labels = ds.get_pairwise_features("gCP")

    assert features.shape == (1, 1)
    assert labels == ["gCP"]
    assert np.isclose(features[0, 0], np.exp(-3 * d))


def test_gcp_heteronuclear_dimer():
    """C-O dimer: Z_C=6, Z_O=8 → 6^2.5·8^2.5·exp(-3*d)."""
    d = 1.5
    ds = make_dataset([co_dimer(d)])
    features, _ = ds.get_pairwise_features("gCP")

    expected = 6**2.5 * 8**2.5 * np.exp(-3 * d)
    assert np.isclose(features[0, 0], expected)


def test_gcp_feature_scales_with_distance():
    """gCP value decreases monotonically with increasing distance."""
    distances = [1.0, 2.0, 3.0, 4.0]
    ds = make_dataset([hh_dimer(d) for d in distances])
    features, _ = ds.get_pairwise_features("gCP")

    vals = features[:, 0]
    assert np.all(np.diff(vals) < 0)


def test_gcp_multiple_molecules_independent():
    """Feature for each molecule is independent — batch equals single evaluations."""
    dimers = [hh_dimer(1.0), co_dimer(2.0), hh_dimer(3.0)]
    batch_features, _ = make_dataset(dimers).get_pairwise_features("gCP")

    for i, mol in enumerate(dimers):
        single, _ = make_dataset([mol]).get_pairwise_features("gCP")
        assert np.isclose(batch_features[i, 0], single[0, 0])


def test_gcp_unknown_label_exits():
    ds = make_dataset([hh_dimer(1.0)])
    with pytest.raises(SystemExit):
        ds.get_pairwise_features("unknown")


# ---------------------------------------------------------------------------
# Detrending correctness: if labels = coef * gCP_sum, lstsq must recover coef
# ---------------------------------------------------------------------------


def test_detrend_recovers_coefficient_hh():
    """Labels = k * gCP(H-H dimer) → lstsq coefficient equals k exactly."""
    known_coef = 42.0
    distances = np.linspace(0.8, 4.0, 30)

    ds = make_dataset([hh_dimer(d) for d in distances])
    features, _ = ds.get_pairwise_features("gCP")
    y = known_coef * features[:, 0]

    coef_fit = linalg.lstsq(features, y)[0]
    assert np.isclose(coef_fit[0], known_coef, rtol=1e-6)


def test_detrend_residuals_are_zero_when_signal_is_pure_gcp():
    """After subtracting the fitted trend, residuals vanish for pure gCP signal."""
    known_coef = 7.3
    distances = np.linspace(1.0, 5.0, 20)

    ds = make_dataset([hh_dimer(d) for d in distances])
    features, _ = ds.get_pairwise_features("gCP")
    y = known_coef * features[:, 0]

    coefs = linalg.lstsq(features, y)[0]
    residuals = y - features @ coefs
    assert np.allclose(residuals, 0.0, atol=1e-10)


def test_joint_atomic_and_gcp_detrending_recovers_both_coefficients():
    """Labels = a_H*n_H + a_C*n_C + a_O*n_O + b*gCP.

    Uses homonuclear H-H, C-C, and O-O dimers so element count columns are
    independent and the Z⁴ prefactors (1, 1296, 4096) make the gCP feature
    non-degenerate with atomic counts.
    """
    coef_H = 1.0
    coef_C = 5.0
    coef_O = 8.0
    coef_gcp = 17.5

    # 15 dimers of each type at different distances
    distances = np.linspace(0.8, 4.0, 15)
    hh = [ase.Atoms("HH", positions=[[0, 0, 0], [0, 0, d]]) for d in distances]
    cc = [ase.Atoms("CC", positions=[[0, 0, 0], [0, 0, d]]) for d in distances]
    oo = [ase.Atoms("OO", positions=[[0, 0, 0], [0, 0, d]]) for d in distances]
    molecules = hh + cc + oo
    n = len(distances)

    ds = make_dataset(molecules)
    pair_features, _ = ds.get_pairwise_features("gCP")

    # atomic count columns: H, C, O
    n_H = np.array([2.0] * n + [0.0] * n + [0.0] * n)
    n_C = np.array([0.0] * n + [2.0] * n + [0.0] * n)
    n_O = np.array([0.0] * n + [0.0] * n + [2.0] * n)
    atomic_features = np.column_stack([n_H, n_C, n_O])

    A = np.hstack([atomic_features, pair_features])
    y = coef_H * n_H + coef_C * n_C + coef_O * n_O + coef_gcp * pair_features[:, 0]

    coefs = linalg.lstsq(A, y)[0]
    assert np.isclose(coefs[0], coef_H, rtol=1e-6)
    assert np.isclose(coefs[1], coef_C, rtol=1e-6)
    assert np.isclose(coefs[2], coef_O, rtol=1e-6)
    assert np.isclose(coefs[3], coef_gcp, rtol=1e-6)
    assert np.allclose(y - A @ coefs, 0.0, atol=1e-10)


def test_detrend_mixed_signal_leaves_non_gcp_part():
    """Labels = k*gCP + noise. After detrending, residuals ≈ noise (not zero)."""
    rng = np.random.default_rng(0)
    known_coef = 5.0
    distances = np.linspace(1.0, 4.0, 50)
    noise = rng.normal(0, 0.1, len(distances))

    ds = make_dataset([hh_dimer(d) for d in distances])
    features, _ = ds.get_pairwise_features("gCP")
    y = known_coef * features[:, 0] + noise

    coefs = linalg.lstsq(features, y)[0]
    residuals = y - features @ coefs

    # residuals should correlate with the noise, not with the gCP feature
    assert np.corrcoef(residuals, noise)[0, 1] > 0.99
    assert np.corrcoef(residuals, features[:, 0])[0, 1] < 0.01


# ---------------------------------------------------------------------------
# --detrending term selection
# ---------------------------------------------------------------------------


def test_detrending_defaults_to_atomic():
    assert _parse_detrending(None, None, None) == ("atomic",)


def test_detrending_accepts_comma_separated_terms():
    assert _parse_detrending(None, None, "atomic,charge") == ("atomic", "charge")


def test_detrending_tolerates_whitespace():
    assert _parse_detrending(None, None, " atomic , charge ") == ("atomic", "charge")


def test_detrending_empty_string_disables_all_terms():
    assert _parse_detrending(None, None, "") == ()


def test_detrending_rejects_unknown_term():
    with pytest.raises(click.BadParameter, match="charg"):
        _parse_detrending(None, None, "atomic,charg")


def test_all_documented_terms_are_accepted():
    joined = ",".join(DETRENDING_TERMS)
    assert _parse_detrending(None, None, joined) == DETRENDING_TERMS


# ---------------------------------------------------------------------------
# categorical (charge / spin) columns
# ---------------------------------------------------------------------------


def test_categorical_block_marks_only_the_molecule_own_state():
    _, columns, _ = AutoKRR._categorical_block(
        "charge", np.array([-1, 0, 1]), lambda q: f"q{q:+d}"
    )
    assert np.array_equal(columns, np.eye(3))


def test_categorical_block_rows_sum_to_one():
    values = np.array([-1, -1, 0, 1, 1, 1])
    _, columns, _ = AutoKRR._categorical_block("charge", values, lambda q: f"q{q:+d}")
    assert np.array_equal(columns.sum(axis=1), np.ones(len(values)))


def test_categorical_block_has_one_column_per_observed_state():
    _, columns, labels = AutoKRR._categorical_block(
        "spin", np.array([1, 1, 3]), lambda m: f"M{m}"
    )
    assert columns.shape == (3, 2)
    assert labels == ["M1", "M3"]


def test_categorical_block_labels_contain_no_equals_sign():
    # labels are used as logger keys, so an embedded "=" renders as "q=-1=0.9"
    _, _, labels = AutoKRR._categorical_block(
        "charge", np.array([-1, 0, 1]), lambda q: f"q{q:+d}"
    )
    assert labels == ["q-1", "q+0", "q+1"]


# ---------------------------------------------------------------------------
# joint fit over atomic counts and charge states
# ---------------------------------------------------------------------------


def test_joint_atomic_and_charge_detrending_recovers_both():
    """Labels = a_H*n_H + a_C*n_C + per-charge offset, fitted in one pass."""
    n = 12
    charges = np.array(([-1] * 4 + [0] * 4 + [1] * 4) * 2)
    n_H = np.array([2.0] * n + [0.0] * n)
    n_C = np.array([0.0] * n + [2.0] * n)
    offsets = {-1: 0.9, 0: 0.0, 1: -0.9}

    _, charge_columns, _ = AutoKRR._categorical_block(
        "charge", charges, lambda q: f"q{q:+d}"
    )
    A = np.hstack([np.column_stack([n_H, n_C]), charge_columns])
    y = 1.0 * n_H + 5.0 * n_C + np.array([offsets[q] for q in charges])

    assert np.allclose(y - A @ linalg.lstsq(A, y)[0], 0.0, atol=1e-9)


def test_joint_fit_centres_every_charge_state():
    """Residuals are orthogonal to each indicator column, so every charge state
    ends up with zero mean -- this is what collapses a multimodal label
    distribution before the kernel sees it."""
    rng = np.random.default_rng(0)
    charges = rng.choice([-1, 0, 1], size=60)
    offsets = {-1: 0.9, 0: 0.0, 1: -0.9}

    _, charge_columns, _ = AutoKRR._categorical_block(
        "charge", charges, lambda q: f"q{q:+d}"
    )
    n_H = rng.integers(2, 10, size=len(charges)).astype(float)
    A = np.hstack([n_H.reshape(-1, 1), charge_columns])
    y = 0.5 * n_H + np.array([offsets[q] for q in charges]) + rng.normal(0, 0.01, 60)

    residual = y - A @ linalg.lstsq(A, y)[0]
    for state in np.unique(charges):
        assert abs(residual[charges == state].mean()) < 1e-9


def test_charge_detrending_alone_still_centres_every_state():
    """With no atomic block there is nothing else to supply an intercept, which
    is exactly where dropping a reference state would leave one state
    unrepresentable."""
    rng = np.random.default_rng(1)
    charges = rng.choice([-1, 0, 1], size=60)
    offsets = {-1: 0.9, 0: 0.0, 1: -0.9}

    _, A, _ = AutoKRR._categorical_block("charge", charges, lambda q: f"q{q:+d}")
    y = np.array([offsets[q] for q in charges])

    residual = y - A @ linalg.lstsq(A, y)[0]
    assert np.allclose(residual, 0.0, atol=1e-9)

import pathlib

import numpy as np
import pytest

from nablachem.krr.dataset import DataSet
from nablachem.krr.features import SLATMGlobal, SLATMLocal
from nablachem.krr import kernels
from nablachem.krr.matrix import GlobalKernelMatrix, LocalKernelMatrix, ElementalKernelMatrix

DATA_FILE = pathlib.Path(__file__).parent / "data" / "molecules.jsonl"


@pytest.fixture(scope="module")
def slatm_global_dataset():
    ds = DataSet(str(DATA_FILE), "A")
    SLATMGlobal().build([ds])
    return ds


@pytest.fixture(scope="module")
def slatm_local_dataset():
    ds = DataSet(str(DATA_FILE), "A")
    SLATMLocal().build([ds])
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
        X,
        train_counts,
    ).compute_train_kernel_matrix_exact(sigma, ntrain)
    K_exponential = LocalKernelMatrix(
        X, train_counts, kernels.Exponential(), X, train_counts
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
    kmat = GlobalKernelMatrix(X_train, kernel, X_holdout)
    batch_0 = kmat.compute_test_kernel_matrix(10.0, len(reps), batch=0)
    batch_1 = kmat.compute_test_kernel_matrix(10.0, len(reps), batch=1)
    batch_2 = kmat.compute_test_kernel_matrix(10.0, len(reps), batch=2)
    assert batch_2 is None

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
    X_holdout = np.concatenate(holdout, axis=0)

    kmat = LocalKernelMatrix(
        X_train,
        train_counts,
        kernels.Gaussian(),
        X_holdout,
        np.array([rep.shape[0] for rep in holdout]),
    )
    batch_0 = kmat.compute_test_kernel_matrix(10.0, len(reps), batch=0)
    batch_1 = kmat.compute_test_kernel_matrix(10.0, len(reps), batch=1)
    batch_2 = kmat.compute_test_kernel_matrix(10.0, len(reps), batch=2)
    assert batch_2 is None

    K_train = np.concatenate([batch_0, batch_1], axis=0)
    actual = np.where(K_train[:, 1] > 0.9)[0]
    assert np.array_equal(actual, powers)
    actual = np.where(K_train[:, 0] < 0.9)[0]
    assert np.array_equal(actual, powers)


def test_local_kernel_holdout_count(slatm_local_dataset):
    """Test that LocalKernelMatrix correctly counts holdout molecules."""
    reps = slatm_local_dataset.representations
    train_counts = np.array([rep.shape[0] for rep in reps])
    X_train = np.concatenate(reps, axis=0)

    kmat = LocalKernelMatrix(
        X_train,
        train_counts,
        kernels.Gaussian(),
        X_train,
        np.array([rep.shape[0] for rep in reps]),
    )
    assert kmat._holdout_molecule_count == len(reps)


def test_global_kernel_holdout_count(slatm_global_dataset):
    """Test that GlobalKernelMatrix correctly counts holdout molecules."""
    reps = slatm_global_dataset.representations
    X_train = np.stack(reps, axis=0)

    kmat = GlobalKernelMatrix(X_train, kernels.Gaussian(), X_train)
    assert kmat._holdout_molecule_count == len(reps)


def test_evaluate_models_uses_per_model_sigma():
    from nablachem.krr.krr import AutoKRR

    n_train_max = 8
    n_holdout = 4
    n_all = n_train_max + n_holdout
    X = np.arange(n_all, dtype=float).reshape(-1, 1)
    y = X[:, 0].copy()

    class _MockDataset:
        def __init__(self):
            self.representations = [X[i] for i in range(n_all)]
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
    kmat = GlobalKernelMatrix(X_train, kernels.Gaussian(), X_holdout)

    shift = np.mean(y[:4])
    y_train_4 = y[:4] - shift
    K_train_4 = kmat.compute_train_kernel_matrix(sigma_map[4], 4)
    alpha_4 = np.linalg.solve(K_train_4 + lam * np.eye(4), y_train_4)
    y_test_4 = y[n_train_max:] - shift
    K_test_4 = kmat.compute_test_kernel_matrix(sigma_map[4], 4, batch=0)
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

    kmat = ElementalKernelMatrix(
        X_train,
        train_counts,
        kernel,
        X_holdout,
        holdout_counts,
        nuclear_charges=train_charges,
        holdout_nuclear_charges=holdout_charges,
    )
    K_test = kmat.compute_test_kernel_matrix(sigma, ntrain=2, batch=0)

    assert K_test.shape == (1, 2)
    assert np.isclose(K_test[0, 0], expected_K_test_0), (
        f"K_test[0,0]={K_test[0,0]:.6f} but expected {expected_K_test_0:.6f}"
    )
    assert np.isclose(K_test[0, 1], expected_K_test_1), (
        f"K_test[0,1]={K_test[0,1]:.6f} but expected {expected_K_test_1:.6f}"
    )

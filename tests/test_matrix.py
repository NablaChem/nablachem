import pathlib

import numpy as np
import pytest

from nablachem.krr.dataset import DataSet
from nablachem.krr.features import SLATMGlobal, SLATMLocal
from nablachem.krr import kernels
from nablachem.krr.matrix import GlobalKernelMatrix, LocalKernelMatrix

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

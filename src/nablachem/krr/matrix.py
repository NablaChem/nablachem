import numpy as np
from .kernels import Kernel


class KernelMatrix:
    """Base class for kernel matrix computation and management"""

    def __init__(
        self, X: np.ndarray, kernel_func: Kernel, X_holdout: np.ndarray = None
    ):
        """Initialize kernel matrix with training and optional holdout data

        Args:
            X: Training data
            X_holdout: Holdout/test data (optional)
            kernel_func: Kernel function to use (optional)
        """
        self._X = X
        self._X_holdout = X_holdout
        self._kernel_func = kernel_func

        # Compute distance matrices
        self._D2 = self._dist_squared(X)
        if X_holdout is not None:
            self._D2_test = self._dist_squared(X, X_holdout)

    @staticmethod
    def _dist_squared(X_one: np.ndarray, X_other: np.ndarray = None) -> np.ndarray:
        """Compute squared Euclidean distances between points"""
        X_one = np.array(X_one)
        X_one_norm_sq = np.sum(X_one**2, axis=1)
        if X_other is None:
            X_other_norm_sq = X_one_norm_sq
            X_other = X_one
        else:
            X_other = np.array(X_other)
            X_other_norm_sq = np.sum(X_other**2, axis=1)
        D2 = X_other_norm_sq[:, None] + X_one_norm_sq[None, :] - 2 * (X_other @ X_one.T)
        D2 = np.maximum(D2, 0.0)
        return D2

    def compute_train_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute training kernel matrix for given sigma and training size"""
        raise NotImplementedError(
            "Subclasses must implement compute_train_kernel_matrix"
        )

    def compute_test_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute test kernel matrix for given sigma and training size"""
        raise NotImplementedError(
            "Subclasses must implement compute_test_kernel_matrix"
        )


class LocalKernelMatrix(KernelMatrix):
    """Kernel matrix for local (atom-based) representations with approximation cache"""

    def __init__(
        self,
        X: np.ndarray,
        train_counts: np.ndarray,
        kernel_func: Kernel,
        X_holdout: np.ndarray = None,
        holdout_counts: np.ndarray = None,
    ):
        """Initialize local kernel matrix

        Args:
            X: Concatenated atom representations
            train_counts: Number of atoms per training molecule
            X_holdout: Concatenated holdout atom representations (optional)
            holdout_counts: Number of atoms per holdout molecule (optional)
        """
        super().__init__(
            X,
            kernel_func,
            X_holdout,
        )
        self._train_counts = train_counts
        self._holdout_counts = holdout_counts

        # Compute holdout self-distances if needed
        if X_holdout is not None and holdout_counts is not None:
            self._test_self = []
            start = 0
            for count in holdout_counts:
                end = start + count
                self._test_self.append(
                    self._dist_squared(X_holdout[start:end], X_holdout[start:end])
                )
                start = end
        self._approx_fail_sigma = dict()
        self._kernel_func.approx_prepare(train_counts, self._D2)

    def length_scale(self, ntrain: int) -> float:
        # get median nearest neighbor distance for first ntrain points
        nentries = sum(self._train_counts[:ntrain])
        section = self._D2[:nentries, :nentries].copy()
        np.fill_diagonal(section, np.inf)
        nnvals = np.amin(section, axis=0)
        return np.median(nnvals) ** 0.5

    @staticmethod
    def aggregate_atomic_kernel(
        K_atom_AB: np.ndarray, counts_A: np.ndarray, counts_B: np.ndarray
    ) -> np.ndarray:
        """Aggregate atomic kernel matrix to molecular kernel matrix"""
        counts_A = np.asarray(counts_A)
        counts_B = np.asarray(counts_B)

        starts_A = np.concatenate(([0], np.cumsum(counts_A)))
        starts_B = np.concatenate(([0], np.cumsum(counts_B)))

        # 2D prefix sum with 1-based indexing
        S = np.zeros((K_atom_AB.shape[0] + 1, K_atom_AB.shape[1] + 1))
        S[1:, 1:] = K_atom_AB
        S = S.cumsum(axis=0).cumsum(axis=1)

        # block boundaries
        a0 = starts_A[:-1][:, None]  # (MA,1)
        a1 = starts_A[1:][:, None]  # (MA,1)
        b0 = starts_B[:-1][None, :]  # (1,MB)
        b1 = starts_B[1:][None, :]  # (1,MB)

        # rectangular block extraction
        Kp = S[a1, b1] - S[a0, b1] - S[a1, b0] + S[a0, b0]
        return Kp

    def compute_train_kernel_matrix(self, sigma, ntrain):
        # approx only pays off for larger ntrain
        if ntrain <= 128:
            return self.compute_train_kernel_matrix_exact(sigma, ntrain)

        failsigma = self._approx_fail_sigma.get(ntrain, None)
        if failsigma is not None and sigma >= failsigma:
            return self.compute_train_kernel_matrix_exact(sigma, ntrain)
        approx = self.compute_train_kernel_matrix_approx(sigma, ntrain)
        if approx is None:
            self._approx_fail_sigma[ntrain] = sigma
            res = self.compute_train_kernel_matrix_exact(sigma, ntrain)
            return res
        return approx

    def compute_train_kernel_matrix_approx(
        self, sigma: float, ntrain: int
    ) -> np.ndarray:
        """Compute training local kernel matrix using approximation"""
        return self._kernel_func.approx(sigma, ntrain)

    def compute_train_kernel_matrix_exact(
        self, sigma: float, ntrain: int
    ) -> np.ndarray:
        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)

        # Compute atomic kernel between test and train
        K_atom = self._kernel_func.exact(np.sqrt(self._D2[:natoms, :natoms]) / sigma)
        K_atom_train = self.aggregate_atomic_kernel(
            K_atom, atom_counts_A, atom_counts_A
        )

        d_train_sqrt = np.sqrt(np.diag(K_atom_train))
        K_train = K_atom_train / np.outer(d_train_sqrt, d_train_sqrt)

        return K_train

    def compute_test_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute test kernel matrix for local representations"""
        if self._X_holdout is None:
            raise ValueError("Holdout data not provided")

        atom_counts_A = self._train_counts[:ntrain]
        atom_counts_B = self._holdout_counts
        natoms = sum(atom_counts_A)

        # Compute atomic kernel between test and train
        K_atom = self._kernel_func.exact(np.sqrt(self._D2_test[:, :natoms]) / sigma)
        K_test = self.aggregate_atomic_kernel(K_atom, atom_counts_B, atom_counts_A)

        # Compute normalization factors
        # For training: get unnormalized diagonal first
        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)
        K_atom_train = self._kernel_func.exact(
            np.sqrt(self._D2[:natoms, :natoms]) / sigma
        )
        K_train_unnorm = self.aggregate_atomic_kernel(
            K_atom_train, atom_counts_A, atom_counts_A
        )
        d_train_sqrt = np.sqrt(np.diag(K_train_unnorm))

        # For test: self-kernel diagonal
        d_test = np.sqrt(
            [
                self._kernel_func.exact(np.sqrt(test_self_dist) / sigma).sum()
                for test_self_dist in self._test_self
            ]
        )

        # Apply normalization
        K_test /= np.outer(d_test, d_train_sqrt)

        return K_test


class GlobalKernelMatrix(KernelMatrix):
    """Kernel matrix for global (molecule-based) representations"""

    def compute_train_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute training kernel matrix for global representations"""
        D2_train = self._D2[:ntrain, :ntrain]
        # K_train = np.exp(-D2_train / sigma**2)
        K_train = self._kernel_func(np.sqrt(D2_train) / sigma)
        return K_train

    def compute_test_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute test kernel matrix for global representations"""
        if self._X_holdout is None:
            raise ValueError("Holdout data not provided")

        D2_test = self._D2_test[:, :ntrain]
        K_test = self._kernel_func(np.sqrt(D2_test) / sigma)
        return K_test

    def length_scale(self, ntrain: int) -> float:
        # get median nearest neighbor distance for first ntrain points
        section = self._D2[:ntrain, :ntrain].copy()
        np.fill_diagonal(section, np.inf)
        nnvals = np.amin(section, axis=0)
        return np.median(nnvals) ** 0.5

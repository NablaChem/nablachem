import numpy as np
from .kernels import Kernel


class KernelMatrix:
    """Base class for kernel matrix computation and management"""

    def __init__(self, X: np.ndarray, kernel_func: Kernel):
        """Initialize kernel matrix with training data

        Args:
            X: Training data
            kernel_func: Kernel function to use
        """
        self._X = X
        self._kernel_func = kernel_func
        self._batch_size = 100

        self._D2 = self._dist_squared(X)

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


class _LocalKernelMatrix(KernelMatrix):
    """Kernel matrix for local (atom-based) representations with approximation cache"""

    def __init__(
        self,
        X: np.ndarray,
        train_counts: np.ndarray,
        kernel_func: Kernel,
        elemental: bool = False,
        nuclear_charges: np.ndarray = None,
    ):
        """Initialize local kernel matrix

        Args:
            X: Concatenated atom representations
            train_counts: Number of atoms per training molecule
            elemental: If True, zero contributions from cross-element atom pairs
            nuclear_charges: Nuclear charges per train atom (required when elemental=True)
        """
        if elemental and nuclear_charges is None:
            raise ValueError("nuclear_charges is required when elemental=True")

        super().__init__(X, kernel_func)
        self._train_counts = train_counts
        self._elemental = elemental

        if elemental:
            self._nuclear_charges = np.asarray(nuclear_charges, dtype=float)

        self._approx_fail_sigma = dict()
        self._d_train_cache: dict[tuple, np.ndarray] = {}
        self._kernel_func.approx_prepare(train_counts, self._X)

    def length_scale(self, ntrain: int) -> float:
        # get median nearest neighbor distance for first ntrain points
        nentries = sum(self._train_counts[:ntrain])
        section = self._D2[:nentries, :nentries]
        # diagonal is always 0 (self-distance), so the 2nd-smallest per column
        # is the nearest-neighbour distance — np.partition avoids a full copy
        nnvals = np.partition(section, 1, axis=0)[1]
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

    def compute_test_kernel_matrix(
        self,
        sigma: float,
        ntrain: int,
        X_batch: np.ndarray,
        counts_batch: np.ndarray,
        nc_batch: np.ndarray = None,
    ) -> np.ndarray:
        """Compute test kernel matrix for local representations"""
        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)

        # Compute atomic kernel between test and train
        D2 = self._dist_squared(self._X[:natoms], X_batch)
        if self._elemental:
            z_train = self._nuclear_charges[:natoms]
            cross = nc_batch[:, None] != z_train[None, :]
            D2 = D2.copy()
            D2[cross] = np.inf
        K_atom = self._kernel_func.exact(np.sqrt(D2) / sigma)
        K_test = self.aggregate_atomic_kernel(K_atom, counts_batch, atom_counts_A)

        # Compute normalization factors
        # For training: get unnormalized diagonal (cached per sigma/ntrain)
        cache_key = (sigma, ntrain)
        if cache_key not in self._d_train_cache:
            K_atom_train = self._kernel_func.exact(
                np.sqrt(self._D2[:natoms, :natoms]) / sigma
            )
            K_train_unnorm = self.aggregate_atomic_kernel(
                K_atom_train, atom_counts_A, atom_counts_A
            )
            self._d_train_cache[cache_key] = np.sqrt(np.diag(K_train_unnorm))
        d_train_sqrt = self._d_train_cache[cache_key]

        # For test: self-kernel diagonal computed inline for this batch
        test_self_list = []
        atom_start = 0
        for count in counts_batch:
            d2 = self._dist_squared(
                X_batch[atom_start : atom_start + count],
                X_batch[atom_start : atom_start + count],
            )
            if self._elemental:
                z = nc_batch[atom_start : atom_start + count]
                d2[z[:, None] != z[None, :]] = np.inf
            test_self_list.append(d2)
            atom_start += count

        d_test = np.sqrt(
            [
                self._kernel_func.exact(np.sqrt(test_self_dist) / sigma).sum()
                for test_self_dist in test_self_list
            ]
        )

        # Apply normalization
        K_test /= np.outer(d_test, d_train_sqrt)

        return K_test


class ElementalKernelMatrix(_LocalKernelMatrix):

    def __init__(self, *args, **kwargs):
        kwargs["elemental"] = True
        super().__init__(*args, **kwargs)


class LocalKernelMatrix(_LocalKernelMatrix):
    def __init__(self, *args, **kwargs):
        kwargs["elemental"] = False
        super().__init__(*args, **kwargs)


class GlobalKernelMatrix(KernelMatrix):
    """Kernel matrix for global (molecule-based) representations"""

    def __init__(self, X: np.ndarray, kernel_func: Kernel):
        super().__init__(X, kernel_func)
        self._D2 = self._dist_squared(X)

    def compute_train_kernel_matrix(self, sigma: float, ntrain: int) -> np.ndarray:
        """Compute training kernel matrix for global representations"""
        D2_train = self._D2[:ntrain, :ntrain]
        # K_train = np.exp(-D2_train / sigma**2)
        K_train = self._kernel_func(np.sqrt(D2_train) / sigma)
        return K_train

    def compute_test_kernel_matrix(
        self,
        sigma: float,
        ntrain: int,
        X_batch: np.ndarray,
        counts_batch=None,
        nc_batch=None,
    ) -> np.ndarray:
        """Compute test kernel matrix for global representations"""
        D2_test = self._dist_squared(self._X[:ntrain], X_batch)
        K_test = self._kernel_func(np.sqrt(D2_test) / sigma)
        return K_test

    def length_scale(self, ntrain: int) -> float:
        # get median nearest neighbor distance for first ntrain points
        section = self._D2[:ntrain, :ntrain].copy()
        np.fill_diagonal(section, np.inf)
        nnvals = np.amin(section, axis=0)
        return np.median(nnvals) ** 0.5

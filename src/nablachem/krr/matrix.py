import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


class KernelMatrix:
    """Base class for kernel matrix computation and management"""

    def __init__(
        self, X: np.ndarray, X_holdout: np.ndarray = None, kernel_func: callable = None
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
        super().__init__(X, X_holdout)
        self._train_counts = train_counts
        self._holdout_counts = holdout_counts

        # Local approximation cache will be built on demand
        self._cache_built = False

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
        self.build_cache(max_molecules=len(train_counts))
        self._approx_fail_sigma = dict()

    def build_cache(self, max_molecules: int):
        """Build local approximation cache for efficient kernel computation"""
        self._local_grid = 1.5 ** np.linspace(-15, 15, 100)
        self._local_ymax = 20.0

        # Chebyshev polynomial coefficients for exp approximation
        cheby_p = [
            1.2783333716342860e-01,
            -2.4252536276891104e-01,
            2.0716160177307505e-01,
            -1.5966072205968104e-01,
            1.1136516853726638e-01,
            -7.0568587229867946e-02,
            4.0796581307398473e-02,
            -2.1612689660989802e-02,
            1.0538815782012821e-02,
            -4.7505844097694584e-03,
            1.9877638444287539e-03,
            -7.7505672091777230e-04,
            2.8263905844468264e-04,
            -9.6722980858327177e-05,
            3.1159309411000033e-05,
            -9.4769211836987947e-06,
            2.7285817731910956e-06,
            -7.4564575189212374e-07,
            1.9431609391984766e-07,
            -5.0571492223751847e-08,
            1.1357243950084354e-08,
        ]

        P = Chebyshev(cheby_p, domain=[0, self._local_ymax])
        Q = P.convert(kind=np.polynomial.Polynomial)
        self._exp_coef = Q.coef

        # Build power moments cache
        nmols = max_molecules
        atoms_per_mol = self._train_counts[:nmols]
        grid = self._local_grid
        D2 = self._D2
        npowers = len(cheby_p)
        npairs = nmols * (nmols + 1) // 2

        power_moments = np.zeros((npairs, npowers, len(grid)), dtype=np.float64)
        pair_idx = 0
        for i in range(nmols):
            for j in range(i, nmols):
                x = D2[
                    sum(atoms_per_mol[:i]) : sum(atoms_per_mol[: i + 1]),
                    sum(atoms_per_mol[:j]) : sum(atoms_per_mol[: j + 1]),
                ].flatten()
                x = np.sort(x)

                cum_moments = np.zeros((len(cheby_p), len(x) + 1))
                cum_moments[0, 1:] = np.cumsum(np.ones_like(x))
                for k in range(1, len(cheby_p)):
                    cum_moments[k, 1:] = np.cumsum(x**k)

                select_indices = np.minimum(
                    np.searchsorted(x, grid, side="right"), len(x) - 1
                )

                power_moments[pair_idx, :, :] = cum_moments[:, select_indices]
                pair_idx += 1

        self._local_power_moments = power_moments
        self._cache_built = True

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
        if not self._cache_built:
            raise RuntimeError("Cache must be built before computing kernel matrix")

        q = sigma**2
        cutoff = np.searchsorted(self._local_grid, self._local_ymax * q) - 1
        cutoff = max(0, min(cutoff, len(self._local_grid) - 1))

        moments = self._local_power_moments[:, :, cutoff]
        triu = moments * self._exp_coef
        triu /= q ** np.arange(len(self._exp_coef))
        triu = np.sum(triu, axis=1)

        # build full K matrix
        nmols = len(self._train_counts)
        K = np.zeros((nmols, nmols))
        pair_idx = 0
        for i in range(nmols):
            for j in range(i, nmols):
                K[i, j] = triu[pair_idx]
                K[j, i] = K[i, j]
                pair_idx += 1

        K_sub = K[:ntrain, :ntrain]

        # normalize
        d = np.diag(K_sub)
        d_sqrt = np.sqrt(d)
        K_sub /= np.outer(d_sqrt, d_sqrt)

        if np.max(K_sub) > 1.0 + 1e-8 or np.min(K_sub) > 0.1:
            return None
        return K_sub

    def compute_train_kernel_matrix_exact(
        self, sigma: float, ntrain: int
    ) -> np.ndarray:
        if self._X_holdout is None:
            raise ValueError("Holdout data not provided")

        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)

        # Compute atomic kernel between test and train
        K_atom = np.exp(-self._D2[:natoms, :natoms] / sigma**2)
        K_atom_train = self.aggregate_atomic_kernel(
            K_atom, atom_counts_A, atom_counts_A
        )

        d_train_sqrt = np.sqrt(np.diag(K_atom_train))

        # Apply normalization
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
        K_atom = np.exp(-self._D2_test[:, :natoms] / sigma**2)
        K_test = self.aggregate_atomic_kernel(K_atom, atom_counts_B, atom_counts_A)

        # Compute normalization factors
        # For training: get unnormalized diagonal first
        atom_counts_A = self._train_counts[:ntrain]
        natoms = sum(atom_counts_A)
        K_atom_train = np.exp(-self._D2[:natoms, :natoms] / sigma**2)
        K_train_unnorm = self.aggregate_atomic_kernel(
            K_atom_train, atom_counts_A, atom_counts_A
        )
        d_train_sqrt = np.sqrt(np.diag(K_train_unnorm))

        # For test: self-kernel diagonal
        d_test = np.sqrt(
            [
                np.exp(-test_self_dist / sigma**2).sum()
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

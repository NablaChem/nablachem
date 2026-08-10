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
        charges_arg = self._nuclear_charges if elemental else None
        self._kernel_func.approx_prepare(
            train_counts, self._X, nuclear_charges=charges_arg
        )

    def length_scale(self, ntrain: int) -> float:
        return self._kernel_func._chebytrick.length_scale(ntrain)

    @staticmethod
    def _pack_batches(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pack molecules into ~BATCH_TARGET-atom batches.

        Returns (batch_mol_edges, atom_offsets): batch P covers molecules
        batch_mol_edges[P]..batch_mol_edges[P+1], and atom_offsets is the
        prefix-sum of counts (length nmols+1).
        """
        BATCH_TARGET = 1000
        counts = np.asarray(counts)
        nmols = len(counts)
        atom_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

        batches = [0]
        cur = 0
        for m in range(nmols):
            cur += int(counts[m])
            if cur >= BATCH_TARGET:
                batches.append(m + 1)
                cur = 0
        if batches[-1] < nmols:
            batches.append(nmols)

        return np.asarray(batches, dtype=np.int64), atom_offsets

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
        counts_A = np.asarray(self._train_counts[:ntrain])
        natoms = int(counts_A.sum())
        X_A = self._X[:natoms]
        z_A = self._nuclear_charges[:natoms] if self._elemental else None

        batches, atom_off = self._pack_batches(counts_A)
        nbatches = len(batches) - 1

        K = np.zeros((ntrain, ntrain))

        for P in range(nbatches):
            iP0, iP1 = int(batches[P]), int(batches[P + 1])
            aP0, aP1 = int(atom_off[iP0]), int(atom_off[iP1])
            X_P = X_A[aP0:aP1]
            counts_P = counts_A[iP0:iP1]
            z_P = z_A[aP0:aP1] if z_A is not None else None

            D2 = self._dist_squared(X_P)
            if z_P is not None:
                D2[z_P[:, None] != z_P[None, :]] = np.inf
            K_atom_block = self._kernel_func.exact(np.sqrt(D2) / sigma)
            K[iP0:iP1, iP0:iP1] = self.aggregate_atomic_kernel(
                K_atom_block, counts_P, counts_P
            )

            for Q in range(P + 1, nbatches):
                iQ0, iQ1 = int(batches[Q]), int(batches[Q + 1])
                aQ0, aQ1 = int(atom_off[iQ0]), int(atom_off[iQ1])
                X_Q = X_A[aQ0:aQ1]
                counts_Q = counts_A[iQ0:iQ1]

                # rows: batch-P atoms, cols: batch-Q atoms
                D2 = self._dist_squared(X_Q, X_P)
                if z_A is not None:
                    z_Q = z_A[aQ0:aQ1]
                    D2[z_P[:, None] != z_Q[None, :]] = np.inf
                K_atom_block = self._kernel_func.exact(np.sqrt(D2) / sigma)
                K_block = self.aggregate_atomic_kernel(K_atom_block, counts_P, counts_Q)
                K[iP0:iP1, iQ0:iQ1] = K_block
                K[iQ0:iQ1, iP0:iP1] = K_block.T

        d_train_sqrt = np.sqrt(np.diag(K))
        K /= np.outer(d_train_sqrt, d_train_sqrt)
        self._d_train_cache[(sigma, ntrain)] = d_train_sqrt

        return K

    def compute_test_kernel_matrix(
        self,
        sigma: float,
        ntrain: int,
        X_batch: np.ndarray,
        counts_batch: np.ndarray,
        nc_batch: np.ndarray = None,
    ) -> np.ndarray:
        """Compute test kernel matrix for local representations"""
        counts_A = np.asarray(self._train_counts[:ntrain])
        natoms_A = int(counts_A.sum())
        X_A = self._X[:natoms_A]
        z_A = self._nuclear_charges[:natoms_A] if self._elemental else None

        counts_B = np.asarray(counts_batch)
        n_test_mols = len(counts_B)
        X_B = np.asarray(X_batch)
        z_B = np.asarray(nc_batch) if z_A is not None else None

        batches_A, atom_off_A = self._pack_batches(counts_A)
        batches_B, atom_off_B = self._pack_batches(counts_B)

        K_test = np.zeros((n_test_mols, ntrain))

        for P in range(len(batches_B) - 1):
            iP0, iP1 = int(batches_B[P]), int(batches_B[P + 1])
            aP0, aP1 = int(atom_off_B[iP0]), int(atom_off_B[iP1])
            X_BP = X_B[aP0:aP1]
            counts_BP = counts_B[iP0:iP1]
            z_BP = z_B[aP0:aP1] if z_B is not None else None

            for Q in range(len(batches_A) - 1):
                iQ0, iQ1 = int(batches_A[Q]), int(batches_A[Q + 1])
                aQ0, aQ1 = int(atom_off_A[iQ0]), int(atom_off_A[iQ1])
                X_AQ = X_A[aQ0:aQ1]
                counts_AQ = counts_A[iQ0:iQ1]

                # rows: test-batch-P atoms, cols: train-batch-Q atoms
                D2 = self._dist_squared(X_AQ, X_BP)
                if z_A is not None:
                    z_AQ = z_A[aQ0:aQ1]
                    D2[z_BP[:, None] != z_AQ[None, :]] = np.inf
                K_atom_block = self._kernel_func.exact(np.sqrt(D2) / sigma)
                K_test[iP0:iP1, iQ0:iQ1] = self.aggregate_atomic_kernel(
                    K_atom_block, counts_BP, counts_AQ
                )

        cache_key = (sigma, ntrain)
        if cache_key not in self._d_train_cache:
            d_train = np.empty(ntrain)
            atom_start = 0
            for i, count in enumerate(counts_A):
                count = int(count)
                atoms_i = X_A[atom_start : atom_start + count]
                d2 = self._dist_squared(atoms_i)
                if z_A is not None:
                    z_i = z_A[atom_start : atom_start + count]
                    d2[z_i[:, None] != z_i[None, :]] = np.inf
                d_train[i] = self._kernel_func.exact(np.sqrt(d2) / sigma).sum()
                atom_start += count
            self._d_train_cache[cache_key] = np.sqrt(d_train)
        d_train_sqrt = self._d_train_cache[cache_key]

        d_test = np.empty(n_test_mols)
        atom_start = 0
        for i, count in enumerate(counts_B):
            count = int(count)
            atoms_i = X_B[atom_start : atom_start + count]
            d2 = self._dist_squared(atoms_i)
            if z_B is not None:
                z_i = z_B[atom_start : atom_start + count]
                d2[z_i[:, None] != z_i[None, :]] = np.inf
            d_test[i] = self._kernel_func.exact(np.sqrt(d2) / sigma).sum()
            atom_start += count
        d_test_sqrt = np.sqrt(d_test)

        K_test /= np.outer(d_test_sqrt, d_train_sqrt)

        return K_test


class ElementalKernelMatrix(_LocalKernelMatrix):

    def __init__(self, *args, **kwargs):
        kwargs["elemental"] = True
        super().__init__(*args, **kwargs)


class LocalKernelMatrix(_LocalKernelMatrix):
    def __init__(self, *args, **kwargs):
        kwargs["elemental"] = False
        super().__init__(*args, **kwargs)


class AlchemicalKernelMatrix(_LocalKernelMatrix):
    """Local kernel with per-element-pair multiplicative weights.

    weights: dict {(Z1, Z2): float} with Z1 <= Z2, values already abs-ed.
    All pairs that appear in the data must be present — missing keys fail at
    matrix build time (KeyError → fail late by design).
    """

    def __init__(self, X, train_counts, kernel_func, nuclear_charges, weights):
        self._alch_charges = np.asarray(nuclear_charges, dtype=float)
        # Build a dense lookup table indexed by nuclear charge for fast access.
        max_Z = max(max(k) for k in weights) + 1
        self._wtable = np.zeros((max_Z, max_Z))
        for (z1, z2), w in weights.items():
            self._wtable[z1, z2] = w
            self._wtable[z2, z1] = w
        super().__init__(X, train_counts, kernel_func, elemental=False)

    def _W(self, z_row, z_col):
        return self._wtable[z_row.astype(int)[:, None], z_col.astype(int)[None, :]]

    def compute_train_kernel_matrix(self, sigma, ntrain):
        counts_A = np.asarray(self._train_counts[:ntrain])
        natoms = int(counts_A.sum())
        X_A = self._X[:natoms]
        z_A = self._alch_charges[:natoms]

        batches, atom_off = self._pack_batches(counts_A)
        nbatches = len(batches) - 1
        K = np.zeros((ntrain, ntrain))

        for P in range(nbatches):
            iP0, iP1 = int(batches[P]), int(batches[P + 1])
            aP0, aP1 = int(atom_off[iP0]), int(atom_off[iP1])
            X_P = X_A[aP0:aP1]
            z_P = z_A[aP0:aP1]
            counts_P = counts_A[iP0:iP1]

            D2 = self._dist_squared(X_P)
            K_atom = self._kernel_func.exact(np.sqrt(D2) / sigma) * self._W(z_P, z_P)
            K[iP0:iP1, iP0:iP1] = self.aggregate_atomic_kernel(
                K_atom, counts_P, counts_P
            )

            for Q in range(P + 1, nbatches):
                iQ0, iQ1 = int(batches[Q]), int(batches[Q + 1])
                aQ0, aQ1 = int(atom_off[iQ0]), int(atom_off[iQ1])
                X_Q = X_A[aQ0:aQ1]
                z_Q = z_A[aQ0:aQ1]
                counts_Q = counts_A[iQ0:iQ1]

                D2 = self._dist_squared(X_Q, X_P)
                K_atom = self._kernel_func.exact(np.sqrt(D2) / sigma) * self._W(
                    z_P, z_Q
                )
                K_block = self.aggregate_atomic_kernel(K_atom, counts_P, counts_Q)
                K[iP0:iP1, iQ0:iQ1] = K_block
                K[iQ0:iQ1, iP0:iP1] = K_block.T

        d_train_sqrt = np.sqrt(np.diag(K))
        K /= np.outer(d_train_sqrt, d_train_sqrt)
        self._d_train_cache[(sigma, ntrain)] = d_train_sqrt
        return K

    def compute_test_kernel_matrix(
        self, sigma, ntrain, X_batch, counts_batch, nc_batch=None
    ):
        counts_A = np.asarray(self._train_counts[:ntrain])
        natoms_A = int(counts_A.sum())
        X_A = self._X[:natoms_A]
        z_A = self._alch_charges[:natoms_A]

        counts_B = np.asarray(counts_batch)
        n_test_mols = len(counts_B)
        X_B = np.asarray(X_batch)
        z_B = np.asarray(nc_batch, dtype=float)

        batches_A, atom_off_A = self._pack_batches(counts_A)
        batches_B, atom_off_B = self._pack_batches(counts_B)

        K_test = np.zeros((n_test_mols, ntrain))

        for P in range(len(batches_B) - 1):
            iP0, iP1 = int(batches_B[P]), int(batches_B[P + 1])
            aP0, aP1 = int(atom_off_B[iP0]), int(atom_off_B[iP1])
            X_BP = X_B[aP0:aP1]
            z_BP = z_B[aP0:aP1]
            counts_BP = counts_B[iP0:iP1]

            for Q in range(len(batches_A) - 1):
                iQ0, iQ1 = int(batches_A[Q]), int(batches_A[Q + 1])
                aQ0, aQ1 = int(atom_off_A[iQ0]), int(atom_off_A[iQ1])
                X_AQ = X_A[aQ0:aQ1]
                z_AQ = z_A[aQ0:aQ1]
                counts_AQ = counts_A[iQ0:iQ1]

                D2 = self._dist_squared(X_AQ, X_BP)
                K_atom = self._kernel_func.exact(np.sqrt(D2) / sigma) * self._W(
                    z_BP, z_AQ
                )
                K_test[iP0:iP1, iQ0:iQ1] = self.aggregate_atomic_kernel(
                    K_atom, counts_BP, counts_AQ
                )

        cache_key = (sigma, ntrain)
        if cache_key not in self._d_train_cache:
            d_train = np.empty(ntrain)
            atom_start = 0
            for i, count in enumerate(counts_A):
                count = int(count)
                z_i = z_A[atom_start : atom_start + count]
                atoms_i = X_A[atom_start : atom_start + count]
                d2 = self._dist_squared(atoms_i)
                d_train[i] = (
                    self._kernel_func.exact(np.sqrt(d2) / sigma) * self._W(z_i, z_i)
                ).sum()
                atom_start += count
            self._d_train_cache[cache_key] = np.sqrt(d_train)
        d_train_sqrt = self._d_train_cache[cache_key]

        d_test = np.empty(n_test_mols)
        atom_start = 0
        for i, count in enumerate(counts_B):
            count = int(count)
            z_i = z_B[atom_start : atom_start + count]
            atoms_i = X_B[atom_start : atom_start + count]
            d2 = self._dist_squared(atoms_i)
            d_test[i] = (
                self._kernel_func.exact(np.sqrt(d2) / sigma) * self._W(z_i, z_i)
            ).sum()
            atom_start += count
        d_test_sqrt = np.sqrt(d_test)

        K_test /= np.outer(d_test_sqrt, d_train_sqrt)
        return K_test


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

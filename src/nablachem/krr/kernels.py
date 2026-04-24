import sys
import numpy as np
from scipy.special import gamma, kv
import inspect

from numpy.polynomial.chebyshev import Chebyshev
from numba import njit


class Kernel:
    def __init__(self):
        pass

    def __call__(self, dr, **kwargs):
        return self.exact(dr, **kwargs)


class Matern32(Kernel):
    def exact(self, dr):
        scaled_r = np.sqrt(3) * dr
        return (1 + scaled_r) * np.exp(-scaled_r)


class Matern52(Kernel):
    def exact(self, dr):
        scaled_r = np.sqrt(5) * dr
        return (1 + scaled_r + scaled_r**2 / 3) * np.exp(-scaled_r)


class MaternGeneral(Kernel):
    def exact(self, dr, nu):
        if nu == 0.5:
            return Exponential().exact(dr)
        elif nu == 1.5:
            return Matern32().exact(dr)
        elif nu == 2.5:
            return Matern52().exact(dr)
        else:
            scaled_r = np.sqrt(2 * nu) * dr

            # Handle zero values consistently for scalar and array inputs
            zero_mask = scaled_r == 0

            # For numerical stability, replace zeros with small value
            scaled_r_safe = np.where(zero_mask, 1e-10, scaled_r)

            const = (2 ** (1 - nu)) / gamma(nu)
            bessel_part = kv(nu, scaled_r_safe) * (scaled_r_safe) ** nu

            # Compute proper normalization: the value at exactly dr=0
            # Use the limit: lim_{r->0} K_ν(r) * r^ν = 2^(ν-1) * Γ(ν) for ν > 0
            norm_bessel_part = (2 ** (nu - 1)) * gamma(nu)
            norm_const = const * norm_bessel_part

            result = (const * bessel_part) / norm_const

            # Ensure k(0) = 1 exactly by setting zero positions to 1.0
            if np.isscalar(zero_mask):
                if zero_mask:
                    result = 1.0
            else:
                result = np.where(zero_mask, 1.0, result)

            return result


class RationalQuadratic(Kernel):
    def exact(self, dr, alpha):
        return (1 + dr**2 / (2 * alpha)) ** (-alpha)


class InverseMultiquadric(Kernel):
    def exact(self, dr):
        return 1 / np.sqrt(1 + dr**2)  # k(0) = 1/sqrt(1) = 1


class InverseQuadratic(Kernel):
    def exact(self, dr):
        return 1 / (1 + dr**2)  # k(0) = 1/(1) = 1


class Power(Kernel):
    def exact(self, dr, alpha):
        return (1 + dr**2) ** (-alpha)  # k(0) = 1^(-alpha) = 1


class GeneralizedCauchy(Kernel):
    def exact(self, dr, alpha, beta):
        return (1 + dr**beta) ** (-alpha / beta)  # k(0) = 1^(-alpha/beta) = 1


class _Wendland(Kernel):
    def exact(self, dr, d):
        k = self._k
        l = int(np.floor(d / 2) + k + 1)
        if k == 0:
            p = lambda r: 1
        elif k == 1:
            p = lambda r: (l + 1) * r + 1
        elif k == 2:
            p = lambda r: (l + 3) * (l + 1) * r**2 + 3 * (l + 2) * r + 3
        elif k == 3:
            p = (
                lambda r: (l + 5) * (l + 3) * (l + 1) * r**3
                + (45 + 6 * l * (l + 6)) * r**2
                + (15 * (l + 3)) * r
                + 15
            )
        elif k == 4:
            p = (
                lambda r: (l + 7) * (l + 5) * (l + 3) * (l + 1) * r**4
                + (5 * (l + 4) * (21 + 2 * l * (8 + l))) * r**3
                + (45 * (14 + l * (l + 8))) * r**2
                + (105 * (l + 4)) * r
                + 105
            )
        else:
            raise NotImplementedError()
        p0 = p(0)
        e = l + k
        return np.maximum(1 - dr, 0) ** e * p(dr) / p0


class WendlandK0(_Wendland):
    _k = 0


class WendlandK1(_Wendland):
    _k = 1


class WendlandK2(_Wendland):
    _k = 2


class WendlandK3(_Wendland):
    _k = 3


class WendlandK4(_Wendland):
    _k = 4


class WuC2(Kernel):
    def exact(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        return pos_part**4 * (4 * dr + 1)  # k(0) = 1^4 * (0 + 1) = 1


class WuC4(Kernel):
    def exact(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        return pos_part**6 * (35 * dr**2 + 18 * dr + 3) / 3


class WuC6(Kernel):
    def exact(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        return pos_part**8 * (16 * dr**3 + 25 * dr**2 + 10 * dr + 1)  # k(0) = 1


class Bump(Kernel):
    def exact(self, dr):
        inside_mask = dr < 1
        k_vals = np.zeros_like(dr) if hasattr(dr, "__len__") else 0.0
        if hasattr(dr, "__len__"):
            k_vals[inside_mask] = np.exp(-1 / (1 - dr[inside_mask] ** 2))
        else:
            if dr < 1:
                k_vals = np.exp(-1 / (1 - dr**2))
        norm_const = np.exp(-1)  # k(0) = exp(-1/(1-0)) = exp(-1)
        return k_vals / norm_const


class Sigmoid(Kernel):
    def exact(self, dr, a, b):
        arg = a - b * dr
        k_vals = 1 / (1 + np.exp(-arg))
        norm_const = 1 / (1 + np.exp(-a))  # k(0) = 1/(1+exp(-a))
        return k_vals / norm_const


class Polynomial(Kernel):
    def exact(self, dr, alpha, beta):
        k_vals = (1 + dr**2) ** alpha * np.exp(-beta * dr**2)
        norm_const = (1 + 0) ** alpha * np.exp(-beta * 0)  # k(0) = 1 * 1 = 1
        return k_vals / norm_const


@njit(inline="always")
def _grid_bin(grid, x):
    """Smallest m with grid[m] >= x. Returns len(grid) if x exceeds all bins."""
    lo = 0
    hi = len(grid)
    while lo < hi:
        mid = (lo + hi) >> 1
        if grid[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _build_power_moments(
    power_moments,
    X,
    atoms_per_mol,
    power,
    ncheby,
    grid,
    charges,
    use_elemental,
    anchor_bucket_of_mol,
    nn_per_anchor,
):
    BATCH_TARGET = 1000
    nmols = len(atoms_per_mol)
    ngrid = len(grid)

    off = np.empty(nmols + 1, dtype=np.int64)
    off[0] = 0
    for m in range(nmols):
        off[m + 1] = off[m] + atoms_per_mol[m]
    total_atoms = off[nmols]

    all_norms = np.empty(total_atoms, dtype=np.float64)
    for a in range(total_atoms):
        s = 0.0
        for f in range(X.shape[1]):
            s += X[a, f] * X[a, f]
        all_norms[a] = s

    # Pack molecules into ~BATCH_TARGET-atom batches; one BLAS call per batch pair.
    batch = np.empty(nmols + 2, dtype=np.int64)
    batch[0] = 0
    nb = 0
    cur = 0
    for m in range(nmols):
        cur += atoms_per_mol[m]
        if cur >= BATCH_TARGET:
            nb += 1
            batch[nb] = m + 1
            cur = 0
    if batch[nb] < nmols:
        nb += 1
        batch[nb] = nmols

    # NN tracking is only active for atom pairs where the "row" atom is in batch 0.
    batch0_atom_end = off[batch[1]]

    for P in range(nb):
        aP0, aP1 = off[batch[P]], off[batch[P + 1]]
        for Q in range(P, nb):
            aQ0, aQ1 = off[batch[Q]], off[batch[Q + 1]]
            G = X[aP0:aP1] @ X[aQ0:aQ1].T

            for i in range(batch[P], batch[P + 1]):
                ai, ni = off[i], atoms_per_mol[i]
                ai_l = ai - aP0
                j_lo = i if P == Q else batch[Q]
                bk_i = anchor_bucket_of_mol[i]
                for j in range(j_lo, batch[Q + 1]):
                    aj, nj = off[j], atoms_per_mol[j]
                    aj_l = aj - aQ0
                    pair_idx = i * nmols - i * (i - 1) // 2 + (j - i)
                    bk_j = anchor_bucket_of_mol[j]

                    contrib = np.zeros((ncheby, ngrid), dtype=np.float64)
                    x_max = -1.0  # sentinel; any real x_val is >= 0 after clamp
                    for b in range(ni):
                        nb_sq = all_norms[ai + b]
                        row = ai_l + b
                        for a in range(nj):
                            if use_elemental and charges[ai + b] != charges[aj + a]:
                                continue
                            d = all_norms[aj + a] + nb_sq - 2.0 * G[row, aj_l + a]
                            if d < 0.0:
                                d = 0.0
                            x_val = np.sqrt(d) if power == 1 else d
                            if x_val > x_max:
                                x_max = x_val
                            bi = _grid_bin(grid, x_val)
                            if bi < ngrid:
                                contrib[0, bi] += 1.0
                                xk = 1.0
                                for p in range(1, ncheby):
                                    xk *= x_val
                                    contrib[p, bi] += xk

                            # NN tracking (P==0 only): update per-sample-atom NN
                            # against the pool-atom's mol-anchor-bucket. Skip the
                            # identity pair (same atom).
                            if P == 0 and not (i == j and b == a):
                                s_i = ai + b
                                if d < nn_per_anchor[s_i, bk_j]:
                                    nn_per_anchor[s_i, bk_j] = d
                                s_j = aj + a
                                if s_j < batch0_atom_end:
                                    if d < nn_per_anchor[s_j, bk_i]:
                                        nn_per_anchor[s_j, bk_i] = d

                    if x_max < 0.0:
                        continue

                    # Cap-to-(n-1) quirk: exclude one copy of the pair's max.
                    bi = _grid_bin(grid, x_max)
                    if bi < ngrid:
                        contrib[0, bi] -= 1.0
                        xk = 1.0
                        for p in range(1, ncheby):
                            xk *= x_max
                            contrib[p, bi] -= xk

                    for p in range(ncheby):
                        run = 0.0
                        for mm in range(ngrid):
                            run += contrib[p, mm]
                            power_moments[pair_idx, p, mm] = run


class ExponentialToChebychev:
    def __init__(
        self,
        atoms_per_mol: np.ndarray,
        X: np.ndarray,
        power: int,
        nuclear_charges: np.ndarray = None,
    ):
        self._local_grid = 1.5 ** np.linspace(-15, 15, 20)
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
        nmols = len(atoms_per_mol)
        self._nmols = nmols
        grid = self._local_grid
        ncheby = len(cheby_p)
        npairs = nmols * (nmols + 1) // 2
        if power not in (1, 2):
            raise NotImplementedError(
                "Only power=1 (for exp(-r)) and power=2 (for exp(-r^2)) are implemented"
            )

        power_moments = np.zeros((npairs, ncheby, len(grid)), dtype=np.float64)
        if nuclear_charges is None:
            charges_arr = np.zeros(0, dtype=np.float64)
            use_elemental = False
        else:
            charges_arr = np.asarray(nuclear_charges, dtype=np.float64)
            use_elemental = True

        # Anchor setup for per-atom NN length-scale tracking (piggybacks on the
        # moments build). Anchors are powers of 2 starting at 4, capped by nmols;
        # nmols is appended as the last anchor when it isn't already a power of 2.
        atoms_per_mol_i64 = np.asarray(atoms_per_mol, dtype=np.int64)
        atom_offsets = np.concatenate(([0], np.cumsum(atoms_per_mol_i64))).astype(
            np.int64
        )
        anchors_list = []
        k = 4
        while k < nmols:
            anchors_list.append(k)
            k *= 2
        anchors_list.append(nmols)
        anchors = np.asarray(anchors_list, dtype=np.int64)
        n_anchors = len(anchors)
        # bucket[j] = smallest i with anchors[i] > j  (mol j contributes to anchor i onward)
        anchor_bucket_of_mol = np.searchsorted(anchors, np.arange(nmols), side="right")
        anchor_bucket_of_mol = anchor_bucket_of_mol.astype(np.int64)

        # Determine size of batch 0 (must match the packing inside _build_power_moments).
        BATCH_TARGET = 1000
        batch0_end_mol = 0
        cur = 0
        for m in range(nmols):
            cur += int(atoms_per_mol_i64[m])
            if cur >= BATCH_TARGET:
                batch0_end_mol = m + 1
                break
        if batch0_end_mol == 0:
            batch0_end_mol = nmols
        batch0_atom_end = int(atom_offsets[batch0_end_mol])

        nn_per_anchor = np.full((batch0_atom_end, n_anchors), np.inf, dtype=np.float64)

        _build_power_moments(
            power_moments,
            X,
            atoms_per_mol_i64,
            power,
            ncheby,
            self._local_grid,
            charges_arr,
            use_elemental,
            anchor_bucket_of_mol,
            nn_per_anchor,
        )

        self._local_power_moments = power_moments
        self._cache_built = True

        # Post-process: cumulative min across buckets, then median per anchor
        # over sample atoms belonging to mols < anchor. Runs in Python per the
        # user's preference to keep this logic outside the numba hot loop.
        sample_mol_id = np.empty(batch0_atom_end, dtype=np.int64)
        for m in range(batch0_end_mol):
            sample_mol_id[atom_offsets[m] : atom_offsets[m + 1]] = m

        nn_cum = np.minimum.accumulate(nn_per_anchor, axis=1)
        length_scale_by_anchor = np.empty(n_anchors, dtype=np.float64)
        for k_idx in range(n_anchors):
            valid = (sample_mol_id < anchors[k_idx]) & np.isfinite(nn_cum[:, k_idx])
            if valid.any():
                length_scale_by_anchor[k_idx] = float(
                    np.sqrt(np.median(nn_cum[valid, k_idx]))
                )
            else:
                length_scale_by_anchor[k_idx] = 1.0
        self._anchors = anchors
        self._length_scale_by_anchor = length_scale_by_anchor

    def length_scale(self, ntrain: int) -> float:
        """Median nearest-neighbour atomic distance heuristic.

        Returns the value at the anchor closest to ntrain in log2 space.
        """
        anchors = self._anchors
        values = self._length_scale_by_anchor
        if ntrain <= anchors[0]:
            return float(values[0])
        if ntrain >= anchors[-1]:
            return float(values[-1])
        log_ntrain = np.log2(float(ntrain))
        idx = int(np.argmin(np.abs(np.log2(anchors.astype(np.float64)) - log_ntrain)))
        return float(values[idx])

    def __call__(self, q, ntrain):
        cutoff = np.searchsorted(self._local_grid, self._local_ymax * q) - 1
        cutoff = max(0, min(cutoff, len(self._local_grid) - 1))

        moments = self._local_power_moments[:, :, cutoff]
        triu = moments * self._exp_coef
        triu /= q ** np.arange(len(self._exp_coef))
        triu = np.sum(triu, axis=1)

        # build full K matrix
        nmols = self._nmols
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


class Gaussian(Kernel):
    def exact(self, dr):
        return np.exp(-(dr**2))

    def approx_prepare(
        self,
        atoms_per_mol: np.ndarray,
        X: np.ndarray,
        nuclear_charges: np.ndarray = None,
    ):
        self._chebytrick = ExponentialToChebychev(
            atoms_per_mol, X=X, power=2, nuclear_charges=nuclear_charges
        )

    def approx(self, sigma: float, ntrain: int) -> np.ndarray:
        return self._chebytrick(sigma**2, ntrain)


class Exponential(Kernel):
    def exact(self, dr):
        return np.exp(-dr)

    def approx_prepare(
        self,
        atoms_per_mol: np.ndarray,
        X: np.ndarray,
        nuclear_charges: np.ndarray = None,
    ):
        self._chebytrick = ExponentialToChebychev(
            atoms_per_mol, X=X, power=1, nuclear_charges=nuclear_charges
        )

    def approx(self, sigma: float, ntrain: int) -> np.ndarray:
        return self._chebytrick(sigma, ntrain)


def list_available():
    """Return names of all public Kernel subclasses."""
    result = []
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if issubclass(cls, Kernel) and cls is not Kernel and not name.startswith("_"):
            result.append(name)
    return result

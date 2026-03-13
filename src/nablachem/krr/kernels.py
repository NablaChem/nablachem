import sys
import numpy as np
from scipy.special import gamma, kv
import inspect

from numpy.polynomial.chebyshev import Chebyshev


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


class ExponentialToChebychev:
    def __init__(self, atoms_per_mol: np.ndarray, Ds: np.ndarray):
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
        nmols = len(atoms_per_mol)
        self._nmols = nmols
        grid = self._local_grid
        npowers = len(cheby_p)
        npairs = nmols * (nmols + 1) // 2

        power_moments = np.zeros((npairs, npowers, len(grid)), dtype=np.float64)
        pair_idx = 0
        atoms_per_mol_cumsum = np.concatenate([[0], np.cumsum(atoms_per_mol)])
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

    def approx_prepare(self, atoms_per_mol: np.ndarray, Ds: np.ndarray):
        self._chebytrick = ExponentialToChebychev(atoms_per_mol, Ds=Ds)

    def approx(self, sigma: float, ntrain: int) -> np.ndarray:
        return self._chebytrick(sigma**2, ntrain)


class Exponential(Kernel):
    def exact(self, dr):
        return np.exp(-dr)

    def approx_prepare(self, atoms_per_mol: np.ndarray, Ds: np.ndarray):
        self._chebytrick = ExponentialToChebychev(atoms_per_mol, Ds=np.sqrt(Ds))

    def approx(self, sigma: float, ntrain: int) -> np.ndarray:
        return self._chebytrick(sigma, ntrain)


def list_available():
    """Return names of all public Kernel subclasses."""
    result = []
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if issubclass(cls, Kernel) and cls is not Kernel and not name.startswith("_"):
            result.append(name)
    return result

import numpy as np
from scipy.special import gamma, kv
import inspect

from numpy.polynomial.chebyshev import Chebyshev


class Kernel:
    def __init__(self):
        pass

    def gaussian(self, dr):
        return np.exp(-(dr**2))

    def exponential(self, dr):
        return np.exp(-dr)

    def matern_32(self, dr):
        scaled_r = np.sqrt(3) * dr
        return (1 + scaled_r) * np.exp(-scaled_r)

    def matern_52(self, dr):
        scaled_r = np.sqrt(5) * dr
        return (1 + scaled_r + scaled_r**2 / 3) * np.exp(-scaled_r)

    def matern_general(self, dr, nu):
        if nu == 0.5:
            return self.exponential(dr)
        elif nu == 1.5:
            return self.matern_32(dr)
        elif nu == 2.5:
            return self.matern_52(dr)
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

    def rational_quadratic(self, dr, alpha):
        return (1 + dr**2 / (2 * alpha)) ** (-alpha)

    def inverse_multiquadric(self, dr):
        return 1 / np.sqrt(1 + dr**2)  # k(0) = 1/sqrt(1) = 1

    def inverse_quadratic(self, dr):
        return 1 / (1 + dr**2)  # k(0) = 1/(1) = 1

    def power(self, dr, alpha):
        return (1 + dr**2) ** (-alpha)  # k(0) = 1^(-alpha) = 1

    def generalized_cauchy(self, dr, alpha, beta):
        return (1 + dr**beta) ** (-alpha / beta)  # k(0) = 1^(-alpha/beta) = 1

    def _wendland(self, dr: float, k: int, d: int) -> float:
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

    def wendland_k0(self, dr, d):
        return self._wendland(dr, k=0, d=d)

    def wendland_k1(self, dr, d):
        return self._wendland(dr, k=1, d=d)

    def wendland_k2(self, dr, d):
        return self._wendland(dr, k=2, d=d)

    def wendland_k3(self, dr, d):
        return self._wendland(dr, k=3, d=d)

    def wendland_k4(self, dr, d):
        return self._wendland(dr, k=4, d=d)

    def wu_c2(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**4 * (4 * dr + 1)
        return k_vals  # k(0) = 1^4 * (0 + 1) = 1

    def wu_c4(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**6 * (35 * dr**2 + 18 * dr + 3) / 3
        return k_vals

    def wu_c6(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**8 * (16 * dr**3 + 25 * dr**2 + 10 * dr + 1)
        return k_vals  # k(0) = 1^8 * (0 + 0 + 0 + 1) = 1

    def bump(self, dr):
        inside_mask = dr < 1
        k_vals = np.zeros_like(dr) if hasattr(dr, "__len__") else 0.0
        if hasattr(dr, "__len__"):
            k_vals[inside_mask] = np.exp(-1 / (1 - dr[inside_mask] ** 2))
        else:
            if dr < 1:
                k_vals = np.exp(-1 / (1 - dr**2))
        norm_const = np.exp(-1)  # k(0) = exp(-1/(1-0)) = exp(-1)
        return k_vals / norm_const

    def sigmoid_kernel(self, dr, a, b):
        arg = a - b * dr
        k_vals = 1 / (1 + np.exp(-arg))
        norm_const = 1 / (1 + np.exp(-a))  # k(0) = 1/(1+exp(-a))
        return k_vals / norm_const

    def polynomial_kernel(self, dr, alpha, beta):
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
        for i in range(nmols):
            for j in range(i, nmols):
                x = Ds[
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


class GaussianKernel(Kernel):
    def exact(self, dr):
        return np.exp(-(dr**2))

    def approx_prepare(self, atoms_per_mol: np.ndarray, Ds: np.ndarray):
        self._chebytrick = ExponentialToChebychev(atoms_per_mol, Ds=Ds)

    def approx(self, sigma: float, ntrain: int) -> np.ndarray:
        return self._chebytrick(sigma**2, ntrain)


class ExponentialKernel(Kernel):
    def exact(self, dr):
        return np.exp(-dr)

    def approx_prepare(self, atoms_per_mol: np.ndarray, Ds: np.ndarray):
        self._chebytrick = ExponentialToChebychev(atoms_per_mol, Ds=np.sqrt(Ds))

    def approx(self, sigma: float, ntrain: int) -> np.ndarray:
        return self._chebytrick(sigma, ntrain)


def list_available():
    """Return string names of all kernel methods from the Kernel class."""
    kernel_methods = []
    for name, method in inspect.getmembers(Kernel, predicate=inspect.isfunction):
        if not name.startswith("_") and name != "list_available":
            kernel_methods.append(name)
    return kernel_methods

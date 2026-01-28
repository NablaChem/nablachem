import numpy as np
from scipy.special import gamma, kv


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

    def wendland_c0(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        return pos_part  # k(0) = 1

    def wendland_c2(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**3 * (3 * dr + 1)
        return k_vals  # k(0) = 1^3 * (0 + 1) = 1

    def wendland_c4(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**5 * (8 * dr**2 + 5 * dr + 1)
        return k_vals  # k(0) = 1^5 * (0 + 0 + 1) = 1

    def wendland_c6(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**7 * (21 * dr**3 + 19 * dr**2 + 7 * dr + 1)
        return k_vals  # k(0) = 1^7 * (0 + 0 + 0 + 1) = 1

    def wu_c2(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**4 * (4 * dr + 1)
        return k_vals  # k(0) = 1^4 * (0 + 1) = 1

    def wu_c4(self, dr):
        pos_part = np.maximum(1 - dr, 0)
        k_vals = pos_part**6 * (35 * dr**2 + 18 * dr + 3) / 3
        norm_const = 3 / 3  # k(0) = 1^6 * 3 / 3 = 1
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

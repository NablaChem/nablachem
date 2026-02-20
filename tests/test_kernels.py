import pytest
import numpy as np
from scipy.linalg import eigvals
from nablachem.krr.kernels import Kernel


# Define reasonable hyperparameters for each kernel
KERNEL_PARAMS = {
    "gaussian": {},
    "exponential": {},
    "matern_32": {},
    "matern_52": {},
    "matern_general": {"nu": 1.5},
    "rational_quadratic": {"alpha": 2.0},
    "inverse_multiquadric": {},
    "inverse_quadratic": {},
    "power": {"alpha": 1.0},
    "generalized_cauchy": {"alpha": 1.0, "beta": 2.0},
    "wendland_k0": {"d": 1},
    "wendland_k1": {"d": 1},
    "wendland_k2": {"d": 1},
    "wendland_k3": {"d": 1},
    "wendland_k4": {"d": 1},
    "wu_c2": {},
    "wu_c4": {},
    "wu_c6": {},
    "bump": {},
    "sigmoid_kernel": {"a": 2.0, "b": 1.0},
    "polynomial_kernel": {"alpha": -1.0, "beta": 1.0},
}


@pytest.fixture
def kernel():
    return Kernel()


class TestKernels:
    def setup_method(self):
        # Define which kernels have compact support (exactly zero beyond some point)
        self.compact_kernels = [
            "wendland_k0",
            "wendland_k1",
            "wendland_k2",
            "wendland_k3",
            "wendland_k4",
            "wu_c2",
            "wu_c4",
            "wu_c6",
            "bump",
        ]

        # Define kernel-specific validity ranges (beyond which they become invalid/problematic)
        self.kernel_ranges = {
            "bump": 1.0,  # undefined for dr >= 1
            "wendland_k0": np.inf,  # valid everywhere, but zero for dr > 1
            "wendland_k1": np.inf,
            "wendland_k2": np.inf,
            "wendland_k3": np.inf,
            "wendland_k4": np.inf,
            "wu_c2": np.inf,
            "wu_c4": np.inf,
            "wu_c6": np.inf,
        }

    def get_test_range(self, kernel_name):
        """Get appropriate test range for each kernel"""
        if kernel_name in self.kernel_ranges:
            max_r = self.kernel_ranges[kernel_name]
            if max_r < np.inf:
                return np.linspace(0, max_r * 0.99, 100)  # Stay just within valid range

        # For compact support kernels, test around their support
        if kernel_name in self.compact_kernels:
            return np.linspace(0, 2.0, 200)  # Test beyond support boundary

        # For other kernels, use extended range to stress test
        return np.linspace(0, 20, 500)

    @pytest.mark.parametrize("kernel_name", list(KERNEL_PARAMS.keys()))
    def test_exact_normalization(self, kernel, kernel_name):
        """Test that k(0) = 1 for all kernels"""
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        k_zero = method(0.0, **params)

        # Use tight tolerance for numerical precision
        assert (
            abs(k_zero - 1.0) < 1e-14
        ), f"{kernel_name}: k(0) = {k_zero:.16f}, expected 1.0"

    @pytest.mark.parametrize("kernel_name", list(KERNEL_PARAMS.keys()))
    @pytest.mark.parametrize("n", [3, 5, 8, 12, 15])
    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
    def test_positive_semi_definiteness(self, kernel, kernel_name, n, seed):
        """Test PSD property with multiple matrix sizes and seeds"""
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        np.random.seed(seed)

        # Generate random points in higher dimensions for stress testing
        points = np.random.randn(n, 4) * 5  # Larger scale

        # Compute distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])

        try:
            gram_matrix = method(dist_matrix, **params)

            # Check for any NaN or inf values first
            assert np.all(
                np.isfinite(gram_matrix)
            ), f"{kernel_name}: Non-finite values in Gram matrix (n={n}, seed={seed})"

            # Check symmetry
            assert np.allclose(
                gram_matrix, gram_matrix.T, atol=1e-12
            ), f"{kernel_name}: Gram matrix not symmetric (n={n}, seed={seed})"

            # Check PSD with tight tolerance
            eigenvals_real = np.real(eigvals(gram_matrix))
            min_eigenval = np.min(eigenvals_real)

            assert (
                min_eigenval >= -1e-12
            ), f"{kernel_name}: Negative eigenvalue {min_eigenval:.2e} (n={n}, seed={seed})"

        except Exception as e:
            pytest.fail(f"{kernel_name} failed PSD test (n={n}, seed={seed}): {str(e)}")

    @pytest.mark.parametrize("kernel_name", list(KERNEL_PARAMS.keys()))
    def test_boundedness(self, kernel, kernel_name):
        """Test boundedness within each kernel's valid domain"""
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        test_points = self.get_test_range(kernel_name)

        try:
            k_vals = method(test_points, **params)

            # Check for non-finite values
            assert np.all(
                np.isfinite(k_vals)
            ), f"{kernel_name}: Non-finite values found"

            # For non-compact kernels, check strict boundedness [0,1]
            if kernel_name not in self.compact_kernels:
                assert np.all(
                    k_vals >= -1e-14
                ), f"{kernel_name}: Negative values found: min = {np.min(k_vals)}"
                assert np.all(
                    k_vals <= 1 + 1e-14
                ), f"{kernel_name}: Values > 1 found: max = {np.max(k_vals)}"

            # For compact kernels, values beyond support should be exactly zero
            if kernel_name in self.compact_kernels:
                beyond_support = test_points > 1.0
                if np.any(beyond_support):
                    vals_beyond = k_vals[beyond_support]
                    assert np.all(
                        np.abs(vals_beyond) < 1e-14
                    ), f"{kernel_name}: Non-zero values beyond support: {vals_beyond[vals_beyond != 0]}"

        except Exception as e:
            pytest.fail(f"{kernel_name} failed boundedness test: {str(e)}")

    def test_known_kernels_only(self, kernel):
        """Test that all kernel methods are in the known parameters list"""
        # Get all callable methods from the kernel object
        kernel_methods = [
            method
            for method in dir(kernel)
            if not method.startswith("_") and callable(getattr(kernel, method))
        ]

        # Check that all found methods are in our known kernel parameters
        unknown_kernels = [
            method for method in kernel_methods if method not in KERNEL_PARAMS
        ]

        assert (
            len(unknown_kernels) == 0
        ), f"Found unknown kernel methods not in KERNEL_PARAMS: {unknown_kernels}"

    @pytest.mark.parametrize(
        "kernel_name",
        [
            "gaussian",
            "exponential",
            "matern_32",
            "matern_52",
            "matern_general",
            "rational_quadratic",
            "inverse_multiquadric",
            "inverse_quadratic",
            "power",
            "generalized_cauchy",
            "sigmoid_kernel",
        ],
    )
    def test_monotonic_decrease(self, kernel, kernel_name):
        """Test monotonic decrease for kernels that have this property"""
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        # Use dense sampling to catch any violations
        test_points = np.linspace(0, 10, 1000)

        try:
            k_vals = method(test_points, **params)

            # Check monotonic decrease with numerical tolerance
            diffs = np.diff(k_vals)
            violations = np.sum(diffs > 1e-14)

            assert violations == 0, (
                f"{kernel_name}: {violations} monotonicity violations found. "
                f"Max increase: {np.max(diffs)}"
            )

        except Exception as e:
            pytest.fail(f"{kernel_name} failed monotonicity test: {str(e)}")

    @pytest.mark.parametrize(
        "kernel_name,params",
        [
            ("matern_general", {"nu": 0.01}),
            ("matern_general", {"nu": 10.0}),
            ("matern_general", {"nu": 0.5001}),
            ("rational_quadratic", {"alpha": 0.001}),
            ("rational_quadratic", {"alpha": 100.0}),
            ("power", {"alpha": 0.1}),
            ("power", {"alpha": 10.0}),
            ("generalized_cauchy", {"alpha": 0.1, "beta": 0.1}),
            ("generalized_cauchy", {"alpha": 10.0, "beta": 3.0}),
            ("generalized_cauchy", {"alpha": 1.0, "beta": 0.01}),
        ],
    )
    def test_extreme_parameter_stability(self, kernel, kernel_name, params):
        """Test kernels with extreme parameter values for numerical stability"""
        method = getattr(kernel, kernel_name)
        test_points = np.array([0.0, 0.001, 0.1, 1.0, 5.0, 10.0])

        try:
            k_vals = method(test_points, **params)

            # Check for numerical pathologies
            assert np.all(
                np.isfinite(k_vals)
            ), f"{kernel_name} with {params}: Non-finite values"

            assert (
                abs(k_vals[0] - 1.0) < 1e-12
            ), f"{kernel_name} with {params}: k(0) = {k_vals[0]}, not 1.0"

            assert np.all(k_vals >= 0), f"{kernel_name} with {params}: Negative values"

        except Exception as e:
            pytest.fail(f"{kernel_name} with {params} failed stability test: {str(e)}")

    @pytest.mark.parametrize(
        "kernel_name,decay_type",
        [
            ("gaussian", "fast"),
            ("exponential", "fast"),
            ("matern_32", "fast"),
            ("matern_52", "fast"),
            ("matern_general", "fast"),
            ("sigmoid_kernel", "fast"),
            ("polynomial_kernel", "fast"),
            ("rational_quadratic", "slow"),
            ("inverse_multiquadric", "slow"),
            ("inverse_quadratic", "slow"),
            ("power", "slow"),
            ("generalized_cauchy", "slow"),
        ],
    )
    def test_large_distance_behavior(self, kernel, kernel_name, decay_type):
        """Test kernel behavior at large distances"""
        large_distances = np.array([50, 100, 500, 1000])
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        try:
            k_vals = method(large_distances, **params)

            if decay_type == "fast":
                # Fast decay kernels should approach zero quickly
                assert np.all(
                    k_vals < 0.01
                ), f"{kernel_name}: Values too large at infinity: {k_vals}"

                # Should be monotonically decreasing even at large distances
                assert np.all(
                    np.diff(k_vals) <= 1e-10
                ), f"{kernel_name}: Not monotonic at large distances"

            else:  # slow decay
                # Slow decay kernels should still decay, but allow larger values
                assert np.all(
                    k_vals < 0.1
                ), f"{kernel_name}: Values too large even for slow decay: {k_vals}"

                # Should be monotonically decreasing
                assert np.all(
                    np.diff(k_vals) <= 1e-10
                ), f"{kernel_name}: Not monotonic at large distances"

                # Should show clear decay from first to last point
                assert (
                    k_vals[-1] < k_vals[0] * 0.9
                ), f"{kernel_name}: Insufficient decay from {k_vals[0]:.6f} to {k_vals[-1]:.6f}"

        except Exception as e:
            pytest.fail(f"{kernel_name} failed infinity test: {str(e)}")

    @pytest.mark.parametrize(
        "kernel_name",
        [
            "wendland_k0",
            "wendland_k1",
            "wendland_k2",
            "wendland_k3",
            "wendland_k4",
            "wu_c2",
            "wu_c4",
            "wu_c6",
        ],
    )
    def test_compact_support_zeros(self, kernel, kernel_name):
        """Test that compact support kernels are zero outside their support"""
        test_points_outside = np.array([1.1, 1.5, 2.0, 5.0, 10.0])

        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        k_vals = method(test_points_outside, **params)

        # Should be exactly zero outside support
        assert np.all(
            k_vals == 0.0
        ), f"{kernel_name}: Non-zero values outside support: {k_vals[k_vals != 0]}"

    @pytest.mark.parametrize("kernel_name", list(KERNEL_PARAMS.keys()))
    @pytest.mark.parametrize("scalar_val", [0.0, 0.5, 1.0, 2.5, 5.0])
    def test_array_vs_scalar_consistency(self, kernel, kernel_name, scalar_val):
        """Test that scalar and array inputs give identical results"""
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        try:
            # Scalar input
            k_scalar = method(scalar_val, **params)

            # Array input with single element
            k_array_single = method(np.array([scalar_val]), **params)

            # Array input with multiple identical elements
            k_array_multiple = method(np.array([scalar_val] * 5), **params)

            # All should give identical results
            assert (
                abs(k_scalar - k_array_single[0]) < 1e-15
            ), f"{kernel_name}: Scalar vs single-array mismatch at {scalar_val}"

            assert np.all(
                np.abs(k_array_multiple - k_scalar) < 1e-15
            ), f"{kernel_name}: Array consistency failure at {scalar_val}"

        except Exception as e:
            pytest.fail(
                f"{kernel_name} failed consistency test at {scalar_val}: {str(e)}"
            )

    @pytest.mark.parametrize(
        "kernel_name", [k for k in KERNEL_PARAMS.keys() if k != "bump"]
    )
    def test_multidimensional_array_element_wise_consistency(self, kernel, kernel_name):
        """Test that 2D and 3D arrays are handled element-wise consistently"""
        # Test values - use safe ranges for all kernels
        base_values = [0.0, 0.1, 0.5, 1.0, 2.0]

        # Create test arrays of different shapes with the same elements
        array_1d = np.array(base_values)
        array_2d = np.array([[0.0, 0.1, 0.5], [1.0, 2.0, 0.0]])
        array_3d = np.array([[[0.0, 0.1], [0.5, 1.0]], [[2.0, 0.0], [0.1, 0.5]]])

        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        try:
            # Test 1D array
            result_1d = method(array_1d, **params)
            assert (
                result_1d.shape == array_1d.shape
            ), f"{kernel_name}: 1D shape mismatch"

            # Test 2D array
            result_2d = method(array_2d, **params)
            assert (
                result_2d.shape == array_2d.shape
            ), f"{kernel_name}: 2D shape mismatch"

            # Test 3D array
            result_3d = method(array_3d, **params)
            assert (
                result_3d.shape == array_3d.shape
            ), f"{kernel_name}: 3D shape mismatch"

            # Test element-wise consistency: each element should match scalar result
            for i in range(len(base_values)):
                scalar_result = method(base_values[i], **params)
                array_result = result_1d[i]

                assert (
                    abs(scalar_result - array_result) < 1e-14
                ), f"{kernel_name}: 1D element {i} inconsistent - scalar={scalar_result}, array={array_result}"

            # Test 2D element-wise consistency
            flat_2d = array_2d.flatten()
            flat_result_2d = result_2d.flatten()

            for i, val in enumerate(flat_2d):
                scalar_result = method(val, **params)
                assert (
                    abs(scalar_result - flat_result_2d[i]) < 1e-14
                ), f"{kernel_name}: 2D element at flat index {i} inconsistent"

            # Test 3D element-wise consistency
            flat_3d = array_3d.flatten()
            flat_result_3d = result_3d.flatten()

            for i, val in enumerate(flat_3d):
                scalar_result = method(val, **params)
                assert (
                    abs(scalar_result - flat_result_3d[i]) < 1e-14
                ), f"{kernel_name}: 3D element at flat index {i} inconsistent"

            # Test that reshaping doesn't affect results
            reshaped_input = array_1d.reshape(1, -1)
            reshaped_result = method(reshaped_input, **params)

            assert np.allclose(
                result_1d, reshaped_result.flatten(), atol=1e-15
            ), f"{kernel_name}: Reshaping affects results"

        except Exception as e:
            pytest.fail(f"{kernel_name} failed multidimensional test: {str(e)}")

    @pytest.mark.parametrize(
        "kernel_name,params",
        [
            ("matern_general", {"nu": 0.01}),
            ("rational_quadratic", {"alpha": 0.001}),
            ("generalized_cauchy", {"alpha": 1.0, "beta": 0.01}),
        ],
    )
    @pytest.mark.parametrize(
        "array_name,test_array",
        [
            ("1d_zero", np.array([0.0])),
            ("1d_mixed", np.array([0.0, 0.1, 0.5])),
            ("2d_with_zero", np.array([[0.0, 0.1], [0.5, 1.0]])),
            ("3d_with_zero", np.array([[[0.0, 0.1]], [[0.5, 1.0]]])),
        ],
    )
    def test_extreme_parameter_multidimensional_consistency(
        self, kernel, kernel_name, params, array_name, test_array
    ):
        """Test multidimensional consistency with extreme parameters"""
        method = getattr(kernel, kernel_name)

        try:
            # Get scalar reference value
            scalar_result = method(0.0, **params)

            result = method(test_array, **params)

            # Extract the value at the zero position
            if array_name == "1d_zero":
                zero_value = result[0]
            elif array_name == "1d_mixed":
                zero_value = result[0]  # First element is 0.0
            elif array_name == "2d_with_zero":
                zero_value = result[0, 0]  # First element is 0.0
            elif array_name == "3d_with_zero":
                zero_value = result[0, 0, 0]  # First element is 0.0

            # Check for consistency between scalar and array evaluation
            assert abs(scalar_result - zero_value) < 1e-12, (
                f"{kernel_name} with {params}: Inconsistency in {array_name} - "
                f"scalar={scalar_result:.10f}, array_zero={zero_value:.10f}"
            )

        except Exception as e:
            pytest.fail(
                f"{kernel_name} with {params} failed extreme multidimensional test: {str(e)}"
            )

    @pytest.mark.parametrize(
        "kernel_name", [k for k in KERNEL_PARAMS.keys() if k != "bump"]
    )
    def test_broadcasting_behavior(self, kernel, kernel_name):
        """Test that kernels handle broadcasting correctly"""
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        try:
            # Test broadcasting with different shaped arrays
            scalar = 1.0
            array_1d = np.array([1.0, 2.0, 3.0])
            array_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

            # All should work without error
            result_scalar = method(scalar, **params)
            result_1d = method(array_1d, **params)
            result_2d = method(array_2d, **params)

            # Check shapes are preserved
            assert (
                np.isscalar(result_scalar) or result_scalar.shape == ()
            ), f"{kernel_name}: Scalar result wrong shape"
            assert (
                result_1d.shape == array_1d.shape
            ), f"{kernel_name}: 1D result wrong shape"
            assert (
                result_2d.shape == array_2d.shape
            ), f"{kernel_name}: 2D result wrong shape"

            # Check that equivalent elements give same results
            assert (
                abs(result_scalar - result_1d[0]) < 1e-15
            ), f"{kernel_name}: Broadcasting inconsistency scalar vs 1D"
            assert (
                abs(result_scalar - result_2d[0, 0]) < 1e-15
            ), f"{kernel_name}: Broadcasting inconsistency scalar vs 2D"

        except Exception as e:
            pytest.fail(f"{kernel_name} failed broadcasting test: {str(e)}")

    @pytest.mark.parametrize("kernel_name", list(KERNEL_PARAMS.keys()))
    def test_numerical_regression_grid(self, kernel, kernel_name):
        """Test that kernel values on grid [0.1, 0.2, ..., 1.0] remain unchanged"""
        method = getattr(kernel, kernel_name)
        params = KERNEL_PARAMS[kernel_name]

        # Test grid from 0.1 to 1.0 with 0.1 step
        test_grid = np.arange(0.1, 1.1, 0.1)

        # Expected values computed once and stored for regression testing
        # These values should never change unless there's an intentional kernel modification
        expected_values = {
            "gaussian": [0.9900498337491681, 0.9607894391523232, 0.9139311852712282,
                        0.8521437889662113, 0.7788007830714049, 0.697676326071031,
                        0.612626394184416, 0.5272924240430485, 0.4448580662229411,
                        0.36787944117144233],
            "exponential": [0.9048374180359595, 0.8187307530779818, 0.7408182206817179,
                           0.6703200460356393, 0.6065306597126334, 0.5488116360940264,
                           0.49658530379140947, 0.44932896411722156, 0.4065696597405991,
                           0.36787944117144233],
            "matern_32": [0.9866245648897064, 0.9522113614772348, 0.9037901598990385,
                         0.846686862268961, 0.7848876539574506, 0.7213304237515004,
                         0.6581373763165839, 0.5968001712848926, 0.538326805558179,
                         0.4833577245965077],
            "matern_52": [0.9917592361711776, 0.9679861199640714, 0.930965342775005,
                         0.8835453294128766, 0.8286491424181255, 0.7689931092516178,
                         0.7069426819040977, 0.64445632646425, 0.5830835509043292,
                         0.5239941088318203],
            "matern_general": [0.9866245648897064, 0.9522113614772348, 0.9037901598990385,
                              0.846686862268961, 0.7848876539574506, 0.7213304237515004,
                              0.6581373763165839, 0.5968001712848926, 0.538326805558179,
                              0.4833577245965077],
            "rational_quadratic": [0.9950186876947283, 0.9802960494069208, 0.9564744352317359,
                                  0.9245562130177514, 0.8858131487889274, 0.84167999326656,
                                  0.7936468569104319, 0.7431629013079665, 0.6915599431191946,
                                  0.64],
            "inverse_multiquadric": [0.9950371902099893, 0.9805806756909201, 0.9578262852211513,
                                    0.9284766908852592, 0.8944271909999159, 0.8574929257125441,
                                    0.8192319205190405, 0.7808688094430303, 0.7432941462471663,
                                    0.7071067811865475],
            "inverse_quadratic": [0.9900990099009901, 0.9615384615384615, 0.9174311926605504,
                                 0.8620689655172413, 0.8, 0.7352941176470588,
                                 0.6711409395973154, 0.6097560975609756, 0.5524861878453039,
                                 0.5],
            "power": [0.9900990099009901, 0.9615384615384615, 0.9174311926605504,
                     0.8620689655172413, 0.8, 0.7352941176470588,
                     0.6711409395973154, 0.6097560975609756, 0.5524861878453039,
                     0.5],
            "generalized_cauchy": [0.9950371902099892, 0.9805806756909201, 0.9578262852211513,
                                  0.9284766908852593, 0.8944271909999159, 0.8574929257125442,
                                  0.8192319205190404, 0.7808688094430303, 0.7432941462471663,
                                  0.7071067811865476],
            "wendland_k0": [0.9, 0.8, 0.7, 0.6, 0.5, 0.3999999999999999,
                           0.29999999999999993, 0.19999999999999996, 0.09999999999999998, 0.0],
            "wendland_k1": [0.9477000000000001, 0.8192000000000003, 0.6516999999999998,
                           0.47519999999999996, 0.3125, 0.17919999999999991,
                           0.08369999999999994, 0.027199999999999985, 0.003699999999999998, 0.0],
            "wendland_k2": [0.9329742000000002, 0.7602176000000003, 0.5411853999999999,
                           0.33281279999999996, 0.171875, 0.07045119999999994,
                           0.02046059999999998, 0.0032383999999999967, 0.00011979999999999988, 0.0],
            "wendland_k3": [0.9140253759000002, 0.6979321856000003, 0.44281907109999985,
                           0.22909962239999998, 0.0927734375, 0.027158118399999973,
                           0.004901723099999993, 0.00037775359999999954, 3.7998999999999947e-06, 0.0],
            "wendland_k4": [0.8944653558463717, 0.6388457069421718, 0.3607116692913999,
                           0.15684119838719995, 0.049769810267857144, 0.010400915221942841,
                           0.0011663791505999977, 4.376154697142849e-05, 1.196927714285712e-07, 0.0],
            "wu_c2": [0.9185399999999999, 0.7372800000000002, 0.5282199999999999,
                     0.33696, 0.1875, 0.08703999999999994,
                     0.030779999999999974, 0.006719999999999994, 0.0004599999999999995, 0.0],
            "wu_c4": [0.9123070500000002, 0.6990506666666669, 0.4529486499999999,
                     0.24572159999999996, 0.10807291666666667, 0.03604479999999996,
                     0.00795824999999999, 0.0008490666666666657, 1.584999999999998e-05, 0.0],
            "wu_c6": [0.9754386978600003, 0.6925634764800004, 0.3852040028199999,
                     0.16836470783999996, 0.0556640625, 0.012750684159999979,
                     0.001688670179999997, 8.497151999999987e-05, 4.1913999999999924e-07, 0.0],
            "bump": [0.9899498337660452, 0.9591894571091382, 0.9058322914025678,
                    0.8265654376242381, 0.7165313105737893, 0.5697828247309229,
                    0.38259269556636644, 0.16901331540606596, 0.014077776007559552, 0.0],
            "sigmoid_kernel": [0.987618541644215, 0.9742867642904025, 0.9599654177527794,
                              0.9446198289440934, 0.9282211494963358, 0.9107476723245048,
                              0.8921861830498622, 0.8725333027482136, 0.851796769486964,
                              0.8299965984314522],
            "polynomial_kernel": [0.9802473601476912, 0.9238359991849261, 0.8384689773130534,
                                 0.7346067146260442, 0.623040626457124, 0.5129972985816404,
                                 0.41115865381504424, 0.3215197707579564, 0.2457779371397465,
                                 0.18393972058572117]
        }

        try:
            computed_values = method(test_grid, **params)
            expected = expected_values[kernel_name]

            # Use appropriate tolerance for numerical precision
            tolerance = 1e-14

            for i, (computed, expected_val) in enumerate(zip(computed_values, expected)):
                assert abs(computed - expected_val) < tolerance, (
                    f"{kernel_name}: Value mismatch at grid point {test_grid[i]:.1f}. "
                    f"Expected {expected_val:.16f}, got {computed:.16f}, "
                    f"diff = {abs(computed - expected_val):.2e}"
                )

        except Exception as e:
            pytest.fail(f"{kernel_name} failed numerical regression test: {str(e)}")

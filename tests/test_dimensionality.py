import numpy as np
import pytest
from nablachem.dimensionality import Estimator

# CO example from your script
CO_grad = np.array(
    [
        -14.6484047,
        -22.2426119,
        1.61029745e-16,
        -2.34101813e-16,
        0.0212353699,
        -1.61029745e-16,
        2.34101813e-16,
        -0.0212353699,
    ]
)
CO_Hess = np.array(
    [
        [
            -1.01902924,
            0.5060515,
            1.666574e-16,
            -1.2600557e-16,
            -0.263126131,
            -1.666574e-16,
            1.2600557e-16,
            0.263126131,
        ],
        [
            0.5060515,
            -1.755712,
            2.20070795e-16,
            1.82402176e-16,
            -0.055175395,
            -2.20070795e-16,
            -1.82402176e-16,
            0.055175395,
        ],
        [
            1.666574e-16,
            2.20070795e-16,
            -0.0101874277,
            -6.96725771e-16,
            4.45826706e-16,
            0.0101874277,
            6.26688006e-16,
            -4.45826707e-16,
        ],
        [
            -1.2600557e-16,
            1.82402176e-16,
            -6.24953282e-16,
            -0.0101874277,
            -3.90736933e-16,
            6.88052153e-16,
            0.0101874277,
            3.90736935e-16,
        ],
        [
            -0.263126131,
            -0.055175395,
            4.45826707e-16,
            -3.90736935e-16,
            1.60960967,
            -4.45826706e-16,
            3.90736933e-16,
            -1.60960967,
        ],
        [
            -1.666574e-16,
            -2.20070795e-16,
            0.0101874277,
            6.88052153e-16,
            -4.45826706e-16,
            -0.0101874277,
            -7.11433721e-16,
            4.45826706e-16,
        ],
        [
            1.2600557e-16,
            -1.82402176e-16,
            6.26688006e-16,
            0.0101874277,
            3.90736933e-16,
            -5.83825558e-16,
            -0.0101874277,
            -3.90736933e-16,
        ],
        [
            0.263126131,
            0.055175395,
            -4.45826707e-16,
            3.90736935e-16,
            -1.60960967,
            4.45826707e-16,
            -3.90736935e-16,
            1.60960967,
        ],
    ]
)


@pytest.fixture
def co_id_instance():
    natoms = len(CO_grad) // 4
    scaling_groups = [True] * natoms + [False] * (3 * natoms)
    return Estimator(
        CO_grad, CO_Hess, CO_grad, CO_Hess, dt=1e-6, scaling_groups=scaling_groups
    )


def test_getID_outputs_expected_keys(co_id_instance):
    result = co_id_instance.getID()
    assert isinstance(result, dict)
    assert "ID" in result and "Error" in result and "natoms" in result
    assert len(result["ID"]) == len(result["Error"])


def test_constant_input_raises():
    g = np.zeros(8)
    h = np.zeros((8, 8))
    natoms = 2
    scaling_groups = [True] * natoms + [False] * (3 * natoms)
    with pytest.raises(NotImplementedError):
        Estimator(g, h, g, h, dt=1e-6, scaling_groups=scaling_groups).getID()


def test_random_input_runs():
    g = np.random.rand(8)
    h = np.random.rand(8, 8)
    natoms = len(g) // 4
    scaling_groups = [True] * natoms + [False] * (3 * natoms)
    id_instance = Estimator(g, h, g, h, dt=1e-6, scaling_groups=scaling_groups)
    result = id_instance.getID()
    assert isinstance(result["ID"], list)
    assert len(result["Error"]) == len(result["ID"])


def test_gradient_projection_check():
    g = np.random.rand(8)
    h = np.eye(8)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    scaling_groups = [True, True] + [False] * 6
    id_instance = Estimator(g, h, g, h, dt=1e-6, scaling_groups=scaling_groups)
    score = id_instance._gradient_check([0, 1], eigenvalues, eigenvectors)
    assert 0 <= score <= len(g)

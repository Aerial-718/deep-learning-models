import numpy as np

from common.gradcheck import check_gradient, numerical_gradient, relative_error


def test_numerical_gradient_matches_cubic_derivative() -> None:
    x = np.asarray([[-1.5, 0.2], [0.7, 2.0]], dtype=np.float64)
    numerical = numerical_gradient(lambda: float(np.sum(x**3)), x)
    analytic = 3.0 * x**2
    result = check_gradient("cubic", analytic, numerical)
    assert result.passed, result


def test_relative_error_handles_all_zero_arrays() -> None:
    zeros = np.zeros((2, 3))
    assert relative_error(zeros, zeros) == 0.0


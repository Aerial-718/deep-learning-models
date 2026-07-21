import numpy as np
import pytest

from common.gradcheck import check_gradient, numerical_gradient
from lstm.numpy_impl import init_parameters, sequence_backward, sequence_forward

pytestmark = pytest.mark.core


def test_sequence_backward_matches_finite_differences() -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(2, 2, 2)).astype(np.float64)
    h0 = rng.normal(size=(2, 2)).astype(np.float64)
    c0 = rng.normal(size=(2, 2)).astype(np.float64)
    params = init_parameters(2, 2, rng)
    outputs, (h_last, c_last), cache = sequence_forward(x, (h0, c0), params)
    doutputs = rng.normal(size=outputs.shape)
    dh_last = rng.normal(size=h_last.shape)
    dc_last = rng.normal(size=c_last.shape)
    dx, (dh0, dc0), gradients = sequence_backward(
        doutputs, cache, dh_last=dh_last, dc_last=dc_last
    )

    def objective() -> float:
        current_outputs, (current_h, current_c), _ = sequence_forward(x, (h0, c0), params)
        return float(
            np.sum(current_outputs * doutputs)
            + np.sum(current_h * dh_last)
            + np.sum(current_c * dc_last)
        )

    checks = [
        check_gradient("x", dx, numerical_gradient(objective, x)),
        check_gradient("h0", dh0, numerical_gradient(objective, h0)),
        check_gradient("c0", dc0, numerical_gradient(objective, c0)),
    ]
    for name, parameter in params.items():
        checks.append(
            check_gradient(name, gradients[name], numerical_gradient(objective, parameter))
        )
    assert all(result.passed for result in checks), checks


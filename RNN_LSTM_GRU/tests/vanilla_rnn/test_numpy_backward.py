import numpy as np
import pytest

from common.gradcheck import check_gradient, numerical_gradient
from vanilla_rnn.numpy_impl import (
    clip_gradients,
    init_parameters,
    sequence_backward,
    sequence_forward,
)

pytestmark = pytest.mark.core


def test_sequence_backward_matches_finite_differences() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(2, 3, 2)).astype(np.float64)
    h0 = rng.normal(size=(2, 2)).astype(np.float64)
    params = init_parameters(2, 2, rng)
    outputs, h_last, cache = sequence_forward(x, h0, params)
    doutputs = rng.normal(size=outputs.shape)
    dh_last = rng.normal(size=h_last.shape)
    dx, dh0, gradients = sequence_backward(doutputs, cache, dh_last)

    def objective() -> float:
        current_outputs, current_last, _ = sequence_forward(x, h0, params)
        return float(np.sum(current_outputs * doutputs) + np.sum(current_last * dh_last))

    checks = [
        check_gradient("x", dx, numerical_gradient(objective, x)),
        check_gradient("h0", dh0, numerical_gradient(objective, h0)),
    ]
    for name, parameter in params.items():
        checks.append(
            check_gradient(name, gradients[name], numerical_gradient(objective, parameter))
        )
    assert all(result.passed for result in checks), checks


def test_global_norm_clipping_preserves_direction_and_input() -> None:
    gradients = {
        "a": np.asarray([3.0, 4.0]),
        "b": np.asarray([0.0, 12.0]),
    }
    originals = {name: value.copy() for name, value in gradients.items()}
    clipped, norm = clip_gradients(gradients, max_norm=6.5)
    assert np.isclose(norm, 13.0)
    assert np.isclose(np.sqrt(sum(np.sum(value**2) for value in clipped.values())), 6.5)
    for name in gradients:
        np.testing.assert_array_equal(gradients[name], originals[name])
        nonzero = gradients[name] != 0
        np.testing.assert_allclose(clipped[name][nonzero] / gradients[name][nonzero], 0.5)

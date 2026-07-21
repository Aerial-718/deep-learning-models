import numpy as np
import pytest

from vanilla_rnn.numpy_impl import init_parameters, sequence_forward, step_forward

pytestmark = pytest.mark.core


def test_init_shapes_dtype_and_reproducibility() -> None:
    first = init_parameters(3, 5, np.random.default_rng(42))
    second = init_parameters(3, 5, np.random.default_rng(42))
    assert set(first) == {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}
    assert first["weight_ih"].shape == (5, 3)
    assert first["weight_hh"].shape == (5, 5)
    assert first["bias_ih"].shape == first["bias_hh"].shape == (5,)
    for name in first:
        assert first[name].dtype == np.float64
        np.testing.assert_array_equal(first[name], second[name])


def test_step_matches_fixed_example() -> None:
    params = {
        "weight_ih": np.asarray([[0.2, -0.1], [0.3, 0.4]]),
        "weight_hh": np.asarray([[0.1, 0.2], [-0.2, 0.1]]),
        "bias_ih": np.asarray([0.01, -0.02]),
        "bias_hh": np.asarray([0.03, 0.04]),
    }
    x = np.asarray([[0.5, -1.0], [1.0, 0.25]])
    h = np.asarray([[0.1, 0.2], [-0.3, 0.4]])
    actual, cache = step_forward(x, h, params)
    expected = np.tanh(
        x @ params["weight_ih"].T
        + params["bias_ih"]
        + h @ params["weight_hh"].T
        + params["bias_hh"]
    )
    np.testing.assert_allclose(actual, expected)
    assert isinstance(cache, dict)


def test_sequence_matches_repeated_steps() -> None:
    rng = np.random.default_rng(4)
    params = init_parameters(3, 2, rng)
    x = rng.normal(size=(2, 4, 3))
    h0 = rng.normal(size=(2, 2))
    outputs, h_last, cache = sequence_forward(x, h0, params)
    expected_steps = []
    h = h0
    for time in range(x.shape[1]):
        h, _ = step_forward(x[:, time], h, params)
        expected_steps.append(h)
    expected = np.stack(expected_steps, axis=1)
    np.testing.assert_allclose(outputs, expected)
    np.testing.assert_allclose(h_last, expected[:, -1])
    assert len(cache) == x.shape[1]


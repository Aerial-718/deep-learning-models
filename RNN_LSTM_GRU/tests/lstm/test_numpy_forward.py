import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lstm.numpy_impl import init_parameters, sequence_forward, step_forward

pytestmark = pytest.mark.core


def test_init_shapes_dtype_and_reproducibility() -> None:
    first = init_parameters(3, 5, np.random.default_rng(42))
    second = init_parameters(3, 5, np.random.default_rng(42))
    assert set(first) == {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}
    assert first["weight_ih"].shape == (20, 3)
    assert first["weight_hh"].shape == (20, 5)
    assert first["bias_ih"].shape == first["bias_hh"].shape == (20,)
    for name in first:
        assert first[name].dtype == np.float64
        np.testing.assert_array_equal(first[name], second[name])


def test_step_matches_official_lstm_cell() -> None:
    rng = np.random.default_rng(8)
    params = init_parameters(3, 2, rng)
    x = rng.normal(size=(2, 3))
    h = rng.normal(size=(2, 2))
    c = rng.normal(size=(2, 2))
    actual_h, actual_c, cache = step_forward(x, (h, c), params)

    official = torch.nn.LSTMCell(3, 2).double()
    with torch.no_grad():
        for name, parameter in official.named_parameters():
            parameter.copy_(torch.from_numpy(params[name]))
    expected_h, expected_c = official(
        torch.from_numpy(x), (torch.from_numpy(h), torch.from_numpy(c))
    )
    np.testing.assert_allclose(actual_h, expected_h.detach().numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_c, expected_c.detach().numpy(), rtol=1e-10, atol=1e-10)
    assert isinstance(cache, dict)


def test_sequence_matches_repeated_steps() -> None:
    rng = np.random.default_rng(18)
    params = init_parameters(3, 2, rng)
    x = rng.normal(size=(2, 4, 3))
    h0 = rng.normal(size=(2, 2))
    c0 = rng.normal(size=(2, 2))
    outputs, (h_last, c_last), cache = sequence_forward(x, (h0, c0), params)
    steps = []
    h, c = h0, c0
    for time in range(x.shape[1]):
        h, c, _ = step_forward(x[:, time], (h, c), params)
        steps.append(h)
    np.testing.assert_allclose(outputs, np.stack(steps, axis=1))
    np.testing.assert_allclose(h_last, h)
    np.testing.assert_allclose(c_last, c)
    assert len(cache) == x.shape[1]


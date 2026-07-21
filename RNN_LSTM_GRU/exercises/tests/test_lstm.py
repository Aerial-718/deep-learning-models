import numpy as np
import torch

from exercises.lstm.language_model import CharLanguageModel
from exercises.lstm.numpy_impl import init_parameters, sequence_backward, sequence_forward, step_forward
from exercises.lstm.torch_impl import ManualLSTMCell, ManualRecurrentLayer
from lstm.numpy_impl import init_parameters as reference_init
from lstm.numpy_impl import sequence_backward as reference_backward
from lstm.numpy_impl import sequence_forward as reference_forward
from lstm.numpy_impl import step_forward as reference_step


def test_l01_init() -> None:
    params = init_parameters(3, 4, np.random.default_rng(42))
    assert params["weight_ih"].shape == (16, 3)
    assert params["weight_hh"].shape == (16, 4)
    assert all(value.dtype == np.float64 for value in params.values())


def test_l02_step() -> None:
    rng = np.random.default_rng(2)
    params = reference_init(3, 4, rng)
    x = rng.normal(size=(2, 3))
    state = (rng.normal(size=(2, 4)), rng.normal(size=(2, 4)))
    actual_h, actual_c, _ = step_forward(x, state, params)
    expected_h, expected_c, _ = reference_step(x, state, params)
    np.testing.assert_allclose(actual_h, expected_h)
    np.testing.assert_allclose(actual_c, expected_c)


def test_l03_sequence() -> None:
    rng = np.random.default_rng(3)
    params = reference_init(3, 4, rng)
    x = rng.normal(size=(2, 5, 3))
    state = (rng.normal(size=(2, 4)), rng.normal(size=(2, 4)))
    actual = sequence_forward(x, state, params)
    expected = reference_forward(x, state, params)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1][0], expected[1][0])
    np.testing.assert_allclose(actual[1][1], expected[1][1])


def test_l05_backward() -> None:
    rng = np.random.default_rng(5)
    params = reference_init(2, 2, rng)
    x = rng.normal(size=(2, 2, 2))
    state = (rng.normal(size=(2, 2)), rng.normal(size=(2, 2)))
    outputs, _, cache = sequence_forward(x, state, params)
    _, _, reference_cache = reference_forward(x, state, params)
    doutputs = rng.normal(size=outputs.shape)
    dh, dc = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    actual = sequence_backward(doutputs, cache, dh, dc)
    expected = reference_backward(doutputs, reference_cache, dh, dc)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1][0], expected[1][0])
    np.testing.assert_allclose(actual[1][1], expected[1][1])
    for name in params:
        np.testing.assert_allclose(actual[2][name], expected[2][name])


def test_l06_cell() -> None:
    manual = ManualLSTMCell(3, 4).double()
    official = torch.nn.LSTMCell(3, 4).double()
    official.load_state_dict(manual.state_dict())
    x = torch.randn(2, 3, dtype=torch.float64)
    state = (torch.randn(2, 4, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64))
    actual = manual(x, state)
    expected = official(x, state)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_l07_layer() -> None:
    outputs, (h, c) = ManualRecurrentLayer(3, 4)(torch.randn(2, 5, 3))
    assert outputs.shape == (2, 5, 4)
    torch.testing.assert_close(h, outputs[:, -1])
    assert c.shape == (2, 4)


def test_l08_language_model() -> None:
    logits, (h, c) = CharLanguageModel(7, 3, 4)(torch.randint(0, 7, (2, 5)))
    assert logits.shape == (2, 5, 7)
    assert h.shape == c.shape == (2, 4)


def test_l09_generate() -> None:
    model = CharLanguageModel(7, 3, 4)
    prefix = torch.tensor([[1, 2]])
    generated = model.generate(prefix, 3, top_k=1)
    assert generated.shape == (1, 5)
    torch.testing.assert_close(generated[:, :2], prefix)


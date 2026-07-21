import numpy as np
import torch

from exercises.gru.language_model import CharLanguageModel
from exercises.gru.numpy_impl import init_parameters, sequence_backward, sequence_forward, step_forward
from exercises.gru.torch_impl import ManualGRUCell, ManualRecurrentLayer
from gru.numpy_impl import init_parameters as reference_init
from gru.numpy_impl import sequence_backward as reference_backward
from gru.numpy_impl import sequence_forward as reference_forward
from gru.numpy_impl import step_forward as reference_step


def test_g01_init() -> None:
    params = init_parameters(3, 4, np.random.default_rng(42))
    assert params["weight_ih"].shape == (12, 3)
    assert params["weight_hh"].shape == (12, 4)
    assert all(value.dtype == np.float64 for value in params.values())


def test_g02_step() -> None:
    rng = np.random.default_rng(2)
    params = reference_init(3, 4, rng)
    x, h = rng.normal(size=(2, 3)), rng.normal(size=(2, 4))
    actual, _ = step_forward(x, h, params)
    expected, _ = reference_step(x, h, params)
    np.testing.assert_allclose(actual, expected)


def test_g03_sequence() -> None:
    rng = np.random.default_rng(3)
    params = reference_init(3, 4, rng)
    x, h = rng.normal(size=(2, 5, 3)), rng.normal(size=(2, 4))
    actual = sequence_forward(x, h, params)
    expected = reference_forward(x, h, params)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_g05_backward() -> None:
    rng = np.random.default_rng(5)
    params = reference_init(2, 2, rng)
    x, h = rng.normal(size=(2, 2, 2)), rng.normal(size=(2, 2))
    outputs, _, cache = sequence_forward(x, h, params)
    _, _, reference_cache = reference_forward(x, h, params)
    doutputs, dh = rng.normal(size=outputs.shape), rng.normal(size=(2, 2))
    actual = sequence_backward(doutputs, cache, dh)
    expected = reference_backward(doutputs, reference_cache, dh)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
    for name in params:
        np.testing.assert_allclose(actual[2][name], expected[2][name])


def test_g06_cell() -> None:
    manual = ManualGRUCell(3, 4).double()
    official = torch.nn.GRUCell(3, 4).double()
    official.load_state_dict(manual.state_dict())
    x, h = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)
    torch.testing.assert_close(manual(x, h), official(x, h))


def test_g07_layer() -> None:
    outputs, state = ManualRecurrentLayer(3, 4)(torch.randn(2, 5, 3))
    assert outputs.shape == (2, 5, 4)
    torch.testing.assert_close(state, outputs[:, -1])


def test_g08_language_model() -> None:
    logits, state = CharLanguageModel(7, 3, 4)(torch.randint(0, 7, (2, 5)))
    assert logits.shape == (2, 5, 7)
    assert state.shape == (2, 4)


def test_g09_generate() -> None:
    model = CharLanguageModel(7, 3, 4)
    prefix = torch.tensor([[1, 2]])
    generated = model.generate(prefix, 3, top_k=1)
    assert generated.shape == (1, 5)
    torch.testing.assert_close(generated[:, :2], prefix)


import numpy as np
import torch

from exercises.vanilla_rnn.language_model import CharLanguageModel
from exercises.vanilla_rnn.numpy_impl import (
    clip_gradients,
    init_parameters,
    sequence_backward,
    sequence_forward,
    step_forward,
)
from exercises.vanilla_rnn.torch_impl import ManualRNNCell, ManualRecurrentLayer
from vanilla_rnn.numpy_impl import init_parameters as reference_init
from vanilla_rnn.numpy_impl import sequence_backward as reference_backward
from vanilla_rnn.numpy_impl import sequence_forward as reference_forward
from vanilla_rnn.numpy_impl import step_forward as reference_step


def test_v01_init() -> None:
    params = init_parameters(3, 4, np.random.default_rng(42))
    assert {name: value.shape for name, value in params.items()} == {
        "weight_ih": (4, 3), "weight_hh": (4, 4), "bias_ih": (4,), "bias_hh": (4,)
    }
    assert all(value.dtype == np.float64 for value in params.values())


def test_v02_step() -> None:
    rng = np.random.default_rng(2)
    params = reference_init(3, 4, rng)
    x, h = rng.normal(size=(2, 3)), rng.normal(size=(2, 4))
    actual, _ = step_forward(x, h, params)
    expected, _ = reference_step(x, h, params)
    np.testing.assert_allclose(actual, expected)


def test_v03_sequence() -> None:
    rng = np.random.default_rng(3)
    params = reference_init(3, 4, rng)
    x, h0 = rng.normal(size=(2, 5, 3)), rng.normal(size=(2, 4))
    actual, actual_last, _ = sequence_forward(x, h0, params)
    expected, expected_last, _ = reference_forward(x, h0, params)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_last, expected_last)


def test_v05_backward() -> None:
    rng = np.random.default_rng(5)
    params = reference_init(2, 3, rng)
    x, h0 = rng.normal(size=(2, 3, 2)), rng.normal(size=(2, 3))
    outputs, _, cache = sequence_forward(x, h0, params)
    _, _, reference_cache = reference_forward(x, h0, params)
    doutputs, dh_last = rng.normal(size=outputs.shape), rng.normal(size=(2, 3))
    actual = sequence_backward(doutputs, cache, dh_last)
    expected = reference_backward(doutputs, reference_cache, dh_last)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
    for name in params:
        np.testing.assert_allclose(actual[2][name], expected[2][name])


def test_v06_clip() -> None:
    clipped, norm = clip_gradients({"a": np.array([3.0, 4.0])}, 2.5)
    assert np.isclose(norm, 5.0)
    np.testing.assert_allclose(clipped["a"], np.array([1.5, 2.0]))


def test_v07_cell() -> None:
    manual = ManualRNNCell(3, 4).double()
    official = torch.nn.RNNCell(3, 4).double()
    official.load_state_dict(manual.state_dict())
    x, h = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)
    torch.testing.assert_close(manual(x, h), official(x, h))


def test_v08_layer() -> None:
    outputs, state = ManualRecurrentLayer(3, 4)(torch.randn(2, 5, 3))
    assert outputs.shape == (2, 5, 4)
    torch.testing.assert_close(state, outputs[:, -1])


def test_v09_language_model() -> None:
    logits, state = CharLanguageModel(7, 3, 4)(torch.randint(0, 7, (2, 5)))
    assert logits.shape == (2, 5, 7)
    assert state.shape == (2, 4)


def test_v10_generate() -> None:
    model = CharLanguageModel(7, 3, 4)
    prefix = torch.tensor([[1, 2]])
    generated = model.generate(prefix, 3, top_k=1)
    assert generated.shape == (1, 5)
    torch.testing.assert_close(generated[:, :2], prefix)


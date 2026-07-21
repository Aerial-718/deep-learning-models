import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common.modeling import build_language_model, build_recurrent_layer
from gru.numpy_impl import init_parameters as init_gru
from gru.numpy_impl import sequence_forward as gru_forward
from lstm.numpy_impl import init_parameters as init_lstm
from lstm.numpy_impl import sequence_forward as lstm_forward
from vanilla_rnn.numpy_impl import init_parameters as init_rnn
from vanilla_rnn.numpy_impl import sequence_forward as rnn_forward


@pytest.mark.parametrize("model_name", ["vanilla", "lstm", "gru"])
def test_default_state_inherits_dtype_and_one_step_matches_final_state(model_name: str) -> None:
    layer = build_recurrent_layer(model_name, input_size=3, hidden_size=4).double()
    x = torch.randn(2, 1, 3, dtype=torch.float64)
    outputs, final_state = layer(x)
    assert outputs.dtype == torch.float64
    if model_name == "lstm":
        h_last, c_last = final_state
        torch.testing.assert_close(h_last, outputs[:, 0])
        assert c_last.shape == (2, 4)
    else:
        torch.testing.assert_close(final_state, outputs[:, 0])


@pytest.mark.parametrize("model_name", ["vanilla", "lstm", "gru"])
def test_sequence_layer_rejects_bad_initial_state(model_name: str) -> None:
    layer = build_recurrent_layer(model_name, input_size=3, hidden_size=4)
    x = torch.randn(2, 3, 3)
    bad = torch.zeros(1, 4)
    with pytest.raises(ValueError):
        layer(x, (bad, bad) if model_name == "lstm" else bad)


@pytest.mark.parametrize("model_name", ["vanilla", "lstm", "gru"])
def test_generation_with_zero_new_tokens_preserves_prefix(model_name: str) -> None:
    model = build_language_model(model_name, vocabulary_size=5, embedding_size=3, hidden_size=4)
    prefix = torch.tensor([[0, 1, 2], [2, 3, 4]])
    generated = model.generate(prefix, new_tokens=0)
    assert generated is not prefix
    torch.testing.assert_close(generated, prefix)
    with pytest.raises(ValueError):
        model.generate(prefix[:, :0], new_tokens=1)


def test_numpy_sequence_implementations_reject_empty_time_axis() -> None:
    rng = np.random.default_rng(42)
    x = np.empty((2, 0, 3), dtype=np.float64)
    h0 = np.zeros((2, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        rnn_forward(x, h0, init_rnn(3, 4, rng))
    with pytest.raises(ValueError):
        lstm_forward(x, (h0, h0.copy()), init_lstm(3, 4, rng))
    with pytest.raises(ValueError):
        gru_forward(x, h0, init_gru(3, 4, rng))


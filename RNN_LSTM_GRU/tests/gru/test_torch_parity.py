import pytest

torch = pytest.importorskip("torch")

from gru.torch_impl import ManualGRUCell, ManualRecurrentLayer

pytestmark = pytest.mark.core


def test_manual_cell_matches_official_forward_and_gradients() -> None:
    torch.manual_seed(9)
    manual = ManualGRUCell(3, 4).double()
    official = torch.nn.GRUCell(3, 4).double()
    official.load_state_dict(manual.state_dict())
    x_manual = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    h_manual = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    x_official = x_manual.detach().clone().requires_grad_(True)
    h_official = h_manual.detach().clone().requires_grad_(True)

    output_manual = manual(x_manual, h_manual)
    output_official = official(x_official, h_official)
    torch.testing.assert_close(output_manual, output_official)
    upstream = torch.randn_like(output_manual)
    output_manual.backward(upstream)
    output_official.backward(upstream)
    torch.testing.assert_close(x_manual.grad, x_official.grad)
    torch.testing.assert_close(h_manual.grad, h_official.grad)
    for manual_parameter, official_parameter in zip(manual.parameters(), official.parameters()):
        torch.testing.assert_close(manual_parameter.grad, official_parameter.grad)


def test_sequence_layer_is_batch_first() -> None:
    layer = ManualRecurrentLayer(3, 4)
    x = torch.randn(2, 5, 3)
    outputs, final_state = layer(x)
    assert outputs.shape == (2, 5, 4)
    assert final_state.shape == (2, 4)
    torch.testing.assert_close(final_state, outputs[:, -1])


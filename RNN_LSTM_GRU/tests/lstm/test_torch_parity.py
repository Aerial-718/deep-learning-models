import pytest

torch = pytest.importorskip("torch")

from lstm.torch_impl import ManualLSTMCell, ManualRecurrentLayer

pytestmark = pytest.mark.core


def test_manual_cell_matches_official_forward_and_gradients() -> None:
    torch.manual_seed(7)
    manual = ManualLSTMCell(3, 4).double()
    official = torch.nn.LSTMCell(3, 4).double()
    official.load_state_dict(manual.state_dict())
    x_manual = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    h_manual = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    c_manual = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    x_official = x_manual.detach().clone().requires_grad_(True)
    h_official = h_manual.detach().clone().requires_grad_(True)
    c_official = c_manual.detach().clone().requires_grad_(True)

    h_out_manual, c_out_manual = manual(x_manual, (h_manual, c_manual))
    h_out_official, c_out_official = official(x_official, (h_official, c_official))
    torch.testing.assert_close(h_out_manual, h_out_official)
    torch.testing.assert_close(c_out_manual, c_out_official)
    dh = torch.randn_like(h_out_manual)
    dc = torch.randn_like(c_out_manual)
    torch.autograd.backward((h_out_manual, c_out_manual), (dh, dc))
    torch.autograd.backward((h_out_official, c_out_official), (dh, dc))
    torch.testing.assert_close(x_manual.grad, x_official.grad)
    torch.testing.assert_close(h_manual.grad, h_official.grad)
    torch.testing.assert_close(c_manual.grad, c_official.grad)
    for manual_parameter, official_parameter in zip(manual.parameters(), official.parameters()):
        torch.testing.assert_close(manual_parameter.grad, official_parameter.grad)


def test_sequence_layer_returns_both_final_states() -> None:
    layer = ManualRecurrentLayer(3, 4)
    x = torch.randn(2, 5, 3)
    outputs, (h_last, c_last) = layer(x)
    assert outputs.shape == (2, 5, 4)
    assert h_last.shape == c_last.shape == (2, 4)
    torch.testing.assert_close(h_last, outputs[:, -1])


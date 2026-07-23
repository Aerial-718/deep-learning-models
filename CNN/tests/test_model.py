import pytest
import torch
from torch import nn

from cnn.model import LeNet


def test_model_output_shape_and_parameter_count() -> None:
    model = LeNet()
    output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 61_706


def test_model_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="28×28"):
        LeNet()(torch.randn(2, 1, 32, 32))


def test_optimizer_updates_parameters() -> None:
    torch.manual_seed(0)
    model = LeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.features[0].weight.detach().clone()
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 1, 28, 28)), torch.arange(4))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.features[0].weight)


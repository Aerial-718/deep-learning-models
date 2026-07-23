from pathlib import Path

import torch

from cnn.model import LeNet
from cnn.utils import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = LeNet()
    optimizer = torch.optim.Adam(model.parameters())
    inputs = torch.randn(2, 1, 28, 28)
    expected = model(inputs).detach()
    path = tmp_path / "model.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=3,
        best_accuracy=0.9,
        config={"seed": 7},
        history=[{"epoch": 3}],
    )
    restored = LeNet()
    restored_optimizer = torch.optim.Adam(restored.parameters())
    payload = load_checkpoint(
        path, restored, torch.device("cpu"), restored_optimizer
    )

    assert payload["epoch"] == 3
    assert payload["best_accuracy"] == 0.9
    assert torch.allclose(expected, restored(inputs))
    assert restored_optimizer.state_dict()["param_groups"]


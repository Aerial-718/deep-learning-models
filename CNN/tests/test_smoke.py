from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn.engine import evaluate_model, train_one_epoch
from cnn.model import LeNet
from cnn.predict import predict_paths
from cnn.utils import load_checkpoint, save_checkpoint


def test_offline_train_save_load_predict_smoke(tmp_path: Path) -> None:
    torch.manual_seed(11)
    device = torch.device("cpu")
    inputs = torch.rand(8, 1, 28, 28)
    targets = torch.arange(8) % 10
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    model = LeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_metrics = train_one_epoch(model, loader, criterion, optimizer, device)
    eval_metrics = evaluate_model(
        model, loader, criterion, device, include_confusion=True
    )
    assert train_metrics.sample_count == 8
    assert eval_metrics.confusion_matrix is not None
    assert int(eval_metrics.confusion_matrix.sum()) == 8

    checkpoint_path = tmp_path / "smoke.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        best_accuracy=eval_metrics.accuracy,
        config={},
        history=[],
    )
    restored = LeNet()
    load_checkpoint(checkpoint_path, restored, device)

    image_path = tmp_path / "sample.png"
    Image.new("L", (28, 28), color=0).save(image_path)
    result = predict_paths(restored, [image_path], device)[0]
    assert result["prediction"] in {str(index) for index in range(10)}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


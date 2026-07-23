"""训练和评估循环。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    sample_count: int
    confusion_matrix: torch.Tensor | None = None


def _finalize_metrics(
    loss_sum: float,
    correct: int,
    sample_count: int,
    confusion_matrix: torch.Tensor | None = None,
) -> EpochMetrics:
    if sample_count == 0:
        raise ValueError("数据加载器不包含任何样本")
    return EpochMetrics(
        loss=loss_sum / sample_count,
        accuracy=correct / sample_count,
        sample_count=sample_count,
        confusion_matrix=confusion_matrix,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        sample_count += batch_size

    return _finalize_metrics(loss_sum, correct, sample_count)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
    include_confusion: bool = False,
) -> EpochMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    confusion = (
        torch.zeros((num_classes, num_classes), dtype=torch.int64)
        if include_confusion
        else None
    )

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (predictions == targets).sum().item()
        sample_count += batch_size

        if confusion is not None:
            encoded = (targets * num_classes + predictions).to("cpu")
            confusion += torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    return _finalize_metrics(loss_sum, correct, sample_count, confusion)


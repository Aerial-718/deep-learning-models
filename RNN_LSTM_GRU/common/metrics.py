"""Metrics shared by training and comparison scripts."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def bits_per_character(mean_cross_entropy_nats: float) -> float:
    if mean_cross_entropy_nats < 0:
        raise ValueError("cross entropy must be non-negative")
    return float(mean_cross_entropy_nats / math.log(2.0))


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have equal shape")
    if predictions.size == 0:
        raise ValueError("accuracy is undefined for empty arrays")
    return float(np.mean(predictions == targets))


def parameter_count(model: object, trainable_only: bool = True) -> int:
    """Count parameters on an object implementing PyTorch's parameters API."""

    parameters: Iterable[object] = model.parameters()  # type: ignore[attr-defined]
    total = 0
    for parameter in parameters:
        if not trainable_only or bool(parameter.requires_grad):  # type: ignore[attr-defined]
            total += int(parameter.numel())  # type: ignore[attr-defined]
    return total


def mean_and_sample_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("at least one value is required")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


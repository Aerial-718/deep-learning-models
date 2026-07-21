"""Small training utilities; model-specific recurrence remains in the exercises."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        if self.count == 0:
            raise RuntimeError("average is undefined before the first update")
        return self.total / self.count


@dataclass
class EarlyStopping:
    patience: int
    minimum_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        """Update a metric that should be minimized; return True when stopping."""

        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.best is None or value < self.best - self.minimum_delta:
            self.best = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def save_json(path: str | Path, data: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


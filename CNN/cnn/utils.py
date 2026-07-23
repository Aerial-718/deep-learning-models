"""设备、随机数、检查点和训练产物工具。"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

CLASSES = tuple(str(index) for index in range(10))


def _load_pyplot() -> Any:
    """仅在需要画图时加载 Matplotlib，并处理只读主目录。"""

    if "MPLCONFIGDIR" not in os.environ:
        default_config = Path.home() / ".config" / "matplotlib"
        if not default_config.exists() or not os.access(default_config, os.W_OK):
            user_id = os.getuid() if hasattr(os, "getuid") else "default"
            cache_dir = Path(tempfile.gettempdir()) / f"cnn-matplotlib-{user_id}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 与 PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """解析 auto/cpu/cuda/mps，并对不可用的显式设备给出错误。"""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境没有可用的 CUDA 设备")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("请求了 MPS，但当前环境没有可用的 MPS 设备")
    return torch.device(requested)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_accuracy: float,
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    """原子地保存一个可恢复训练的检查点。"""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "classes": list(CLASSES),
        "config": dict(config),
        "history": list(history),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """加载并校验检查点；可选恢复优化器状态。"""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到检查点：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    required = {"epoch", "best_accuracy", "model_state", "config", "classes"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"检查点缺少字段：{', '.join(sorted(missing))}")
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("检查点类别与 MNIST 的 0–9 类别不一致")

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is None:
            raise ValueError("检查点不包含优化器状态，无法恢复训练")
        optimizer.load_state_dict(optimizer_state)
    return checkpoint


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_learning_curves(
    history: Sequence[Mapping[str, float | int]], path: str | Path
) -> None:
    if not history:
        return
    plt = _load_pyplot()
    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(
        epochs, [100 * item["train_accuracy"] for item in history], label="train"
    )
    axes[1].plot(
        epochs, [100 * item["val_accuracy"] for item in history], label="validation"
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(matrix: torch.Tensor, path: str | Path) -> None:
    plt = _load_pyplot()
    values = matrix.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="MNIST confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(10),
        yticks=range(10),
    )
    threshold = values.max() / 2
    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

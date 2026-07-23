"""MNIST 数据加载与推理图片预处理。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
MNIST_TRAIN_SIZE = 55_000
MNIST_VAL_SIZE = 5_000
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def mnist_transform() -> transforms.Compose:
    """返回训练、验证和测试共用的 MNIST 预处理。"""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def deterministic_split_indices(
    total_size: int, val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """使用局部随机数生成器创建可复现的训练/验证索引。"""

    if total_size <= 0:
        raise ValueError("total_size 必须为正整数")
    if not 0 < val_size < total_size:
        raise ValueError("val_size 必须位于 0 和 total_size 之间")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    return permutation[val_size:], permutation[:val_size]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }


def create_mnist_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """创建固定 55,000/5,000/10,000 划分的数据加载器。"""

    transform = mnist_transform()
    train_dataset = datasets.MNIST(
        root=Path(data_dir), train=True, transform=transform, download=download
    )
    test_dataset = datasets.MNIST(
        root=Path(data_dir), train=False, transform=transform, download=download
    )
    if len(train_dataset) != MNIST_TRAIN_SIZE + MNIST_VAL_SIZE:
        raise RuntimeError(
            f"期望 MNIST 训练集含 60000 个样本，实际为 {len(train_dataset)}"
        )

    train_indices, val_indices = deterministic_split_indices(
        len(train_dataset), MNIST_VAL_SIZE, seed
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    common = _loader_kwargs(batch_size, num_workers, device)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset, shuffle=True, generator=train_generator, **common
    )
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def create_mnist_test_loader(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    download: bool = True,
) -> DataLoader:
    """只创建 MNIST 测试集加载器。"""

    dataset = datasets.MNIST(
        root=Path(data_dir),
        train=False,
        transform=mnist_transform(),
        download=download,
    )
    return DataLoader(
        dataset,
        shuffle=False,
        **_loader_kwargs(batch_size, num_workers, device),
    )


def preprocess_image(path: str | Path, invert: bool = False) -> torch.Tensor:
    """将一张本地图片转换为形状 [1, 28, 28] 的标准化张量。"""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片：{image_path}")

    with Image.open(image_path) as source:
        image = source.convert("L")
        if invert:
            image = ImageOps.invert(image)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        return mnist_transform()(image)


def discover_images(directory: str | Path) -> Sequence[Path]:
    """按文件名排序返回目录中的受支持图片。"""

    image_dir = Path(directory)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"找不到图片目录：{image_dir}")

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"目录中没有受支持的图片：{image_dir}")
    return images


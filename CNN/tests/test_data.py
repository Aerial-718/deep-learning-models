from pathlib import Path

import torch
from PIL import Image

from cnn.data import deterministic_split_indices, preprocess_image


def test_split_is_deterministic_and_complete() -> None:
    first_train, first_val = deterministic_split_indices(100, 20, seed=42)
    second_train, second_val = deterministic_split_indices(100, 20, seed=42)
    assert first_train == second_train
    assert first_val == second_val
    assert len(first_train) == 80
    assert len(first_val) == 20
    assert set(first_train).isdisjoint(first_val)
    assert set(first_train + first_val) == set(range(100))


def test_preprocess_image_shape_and_inversion(tmp_path: Path) -> None:
    image_path = tmp_path / "digit.png"
    Image.new("L", (40, 20), color=255).save(image_path)
    normal = preprocess_image(image_path)
    inverted = preprocess_image(image_path, invert=True)
    assert normal.shape == (1, 28, 28)
    assert normal.dtype == torch.float32
    assert normal.mean() > inverted.mean()


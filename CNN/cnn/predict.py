"""单图和目录批量推理命令。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from .data import discover_images, preprocess_image
from .model import LeNet
from .utils import CLASSES, load_checkpoint, resolve_device, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 LeNet 检查点识别数字图片")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="单张图片")
    source.add_argument("--input-dir", type=Path, help="批量图片目录")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="反转灰度，适用于常见的白底黑字图片",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="计算设备",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON 输出路径；批量模式默认为输入目录下 predictions.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="目录推理的批量大小",
    )
    return parser


@torch.inference_mode()
def predict_paths(
    model: LeNet,
    paths: Sequence[Path],
    device: torch.device,
    invert: bool = False,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")

    model.eval()
    results: list[dict[str, Any]] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack(
            [preprocess_image(path, invert) for path in batch_paths]
        ).to(device)
        probabilities = torch.softmax(model(batch), dim=1).cpu()
        for path, row in zip(batch_paths, probabilities, strict=True):
            predicted_index = int(row.argmax().item())
            results.append(
                {
                    "image": str(path),
                    "prediction": CLASSES[predicted_index],
                    "confidence": float(row[predicted_index].item()),
                    "probabilities": {
                        label: float(probability)
                        for label, probability in zip(
                            CLASSES, row.tolist(), strict=True
                        )
                    },
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model = LeNet().to(device)
    load_checkpoint(args.checkpoint, model, device)

    paths = [args.image] if args.image else list(discover_images(args.input_dir))
    results = predict_paths(model, paths, device, args.invert, args.batch_size)
    for result in results:
        print(
            f"{result['image']} -> {result['prediction']} "
            f"(confidence {result['confidence']:.2%})"
        )
        if len(results) == 1:
            probabilities = result["probabilities"]
            print(
                "各类别概率："
                + ", ".join(
                    f"{label}={probability:.2%}"
                    for label, probability in probabilities.items()
                )
            )

    output = args.output
    if args.input_dir and output is None:
        output = args.input_dir / "predictions.json"
    if output:
        write_json(output, {"checkpoint": str(args.checkpoint), "results": results})
        print(f"JSON 结果：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

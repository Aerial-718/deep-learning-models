"""独立评估命令。"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from .data import create_mnist_test_loader
from .engine import evaluate_model
from .model import LeNet
from .utils import (
    load_checkpoint,
    plot_confusion_matrix,
    resolve_device,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在 MNIST 测试集上评估检查点")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型检查点")
    parser.add_argument("--data-dir", default="data", help="MNIST 数据目录")
    parser.add_argument("--batch-size", type=int, default=256, help="批量大小")
    parser.add_argument("--num-workers", type=int, default=0, help="数据读取进程数")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="计算设备",
    )
    parser.add_argument("--output", type=Path, help="可选的 JSON 评估结果路径")
    parser.add_argument(
        "--confusion-matrix", type=Path, help="可选的混淆矩阵图片路径"
    )
    parser.add_argument("--no-download", action="store_true", help="禁止自动下载")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model = LeNet().to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    loader = create_mnist_test_loader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        device,
        download=not args.no_download,
    )
    metrics = evaluate_model(
        model,
        loader,
        nn.CrossEntropyLoss(),
        device,
        include_confusion=True,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "samples": metrics.sample_count,
        "device": str(device),
    }
    print(
        f"测试集：loss {metrics.loss:.4f}, accuracy {metrics.accuracy:.2%}, "
        f"samples {metrics.sample_count}"
    )
    if args.output:
        write_json(args.output, result)
        print(f"评估结果：{args.output}")
    if args.confusion_matrix and metrics.confusion_matrix is not None:
        plot_confusion_matrix(metrics.confusion_matrix, args.confusion_matrix)
        print(f"混淆矩阵：{args.confusion_matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


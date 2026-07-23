"""MNIST 模型训练命令。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import create_mnist_loaders
from .engine import evaluate_model, train_one_epoch
from .model import LeNet
from .utils import (
    load_checkpoint,
    plot_confusion_matrix,
    plot_learning_curves,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 LeNet 风格的 MNIST 分类器")
    parser.add_argument("--data-dir", default="data", help="MNIST 数据目录")
    parser.add_argument("--output-dir", default="runs/lenet", help="训练产物目录")
    parser.add_argument("--epochs", type=int, default=10, help="总训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批量大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="数据读取进程数")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="计算设备",
    )
    parser.add_argument("--resume", type=Path, help="从 last.pt 等检查点继续训练")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="不自动下载 MNIST（数据必须已存在）",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs 必须为正整数")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须为正整数")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate 必须为正数")
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能为负数")


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device = resolve_device(args.device)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": args.device,
    }
    write_json(output_dir / "config.json", config)

    print(f"设备：{device}")
    print("正在准备 MNIST 数据……")
    train_loader, val_loader, test_loader = create_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
        download=not args.no_download,
    )

    model = LeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1
    best_accuracy = -1.0
    history: list[dict[str, Any]] = []

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, device, optimizer)
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        best_path = output_dir / "best.pt"
        best_accuracy = (
            float(checkpoint["best_accuracy"]) if best_path.is_file() else -1.0
        )
        print(f"已恢复检查点：{args.resume}（下一轮：{start_epoch}）")

    if start_epoch > args.epochs:
        print(
            f"检查点已训练到第 {start_epoch - 1} 轮，"
            f"不低于指定的 {args.epochs} 轮；跳过训练。"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
        }
        history.append(record)

        improved = val_metrics.accuracy > best_accuracy
        if improved:
            best_accuracy = val_metrics.accuracy
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_accuracy,
                config,
                history,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_accuracy,
            config,
            history,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss {train_metrics.loss:.4f}, "
            f"acc {train_metrics.accuracy:.2%} | "
            f"val loss {val_metrics.loss:.4f}, "
            f"acc {val_metrics.accuracy:.2%}"
            + (" | 已保存最佳模型" if improved else "")
        )

    best_path = output_dir / "best.pt"
    if best_path.is_file():
        load_checkpoint(best_path, model, device)
    elif args.resume:
        load_checkpoint(args.resume, model, device)
    else:
        raise RuntimeError("训练没有生成可供测试的检查点")

    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        include_confusion=True,
    )
    summary = {
        "best_validation_accuracy": max(
            (float(item["val_accuracy"]) for item in history), default=None
        ),
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "test_samples": test_metrics.sample_count,
        "best_checkpoint": str(best_path if best_path.is_file() else args.resume),
    }
    _write_history_csv(output_dir / "history.csv", history)
    write_json(output_dir / "metrics.json", {"history": history, "summary": summary})
    plot_learning_curves(history, output_dir / "learning_curves.png")
    if test_metrics.confusion_matrix is not None:
        plot_confusion_matrix(
            test_metrics.confusion_matrix, output_dir / "confusion_matrix.png"
        )

    print(
        f"测试集：loss {test_metrics.loss:.4f}, "
        f"accuracy {test_metrics.accuracy:.2%}"
    )
    print(f"产物已保存到：{output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


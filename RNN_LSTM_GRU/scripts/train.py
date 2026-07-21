#!/usr/bin/env python3
"""Unified training entry point for the learner-completed manual models."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common.data import CharVocabulary, contiguous_split, delayed_recall_batch, random_char_batch
from common.metrics import bits_per_character, parameter_count
from common.modeling import DelayedRecallClassifier, ModelName, build_language_model
from common.seed import default_device, seed_everything
from common.training import JsonlLogger, save_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=["vanilla", "lstm", "gru"], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "task" not in config:
        raise ValueError("config must be a mapping containing task")
    return config


def make_run_dir(model_name: str, task: str, requested_name: str | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = requested_name or f"{task}-{model_name}-{timestamp}"
    run_dir = Path("artifacts") / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


@torch.no_grad()
def evaluate_char_model(
    model: torch.nn.Module,
    tokens: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    training = config["training"]
    losses: list[float] = []
    for _ in range(int(training["eval_batches"])):
        x_np, y_np = random_char_batch(
            tokens,
            int(training["batch_size"]),
            int(training["sequence_length"]),
            rng,
        )
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(float(loss.item()))
    mean_loss = float(np.mean(losses))
    model.train()
    return {"loss": mean_loss, "bpc": bits_per_character(mean_loss)}


def train_char_model(
    model_name: ModelName,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
) -> int:
    text = Path(config["data"]["path"]).read_text(encoding="utf-8")
    vocabulary = CharVocabulary.build(text)
    all_tokens = vocabulary.encode(text)
    train_tokens, validation_tokens, test_tokens = contiguous_split(
        all_tokens,
        tuple(float(value) for value in config["data"]["split"]),
    )
    metadata_path = Path(config["data"]["path"]).with_suffix(
        Path(config["data"]["path"]).suffix + ".metadata.json"
    )
    if metadata_path.exists():
        save_json(
            run_dir / "corpus_metadata.json",
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )
    model_config = config["model"]
    model = build_language_model(
        model_name,
        vocabulary.size,
        int(model_config["embedding_size"]),
        int(model_config["hidden_size"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    rng = np.random.default_rng(seed)
    best_validation = float("inf")
    training = config["training"]
    last_log_time = time.perf_counter()

    save_json(run_dir / "vocabulary.json", {"chars": list(vocabulary.chars)})
    for step in range(1, int(training["steps"]) + 1):
        x_np, y_np = random_char_batch(
            train_tokens,
            int(training["batch_size"]),
            int(training["sequence_length"]),
            rng,
        )
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        )
        optimizer.step()

        if step % int(training["log_interval"]) == 0:
            now = time.perf_counter()
            interval = int(training["log_interval"])
            tokens_per_second = (
                interval
                * int(training["batch_size"])
                * int(training["sequence_length"])
                / max(now - last_log_time, 1e-12)
            )
            logger.log(
                {
                    "split": "train",
                    "step": step,
                    "loss": float(loss.item()),
                    "bpc": bits_per_character(float(loss.item())),
                    "gradient_norm": float(gradient_norm),
                    "tokens_per_second": tokens_per_second,
                    "peak_memory_bytes": peak_memory_bytes(device),
                }
            )
            last_log_time = now
        if step % int(training["eval_interval"]) == 0 or step == int(training["steps"]):
            validation = evaluate_char_model(model, validation_tokens, config, device, seed + step)
            logger.log({"split": "validation", "step": step, **validation})
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                save_checkpoint(
                    run_dir / "best.pt",
                    {
                        "model_state": model.state_dict(),
                        "model_name": model_name,
                        "config": config,
                        "seed": seed,
                        "vocabulary": vocabulary.chars,
                        "step": step,
                    },
                )

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate_char_model(model, test_tokens, config, device, seed + 999)
    save_json(run_dir / "test_metrics.json", test_metrics)
    generation = config.get("generation")
    if generation:
        prefix_ids = torch.from_numpy(vocabulary.encode(str(generation["prefix"]))).unsqueeze(0)
        generator = torch.Generator(device=device).manual_seed(seed)
        generated = model.generate(
            prefix_ids.to(device),
            new_tokens=int(generation["new_tokens"]),
            temperature=float(generation["temperature"]),
            top_k=int(generation["top_k"]) if generation.get("top_k") is not None else None,
            generator=generator,
        )
        (run_dir / "sample.txt").write_text(
            vocabulary.decode(generated[0].detach().cpu().tolist()),
            encoding="utf-8",
        )
    return parameter_count(model)


@torch.no_grad()
def evaluate_recall_model(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    training = config["training"]
    num_classes = int(config["data"]["num_classes"])
    results: dict[str, float] = {}
    for delay in config["data"]["delays"]:
        correct = 0
        total = 0
        for _ in range(int(training["eval_batches"])):
            x_np, y_np = delayed_recall_batch(
                int(training["batch_size"]), int(delay), num_classes, rng
            )
            logits = model(torch.from_numpy(x_np).to(device))
            target = torch.from_numpy(y_np).to(device)
            correct += int((logits.argmax(dim=-1) == target).sum().item())
            total += len(y_np)
        results[f"accuracy_delay_{delay}"] = correct / total
    model.train()
    return results


def train_recall_model(
    model_name: ModelName,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    run_dir: Path,
) -> int:
    data = config["data"]
    model_config = config["model"]
    num_classes = int(data["num_classes"])
    model = DelayedRecallClassifier(
        model_name,
        vocabulary_size=num_classes + 2,
        embedding_size=int(model_config["embedding_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    rng = np.random.default_rng(seed)
    training = config["training"]
    last_log_time = time.perf_counter()

    for step in range(1, int(training["steps"]) + 1):
        delay = int(rng.choice(data["delays"]))
        x_np, y_np = delayed_recall_batch(
            int(training["batch_size"]), delay, num_classes, rng
        )
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        )
        optimizer.step()

        if step % int(training["log_interval"]) == 0:
            now = time.perf_counter()
            interval = int(training["log_interval"])
            examples_per_second = (
                interval * int(training["batch_size"]) / max(now - last_log_time, 1e-12)
            )
            logger.log(
                {
                    "split": "train",
                    "step": step,
                    "delay": delay,
                    "loss": float(loss.item()),
                    "gradient_norm": float(gradient_norm),
                    "examples_per_second": examples_per_second,
                    "peak_memory_bytes": peak_memory_bytes(device),
                }
            )
            last_log_time = now
        if step % int(training["eval_interval"]) == 0 or step == int(training["steps"]):
            metrics = evaluate_recall_model(model, config, device, seed + step)
            logger.log({"split": "validation", "step": step, **metrics})

    save_checkpoint(
        run_dir / "best.pt",
        {
            "model_state": model.state_dict(),
            "model_name": model_name,
            "config": config,
            "seed": seed,
            "step": int(training["steps"]),
        },
    )
    save_json(
        run_dir / "test_metrics.json",
        evaluate_recall_model(model, config, device, seed + 999),
    )
    return parameter_count(model)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["seeds"][0])
    seed_everything(seed)
    device = default_device()
    run_dir = make_run_dir(args.model, str(config["task"]), args.run_name)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "environment.json", environment_record(device))
    save_json(run_dir / "run.json", {"model": args.model, "seed": seed})

    if config["task"] == "char_lm":
        parameters = train_char_model(args.model, config, seed, device, run_dir)
    elif config["task"] == "delayed_recall":
        parameters = train_recall_model(args.model, config, seed, device, run_dir)
    else:
        raise ValueError(f"unknown task: {config['task']}")

    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "seed": seed,
        "parameters": parameters,
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

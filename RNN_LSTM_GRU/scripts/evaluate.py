#!/usr/bin/env python3
"""Re-evaluate a saved run using its recorded config and checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.data import CharVocabulary, contiguous_split
from common.modeling import DelayedRecallClassifier, build_language_model
from common.seed import default_device, seed_everything
from scripts.train import evaluate_char_model, evaluate_recall_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_name = checkpoint["model_name"]
    seed = int(checkpoint["seed"])
    seed_everything(seed)
    device = default_device()

    if config["task"] == "char_lm":
        text = Path(config["data"]["path"]).read_text(encoding="utf-8")
        vocabulary = CharVocabulary(tuple(checkpoint["vocabulary"]))
        tokens = vocabulary.encode(text)
        _, _, test_tokens = contiguous_split(
            tokens, tuple(float(value) for value in config["data"]["split"])
        )
        model = build_language_model(
            model_name,
            vocabulary.size,
            int(config["model"]["embedding_size"]),
            int(config["model"]["hidden_size"]),
        )
    else:
        num_classes = int(config["data"]["num_classes"])
        model = DelayedRecallClassifier(
            model_name,
            vocabulary_size=num_classes + 2,
            embedding_size=int(config["model"]["embedding_size"]),
            hidden_size=int(config["model"]["hidden_size"]),
            num_classes=num_classes,
        )

    model.load_state_dict(checkpoint["model_state"])
    if config["task"] == "char_lm":
        metrics = evaluate_char_model(model.to(device), test_tokens, config, device, seed + 999)
        generation = config.get("generation")
        if generation:
            prefix_ids = torch.from_numpy(
                vocabulary.encode(str(generation["prefix"]))
            ).unsqueeze(0)
            generator = torch.Generator(device=device).manual_seed(seed)
            generated = model.generate(
                prefix_ids.to(device),
                new_tokens=int(generation["new_tokens"]),
                temperature=float(generation["temperature"]),
                top_k=int(generation["top_k"]) if generation.get("top_k") is not None else None,
                generator=generator,
            )
            metrics["sample"] = vocabulary.decode(generated[0].detach().cpu().tolist())
    else:
        metrics = evaluate_recall_model(model.to(device), config, device, seed + 999)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

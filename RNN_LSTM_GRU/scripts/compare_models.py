#!/usr/bin/env python3
"""Aggregate completed run directories into a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common.metrics import mean_and_sample_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for run_dir in args.run_dirs:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        throughput_values = [
            float(record[key])
            for record in records
            for key in ("tokens_per_second", "examples_per_second")
            if key in record
        ]
        memory_values = [
            int(record["peak_memory_bytes"])
            for record in records
            if "peak_memory_bytes" in record
        ]
        numeric_results = {
            **{metric: float(value) for metric, value in metrics.items()},
            "parameters": float(run_summary["parameters"]),
            "mean_throughput_per_second": (
                sum(throughput_values) / len(throughput_values) if throughput_values else 0.0
            ),
            "peak_memory_bytes": float(max(memory_values, default=0)),
        }
        for metric, value in numeric_results.items():
            grouped.setdefault((str(run["model"]), metric), []).append(value)
        rows.append(
            {
                "run_dir": str(run_dir),
                "model": run["model"],
                "task": config["task"],
                "seed": run["seed"],
                **numeric_results,
            }
        )

    fieldnames = sorted({key for row in rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for (model, metric), values in sorted(grouped.items()):
        mean, std = mean_and_sample_std(values)
        summary.append({"model": model, "metric": metric, "mean": mean, "sample_std": std})
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()

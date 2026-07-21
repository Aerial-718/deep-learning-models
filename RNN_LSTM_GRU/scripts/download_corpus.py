#!/usr/bin/env python3
"""Download the pinned Tiny Shakespeare learning corpus and verify its hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DEFAULT_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--output", type=Path, default=Path("data/raw/tinyshakespeare.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with urllib.request.urlopen(args.url, timeout=30) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != args.sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch: expected {args.sha256}, received {digest}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    metadata = {
        "url": args.url,
        "sha256": digest,
        "bytes": len(content),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output} ({len(content)} bytes, sha256={digest})")


if __name__ == "__main__":
    main()


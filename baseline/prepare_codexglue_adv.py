#!/usr/bin/env python3
"""Legacy wrapper that reproduces the original adversarial split exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.prepare_dataset import export_legacy_adv_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-zip",
        default="CodeXGLUE/Text-Code/NL-code-search-Adv/dataset.zip",
        help="Path to the official CodeXGLUE AdvTest dataset.zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline/data",
        help="Directory where valid.jsonl and test.jsonl will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = export_legacy_adv_dataset(Path(args.dataset_zip), Path(args.output_dir))
    except Exception as exc:
        print(f"prepare_codexglue_adv failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Legacy wrapper that reproduces the original BM25 baseline outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.bm25 import build_rankings


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="baseline/data",
        help="Directory containing valid.jsonl/test.jsonl prepared by prepare_codexglue_adv.py.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        required=True,
        help="Dataset split to score.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Prediction output path. Defaults to baseline/out/<split>.predictions.jsonl.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidate indices to emit per query.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data_dir = Path(args.data_dir)
        split_path = data_dir / f"{args.split}.jsonl"
        output_path = Path(args.output or f"baseline/out/{args.split}.predictions.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = read_jsonl(split_path)
        docs = [row["function_tokens"] for row in rows]
        queries = [row["docstring_tokens"] for row in rows]
        idx_by_doc_id = [row["idx"] for row in rows]
        urls_by_query_id = [row["url"] for row in rows]
        rankings = build_rankings(docs, queries, top_k=args.top_k)

        with output_path.open("w", encoding="utf-8") as handle:
            for query_id, ranked in enumerate(rankings):
                payload = {
                    "url": urls_by_query_id[query_id],
                    "answers": [idx_by_doc_id[doc_id] for doc_id in ranked],
                }
                handle.write(json.dumps(payload))
                handle.write("\n")
    except Exception as exc:
        print(f"run_bm25_baseline failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "split": args.split,
                "examples": len(rows),
                "output": str(output_path),
                "top_k": args.top_k,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

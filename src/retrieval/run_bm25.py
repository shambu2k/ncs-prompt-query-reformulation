"""Run the canonical Phase 1 BM25 baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.data.loader import DatasetLoader, DEFAULT_MANIFEST_ROOT, DEFAULT_PROCESSED_ROOT
from src.data.schema import CanonicalRecord, DatasetValidationError
from src.rewriting.cache_manager import DEFAULT_REWRITE_ROOT, load_rewrite_records

from .bm25 import build_rankings


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_query_text(record: CanonicalRecord, query_source: str) -> str:
    if query_source == "original":
        return record.query_text
    if query_source.startswith("metadata:"):
        key = query_source.split(":", 1)[1]
        value = record.metadata.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return " ".join(value)
        raise DatasetValidationError(
            f"Query source '{query_source}' is not available as text for record {record.example_id}."
        )
    raise DatasetValidationError(
        f"Unsupported query source '{query_source}'. Use 'original' or 'metadata:<key>'."
    )


def _resolve_query_tokens(record: CanonicalRecord, query_source: str) -> list[str]:
    if query_source == "original":
        return list(record.query_tokens)
    if query_source.startswith("metadata:"):
        key = query_source.split(":", 1)[1]
        value = record.metadata.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
        if isinstance(value, str):
            return value.split()
    raise DatasetValidationError(
        f"Unsupported token source '{query_source}' for record {record.example_id}."
    )


def _resolve_rewrite_cache_path(
    *,
    rewrite_root: Path,
    condition: str,
    rewrite_run_name: str | None,
    rewrite_path: str | None,
) -> Path:
    if rewrite_path:
        return Path(rewrite_path)
    if rewrite_run_name:
        return rewrite_root / condition / rewrite_run_name / "rewrites.jsonl"
    raise DatasetValidationError(
        "query_source=rewritten requires --rewrite-run-name or --rewrite-path."
    )


def _load_rewrites_by_example_id(
    *,
    rewrite_root: Path,
    condition: str,
    rewrite_run_name: str | None,
    rewrite_path: str | None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    cache_path = _resolve_rewrite_cache_path(
        rewrite_root=rewrite_root,
        condition=condition,
        rewrite_run_name=rewrite_run_name,
        rewrite_path=rewrite_path,
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Rewrite cache not found: {cache_path}")
    return cache_path, load_rewrite_records(cache_path)


def _tokenize_records(
    records: list[CanonicalRecord],
    *,
    tokenization_mode: str,
    query_source: str,
    rewrites_by_example_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    if query_source == "rewritten":
        if rewrites_by_example_id is None:
            raise DatasetValidationError("Missing rewrite cache for query_source=rewritten.")
        missing = [record.example_id for record in records if record.example_id not in rewrites_by_example_id]
        if missing:
            raise DatasetValidationError(
                f"Rewrite cache is missing {len(missing)} example_ids; sample={missing[0]}."
            )
        if tokenization_mode == "pretokenized":
            queries = []
            for record in records:
                payload = rewrites_by_example_id[record.example_id]
                tokens = payload.get("rewritten_query_tokens")
                if isinstance(tokens, list) and all(isinstance(item, str) for item in tokens):
                    queries.append(list(tokens))
                else:
                    rewritten_query = payload.get("rewritten_query")
                    if not isinstance(rewritten_query, str):
                        raise DatasetValidationError(
                            f"Rewrite cache record {record.example_id} is missing rewritten_query."
                        )
                    queries.append(rewritten_query.split())
            return [list(record.code_tokens) for record in records], queries
        if tokenization_mode == "whitespace":
            return (
                [record.code_text.split() for record in records],
                [
                    str(rewrites_by_example_id[record.example_id]["rewritten_query"]).split()
                    for record in records
                ],
            )
        raise DatasetValidationError(f"Unsupported tokenization mode '{tokenization_mode}'.")

    if tokenization_mode == "pretokenized":
        return (
            [list(record.code_tokens) for record in records],
            [_resolve_query_tokens(record, query_source) for record in records],
        )
    if tokenization_mode == "whitespace":
        return (
            [record.code_text.split() for record in records],
            [_resolve_query_text(record, query_source).split() for record in records],
        )
    raise DatasetValidationError(
        f"Unsupported tokenization mode '{tokenization_mode}'."
    )


def _prediction_rows(
    records: list[CanonicalRecord],
    rankings: list[list[int]],
) -> list[dict[str, Any]]:
    idx_by_doc_id = [record.metadata["idx"] for record in records]
    urls_by_query_id = [record.metadata["url"] for record in records]
    return [
        {"url": urls_by_query_id[query_id], "answers": [idx_by_doc_id[doc_id] for doc_id in ranked]}
        for query_id, ranked in enumerate(rankings)
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 on a canonical processed split.")
    parser.add_argument("--split", choices=["valid", "test"], required=True, help="Split to score.")
    parser.add_argument(
        "--condition",
        choices=["clean", "adv"],
        required=True,
        help="Condition to score.",
    )
    parser.add_argument("--run-name", required=True, help="Run directory name under results/bm25/.")
    parser.add_argument(
        "--processed-root",
        default=str(DEFAULT_PROCESSED_ROOT),
        help="Directory containing canonical processed JSONL files.",
    )
    parser.add_argument(
        "--manifest-root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Directory containing dataset manifests.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory where BM25 result artifacts will be written.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to emit per query.")
    parser.add_argument(
        "--tokenization-mode",
        choices=["pretokenized", "whitespace"],
        default="pretokenized",
        help="Whether to use canonical token lists or whitespace tokenization.",
    )
    parser.add_argument(
        "--query-source",
        default="original",
        help="Query source. Use 'original', 'rewritten', or 'metadata:<key>'.",
    )
    parser.add_argument(
        "--rewrite-root",
        default=str(DEFAULT_REWRITE_ROOT),
        help="Root directory for cached rewritten queries.",
    )
    parser.add_argument(
        "--rewrite-run-name",
        default=None,
        help="Rewrite run name under rewritten_queries/<condition>/ when query_source=rewritten.",
    )
    parser.add_argument(
        "--rewrite-path",
        default=None,
        help="Explicit path to rewrites.jsonl when query_source=rewritten.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        loader = DatasetLoader(processed_root=Path(args.processed_root), manifest_root=Path(args.manifest_root))
        records = loader.load_split(args.split, args.condition)
        manifests = loader.get_manifest(args.split, args.condition)
        rewrite_cache_path = None
        rewrites_by_example_id = None
        if args.query_source == "rewritten":
            rewrite_cache_path, rewrites_by_example_id = _load_rewrites_by_example_id(
                rewrite_root=Path(args.rewrite_root),
                condition=args.condition,
                rewrite_run_name=args.rewrite_run_name,
                rewrite_path=args.rewrite_path,
            )

        document_tokens, query_tokens = _tokenize_records(
            records,
            tokenization_mode=args.tokenization_mode,
            query_source=args.query_source,
            rewrites_by_example_id=rewrites_by_example_id,
        )
        rankings = build_rankings(document_tokens, query_tokens, top_k=args.top_k)
        predictions = _prediction_rows(records, rankings)

        run_dir = Path(args.results_root) / "bm25" / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row))
                handle.write("\n")

        metadata = {
            "run_name": args.run_name,
            "method": "bm25",
            "timestamp": _utc_now(),
            "split": args.split,
            "condition": args.condition,
            "top_k": args.top_k,
            "tokenization_mode": args.tokenization_mode,
            "query_source": args.query_source,
            "corpus_size": len(records),
            "prediction_path": str(prediction_path),
            "reference_path": str(loader.split_path(args.split, args.condition)),
            "manifest_path": str(loader.manifest_path(args.split, args.condition)),
            "rewrite_cache_path": str(rewrite_cache_path) if rewrite_cache_path else None,
            "rewrite_run_name": args.rewrite_run_name,
            "implementation_id": "src.retrieval.run_bm25",
            "source_task": manifests["source_task"],
        }
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run_log_path = run_dir / "run.log"
        run_log_path.write_text(
            "\n".join(
                [
                    f"timestamp={metadata['timestamp']}",
                    f"split={args.split}",
                    f"condition={args.condition}",
                    f"query_source={args.query_source}",
                    f"tokenization_mode={args.tokenization_mode}",
                    f"top_k={args.top_k}",
                    f"corpus_size={len(records)}",
                    f"prediction_path={prediction_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"run_bm25 failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score a Phase 1 run and emit standardized evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.data.loader import DatasetLoader, DEFAULT_MANIFEST_ROOT, DEFAULT_PROCESSED_ROOT

from .official_wrapper import DEFAULT_EVALUATOR_PATH, score_prediction_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_run_metadata(results_root: Path, run_name: str) -> tuple[Path, dict[str, Any]]:
    run_dir = results_root / "bm25" / run_name
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return run_dir, metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 1 retrieval run.")
    parser.add_argument("--run-name", required=True, help="Run directory name under results/bm25/.")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory containing retrieval results and evaluation outputs.",
    )
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
        "--official-evaluator",
        default=str(DEFAULT_EVALUATOR_PATH),
        help="Path to the official CodeXGLUE evaluator.py module.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results_root = Path(args.results_root)
        _, run_metadata = _resolve_run_metadata(results_root, args.run_name)
        split = run_metadata["split"]
        condition = run_metadata["condition"]
        prediction_path = Path(run_metadata["prediction_path"])

        loader = DatasetLoader(processed_root=Path(args.processed_root), manifest_root=Path(args.manifest_root))
        records = loader.load_split(split, condition)
        scores = score_prediction_file(
            records,
            prediction_path,
            evaluator_path=Path(args.official_evaluator),
        )

        evaluation_result = {
            "run_name": args.run_name,
            "method": run_metadata.get("method", "bm25"),
            "split": split,
            "condition": condition,
            "metrics": {"mrr": float(scores["MRR"])},
            "prediction_path": str(prediction_path),
            "reference_path": str(loader.split_path(split, condition)),
            "timestamp": _utc_now(),
        }

        evaluations_dir = results_root / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        output_path = evaluations_dir / f"{args.run_name}.json"
        output_path.write_text(
            json.dumps(evaluation_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"evaluate failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(evaluation_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

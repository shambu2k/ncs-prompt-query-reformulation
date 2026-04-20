"""Compare retrieval runs across baselines and the evolved prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "configs" / "experiment.yaml"


def _load_json_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def _load_evaluation(results_root: Path, run_name: str) -> dict[str, Any]:
    path = results_root / "evaluations" / f"{run_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _method_row(label: str, clean_result: dict[str, Any], adv_result: dict[str, Any]) -> dict[str, Any]:
    clean_mrr = float(clean_result["metrics"]["mrr"])
    adv_mrr = float(adv_result["metrics"]["mrr"])
    robustness_drop = clean_mrr - adv_mrr
    relative_drop = robustness_drop / clean_mrr if clean_mrr else None
    return {
        "method": label,
        "clean_mrr": clean_mrr,
        "adv_mrr": adv_mrr,
        "robustness_drop": robustness_drop,
        "relative_drop": relative_drop,
        "clean_run_name": clean_result["run_name"],
        "adv_run_name": adv_result["run_name"],
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Clean MRR | Adv MRR | Robustness Drop | Relative Drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        rel = "n/a" if row["relative_drop"] is None else f"{row['relative_drop']:.4f}"
        lines.append(
            f"| {row['method']} | {row['clean_mrr']:.4f} | {row['adv_mrr']:.4f} | "
            f"{row['robustness_drop']:.4f} | {rel} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval runs across baselines and the evolved prompt."
    )
    parser.add_argument("--baseline-clean", required=True, help="Evaluation run name for the raw clean condition.")
    parser.add_argument("--baseline-adv", required=True, help="Evaluation run name for the raw adversarial condition.")
    parser.add_argument("--candidate-clean", required=True, help="Evaluation run name for the fixed-prompt clean condition.")
    parser.add_argument("--candidate-adv", required=True, help="Evaluation run name for the fixed-prompt adversarial condition.")
    parser.add_argument("--comparison-name", required=True, help="Output name under results/comparisons/.")
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Display label for the raw baseline method.",
    )
    parser.add_argument(
        "--candidate-label",
        default=None,
        help="Display label for the fixed-prompt method.",
    )
    # Phase 3 evolved prompt (optional — omit for Phase 2 comparisons)
    parser.add_argument(
        "--evolved-clean",
        default=None,
        help="Evaluation run name for the evolved-prompt clean condition.",
    )
    parser.add_argument(
        "--evolved-adv",
        default=None,
        help="Evaluation run name for the evolved-prompt adversarial condition.",
    )
    parser.add_argument(
        "--evolved-label",
        default=None,
        help="Display label for the evolved-prompt method (default: Evolved Prompt Rewrite).",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Results root containing evaluations and comparison outputs.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Optional JSON-compatible YAML experiment config.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_json_yaml(Path(args.config)) if args.config else {}
        baseline_label = args.baseline_label or config.get("raw_baseline_label", "Raw Query BM25")
        candidate_label = args.candidate_label or config.get("fixed_prompt_label", "Fixed Prompt Rewrite")
        evolved_label = args.evolved_label or config.get("evolved_prompt_label", "Evolved Prompt Rewrite")
        results_root = Path(args.results_root)

        baseline_clean = _load_evaluation(results_root, args.baseline_clean)
        baseline_adv = _load_evaluation(results_root, args.baseline_adv)
        candidate_clean = _load_evaluation(results_root, args.candidate_clean)
        candidate_adv = _load_evaluation(results_root, args.candidate_adv)

        rows = [
            _method_row(baseline_label, baseline_clean, baseline_adv),
            _method_row(candidate_label, candidate_clean, candidate_adv),
        ]

        has_evolved = args.evolved_clean is not None and args.evolved_adv is not None
        if has_evolved:
            evolved_clean = _load_evaluation(results_root, args.evolved_clean)
            evolved_adv = _load_evaluation(results_root, args.evolved_adv)
            rows.append(_method_row(evolved_label, evolved_clean, evolved_adv))

        payload: dict[str, Any] = {
            "comparison_name": args.comparison_name,
            "rows": rows,
            "baseline_delta_clean": rows[1]["clean_mrr"] - rows[0]["clean_mrr"],
            "baseline_delta_adv": rows[1]["adv_mrr"] - rows[0]["adv_mrr"],
            "robustness_delta": rows[1]["robustness_drop"] - rows[0]["robustness_drop"],
        }
        if has_evolved:
            payload["evolved_delta_clean"] = rows[2]["clean_mrr"] - rows[0]["clean_mrr"]
            payload["evolved_delta_adv"] = rows[2]["adv_mrr"] - rows[0]["adv_mrr"]
            payload["evolved_vs_fixed_clean"] = rows[2]["clean_mrr"] - rows[1]["clean_mrr"]
            payload["evolved_vs_fixed_adv"] = rows[2]["adv_mrr"] - rows[1]["adv_mrr"]

        output_dir = results_root / "comparisons"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{args.comparison_name}.json"
        md_path = output_dir / f"{args.comparison_name}.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(_markdown_table(rows), encoding="utf-8")
    except Exception as exc:
        print(f"compare_runs failed: {exc}", file=sys.stderr)
        return 1

    print(_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

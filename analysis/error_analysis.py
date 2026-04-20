"""Compute structured Phase 4 error analysis over the final runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .common import (
    DEFAULT_PHASE4_ROOT,
    DEFAULT_RESULTS_ROOT,
    build_case_rows,
    load_experiment_config,
    markdown_table,
    short_float,
    summarize_case_rows,
    write_csv,
    write_json,
    write_text,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 quantitative error analysis.")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--condition", default="adv", choices=["clean", "adv"])
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_PHASE4_ROOT / "error_analysis"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config()
    try:
        case_rows, runs = build_case_rows(
            split=args.split,
            condition=args.condition,
            results_root=Path(args.results_root),
            config=config,
        )
        summary = summarize_case_rows(case_rows)
    except Exception as exc:
        print(f"error_analysis failed: {exc}", file=sys.stderr)
        return 1

    output_root = Path(args.output_root)
    write_json(output_root / "error_summary.json", summary)

    per_query_rows = []
    for row in case_rows:
        per_query_rows.append(
            {
                "example_id": row["example_id"],
                "raw_rank": row["raw_rank"],
                "fixed_rank": row["fixed_rank"],
                "evolved_rank": row["evolved_rank"],
                "raw_rr": short_float(row["raw_rr"]),
                "fixed_rr": short_float(row["fixed_rr"]),
                "evolved_rr": short_float(row["evolved_rr"]),
                "fixed_outcome": row["fixed_outcome"],
                "evolved_outcome": row["evolved_outcome"],
                "evolved_vs_fixed_outcome": row["evolved_vs_fixed_outcome"],
                "tags": ",".join(row["tags"]),
                "query_text": row["query_text"],
            }
        )
    write_csv(output_root / "per_query_deltas.csv", per_query_rows)

    category_rows = []
    for tag, bucket in sorted(summary["tags"].items()):
        category_rows.append(
            {
                "tag": tag,
                "count": bucket["count"],
                "fixed_improved": bucket["fixed_improved"],
                "fixed_worse": bucket["fixed_worse"],
                "evolved_improved": bucket["evolved_improved"],
                "evolved_worse": bucket["evolved_worse"],
            }
        )
    write_csv(output_root / "category_summary.csv", category_rows)

    md = [
        "# Phase 4 Error Analysis",
        "",
        f"- Split: `{args.split}`",
        f"- Condition: `{args.condition}`",
        f"- Raw run: `{runs['raw'].run_name}`",
        f"- Fixed run: `{runs['fixed'].run_name}`",
        f"- Evolved run: `{runs['evolved'].run_name}`",
        "",
        "## Outcome Summary",
        "",
        markdown_table(
            [
                {
                    "comparison": "fixed_vs_raw",
                    "improved": summary["fixed"]["improved"],
                    "worse": summary["fixed"]["worse"],
                    "same": summary["fixed"]["same"],
                    "mean_delta_rr": short_float(summary["fixed"]["mean_delta_rr"]),
                },
                {
                    "comparison": "evolved_vs_raw",
                    "improved": summary["evolved"]["improved"],
                    "worse": summary["evolved"]["worse"],
                    "same": summary["evolved"]["same"],
                    "mean_delta_rr": short_float(summary["evolved"]["mean_delta_rr"]),
                },
                {
                    "comparison": "evolved_vs_fixed",
                    "improved": summary["evolved_vs_fixed"]["improved"],
                    "worse": summary["evolved_vs_fixed"]["worse"],
                    "same": summary["evolved_vs_fixed"]["same"],
                    "mean_delta_rr": short_float(summary["evolved_vs_fixed"]["mean_delta_rr"]),
                },
            ],
            headers=[
                ("comparison", "Comparison"),
                ("improved", "Improved"),
                ("worse", "Worse"),
                ("same", "Same"),
                ("mean_delta_rr", "Mean Delta RR"),
            ],
        ),
    ]
    if category_rows:
        md.extend(
            [
                "## Category Coverage",
                "",
                markdown_table(
                    category_rows,
                    headers=[
                        ("tag", "Tag"),
                        ("count", "Count"),
                        ("fixed_improved", "Fixed Improved"),
                        ("fixed_worse", "Fixed Worse"),
                        ("evolved_improved", "Evolved Improved"),
                        ("evolved_worse", "Evolved Worse"),
                    ],
                ),
            ]
        )
    write_text(output_root / "error_analysis.md", "\n".join(md))
    print(f"wrote {len(case_rows)} per-query rows to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


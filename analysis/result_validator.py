"""Validate the final Phase 4 experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from .common import (
    DEFAULT_EVOLUTION_SUMMARY_PATH,
    DEFAULT_FIXED_PROMPT_VERSION,
    DEFAULT_PHASE4_ROOT,
    DEFAULT_RESULTS_ROOT,
    comparison_rows_for_split,
    duplicate_signatures,
    load_evolution_summary,
    load_experiment_config,
    load_records_for_run,
    load_run_matrix,
    markdown_table,
    recompute_mrr,
    scan_run_inventory,
    short_float,
    write_csv,
    write_json,
    write_text,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate saved Phase 4 results against raw artifacts.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_PHASE4_ROOT / "validation"))
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--conditions", nargs="+", default=["clean", "adv"])
    parser.add_argument("--fixed-prompt-version", default=DEFAULT_FIXED_PROMPT_VERSION)
    parser.add_argument("--evolution-summary", default=str(DEFAULT_EVOLUTION_SUMMARY_PATH))
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args(argv)


def _validate_run(
    run,
    *,
    fixed_prompt_version: str,
    evolution_summary: dict[str, Any],
    tolerance: float,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    records = load_records_for_run(run)
    recomputed = recompute_mrr(run)
    saved = run.mrr
    delta = abs(recomputed - saved)
    if delta > tolerance:
        errors.append(
            f"{run.run_name}: saved MRR {saved:.10f} != recomputed {recomputed:.10f}."
        )

    if run.evaluation.get("split") != run.split:
        errors.append(f"{run.run_name}: evaluation split mismatch.")
    if run.evaluation.get("condition") != run.condition:
        errors.append(f"{run.run_name}: evaluation condition mismatch.")
    if run.retrieval_metadata.get("split") != run.split:
        errors.append(f"{run.run_name}: retrieval metadata split mismatch.")
    if run.retrieval_metadata.get("condition") != run.condition:
        errors.append(f"{run.run_name}: retrieval metadata condition mismatch.")
    if Path(run.evaluation["prediction_path"]) != Path(run.retrieval_metadata["prediction_path"]):
        errors.append(f"{run.run_name}: prediction path differs between evaluation and retrieval metadata.")
    if Path(run.evaluation["reference_path"]) != Path(run.retrieval_metadata["reference_path"]):
        errors.append(f"{run.run_name}: reference path differs between evaluation and retrieval metadata.")

    if run.method_key == "raw":
        if run.retrieval_metadata.get("query_source") != "original":
            errors.append(f"{run.run_name}: raw baseline must use query_source=original.")
        if run.rewrite_metadata is not None:
            warnings.append(f"{run.run_name}: raw run unexpectedly has rewrite metadata.")
    else:
        if run.retrieval_metadata.get("query_source") != "rewritten":
            errors.append(f"{run.run_name}: rewritten methods must use query_source=rewritten.")
        if run.rewrite_metadata is None:
            errors.append(f"{run.run_name}: missing rewrite metadata.")
        else:
            if run.rewrite_metadata.get("split") != run.split:
                errors.append(f"{run.run_name}: rewrite split mismatch.")
            if run.rewrite_metadata.get("condition") != run.condition:
                errors.append(f"{run.run_name}: rewrite condition mismatch.")
            if int(run.rewrite_metadata.get("record_count", -1)) != len(records):
                errors.append(f"{run.run_name}: rewrite record_count does not match dataset size.")

        if run.method_key == "fixed" and run.rewrite_metadata is not None:
            if run.rewrite_metadata.get("prompt_version") != fixed_prompt_version:
                errors.append(
                    f"{run.run_name}: fixed-prompt run must use prompt_version={fixed_prompt_version}."
                )

        if run.method_key == "evolved" and run.rewrite_metadata is not None:
            best_prompt_path = Path(evolution_summary["best_prompt_path"]).resolve()
            rewrite_prompt_path = Path(run.rewrite_metadata["prompt_path"]).resolve()
            if rewrite_prompt_path != best_prompt_path:
                errors.append(
                    f"{run.run_name}: evolved run does not use best prompt {best_prompt_path}."
                )

    return errors, warnings, {
        "run_name": run.run_name,
        "method_key": run.method_key,
        "method": run.label,
        "split": run.split,
        "condition": run.condition,
        "saved_mrr": saved,
        "recomputed_mrr": recomputed,
        "mrr_delta": delta,
        "evaluation_path": str(run.evaluation_path),
        "prediction_path": str(run.prediction_path),
        "reference_path": str(run.reference_path),
        "retrieval_metadata_path": str(run.retrieval_metadata_path),
        "rewrite_metadata_path": None if run.rewrite_metadata_path is None else str(run.rewrite_metadata_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    config = load_experiment_config()
    evolution_summary = load_evolution_summary(Path(args.evolution_summary))

    try:
        matrix = load_run_matrix(
            splits=args.splits,
            conditions=args.conditions,
            results_root=results_root,
            config=config,
        )
    except Exception as exc:
        print(f"result_validator failed: {exc}", file=sys.stderr)
        return 1

    validation_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for split in args.splits:
        for condition in args.conditions:
            for method_key in ("raw", "fixed", "evolved"):
                run = matrix[split][condition][method_key]
                run_errors, run_warnings, row = _validate_run(
                    run,
                    fixed_prompt_version=args.fixed_prompt_version,
                    evolution_summary=evolution_summary,
                    tolerance=args.tolerance,
                )
                validation_rows.append(row)
                errors.extend(run_errors)
                warnings.extend(run_warnings)
                trace_rows.append(
                    {
                        "split": split,
                        "condition": condition,
                        "method_key": method_key,
                        "method": run.label,
                        "metric": "MRR",
                        "value": short_float(run.mrr),
                        "evaluation_path": str(run.evaluation_path),
                        "prediction_path": str(run.prediction_path),
                        "reference_path": str(run.reference_path),
                        "retrieval_metadata_path": str(run.retrieval_metadata_path),
                        "rewrite_metadata_path": "" if run.rewrite_metadata_path is None else str(run.rewrite_metadata_path),
                    }
                )

        for row in comparison_rows_for_split(split=split, results_root=results_root, config=config):
            trace_rows.append(
                {
                    "split": split,
                    "condition": "clean+adv",
                    "method_key": row["method_key"],
                    "method": row["method"],
                    "metric": "RobustnessDrop",
                    "value": short_float(row["robustness_drop"]),
                    "evaluation_path": f'{row["clean_evaluation_path"]} | {row["adv_evaluation_path"]}',
                    "prediction_path": "",
                    "reference_path": "",
                    "retrieval_metadata_path": "",
                    "rewrite_metadata_path": "",
                }
            )

    inventory_rows = scan_run_inventory(results_root)
    duplicate_rows = duplicate_signatures(inventory_rows)
    for duplicate in duplicate_rows:
        warnings.append(
            "duplicate run signature: "
            f"{duplicate['signature']} -> {', '.join(duplicate['run_names'])}"
        )

    report = {
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "validation_rows": validation_rows,
        "duplicate_signatures": duplicate_rows,
        "evolution_summary_path": str(Path(args.evolution_summary)),
    }
    write_json(output_root / "validation_report.json", report)
    write_csv(output_root / "traceability.csv", trace_rows)
    write_csv(output_root / "run_inventory.csv", inventory_rows)

    summary_rows = [
        {
            "run_name": row["run_name"],
            "method": row["method"],
            "split": row["split"],
            "condition": row["condition"],
            "saved_mrr": short_float(row["saved_mrr"]),
            "recomputed_mrr": short_float(row["recomputed_mrr"]),
            "mrr_delta": f"{row['mrr_delta']:.6g}",
        }
        for row in validation_rows
    ]
    summary_md = [
        "# Phase 4 Validation Summary",
        "",
        f"- Status: `{report['status']}`",
        f"- Errors: `{len(errors)}`",
        f"- Warnings: `{len(warnings)}`",
        f"- Evolution summary: `{args.evolution_summary}`",
        "",
        "## Run Checks",
        "",
        markdown_table(
            summary_rows,
            headers=[
                ("run_name", "Run"),
                ("method", "Method"),
                ("split", "Split"),
                ("condition", "Condition"),
                ("saved_mrr", "Saved MRR"),
                ("recomputed_mrr", "Recomputed MRR"),
                ("mrr_delta", "Abs Delta"),
            ],
        ),
    ]
    if warnings:
        summary_md.extend(["## Warnings", ""])
        summary_md.extend(f"- {warning}" for warning in warnings[:50])
        summary_md.append("")
    if errors:
        summary_md.extend(["## Errors", ""])
        summary_md.extend(f"- {error}" for error in errors[:50])
        summary_md.append("")
    write_text(output_root / "validation_summary.md", "\n".join(summary_md))

    print(f"validation status={report['status']} errors={len(errors)} warnings={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Generate paper-ready tables from validated experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .common import (
    DEFAULT_EVOLUTION_RUN_NAME,
    DEFAULT_FIGURES_ROOT,
    DEFAULT_PAPER_ROOT,
    DEFAULT_PHASE4_ROOT,
    DEFAULT_REPORTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    comparison_rows_for_split,
    evolution_progress_rows,
    latex_escape,
    load_evolution_history,
    load_experiment_config,
    markdown_table,
    short_float,
    write_csv,
    write_json,
    write_text,
)


def _comparison_markdown(rows: list[dict[str, object]]) -> str:
    printable = [
        {
            "method": row["method"],
            "clean_mrr": short_float(float(row["clean_mrr"])),
            "adv_mrr": short_float(float(row["adv_mrr"])),
            "robustness_drop": short_float(float(row["robustness_drop"])),
        }
        for row in rows
    ]
    return markdown_table(
        printable,
        headers=[
            ("method", "Method"),
            ("clean_mrr", "Clean MRR"),
            ("adv_mrr", "Adv MRR"),
            ("robustness_drop", "Robustness Drop"),
        ],
    )


def _comparison_latex(rows: list[dict[str, object]], *, caption: str, label: str) -> str:
    body = []
    for row in rows:
        body.append(
            f"{latex_escape(str(row['method']))} & "
            f"{float(row['clean_mrr']):.4f} & "
            f"{float(row['adv_mrr']):.4f} & "
            f"{float(row['robustness_drop']):.4f} \\\\"
        )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\begin{tabular}{lrrr}\n"
        "\\hline\n"
        "Method & Clean MRR & Adv MRR & Robustness Drop \\\\\n"
        "\\hline\n"
        + "\n".join(body)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        f"\\caption{{{latex_escape(caption)}}}\n"
        f"\\label{{{latex_escape(label)}}}\n"
        "\\end{table}\n"
    )


def _evolution_markdown(rows: list[dict[str, object]]) -> str:
    printable = [
        {
            "iteration": row["iteration"],
            "prompt_id": row["prompt_id"],
            "dev_mrr": short_float(float(row["dev_mrr"])),
            "accepted": row["accepted"],
        }
        for row in rows
    ]
    return markdown_table(
        printable,
        headers=[
            ("iteration", "Iteration"),
            ("prompt_id", "Prompt ID"),
            ("dev_mrr", "Dev MRR"),
            ("accepted", "Accepted"),
        ],
    )


def _evolution_latex(rows: list[dict[str, object]], *, caption: str, label: str) -> str:
    body = []
    for row in rows:
        body.append(
            f"{row['iteration']} & {latex_escape(str(row['prompt_id']))} & "
            f"{float(row['dev_mrr']):.4f} & {latex_escape(str(row['accepted']))} \\\\"
        )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\begin{tabular}{rlrl}\n"
        "\\hline\n"
        "Iteration & Prompt ID & Dev MRR & Accepted \\\\\n"
        "\\hline\n"
        + "\n".join(body)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        f"\\caption{{{latex_escape(caption)}}}\n"
        f"\\label{{{latex_escape(label)}}}\n"
        "\\end{table}\n"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 4 tables.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_PHASE4_ROOT / "tables"))
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--figures-root", default=str(DEFAULT_FIGURES_ROOT))
    parser.add_argument("--paper-root", default=str(DEFAULT_PAPER_ROOT))
    parser.add_argument("--evolution-run-name", default=DEFAULT_EVOLUTION_RUN_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    figures_root = Path(args.figures_root)
    paper_root = Path(args.paper_root)
    config = load_experiment_config()

    try:
        valid_rows = comparison_rows_for_split(split="valid", results_root=results_root, config=config)
        test_rows = comparison_rows_for_split(split="test", results_root=results_root, config=config)
        history = load_evolution_history(run_name=args.evolution_run_name)
        evolution_rows = evolution_progress_rows(history)
    except Exception as exc:
        print(f"generate_tables failed: {exc}", file=sys.stderr)
        return 1

    write_json(output_root / "comparison_valid.json", valid_rows)
    write_json(output_root / "comparison_test.json", test_rows)
    write_csv(output_root / "comparison_valid.csv", valid_rows)
    write_csv(output_root / "comparison_test.csv", test_rows)
    write_text(output_root / "comparison_valid.md", _comparison_markdown(valid_rows))
    write_text(output_root / "comparison_test.md", _comparison_markdown(test_rows))

    valid_tex = _comparison_latex(valid_rows, caption="Validation split comparison.", label="tab:phase4-valid")
    test_tex = _comparison_latex(test_rows, caption="Test split comparison.", label="tab:phase4-test")
    write_text(output_root / "comparison_valid.tex", valid_tex)
    write_text(output_root / "comparison_test.tex", test_tex)

    write_json(output_root / "evolution_progress.json", evolution_rows)
    write_csv(output_root / "evolution_progress.csv", evolution_rows)
    write_text(output_root / "evolution_progress.md", _evolution_markdown(evolution_rows))
    evolution_tex = _evolution_latex(
        evolution_rows,
        caption="Prompt evolution progress on the frozen dev subset.",
        label="tab:evolution-progress",
    )
    write_text(output_root / "evolution_progress.tex", evolution_tex)

    flattened_rows = []
    for row in valid_rows + test_rows:
        flattened_rows.append(
            {
                "split": row["split"],
                "method_key": row["method_key"],
                "method": row["method"],
                "clean_mrr": short_float(float(row["clean_mrr"])),
                "adv_mrr": short_float(float(row["adv_mrr"])),
                "robustness_drop": short_float(float(row["robustness_drop"])),
                "relative_drop": short_float(float(row["relative_drop"])),
                "clean_run_name": row["clean_run_name"],
                "adv_run_name": row["adv_run_name"],
            }
        )
    write_csv(reports_root / "final_results.csv", flattened_rows)

    summary_lines = [
        "# Final Comparison Summary",
        "",
        "## Validation Split",
        "",
        _comparison_markdown(valid_rows),
        "## Test Split",
        "",
        _comparison_markdown(test_rows),
        "## Evolution Progress",
        "",
        _evolution_markdown(evolution_rows),
    ]
    write_text(reports_root / "comparison_summary.md", "\n".join(summary_lines))

    figure_table_root = figures_root / "final_tables"
    write_text(figure_table_root / "comparison_valid.tex", valid_tex)
    write_text(figure_table_root / "comparison_test.tex", test_tex)
    write_text(figure_table_root / "evolution_progress.tex", evolution_tex)
    write_csv(figure_table_root / "final_results.csv", flattened_rows)

    paper_tables = "\n\n".join([valid_tex, test_tex, evolution_tex]) + "\n"
    write_text(paper_root / "final_tables.tex", paper_tables)

    print(f"wrote tables to {output_root}, {reports_root}, {figures_root / 'final_tables'}, and {paper_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


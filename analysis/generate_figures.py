"""Generate publication-ready Phase 4 figures."""

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
    DEFAULT_RESULTS_ROOT,
    build_case_rows,
    comparison_rows_for_split,
    evolution_progress_rows,
    load_evolution_history,
    load_experiment_config,
    write_json,
)


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ModuleNotFoundError as exc:
        raise RuntimeError("generate_figures requires matplotlib. Install it with `.venv/bin/python -m pip install matplotlib`.") from exc
    return plt, Circle


def _save_figure(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 4 figures.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_PHASE4_ROOT / "figures"))
    parser.add_argument("--figures-root", default=str(DEFAULT_FIGURES_ROOT))
    parser.add_argument("--paper-root", default=str(DEFAULT_PAPER_ROOT))
    parser.add_argument("--evolution-run-name", default=DEFAULT_EVOLUTION_RUN_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    figures_root = Path(args.figures_root)
    paper_root = Path(args.paper_root)

    try:
        test_rows = comparison_rows_for_split(split="test", results_root=results_root, config=config)
        valid_rows = comparison_rows_for_split(split="valid", results_root=results_root, config=config)
        history = load_evolution_history(run_name=args.evolution_run_name)
        evolution_rows = evolution_progress_rows(history)
        adv_case_rows, _ = build_case_rows(
            split="test",
            condition="adv",
            results_root=results_root,
            config=config,
        )
        plt, Circle = _import_matplotlib()
    except Exception as exc:
        print(f"generate_figures failed: {exc}", file=sys.stderr)
        return 1

    manifest: dict[str, str] = {}
    paper_fig_root = paper_root / "final_figures"
    paper_fig_root.mkdir(parents=True, exist_ok=True)

    methods = [row["method"] for row in test_rows]
    x_positions = list(range(len(methods)))
    clean_values = [float(row["clean_mrr"]) for row in test_rows]
    adv_values = [float(row["adv_mrr"]) for row in test_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([x - 0.18 for x in x_positions], clean_values, width=0.35, label="Clean", color="#2f6db3")
    ax.bar([x + 0.18 for x in x_positions], adv_values, width=0.35, label="Adv", color="#d37a2c")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("MRR")
    ax.set_title("Test Split MRR by Method")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    mrr_stem = figures_root / "plots" / "test_mrr_comparison"
    _save_figure(fig, mrr_stem)
    plt.close(fig)
    manifest["test_mrr_comparison"] = str(mrr_stem.with_suffix(".png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    valid_drop = [float(row["robustness_drop"]) for row in valid_rows]
    test_drop = [float(row["robustness_drop"]) for row in test_rows]
    ax.bar([x - 0.18 for x in x_positions], valid_drop, width=width, label="Valid", color="#3f8f5f")
    ax.bar([x + 0.18 for x in x_positions], test_drop, width=width, label="Test", color="#bf4b4b")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Clean MRR - Adv MRR")
    ax.set_title("Robustness Drop by Method")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    drop_stem = figures_root / "plots" / "robustness_drop"
    _save_figure(fig, drop_stem)
    plt.close(fig)
    manifest["robustness_drop"] = str(drop_stem.with_suffix(".png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    accepted_x = [int(row["iteration"]) for row in evolution_rows if row["accepted"] == "yes"]
    accepted_y = [float(row["dev_mrr"]) for row in evolution_rows if row["accepted"] == "yes"]
    rejected_x = [int(row["iteration"]) for row in evolution_rows if row["accepted"] != "yes"]
    rejected_y = [float(row["dev_mrr"]) for row in evolution_rows if row["accepted"] != "yes"]
    ax.plot([int(row["iteration"]) for row in evolution_rows], [float(row["dev_mrr"]) for row in evolution_rows], color="#444444", alpha=0.35)
    ax.scatter(accepted_x, accepted_y, color="#2f6db3", label="Accepted", zorder=3)
    if rejected_x:
        ax.scatter(rejected_x, rejected_y, color="#bf4b4b", label="Rejected", zorder=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Dev MRR")
    ax.set_title("Prompt Evolution Progress")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    evolution_stem = figures_root / "plots" / "evolution_progress"
    _save_figure(fig, evolution_stem)
    plt.close(fig)
    manifest["evolution_progress"] = str(evolution_stem.with_suffix(".png"))

    fixed_improved = {row["example_id"] for row in adv_case_rows if row["fixed_outcome"] == "improved"}
    evolved_improved = {row["example_id"] for row in adv_case_rows if row["evolved_outcome"] == "improved"}
    fixed_only = len(fixed_improved - evolved_improved)
    overlap = len(fixed_improved & evolved_improved)
    evolved_only = len(evolved_improved - fixed_improved)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.add_patch(Circle((0.42, 0.5), 0.22, color="#2f6db3", alpha=0.4))
    ax.add_patch(Circle((0.58, 0.5), 0.22, color="#d37a2c", alpha=0.4))
    ax.text(0.32, 0.78, "Fixed Improved", ha="center", va="center", fontsize=11)
    ax.text(0.68, 0.78, "Evolved Improved", ha="center", va="center", fontsize=11)
    ax.text(0.34, 0.50, str(fixed_only), ha="center", va="center", fontsize=18)
    ax.text(0.50, 0.50, str(overlap), ha="center", va="center", fontsize=18)
    ax.text(0.66, 0.50, str(evolved_only), ha="center", va="center", fontsize=18)
    ax.set_title("Improvement Overlap on Test Adversarial Queries")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.15, 0.9)
    ax.axis("off")
    venn_stem = figures_root / "venn" / "test_adv_improvement_overlap"
    _save_figure(fig, venn_stem)
    plt.close(fig)
    manifest["test_adv_improvement_overlap"] = str(venn_stem.with_suffix(".png"))

    for key, png_path in manifest.items():
        png = Path(png_path)
        pdf = png.with_suffix(".pdf")
        (paper_fig_root / png.name).write_bytes(png.read_bytes())
        (paper_fig_root / pdf.name).write_bytes(pdf.read_bytes())

    write_json(output_root / "figure_manifest.json", manifest)
    print(f"wrote figures to {figures_root} and {paper_fig_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sample representative qualitative examples for Phase 4 discussion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .common import (
    DEFAULT_PHASE4_ROOT,
    DEFAULT_RESULTS_ROOT,
    build_case_rows,
    load_experiment_config,
    sample_case_rows,
    short_float,
    write_csv,
    write_text,
)
from .openai_client import OpenAIClientError, chat_completion, maybe_load_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Phase 4 qualitative review examples.")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--condition", default="adv", choices=["clean", "adv"])
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_PHASE4_ROOT / "qualitative_review"))
    parser.add_argument("--annotator", choices=["auto", "heuristic", "openai"], default="auto")
    parser.add_argument("--model", default="gpt-4o-mini")
    return parser.parse_args(argv)


def _annotate_with_openai(row: dict[str, object], *, model: str) -> tuple[str, str]:
    prompt = (
        "You are reviewing BM25 code-search query rewrites.\n"
        "Explain in one sentence why the rewrite improved or failed.\n"
        "Focus on lexical retrieval: identifier splitting, term specificity, noise, or hallucination.\n"
        "Return compact JSON with keys summary and claim_strength.\n"
        f"Original query: {row['query_text']}\n"
        f"Fixed rewrite: {row['fixed_rewrite']}\n"
        f"Evolved rewrite: {row['evolved_rewrite']}\n"
        f"Raw rank: {row['raw_rank']}\n"
        f"Fixed rank: {row['fixed_rank']}\n"
        f"Evolved rank: {row['evolved_rank']}\n"
        f"Tags: {', '.join(row['tags'])}\n"
    )
    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.1,
        max_tokens=120,
    )
    payload = maybe_load_json(response)
    if payload is None:
        return response.strip(), "llm_freeform"
    summary = payload.get("summary")
    claim_strength = payload.get("claim_strength")
    if not isinstance(summary, str):
        summary = response.strip()
    if not isinstance(claim_strength, str):
        claim_strength = "llm_json"
    return summary.strip(), claim_strength


def _annotate_row(row: dict[str, object], *, annotator: str, model: str) -> tuple[str, str]:
    if annotator == "heuristic":
        if row["evolved_vs_fixed_outcome"] == "improved":
            return str(row["evolved_reason"]), "heuristic"
        if row["fixed_outcome"] == "improved":
            return str(row["fixed_reason"]), "heuristic"
        return str(row["evolved_reason"]), "heuristic"

    try:
        return _annotate_with_openai(row, model=model)
    except OpenAIClientError:
        if annotator == "openai":
            raise
        if row["evolved_vs_fixed_outcome"] == "improved":
            return str(row["evolved_reason"]), "heuristic_fallback"
        if row["fixed_outcome"] == "improved":
            return str(row["fixed_reason"]), "heuristic_fallback"
        return str(row["evolved_reason"]), "heuristic_fallback"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config()
    try:
        case_rows, _ = build_case_rows(
            split=args.split,
            condition=args.condition,
            results_root=Path(args.results_root),
            config=config,
        )
        sampled = sample_case_rows(case_rows, args.sample_size)
    except Exception as exc:
        print(f"qualitative_review failed: {exc}", file=sys.stderr)
        return 1

    output_root = Path(args.output_root)
    csv_rows = []
    markdown_sections = [
        "# Phase 4 Qualitative Review",
        "",
        f"- Split: `{args.split}`",
        f"- Condition: `{args.condition}`",
        f"- Sample size: `{len(sampled)}`",
        f"- Annotator: `{args.annotator}`",
        "",
    ]
    for row in sampled:
        try:
            annotation, annotation_source = _annotate_row(
                row,
                annotator=args.annotator,
                model=args.model,
            )
        except Exception as exc:
            print(f"qualitative_review failed during annotation: {exc}", file=sys.stderr)
            return 1

        csv_row = {
            "example_id": row["example_id"],
            "selection_bucket": row["selection_bucket"],
            "tags": ",".join(row["tags"]),
            "raw_rank": row["raw_rank"],
            "fixed_rank": row["fixed_rank"],
            "evolved_rank": row["evolved_rank"],
            "raw_rr": short_float(row["raw_rr"]),
            "fixed_rr": short_float(row["fixed_rr"]),
            "evolved_rr": short_float(row["evolved_rr"]),
            "query_text": row["query_text"],
            "fixed_rewrite": row["fixed_rewrite"],
            "evolved_rewrite": row["evolved_rewrite"],
            "annotation": annotation,
            "annotation_source": annotation_source,
            "manual_note": "",
            "paper_candidate": "",
        }
        csv_rows.append(csv_row)
        markdown_sections.extend(
            [
                f"## Example `{row['example_id']}`",
                "",
                f"- Selection bucket: `{row['selection_bucket']}`",
                f"- Tags: `{', '.join(row['tags']) or 'none'}`",
                f"- Raw rank: `{row['raw_rank']}`",
                f"- Fixed rank: `{row['fixed_rank']}`",
                f"- Evolved rank: `{row['evolved_rank']}`",
                f"- Annotation: {annotation}",
                "",
                "**Original query**",
                "",
                row["query_text"],
                "",
                "**Fixed rewrite**",
                "",
                str(row["fixed_rewrite"]),
                "",
                "**Evolved rewrite**",
                "",
                str(row["evolved_rewrite"]),
                "",
                "**Manual note**",
                "",
                "_Fill in during review._",
                "",
            ]
        )

    write_csv(output_root / "review_samples.csv", csv_rows)
    write_text(output_root / "review_samples.md", "\n".join(markdown_sections))

    jsonl_path = output_root / "review_samples.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in csv_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    print(f"wrote {len(csv_rows)} qualitative samples to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

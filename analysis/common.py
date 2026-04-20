"""Shared helpers for Phase 4 analysis scripts."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.loader import DEFAULT_MANIFEST_ROOT, DEFAULT_PROCESSED_ROOT, DatasetLoader
from src.evaluation.official_wrapper import score_prediction_file
from src.rewriting.cache_manager import load_rewrite_records


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_PHASE4_ROOT = DEFAULT_RESULTS_ROOT / "phase4"
DEFAULT_REPORTS_ROOT = REPO_ROOT / "reports"
DEFAULT_FIGURES_ROOT = REPO_ROOT / "figures"
DEFAULT_PAPER_ROOT = REPO_ROOT / "paper"
DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "src" / "configs" / "experiment.yaml"
DEFAULT_EVOLUTION_SUMMARY_PATH = REPO_ROOT / "evolution_logs" / "summaries" / "evolve_v1_summary.json"

DEFAULT_METHOD_LABELS = {
    "raw": "Raw Query BM25",
    "fixed": "Fixed Prompt Rewrite",
    "evolved": "Evolved Prompt Rewrite",
}
DEFAULT_RUN_NAMES = {
    "raw": "raw_{split}_{condition}",
    "fixed": "rewritten_{split}_{condition}",
    "evolved": "evolved_bm25_{split}_{condition}",
}
DEFAULT_FIXED_PROMPT_VERSION = "fixed_prompt_v1"
DEFAULT_EVOLUTION_RUN_NAME = "evolve_v1"
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "get",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "return",
    "set",
    "that",
    "the",
    "this",
    "to",
    "update",
    "with",
}
VERB_HINTS = {
    "add",
    "build",
    "calculate",
    "check",
    "collect",
    "convert",
    "create",
    "delete",
    "fetch",
    "find",
    "format",
    "generate",
    "get",
    "load",
    "map",
    "parse",
    "remove",
    "render",
    "resolve",
    "return",
    "save",
    "serialize",
    "transform",
    "update",
    "validate",
}


@dataclass(frozen=True)
class RunArtifact:
    """Resolved artifact bundle for one evaluated run."""

    method_key: str
    label: str
    split: str
    condition: str
    run_name: str
    evaluation_path: Path
    retrieval_metadata_path: Path
    evaluation: dict[str, Any]
    retrieval_metadata: dict[str, Any]
    rewrite_metadata_path: Path | None = None
    rewrite_metadata: dict[str, Any] | None = None

    @property
    def mrr(self) -> float:
        return float(self.evaluation["metrics"]["mrr"])

    @property
    def prediction_path(self) -> Path:
        return Path(self.evaluation["prediction_path"])

    @property
    def reference_path(self) -> Path:
        return Path(self.evaluation["reference_path"])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path | str) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {resolved}.")
    return payload


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object rows in {path}.")
                rows.append(payload)
    return rows


def write_json(path: Path | str, payload: Any) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path | str, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    rows_list = list(rows)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows_list:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def write_text(path: Path | str, text: str) -> None:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copyfile(src, dst)


def markdown_table(rows: Sequence[dict[str, Any]], headers: Sequence[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "")) for key, _ in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(ch, ch) for ch in text)


def short_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def load_experiment_config(config_path: Path = DEFAULT_EXPERIMENT_CONFIG) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return load_json(config_path)


def method_label(method_key: str, config: dict[str, Any] | None = None) -> str:
    config = config or {}
    if method_key == "raw":
        return str(config.get("raw_baseline_label", DEFAULT_METHOD_LABELS["raw"]))
    if method_key == "fixed":
        return str(config.get("fixed_prompt_label", DEFAULT_METHOD_LABELS["fixed"]))
    if method_key == "evolved":
        return str(config.get("evolved_prompt_label", DEFAULT_METHOD_LABELS["evolved"]))
    raise ValueError(f"Unknown method_key={method_key}")


def default_run_name(method_key: str, split: str, condition: str) -> str:
    template = DEFAULT_RUN_NAMES.get(method_key)
    if template is None:
        raise ValueError(f"Unknown method_key={method_key}")
    return template.format(split=split, condition=condition)


def load_evolution_summary(path: Path = DEFAULT_EVOLUTION_SUMMARY_PATH) -> dict[str, Any]:
    return load_json(path)


def _load_rewrite_metadata_for_run(retrieval_metadata: dict[str, Any]) -> tuple[Path, dict[str, Any]] | tuple[None, None]:
    rewrite_cache_path = retrieval_metadata.get("rewrite_cache_path")
    if not rewrite_cache_path:
        return None, None
    cache_path = Path(rewrite_cache_path)
    metadata_path = cache_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Rewrite metadata not found: {metadata_path}")
    return metadata_path, load_json(metadata_path)


def resolve_run_artifact(
    *,
    method_key: str,
    split: str,
    condition: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> RunArtifact:
    resolved_run_name = run_name or default_run_name(method_key, split, condition)
    evaluation_path = results_root / "evaluations" / f"{resolved_run_name}.json"
    retrieval_metadata_path = results_root / "bm25" / resolved_run_name / "metadata.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation not found: {evaluation_path}")
    if not retrieval_metadata_path.exists():
        raise FileNotFoundError(f"Retrieval metadata not found: {retrieval_metadata_path}")

    evaluation = load_json(evaluation_path)
    retrieval_metadata = load_json(retrieval_metadata_path)
    rewrite_metadata_path, rewrite_metadata = _load_rewrite_metadata_for_run(retrieval_metadata)
    return RunArtifact(
        method_key=method_key,
        label=method_label(method_key, config=config),
        split=split,
        condition=condition,
        run_name=resolved_run_name,
        evaluation_path=evaluation_path,
        retrieval_metadata_path=retrieval_metadata_path,
        evaluation=evaluation,
        retrieval_metadata=retrieval_metadata,
        rewrite_metadata_path=rewrite_metadata_path,
        rewrite_metadata=rewrite_metadata,
    )


def load_run_matrix(
    *,
    splits: Sequence[str],
    conditions: Sequence[str],
    results_root: Path = DEFAULT_RESULTS_ROOT,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, RunArtifact]]]:
    matrix: dict[str, dict[str, dict[str, RunArtifact]]] = {}
    for split in splits:
        by_condition: dict[str, dict[str, RunArtifact]] = {}
        for condition in conditions:
            by_condition[condition] = {
                "raw": resolve_run_artifact(
                    method_key="raw",
                    split=split,
                    condition=condition,
                    results_root=results_root,
                    config=config,
                ),
                "fixed": resolve_run_artifact(
                    method_key="fixed",
                    split=split,
                    condition=condition,
                    results_root=results_root,
                    config=config,
                ),
                "evolved": resolve_run_artifact(
                    method_key="evolved",
                    split=split,
                    condition=condition,
                    results_root=results_root,
                    config=config,
                ),
            }
        matrix[split] = by_condition
    return matrix


def scan_run_inventory(results_root: Path = DEFAULT_RESULTS_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted((results_root / "bm25").glob("*/metadata.json")):
        run_metadata = load_json(metadata_path)
        run_name = str(run_metadata.get("run_name"))
        evaluation_path = results_root / "evaluations" / f"{run_name}.json"
        evaluation = load_json(evaluation_path) if evaluation_path.exists() else None
        rewrite_metadata_path, rewrite_metadata = _load_rewrite_metadata_for_run(run_metadata)
        rows.append(
            {
                "run_name": run_name,
                "split": run_metadata.get("split"),
                "condition": run_metadata.get("condition"),
                "query_source": run_metadata.get("query_source"),
                "prompt_version": None if rewrite_metadata is None else rewrite_metadata.get("prompt_version"),
                "prompt_path": None if rewrite_metadata is None else rewrite_metadata.get("prompt_path"),
                "rewrite_run_name": run_metadata.get("rewrite_run_name"),
                "retrieval_metadata_path": str(metadata_path),
                "rewrite_metadata_path": None if rewrite_metadata_path is None else str(rewrite_metadata_path),
                "evaluation_path": str(evaluation_path) if evaluation_path.exists() else None,
                "mrr": None if evaluation is None else float(evaluation["metrics"]["mrr"]),
                "timestamp": run_metadata.get("timestamp"),
            }
        )
    return rows


def duplicate_signatures(inventory_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in inventory_rows:
        signature = (
            row.get("split"),
            row.get("condition"),
            row.get("query_source"),
            row.get("prompt_version"),
            row.get("prompt_path"),
        )
        buckets.setdefault(signature, []).append(dict(row))

    duplicates: list[dict[str, Any]] = []
    for signature, rows in buckets.items():
        if len(rows) > 1:
            duplicates.append(
                {
                    "signature": list(signature),
                    "run_names": [row["run_name"] for row in rows],
                    "count": len(rows),
                }
            )
    return sorted(duplicates, key=lambda item: item["count"], reverse=True)


def load_records_for_run(
    run: RunArtifact,
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    manifest_root: Path = DEFAULT_MANIFEST_ROOT,
):
    loader = DatasetLoader(processed_root=processed_root, manifest_root=manifest_root)
    return loader.load_split(run.split, run.condition)


def recompute_mrr(
    run: RunArtifact,
    *,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    manifest_root: Path = DEFAULT_MANIFEST_ROOT,
) -> float:
    records = load_records_for_run(run, processed_root=processed_root, manifest_root=manifest_root)
    scores = score_prediction_file(records, run.prediction_path)
    return float(scores["MRR"])


def load_prediction_rows(prediction_path: Path | str) -> list[dict[str, Any]]:
    rows = load_jsonl(prediction_path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        url = row.get("url")
        answers = row.get("answers")
        if not isinstance(url, str):
            raise ValueError(f"Prediction row missing url in {prediction_path}")
        if not isinstance(answers, list):
            raise ValueError(f"Prediction row missing answers in {prediction_path}")
        normalized.append({"url": url, "answers": list(answers)})
    return normalized


def rank_of_answer(answers: Sequence[Any], correct_answer: Any) -> int | None:
    for index, candidate in enumerate(answers, start=1):
        if candidate == correct_answer:
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank


def rank_map_for_run(run: RunArtifact) -> dict[str, dict[str, Any]]:
    records = load_records_for_run(run)
    predictions = load_prediction_rows(run.prediction_path)
    predictions_by_url = {row["url"]: row["answers"] for row in predictions}
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        url = record.metadata["url"]
        correct_idx = record.metadata["idx"]
        answers = predictions_by_url[url]
        rank = rank_of_answer(answers, correct_idx)
        rows[record.example_id] = {
            "example_id": record.example_id,
            "url": url,
            "correct_idx": correct_idx,
            "rank": rank,
            "rr": reciprocal_rank(rank),
            "top_prediction": None if not answers else answers[0],
        }
    return rows


def load_rewrite_payloads(run: RunArtifact) -> dict[str, dict[str, Any]]:
    if run.rewrite_metadata is None:
        return {}
    cache_path = Path(run.rewrite_metadata["cache_path"])
    return load_rewrite_records(cache_path)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_./:+-]+", text.lower())


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize_text(text) if token not in STOPWORDS}


def infer_query_tags(original_query: str, rewritten_query: str | None = None) -> list[str]:
    tags: list[str] = []
    token_count = len(tokenize_text(original_query))
    has_identifier_marker = bool(
        re.search(r"[A-Za-z0-9]+_[A-Za-z0-9]+|[A-Za-z0-9]+\.[A-Za-z0-9]+|`[^`]+`|[a-z][A-Z]", original_query)
    )
    if has_identifier_marker:
        tags.append("identifier_heavy")
    if token_count <= 6 and not has_identifier_marker:
        tags.append("ambiguous_natural_language")
    lowered_tokens = tokenize_text(original_query)
    if lowered_tokens and lowered_tokens[0] in VERB_HINTS:
        tags.append("functionality_focused")
    if "\n" in original_query or token_count >= 24:
        tags.append("long_form_query")

    if rewritten_query:
        original_tokens = content_tokens(original_query)
        rewritten_tokens = content_tokens(rewritten_query)
        novel_tokens = rewritten_tokens - original_tokens
        if rewritten_tokens and len(novel_tokens) / len(rewritten_tokens) >= 0.5:
            tags.append("hallucination_risk")
        if rewritten_tokens and len(rewritten_tokens) <= max(3, len(tokenize_text(original_query)) // 3):
            overlap = len(rewritten_tokens & original_tokens)
            if overlap <= max(1, len(rewritten_tokens) // 3):
                tags.append("over_generalized")
    return sorted(set(tags))


def outcome_label(delta_rr: float, *, epsilon: float = 1e-12) -> str:
    if delta_rr > epsilon:
        return "improved"
    if delta_rr < -epsilon:
        return "worse"
    return "same"


def infer_reason(
    *,
    original_query: str,
    rewritten_query: str,
    tags: Sequence[str],
    delta_rr: float,
) -> str:
    original_tokens = content_tokens(original_query)
    rewritten_tokens = content_tokens(rewritten_query)
    novel_tokens = rewritten_tokens - original_tokens

    if delta_rr > 0:
        if "identifier_heavy" in tags:
            return "Identifier splitting likely exposed lexical matches already present in code and docstrings."
        if "long_form_query" in tags and len(rewritten_tokens) < len(original_tokens):
            return "The rewrite trimmed noisy context and concentrated BM25 weight on the core behavior terms."
        if "ambiguous_natural_language" in tags:
            return "The rewrite made the intended functionality more explicit for lexical retrieval."
        return "The rewrite appears to align the query more closely with documentation-style vocabulary."

    if "hallucination_risk" in tags and novel_tokens:
        return "The rewrite introduced unsupported terms, which likely diluted the lexical signal."
    if "over_generalized" in tags:
        return "The rewrite became too generic and lost discriminative tokens needed for ranking."
    if len(rewritten_tokens) > max(6, len(original_tokens) * 2):
        return "The rewrite added too much text, which likely spread BM25 weight across irrelevant terms."
    return "The rewrite did not preserve the strongest lexical hooks for the target code example."


def build_case_rows(
    *,
    split: str,
    condition: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, RunArtifact]]:
    runs = {
        key: resolve_run_artifact(
            method_key=key,
            split=split,
            condition=condition,
            results_root=results_root,
            config=config,
        )
        for key in ("raw", "fixed", "evolved")
    }
    records = load_records_for_run(runs["raw"])
    raw_ranks = rank_map_for_run(runs["raw"])
    fixed_ranks = rank_map_for_run(runs["fixed"])
    evolved_ranks = rank_map_for_run(runs["evolved"])
    fixed_rewrites = load_rewrite_payloads(runs["fixed"])
    evolved_rewrites = load_rewrite_payloads(runs["evolved"])

    rows: list[dict[str, Any]] = []
    for record in records:
        raw_rank = raw_ranks[record.example_id]["rank"]
        fixed_rank = fixed_ranks[record.example_id]["rank"]
        evolved_rank = evolved_ranks[record.example_id]["rank"]
        raw_rr = raw_ranks[record.example_id]["rr"]
        fixed_rr = fixed_ranks[record.example_id]["rr"]
        evolved_rr = evolved_ranks[record.example_id]["rr"]
        fixed_rewrite = fixed_rewrites[record.example_id]["rewritten_query"]
        evolved_rewrite = evolved_rewrites[record.example_id]["rewritten_query"]
        merged_tags = sorted(
            set(infer_query_tags(record.query_text, fixed_rewrite))
            | set(infer_query_tags(record.query_text, evolved_rewrite))
        )

        row = {
            "example_id": record.example_id,
            "query_text": record.query_text,
            "fixed_rewrite": fixed_rewrite,
            "evolved_rewrite": evolved_rewrite,
            "raw_rank": raw_rank,
            "fixed_rank": fixed_rank,
            "evolved_rank": evolved_rank,
            "raw_rr": raw_rr,
            "fixed_rr": fixed_rr,
            "evolved_rr": evolved_rr,
            "fixed_delta_rr": fixed_rr - raw_rr,
            "evolved_delta_rr": evolved_rr - raw_rr,
            "evolved_vs_fixed_delta_rr": evolved_rr - fixed_rr,
            "fixed_outcome": outcome_label(fixed_rr - raw_rr),
            "evolved_outcome": outcome_label(evolved_rr - raw_rr),
            "evolved_vs_fixed_outcome": outcome_label(evolved_rr - fixed_rr),
            "tags": merged_tags,
        }
        row["fixed_reason"] = infer_reason(
            original_query=record.query_text,
            rewritten_query=fixed_rewrite,
            tags=merged_tags,
            delta_rr=row["fixed_delta_rr"],
        )
        row["evolved_reason"] = infer_reason(
            original_query=record.query_text,
            rewritten_query=evolved_rewrite,
            tags=merged_tags,
            delta_rr=row["evolved_delta_rr"],
        )
        rows.append(row)

    return rows, runs


def summarize_case_rows(case_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(case_rows),
        "fixed": {"improved": 0, "worse": 0, "same": 0, "mean_delta_rr": 0.0},
        "evolved": {"improved": 0, "worse": 0, "same": 0, "mean_delta_rr": 0.0},
        "evolved_vs_fixed": {"improved": 0, "worse": 0, "same": 0, "mean_delta_rr": 0.0},
        "tags": {},
    }
    if not case_rows:
        return summary

    fixed_total = 0.0
    evolved_total = 0.0
    evolved_vs_fixed_total = 0.0
    for row in case_rows:
        fixed_label = row["fixed_outcome"]
        evolved_label = row["evolved_outcome"]
        evolved_vs_fixed_label = row["evolved_vs_fixed_outcome"]
        summary["fixed"][fixed_label] += 1
        summary["evolved"][evolved_label] += 1
        summary["evolved_vs_fixed"][evolved_vs_fixed_label] += 1
        fixed_total += float(row["fixed_delta_rr"])
        evolved_total += float(row["evolved_delta_rr"])
        evolved_vs_fixed_total += float(row["evolved_vs_fixed_delta_rr"])

        for tag in row["tags"]:
            bucket = summary["tags"].setdefault(
                tag,
                {
                    "count": 0,
                    "fixed_improved": 0,
                    "fixed_worse": 0,
                    "evolved_improved": 0,
                    "evolved_worse": 0,
                },
            )
            bucket["count"] += 1
            if row["fixed_outcome"] == "improved":
                bucket["fixed_improved"] += 1
            elif row["fixed_outcome"] == "worse":
                bucket["fixed_worse"] += 1
            if row["evolved_outcome"] == "improved":
                bucket["evolved_improved"] += 1
            elif row["evolved_outcome"] == "worse":
                bucket["evolved_worse"] += 1

    n = len(case_rows)
    summary["fixed"]["mean_delta_rr"] = fixed_total / n
    summary["evolved"]["mean_delta_rr"] = evolved_total / n
    summary["evolved_vs_fixed"]["mean_delta_rr"] = evolved_vs_fixed_total / n
    return summary


def sample_case_rows(case_rows: Sequence[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    if not case_rows:
        return []

    def score_abs(row: dict[str, Any], key: str) -> tuple[float, str]:
        return (abs(float(row[key])), str(row["example_id"]))

    buckets: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "fixed_success",
            sorted(
                [row for row in case_rows if row["fixed_outcome"] == "improved"],
                key=lambda row: score_abs(row, "fixed_delta_rr"),
                reverse=True,
            ),
        ),
        (
            "fixed_failure",
            sorted(
                [row for row in case_rows if row["fixed_outcome"] == "worse"],
                key=lambda row: score_abs(row, "fixed_delta_rr"),
                reverse=True,
            ),
        ),
        (
            "evolved_success",
            sorted(
                [row for row in case_rows if row["evolved_vs_fixed_outcome"] == "improved"],
                key=lambda row: score_abs(row, "evolved_vs_fixed_delta_rr"),
                reverse=True,
            ),
        ),
        (
            "evolved_failure",
            sorted(
                [row for row in case_rows if row["evolved_vs_fixed_outcome"] == "worse"],
                key=lambda row: score_abs(row, "evolved_vs_fixed_delta_rr"),
                reverse=True,
            ),
        ),
        (
            "identifier_heavy",
            sorted(
                [row for row in case_rows if "identifier_heavy" in row["tags"]],
                key=lambda row: score_abs(row, "evolved_delta_rr"),
                reverse=True,
            ),
        ),
        (
            "ambiguous_natural_language",
            sorted(
                [row for row in case_rows if "ambiguous_natural_language" in row["tags"]],
                key=lambda row: score_abs(row, "evolved_delta_rr"),
                reverse=True,
            ),
        ),
        (
            "functionality_focused",
            sorted(
                [row for row in case_rows if "functionality_focused" in row["tags"]],
                key=lambda row: score_abs(row, "evolved_delta_rr"),
                reverse=True,
            ),
        ),
        (
            "failure_modes",
            sorted(
                [
                    row
                    for row in case_rows
                    if "hallucination_risk" in row["tags"] or "over_generalized" in row["tags"]
                ],
                key=lambda row: score_abs(row, "evolved_delta_rr"),
                reverse=True,
            ),
        ),
    ]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < sample_size:
        made_progress = False
        for bucket_name, bucket_rows in buckets:
            while bucket_rows and bucket_rows[0]["example_id"] in seen:
                bucket_rows.pop(0)
            if not bucket_rows:
                continue
            row = dict(bucket_rows.pop(0))
            row["selection_bucket"] = bucket_name
            selected.append(row)
            seen.add(row["example_id"])
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break

    if len(selected) < sample_size:
        filler = sorted(
            case_rows,
            key=lambda row: (
                max(
                    abs(float(row["fixed_delta_rr"])),
                    abs(float(row["evolved_delta_rr"])),
                    abs(float(row["evolved_vs_fixed_delta_rr"])),
                ),
                str(row["example_id"]),
            ),
            reverse=True,
        )
        for row in filler:
            if row["example_id"] in seen:
                continue
            filled = dict(row)
            filled["selection_bucket"] = "largest_absolute_change"
            selected.append(filled)
            seen.add(row["example_id"])
            if len(selected) >= sample_size:
                break
    return selected


def comparison_rows_for_split(
    *,
    split: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_key in ("raw", "fixed", "evolved"):
        clean_run = resolve_run_artifact(
            method_key=method_key,
            split=split,
            condition="clean",
            results_root=results_root,
            config=config,
        )
        adv_run = resolve_run_artifact(
            method_key=method_key,
            split=split,
            condition="adv",
            results_root=results_root,
            config=config,
        )
        clean_mrr = clean_run.mrr
        adv_mrr = adv_run.mrr
        robustness_drop = clean_mrr - adv_mrr
        rows.append(
            {
                "method_key": method_key,
                "method": clean_run.label,
                "split": split,
                "clean_mrr": clean_mrr,
                "adv_mrr": adv_mrr,
                "robustness_drop": robustness_drop,
                "relative_drop": None if clean_mrr == 0 else robustness_drop / clean_mrr,
                "clean_run_name": clean_run.run_name,
                "adv_run_name": adv_run.run_name,
                "clean_evaluation_path": str(clean_run.evaluation_path),
                "adv_evaluation_path": str(adv_run.evaluation_path),
            }
        )
    return rows


def load_evolution_history(
    path: Path | None = None,
    *,
    run_name: str = DEFAULT_EVOLUTION_RUN_NAME,
) -> dict[str, Any]:
    resolved = path or (REPO_ROOT / "evolution_logs" / "summaries" / f"{run_name}_history.json")
    return load_json(resolved)


def evolution_progress_rows(history: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = history.get("prompts", [])
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        iteration = int(prompt["iteration"])
        status = str(prompt["status"])
        rows.append(
            {
                "iteration": iteration,
                "prompt_id": str(prompt["prompt_id"]),
                "dev_mrr": float(prompt["dev_mrr"]),
                "accepted": "yes" if status == "accepted" else "no",
                "generation_method": str(prompt["generation_method"]),
                "status": status,
            }
        )
    return sorted(rows, key=lambda row: (row["iteration"], row["prompt_id"]))


def format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isfinite(value):
        return value
    return None

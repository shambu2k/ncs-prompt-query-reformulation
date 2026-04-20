"""Thin wrapper around the official CodeXGLUE evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Iterable

from src.data.schema import CanonicalRecord, DatasetValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATOR_PATH = (
    REPO_ROOT / "CodeXGLUE" / "Text-Code" / "NL-code-search-Adv" / "evaluator" / "evaluator.py"
)


def _normalize_json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _normalize_json_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(inner) for inner in value]
    return value


def _load_official_module(evaluator_path: Path) -> Any:
    if not evaluator_path.exists():
        raise FileNotFoundError(f"Official evaluator not found: {evaluator_path}")

    spec = importlib.util.spec_from_file_location("codexglue_official_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load official evaluator module from {evaluator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_answer_mapping(records: Iterable[CanonicalRecord]) -> dict[str, int]:
    answers: dict[str, int] = {}
    for record in records:
        url = record.metadata.get("url")
        idx = record.metadata.get("idx")
        if not isinstance(url, str) or not url:
            raise DatasetValidationError(
                f"Record {record.example_id} is missing metadata.url for evaluation."
            )
        if not isinstance(idx, int):
            raise DatasetValidationError(
                f"Record {record.example_id} is missing integer metadata.idx for evaluation."
            )
        if url in answers:
            raise DatasetValidationError(f"Duplicate evaluation URL detected: {url}")
        answers[url] = idx
    return answers


def score_prediction_file(
    records: Iterable[CanonicalRecord],
    prediction_path: Path | str,
    *,
    evaluator_path: Path | str = DEFAULT_EVALUATOR_PATH,
) -> dict[str, Any]:
    evaluator_path = Path(evaluator_path)
    prediction_path = Path(prediction_path)
    module = _load_official_module(evaluator_path)
    answers = build_answer_mapping(records)
    predictions = module.read_predictions(str(prediction_path))

    expected_urls = set(answers)
    predicted_urls = set(predictions)
    if expected_urls != predicted_urls:
        missing = sorted(expected_urls - predicted_urls)
        extras = sorted(predicted_urls - expected_urls)
        details = []
        if missing:
            details.append(f"missing {len(missing)} urls")
        if extras:
            details.append(f"unexpected {len(extras)} urls")
        raise DatasetValidationError(
            f"Prediction/reference mismatch for {prediction_path}: {', '.join(details)}."
        )

    scores = module.calculate_scores(answers, predictions)
    return _normalize_json_value(scores)

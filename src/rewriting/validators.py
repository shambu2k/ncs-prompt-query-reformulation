"""Validation for rewritten search queries."""

from __future__ import annotations

import re
from dataclasses import dataclass


PROMPT_LEAKAGE_PATTERNS = [
    "rewrite the input query",
    "return only one rewritten search query",
    "original query:",
    "you rewrite software code-search queries",
]

CODE_PATTERNS = [
    r"```",
    r"\bdef\b",
    r"\bclass\b",
    r"\breturn\b",
    r"[{};]",
]


@dataclass(frozen=True)
class RewriteValidationResult:
    """Normalized validation result."""

    rewritten_query: str
    rewritten_query_tokens: list[str]
    validation_status: str
    reason: str | None


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in _normalize_whitespace(text).split(" ") if token]


def validate_rewritten_query(original_query: str, rewritten_query: str) -> RewriteValidationResult:
    original = _normalize_whitespace(original_query)
    candidate = _normalize_whitespace(rewritten_query)

    if not candidate:
        return RewriteValidationResult(original, _tokens(original), "fallback", "empty_output")

    lowered = candidate.lower()
    for pattern in PROMPT_LEAKAGE_PATTERNS:
        if pattern in lowered:
            return RewriteValidationResult(original, _tokens(original), "fallback", "prompt_leakage")

    if not re.search(r"[A-Za-z]", candidate):
        return RewriteValidationResult(original, _tokens(original), "fallback", "malformed_text")

    for pattern in CODE_PATTERNS:
        if re.search(pattern, candidate):
            return RewriteValidationResult(original, _tokens(original), "fallback", "generated_code")

    original_tokens = _tokens(original)
    candidate_tokens = _tokens(candidate)
    if len(candidate_tokens) > max(64, max(1, len(original_tokens)) * 3):
        return RewriteValidationResult(original, _tokens(original), "fallback", "length_explosion")

    if len(candidate) > max(400, len(original) * 4):
        return RewriteValidationResult(original, _tokens(original), "fallback", "length_explosion")

    return RewriteValidationResult(candidate, candidate_tokens, "passed", None)

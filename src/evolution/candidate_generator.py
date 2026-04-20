"""Generate candidate prompt mutations for evolution."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES_ROOT = REPO_ROOT / "prompts" / "candidates"

# Deterministic heuristic mutations applied when LLM provider is unavailable.
# Each entry is (strategy_name, additional_instruction).
_HEURISTIC_MUTATIONS: list[tuple[str, str]] = [
    (
        "abstract_identifiers",
        "In addition: Replace all identifier names with abstract descriptions of their purpose.",
    ),
    (
        "doc_vocabulary",
        "In addition: Use vocabulary that commonly appears in code documentation and docstrings.",
    ),
    (
        "io_behavior",
        "In addition: Describe the function's inputs, outputs, and primary side effects explicitly.",
    ),
    (
        "action_verb",
        "In addition: Begin the rewrite with a strong action verb that describes the primary operation.",
    ),
    (
        "ultra_concise",
        "In addition: Keep the rewrite to at most 10 words while preserving core retrieval intent.",
    ),
    (
        "no_class_names",
        "In addition: Do not include any class names, module names, or package names in the rewrite.",
    ),
    (
        "behavior_only",
        "In addition: Describe only observable behavior, not any implementation details.",
    ),
    (
        "retrieval_terms",
        "In addition: Choose words that a developer searching for this functionality would naturally use.",
    ),
    (
        "no_synonyms",
        "In addition: Avoid paraphrasing that introduces synonyms not present in typical code documentation.",
    ),
    (
        "generalize",
        "In addition: Generalize the query beyond any specific library or framework when possible.",
    ),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _prompt_id(run_name: str, iteration: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{run_name}_iter{iteration:03d}_{ts}"


def _heuristic_mutate(parent_text: str, iteration: int) -> str:
    _, instruction = _HEURISTIC_MUTATIONS[iteration % len(_HEURISTIC_MUTATIONS)]
    return f"{parent_text}\n{instruction}"


def _llm_generate(
    parent_text: str,
    parent_mrr: float | None,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("provider=openai requires the requests package.") from exc

    mrr_line = f"Current dev-set MRR: {parent_mrr:.4f}\n" if parent_mrr is not None else ""
    meta_prompt = (
        "You are improving a code-search query rewrite prompt for BM25 lexical retrieval.\n"
        f"{mrr_line}"
        "Generate one improved version of the prompt below.\n"
        "The improved prompt should help rewrite software search queries so they retrieve "
        "relevant code more accurately using BM25 lexical matching.\n\n"
        "Requirements for the new prompt:\n"
        "- Focus on rewriting for retrieval, not code generation\n"
        "- Emphasize functional behavior and purpose over identifier names\n"
        "- Prefer concise natural language that matches code documentation vocabulary\n"
        "- Do not add hallucination-prone or overly prescriptive instructions\n"
        "- Keep the prompt under 200 words\n"
        "- Preserve the core instruction structure\n\n"
        "Current prompt:\n"
        "---\n"
        f"{parent_text}\n"
        "---\n\n"
        "Return only the new prompt text. No explanation, no preamble."
    )
    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    url = f"{api_base}/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": meta_prompt}],
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


class CandidateGenerator:
    """Generate one candidate prompt per call, with optional LLM or heuristic backend."""

    def __init__(
        self,
        *,
        provider: str = "auto",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        candidates_root: Path = DEFAULT_CANDIDATES_ROOT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.candidates_root = Path(candidates_root)
        self.candidates_root.mkdir(parents=True, exist_ok=True)

        if provider == "auto":
            self.provider = "openai" if os.environ.get("OPENAI_API_KEY") else "heuristic"
        else:
            self.provider = provider

    def generate(
        self,
        *,
        parent_prompt_id: str,
        parent_text: str,
        parent_mrr: float | None,
        iteration: int,
        run_name: str,
    ) -> dict[str, Any]:
        """Generate one candidate and save it to disk. Returns the candidate record."""
        if self.provider == "openai":
            try:
                candidate_text = _llm_generate(
                    parent_text,
                    parent_mrr,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                generation_method = f"openai:{self.model}"
            except Exception as exc:
                candidate_text = _heuristic_mutate(parent_text, iteration)
                generation_method = f"heuristic_fallback:{type(exc).__name__}"
        else:
            candidate_text = _heuristic_mutate(parent_text, iteration)
            generation_method = "heuristic"

        prompt_id = _prompt_id(run_name, iteration)
        record: dict[str, Any] = {
            "prompt_id": prompt_id,
            "parent_prompt_id": parent_prompt_id,
            "prompt_text": candidate_text,
            "iteration": iteration,
            "generator_model": self.model if self.provider == "openai" else "heuristic",
            "generation_method": generation_method,
            "timestamp": _utc_now(),
            "status": "generated",
        }

        candidate_txt = self.candidates_root / f"{prompt_id}.txt"
        candidate_json = self.candidates_root / f"{prompt_id}.json"
        candidate_txt.write_text(candidate_text, encoding="utf-8")
        candidate_json.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        return record

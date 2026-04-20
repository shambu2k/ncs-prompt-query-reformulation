"""Versioned prompt template loading for Phase 2 rewrite baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = REPO_ROOT / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    """Loaded rewrite prompt template."""

    version: str
    text: str
    path: Path


def resolve_prompt_path(prompt_version: str, prompt_path: str | None = None) -> Path:
    if prompt_path:
        return Path(prompt_path)
    return PROMPT_ROOT / f"{prompt_version}.txt"


def load_prompt_template(prompt_version: str, prompt_path: str | None = None) -> PromptTemplate:
    path = resolve_prompt_path(prompt_version, prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt template is empty: {path}")
    return PromptTemplate(version=prompt_version, text=text, path=path)


def build_messages(template: PromptTemplate, query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": template.text},
        {
            "role": "user",
            "content": (
                "Original query:\n"
                f"{query}\n\n"
                "Return only the rewritten software search query."
            ),
        },
    ]

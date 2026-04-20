"""Prompt evolution history tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_ROOT = REPO_ROOT / "evolution_logs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


class PromptHistory:
    """Manage the versioned prompt evolution history for a single run."""

    def __init__(self, run_name: str, history_root: Path = DEFAULT_HISTORY_ROOT) -> None:
        self.run_name = run_name
        self.history_root = Path(history_root)
        self.summaries_dir = self.history_root / "summaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.summaries_dir / f"{run_name}_history.json"
        self._records: list[dict[str, Any]] = []
        self._seen_hashes: set[str] = set()
        self._seen_ids: set[str] = set()
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        for record in data.get("prompts", []):
            self._records.append(record)
            self._seen_hashes.add(prompt_hash(record["prompt_text"]))
            self._seen_ids.add(record["prompt_id"])

    def _save(self) -> None:
        payload = {
            "run_name": self.run_name,
            "updated": _utc_now(),
            "prompts": self._records,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def is_duplicate(self, prompt_text: str) -> bool:
        return prompt_hash(prompt_text) in self._seen_hashes

    def add(self, record: dict[str, Any]) -> None:
        pid = record["prompt_id"]
        if pid in self._seen_ids:
            raise ValueError(f"Duplicate prompt_id: {pid}")
        ph = prompt_hash(record["prompt_text"])
        if ph in self._seen_hashes:
            raise ValueError(f"Duplicate prompt content for prompt_id {pid}.")
        self._records.append(record)
        self._seen_hashes.add(ph)
        self._seen_ids.add(pid)
        self._save()

    def update_status(self, prompt_id: str, status: str) -> None:
        record = self.get_by_id(prompt_id)
        if record is None:
            raise KeyError(f"prompt_id not found: {prompt_id}")
        record["status"] = status
        self._save()

    def set_score(self, prompt_id: str, dev_mrr: float) -> None:
        record = self.get_by_id(prompt_id)
        if record is None:
            raise KeyError(f"prompt_id not found: {prompt_id}")
        record["dev_mrr"] = dev_mrr
        self._save()

    def get_by_id(self, prompt_id: str) -> dict[str, Any] | None:
        for record in self._records:
            if record["prompt_id"] == prompt_id:
                return record
        return None

    def all_records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def accepted_records(self) -> list[dict[str, Any]]:
        return [r for r in self._records if r.get("status") == "accepted"]

    def best_accepted(self) -> dict[str, Any] | None:
        accepted = self.accepted_records()
        if not accepted:
            return None
        return max(accepted, key=lambda r: float(r.get("dev_mrr", 0.0)))

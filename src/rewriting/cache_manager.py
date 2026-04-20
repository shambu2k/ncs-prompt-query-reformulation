"""Resumable cache manager for rewritten queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REWRITE_ROOT = REPO_ROOT / "rewritten_queries"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_rewrite_records(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            example_id = payload["example_id"]
            records[example_id] = payload
    return records


class RewriteCacheManager:
    """Manage rewrite cache files, metadata, and logs."""

    def __init__(
        self,
        *,
        rewrite_root: Path = DEFAULT_REWRITE_ROOT,
        log_root: Path = DEFAULT_LOG_ROOT,
        condition: str,
        run_name: str,
    ) -> None:
        self.run_dir = Path(rewrite_root) / condition / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.run_dir / "rewrites.jsonl"
        self.metadata_path = self.run_dir / "metadata.json"
        self.run_log_path = Path(log_root) / f"{run_name}.log"
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._append_handle = self.cache_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._append_handle.close()

    def load_existing(self) -> dict[str, dict[str, Any]]:
        return load_rewrite_records(self.cache_path)

    def append(self, record: dict[str, Any]) -> None:
        self._append_handle.write(json.dumps(record, sort_keys=True))
        self._append_handle.write("\n")
        self._append_handle.flush()

    def log(self, message: str) -> None:
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_utc_now()} {message}\n")

    def finalize(
        self,
        *,
        ordered_records: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        self.close()
        with self.cache_path.open("w", encoding="utf-8") as handle:
            for record in ordered_records:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

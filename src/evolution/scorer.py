"""Score candidate prompts on the dev subset using inline BM25 retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.data.schema import CanonicalRecord
from src.retrieval.bm25 import BM25Index, ranked_doc_ids


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORE_CACHE_ROOT = REPO_ROOT / "evolution_logs" / "score_cache"
_THREAD_LOCAL = threading.local()


def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def _rewrite_heuristic(query: str) -> str:
    import re
    text = query.replace("`", " ").replace("/", " / ").replace(".", " ").replace("_", " ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _rewrite_openai(
    prompt_text: str,
    query: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenAI provider requires requests package.") from exc

    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    for attempt in range(5):
        response = _requests_session(requests).post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": prompt_text},
                    {
                        "role": "user",
                        "content": (
                            f"Original query:\n{query}\n\n"
                            "Return only the rewritten software search query."
                        ),
                    },
                ],
            },
            timeout=60,
        )
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    response.raise_for_status()
    return ""


def _requests_session(requests_module):
    session = getattr(_THREAD_LOCAL, "requests_session", None)
    if session is None:
        session = requests_module.Session()
        adapter = requests_module.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=128)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _THREAD_LOCAL.requests_session = session
    return session


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_batch_response(response_text: str, expected_ids: list[str]) -> dict[str, str]:
    parsed = json.loads(_strip_code_fences(response_text))
    if not isinstance(parsed, dict):
        raise ValueError("Batch rewrite response must be a JSON object.")

    rewrites: dict[str, str] = {}
    for example_id in expected_ids:
        rewritten = parsed.get(example_id)
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ValueError(f"Batch rewrite response missing id={example_id}.")
        rewrites[example_id] = rewritten.strip()
    return rewrites


def _rewrite_openai_batch(
    prompt_text: str,
    records: list[CanonicalRecord],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, str]:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenAI provider requires requests package.") from exc

    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    payload_records = [{"id": record.example_id, "query": record.query_text} for record in records]
    for attempt in range(5):
        response = _requests_session(requests).post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": prompt_text},
                    {
                        "role": "system",
                        "content": (
                            "You may receive multiple labeled software search queries. "
                            "Rewrite each query independently. "
                            "Return only a compact JSON object that maps every id to one rewritten query. "
                            "Do not omit ids. Do not include markdown fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the following labeled queries.\n"
                            + json.dumps(payload_records, ensure_ascii=False)
                        ),
                    },
                ],
            },
            timeout=90,
        )
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return _parse_batch_response(content, [record.example_id for record in records])
    response.raise_for_status()
    return {}


def _compute_inline_mrr(
    all_records: list[CanonicalRecord],
    dev_indices: list[int],
    rewrites: dict[str, str],
    *,
    top_k: int,
) -> float:
    """Compute MRR using in-memory BM25 over the full corpus.

    Builds the BM25 index once on all code, then evaluates only dev-subset queries.
    Ground truth: for query at position i, the correct document is all_records[i]
    (paired dataset — each record's url maps to its own idx in the answer key).
    """
    document_tokens = [list(record.code_tokens) for record in all_records]
    index = BM25Index(document_tokens)

    mrr_total = 0.0
    for record_idx in dev_indices:
        record = all_records[record_idx]
        rewritten = rewrites[record.example_id]
        query_tokens = rewritten.split()
        ranked = ranked_doc_ids(index, query_tokens, top_k=top_k)
        correct_idx = record.metadata["idx"]
        for rank, doc_id in enumerate(ranked, start=1):
            if all_records[doc_id].metadata["idx"] == correct_idx:
                mrr_total += 1.0 / rank
                break

    return mrr_total / len(dev_indices) if dev_indices else 0.0


def compute_dev_mrr(
    all_records: list[CanonicalRecord],
    dev_indices: list[int],
    *,
    prompt_text: str,
    provider: str = "auto",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 64,
    top_k: int = 100,
    score_cache_root: Path = DEFAULT_SCORE_CACHE_ROOT,
    num_workers: int = 10,
    batch_size: int = 1,
) -> float:
    """Rewrite dev-subset queries with prompt_text, run BM25, return MRR.

    Rewrites are cached under score_cache_root/<prompt_hash>/ to avoid
    rescoring the same prompt twice across resumed runs.
    """
    resolved_provider = provider
    if provider == "auto":
        resolved_provider = "openai" if os.environ.get("OPENAI_API_KEY") else "heuristic"
    allow_heuristic_fallback = provider == "auto"

    ph = _prompt_hash(prompt_text)
    cache_dir = Path(score_cache_root) / ph
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "rewrites.json"

    rewrite_cache: dict[str, str] = {}
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as handle:
            rewrite_cache = json.load(handle)

    dev_records = [all_records[i] for i in dev_indices]
    rewrites: dict[str, str] = {}
    dirty = False

    pending = [r for r in dev_records if r.example_id not in rewrite_cache]
    for record in dev_records:
        if record.example_id in rewrite_cache:
            rewrites[record.example_id] = rewrite_cache[record.example_id]

    def _rewrite_one(record: CanonicalRecord) -> tuple[str, str]:
        if resolved_provider == "openai":
            try:
                return record.example_id, _rewrite_openai(
                    prompt_text, record.query_text,
                    model=model, temperature=temperature, max_tokens=max_tokens,
                )
            except Exception:
                if not allow_heuristic_fallback:
                    raise
        return record.example_id, _rewrite_heuristic(record.query_text)

    def _rewrite_batch(records_batch: list[CanonicalRecord]) -> dict[str, str]:
        if resolved_provider == "openai" and batch_size > 1:
            try:
                return _rewrite_openai_batch(
                    prompt_text,
                    records_batch,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception:
                if not allow_heuristic_fallback:
                    return {example_id: rewritten for example_id, rewritten in map(_rewrite_one, records_batch)}
        return {example_id: rewritten for example_id, rewritten in map(_rewrite_one, records_batch)}

    if pending:
        workers = max(1, num_workers)
        effective_batch_size = max(1, batch_size if resolved_provider == "openai" else 1)
        pending_batches = [
            pending[offset : offset + effective_batch_size]
            for offset in range(0, len(pending), effective_batch_size)
        ]
        with ThreadPoolExecutor(max_workers=min(workers, len(pending_batches))) as executor:
            for batch_rewrites in executor.map(_rewrite_batch, pending_batches):
                for eid, rewritten in batch_rewrites.items():
                    rewrites[eid] = rewritten
                    rewrite_cache[eid] = rewritten
        dirty = True

    if dirty:
        cache_file.write_text(
            json.dumps(rewrite_cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return _compute_inline_mrr(all_records, dev_indices, rewrites, top_k=top_k)

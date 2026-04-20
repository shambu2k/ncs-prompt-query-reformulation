"""Generate and cache Phase 2 rewritten queries."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.data.loader import DEFAULT_MANIFEST_ROOT, DEFAULT_PROCESSED_ROOT, DatasetLoader

from .cache_manager import DEFAULT_LOG_ROOT, DEFAULT_REWRITE_ROOT, RewriteCacheManager
from .prompt_templates import PromptTemplate, build_messages, load_prompt_template
from .validators import RewriteValidationResult, validate_rewritten_query


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "configs" / "rewrite.yaml"
_THREAD_LOCAL = threading.local()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return json.loads(text)


def _split_camel_case(text: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)


def _normalize_identifier_like_text(text: str) -> str:
    text = text.replace("`", " ")
    text = text.replace("/", " / ")
    text = text.replace(".", " ")
    text = text.replace("_", " ")
    text = _split_camel_case(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _heuristic_rewrite(query: str) -> str:
    normalized = _normalize_identifier_like_text(query)
    normalized = re.sub(r"\s+([.,!?])", r"\1", normalized)
    normalized = normalized.strip()
    if normalized and normalized[0].islower():
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


class RewriteProvider:
    def rewrite(self, *, template: PromptTemplate, query: str, model: str, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

    def rewrite_batch(
        self,
        *,
        template: PromptTemplate,
        items: list[tuple[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, str]:
        return {
            example_id: self.rewrite(
                template=template,
                query=query,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for example_id, query in items
        }


class HeuristicRewriteProvider(RewriteProvider):
    def rewrite(self, *, template: PromptTemplate, query: str, model: str, temperature: float, max_tokens: int) -> str:
        del template, model, temperature, max_tokens
        return _heuristic_rewrite(query)


class OpenAICompatibleRewriteProvider(RewriteProvider):
    def __init__(self, *, api_key: str, api_base: str) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")

    def _session(self):
        try:
            import requests
        except ModuleNotFoundError as exc:
            raise RuntimeError("provider=openai requires the requests package.") from exc
        session = getattr(_THREAD_LOCAL, "requests_session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=128)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _THREAD_LOCAL.requests_session = session
        return session

    def rewrite(self, *, template: PromptTemplate, query: str, model: str, temperature: float, max_tokens: int) -> str:
        import time as _time
        url = f"{self.api_base}/chat/completions"
        payload_body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": build_messages(template, query),
        }
        for attempt in range(5):
            response = self._session().post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload_body,
                timeout=60,
            )
            if response.status_code == 429:
                _time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        response.raise_for_status()
        return ""  # unreachable

    def rewrite_batch(
        self,
        *,
        template: PromptTemplate,
        items: list[tuple[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, str]:
        import time as _time

        url = f"{self.api_base}/chat/completions"
        payload_body = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": template.text},
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
                        + json.dumps(
                            [{"id": example_id, "query": query} for example_id, query in items],
                            ensure_ascii=False,
                        )
                    ),
                },
            ],
        }
        for attempt in range(5):
            response = self._session().post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload_body,
                timeout=90,
            )
            if response.status_code == 429:
                _time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            return _parse_batch_response(content, [example_id for example_id, _ in items])
        response.raise_for_status()
        return {}


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


def _resolve_provider(provider_name: str) -> tuple[str, RewriteProvider]:
    if provider_name == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            provider_name = "openai"
        else:
            provider_name = "heuristic"

    if provider_name == "heuristic":
        return provider_name, HeuristicRewriteProvider()
    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai.")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return provider_name, OpenAICompatibleRewriteProvider(api_key=api_key, api_base=api_base)
    raise ValueError(f"Unsupported rewrite provider '{provider_name}'.")


def _record_payload(
    *,
    example_id: str,
    original_query: str,
    validation: RewriteValidationResult,
    prompt_version: str,
    model: str,
    provider: str,
    split: str,
    condition: str,
) -> dict[str, Any]:
    payload = {
        "example_id": example_id,
        "original_query": original_query,
        "rewritten_query": validation.rewritten_query,
        "rewritten_query_tokens": validation.rewritten_query_tokens,
        "prompt_version": prompt_version,
        "model": model,
        "provider": provider,
        "timestamp": _utc_now(),
        "validation_status": validation.validation_status,
        "split": split,
        "condition": condition,
    }
    if validation.reason is not None:
        payload["validation_reason"] = validation.reason
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rewritten queries for Phase 2 baselines.")
    parser.add_argument("--split", choices=["valid", "test"], required=True, help="Split to rewrite.")
    parser.add_argument("--condition", choices=["clean", "adv"], required=True, help="Condition to rewrite.")
    parser.add_argument("--run-name", required=True, help="Rewrite run name under rewritten_queries/<condition>/.")
    parser.add_argument(
        "--prompt-version",
        "--prompt",
        dest="prompt_version",
        default="fixed_prompt_v1",
        help="Frozen prompt version to use.",
    )
    parser.add_argument(
        "--prompt-path",
        default=None,
        help="Optional explicit prompt template path.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "heuristic", "openai"],
        default="auto",
        help="Rewrite backend. 'auto' uses OpenAI when configured, otherwise heuristic.",
    )
    parser.add_argument(
        "--model",
        default="heuristic-fixed-prompt-v1",
        help="Model name recorded in metadata; used by provider=openai.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Rewrite temperature.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max completion tokens for rewrite generation.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate cached rewrites instead of resuming.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument(
        "--processed-root",
        default=str(DEFAULT_PROCESSED_ROOT),
        help="Directory containing canonical processed JSONL files.",
    )
    parser.add_argument(
        "--manifest-root",
        default=str(DEFAULT_MANIFEST_ROOT),
        help="Directory containing canonical dataset manifests.",
    )
    parser.add_argument(
        "--rewrite-root",
        default=str(DEFAULT_REWRITE_ROOT),
        help="Directory where rewrite caches will be stored.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Directory where rewrite logs will be stored.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Optional JSON-compatible YAML config file for default rewrite settings.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel rewrite threads (default: 1). Use 8-16 for OpenAI provider.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of queries per OpenAI request (default: 1). Ignored for heuristic provider.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_json_yaml(Path(args.config)) if args.config else {}
        provider_name = args.provider if args.provider != "auto" else config.get("provider", "auto")
        model_name = args.model if args.model != "heuristic-fixed-prompt-v1" else config.get("model", args.model)
        temperature = args.temperature if args.temperature != 0.0 else float(config.get("temperature", 0.0))
        max_tokens = args.max_tokens if args.max_tokens != 64 else int(config.get("max_tokens", 64))
        overwrite = args.overwrite or bool(config.get("overwrite", False))
        num_workers = args.num_workers if args.num_workers != 1 else int(config.get("num_workers", 1))
        batch_size = args.batch_size if args.batch_size != 1 else int(config.get("batch_size", 1))

        template = load_prompt_template(args.prompt_version, args.prompt_path)
        resolved_provider_name, provider = _resolve_provider(provider_name)
        if resolved_provider_name == "heuristic" and model_name == "heuristic-fixed-prompt-v1":
            model_name = "heuristic-fixed-prompt-v1"

        loader = DatasetLoader(processed_root=Path(args.processed_root), manifest_root=Path(args.manifest_root))
        records = loader.load_split(args.split, args.condition)
        if args.max_examples is not None:
            records = records[: args.max_examples]

        cache = RewriteCacheManager(
            rewrite_root=Path(args.rewrite_root),
            log_root=Path(args.log_root),
            condition=args.condition,
            run_name=args.run_name,
        )
        existing = {} if overwrite else cache.load_existing()
        if overwrite and cache.cache_path.exists():
            cache.close()
            cache.cache_path.unlink()
            cache = RewriteCacheManager(
                rewrite_root=Path(args.rewrite_root),
                log_root=Path(args.log_root),
                condition=args.condition,
                run_name=args.run_name,
            )

        cache.log(
            f"start split={args.split} condition={args.condition} prompt={args.prompt_version} "
            f"provider={resolved_provider_name} model={model_name}"
        )

        generated = 0
        cached = 0
        fallback = 0
        by_example_id = dict(existing)

        pending = [r for r in records if r.example_id not in by_example_id]
        cached = len(records) - len(pending)

        def _rewrite_one(record: Any) -> tuple[Any, Any]:
            try:
                rewritten = provider.rewrite(
                    template=template,
                    query=record.query_text,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                cache.log(f"provider_error example_id={record.example_id} reason={type(exc).__name__}")
                return record, RewriteValidationResult(
                    rewritten_query=record.query_text,
                    rewritten_query_tokens=record.query_tokens,
                    validation_status="fallback",
                    reason=f"provider_error:{type(exc).__name__}",
                )
            validation = validate_rewritten_query(record.query_text, rewritten)
            if validation.validation_status == "fallback":
                cache.log(
                    f"validation_fallback example_id={record.example_id} reason={validation.reason}"
                )
            return record, validation

        def _rewrite_batch(batch: list[Any]) -> list[tuple[Any, RewriteValidationResult]]:
            if resolved_provider_name == "openai" and batch_size > 1:
                try:
                    rewritten_by_id = provider.rewrite_batch(
                        template=template,
                        items=[(record.example_id, record.query_text) for record in batch],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    results: list[tuple[Any, RewriteValidationResult]] = []
                    for record in batch:
                        validation = validate_rewritten_query(
                            record.query_text,
                            rewritten_by_id[record.example_id],
                        )
                        results.append((record, validation))
                    return results
                except Exception:
                    pass
            return [_rewrite_one(record) for record in batch]

        worker_count = max(1, num_workers)
        effective_batch_size = max(1, batch_size if resolved_provider_name == "openai" else 1)
        pending_batches = [
            pending[offset : offset + effective_batch_size]
            for offset in range(0, len(pending), effective_batch_size)
        ]
        with ThreadPoolExecutor(max_workers=min(worker_count, max(1, len(pending_batches)))) as executor:
            futures = {executor.submit(_rewrite_batch, batch): batch for batch in pending_batches}
            results_map: dict[str, Any] = {}
            for future in as_completed(futures):
                for record, validation in future.result():
                    if validation.validation_status == "fallback":
                        fallback += 1
                    payload = _record_payload(
                        example_id=record.example_id,
                        original_query=record.query_text,
                        validation=validation,
                        prompt_version=args.prompt_version,
                        model=model_name,
                        provider=resolved_provider_name,
                        split=args.split,
                        condition=args.condition,
                    )
                    results_map[record.example_id] = payload
                    generated += 1

        for record in pending:
            payload = results_map[record.example_id]
            cache.append(payload)
            by_example_id[record.example_id] = payload

        ordered_records = [by_example_id[record.example_id] for record in records]
        metadata = {
            "run_name": args.run_name,
            "split": args.split,
            "condition": args.condition,
            "prompt_version": args.prompt_version,
            "prompt_path": str(template.path),
            "provider": resolved_provider_name,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "record_count": len(records),
            "generated_count": generated,
            "cached_count": cached,
            "fallback_count": fallback,
            "timestamp": _utc_now(),
            "cache_path": str(cache.cache_path),
        }
        cache.finalize(ordered_records=ordered_records, metadata=metadata)
        cache.log(
            f"finish generated={generated} cached={cached} fallback={fallback} cache={cache.cache_path}"
        )
    except Exception as exc:
        print(f"rewrite_queries failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

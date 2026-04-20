"""Small OpenAI-compatible chat client used for optional Phase 4 annotations."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


class OpenAIClientError(RuntimeError):
    """Raised when the OpenAI-compatible API call fails."""


def chat_completion(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 160,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIClientError("OPENAI_API_KEY is not set.")

    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{api_base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise OpenAIClientError(
            f"OpenAI request failed with status {exc.code}: {details}"
        ) from exc
    except error.URLError as exc:
        raise OpenAIClientError(f"OpenAI request failed: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
        choice = payload["choices"][0]
        message = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise OpenAIClientError("OpenAI response did not match chat completions format.") from exc

    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        combined = "\n".join(part for part in parts if part).strip()
        if combined:
            return combined
    raise OpenAIClientError("OpenAI response did not contain text content.")


def maybe_load_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a raw model response when possible."""

    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


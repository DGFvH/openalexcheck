"""Provider-agnostic LLM access using the user's own one-time API key.

The key arrives with the request, lives only in this object for the duration
of the analysis, and is never written to disk or logs.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import httpx

# Read timeout for the non-streaming providers scales with the requested
# output size (a 16k-token JSON reply takes minutes); capped below the hosting
# platform's function ceiling so a stuck call fails with a clear message.
LLM_READ_TIMEOUT_CAP_S = float(os.environ.get("LLM_READ_TIMEOUT_CAP_S", "280"))

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        provider = (provider or "").strip().lower()
        if provider not in DEFAULT_MODELS:
            raise LLMError(f"Unknown provider '{provider}'. Use anthropic, openai, or gemini.")
        if not api_key or not api_key.strip():
            raise LLMError("An API key is required.")
        self.provider = provider
        self._api_key = api_key.strip()
        self.model = (model or "").strip() or DEFAULT_MODELS[provider]

    def redact(self, text: str) -> str:
        """Strip this client's key from a message before it leaves the server."""
        from .keysafety import redact
        return redact(text, self._api_key)

    def complete_json(self, system: str, user: str, max_tokens: int = 8000,
                      thinking: bool = True) -> dict:
        """Run one completion and parse the response as a JSON object.

        `thinking=False` disables model thinking (Anthropic) — appropriate for
        the mechanical extraction pass, where thinking tokens would otherwise
        eat into `max_tokens` and can truncate the JSON on long documents.
        """
        text, truncated = self._complete(system, user, max_tokens, thinking)
        try:
            return _parse_json(text)
        except LLMError:
            if truncated:
                raise LLMError(
                    "The model's reply was cut off at the token limit before the JSON "
                    "finished — the document is too long for one pass. Split it or "
                    "shorten the reference list and try again."
                )
            raise

    # -- providers ---------------------------------------------------------
    # Each returns (text, truncated) where `truncated` means the reply hit the
    # output-token ceiling (the usual cause of unparseable JSON).

    def _complete(self, system: str, user: str, max_tokens: int, thinking: bool):
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens, thinking)
        if self.provider == "openai":
            return self._openai(system, user, max_tokens)
        return self._gemini(system, user, max_tokens)

    def _anthropic(self, system: str, user: str, max_tokens: int, thinking: bool):
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        try:
            # Always stream: the SDK refuses a NON-streaming request whose
            # max_tokens could take longer than 10 minutes to generate ("Streaming
            # is required for operations that may take longer than 10 minutes").
            # A large max_tokens trips that ceiling, so we stream and accumulate
            # the final message instead.
            with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"} if thinking else {"type": "disabled"},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                response = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("The Anthropic model refused this request.")
        text = "".join(block.text for block in response.content if block.type == "text")
        truncated = response.stop_reason == "max_tokens"
        if not text:
            if truncated:
                raise LLMError(
                    "The model spent the whole token budget before producing any answer. "
                    "Try again, or split the document if it is very long."
                )
            raise LLMError("Anthropic returned an empty response.")
        return text, truncated

    def _openai(self, system: str, user: str, max_tokens: int):
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            body,
            headers={"Authorization": f"Bearer {self._api_key}"},
            provider="OpenAI",
            max_tokens=max_tokens,
        )
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from OpenAI.") from exc
        return text, choice.get("finish_reason") == "length"

    def _gemini(self, system: str, user: str, max_tokens: int):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        data = _post_json(
            url,
            body,
            headers={"x-goog-api-key": self._api_key},
            provider="Gemini",
            max_tokens=max_tokens,
        )
        try:
            cand = data["candidates"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from Gemini.") from exc
        truncated = cand.get("finishReason") == "MAX_TOKENS"
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text and truncated:
            raise LLMError(
                "The model spent the whole token budget before producing any answer. "
                "Try again, or split the document if it is very long."
            )
        return text, truncated


def _read_timeout(max_tokens: int) -> float:
    return min(LLM_READ_TIMEOUT_CAP_S, 60.0 + max_tokens / 20.0)


def _post_json(url: str, body: dict, headers: dict, provider: str,
               max_tokens: int = 8000) -> dict:
    read = _read_timeout(max_tokens)
    timeout = httpx.Timeout(connect=10.0, read=read, write=30.0, pool=10.0)
    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise LLMError(
            f"The {provider} API did not answer within {int(read)} s. Pick a "
            "faster model, split a very long document, or retry."
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Could not reach the {provider} API: {exc}") from exc
    if resp.status_code in (401, 403):
        raise LLMError(f"{provider} rejected the API key.")
    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise LLMError(f"{provider} API error ({resp.status_code}): {detail}")
    return resp.json()


def _extract_first_object(text: str) -> Optional[str]:
    """Return the first balanced {...} object in `text`, correctly skipping
    braces that appear inside string literals (e.g. LaTeX/math in an abstract
    or citation context — a naive depth counter mis-slices on those)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # never balanced → truncated output


def _parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating a BOM, code fences, and prose
    around the object."""
    text = text.lstrip("﻿").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    obj = _extract_first_object(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            pass
    raise LLMError(
        "The model did not return valid JSON. Try again, or use a different model."
    )

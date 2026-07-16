"""Provider-agnostic LLM access using the user's own one-time API key.

The key arrives with the request, lives only in this object for the duration
of the analysis, and is never written to disk or logs.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import httpx

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
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

    def complete_json(self, system: str, user: str, max_tokens: int = 8000) -> dict:
        """Run one completion and parse the response as a JSON object."""
        text = self._complete(system, user, max_tokens)
        return _parse_json(text)

    # -- providers ---------------------------------------------------------

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens)
        if self.provider == "openai":
            return self._openai(system, user, max_tokens)
        return self._gemini(system, user, max_tokens)

    def _anthropic(self, system: str, user: str, max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("The Anthropic model refused this request.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise LLMError("Anthropic returned an empty response.")
        return text

    def _openai(self, system: str, user: str, max_tokens: int) -> str:
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
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from OpenAI.") from exc

    def _gemini(self, system: str, user: str, max_tokens: int) -> str:
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
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from Gemini.") from exc


def _post_json(url: str, body: dict, headers: dict, provider: str) -> dict:
    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=180.0)
    except httpx.HTTPError as exc:
        raise LLMError(f"Could not reach the {provider} API: {exc}") from exc
    if resp.status_code in (401, 403):
        raise LLMError(f"{provider} rejected the API key.")
    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise LLMError(f"{provider} API error ({resp.status_code}): {detail}")
    return resp.json()


def _parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError("The model did not return valid JSON. Try again or use a different model.")

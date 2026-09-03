"""Helpers to guarantee user-supplied API keys never leak.

Keys arrive with a request, live in memory for that request only, and are
never written to disk, logs, or responses. Because error messages can embed
request details (httpx includes the full URL — query string and all — in
HTTPStatusError text), every string that might travel back to the browser or
into a log line must pass through redact().
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote, quote_plus

REDACTED = "•••redacted-key•••"


def redact(text: str, *keys: Optional[str]) -> str:
    """Remove any occurrence of the given secret keys from a message — the raw
    form and the percent-encoded forms httpx uses when it embeds a request URL
    (query string included) in exception text."""
    out = str(text)
    for key in keys:
        if key and key.strip():
            k = key.strip()
            for form in {k, quote(k, safe=""), quote_plus(k)}:
                out = out.replace(form, REDACTED)
    return out

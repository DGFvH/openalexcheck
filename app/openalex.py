"""OpenAlex lookups: resolve extracted references to works and abstracts.

No API key is needed for OpenAlex. Set OPENALEX_MAILTO to join the polite pool.
"""

from __future__ import annotations

import logging
import os
import re
import string
import threading
import time
from contextlib import nullcontext
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

import httpx

OPENALEX_BASE = "https://api.openalex.org"

log = logging.getLogger("phantocite.openalex")
# httpx logs "HTTP Request: GET <full url>" at INFO — for OpenAlex that URL
# carries the user's Premium key as a query parameter. Never let it reach a log.
logging.getLogger("httpx").setLevel(logging.WARNING)

class OpenAlexAuthError(Exception):
    """Raised when OpenAlex rejects the supplied API key."""


# Similarity thresholds on normalized titles
FOUND_THRESHOLD = 0.88
FUZZY_THRESHOLD = 0.55

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

RETRY_AFTER_CAP_S = 10.0


class TokenBucket:
    """Process-wide pacing of OpenAlex requests (they are rate-limited per IP,
    and a shared egress IP is easy to get throttled). Thread-safe; `rate <= 0`
    disables pacing. `clock`/`sleep` are injectable for tests."""

    def __init__(self, rate: float, capacity: float,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rate, self.capacity = float(rate), float(capacity)
        self._clock, self._sleep = clock, sleep
        self._tokens = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self, cancel: Optional[threading.Event] = None) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = min((1 - self._tokens) / self.rate, 0.25)
            if cancel is not None:
                if cancel.wait(wait):
                    raise OpenAlexLookupError("cancelled")
            else:
                self._sleep(wait)


_BUCKET = TokenBucket(rate=float(os.environ.get("OPENALEX_RPS", "8")), capacity=8)
_WARNED = {"mailto": False}


def _client(api_key: Optional[str] = None) -> httpx.Client:
    """OpenAlex client. `api_key` is the user's optional Premium key — one-time
    use, request-scoped, never stored. Sent as the documented `api_key` query
    parameter; any error text derived from these requests must be redact()ed
    because httpx embeds the full URL (query string included) in exceptions."""
    params = {}
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    elif not _WARNED["mailto"]:
        _WARNED["mailto"] = True
        log.warning("OPENALEX_MAILTO is not set — OpenAlex requests go to the common "
                    "(more heavily rate-limited) pool. Set it in the deployment's environment.")
    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()
    return httpx.Client(base_url=OPENALEX_BASE, params=params, timeout=30.0,
                        headers={"User-Agent": "openalexcheck/0.1"})


def normalize_title(title: str) -> str:
    title = (title or "").lower().translate(_PUNCT_TABLE)
    return " ".join(title.split())


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def clean_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()


def reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """OpenAlex stores abstracts as an inverted index; rebuild the plain text."""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda t: t[0])
    return " ".join(word for _, word in positions) or None


def summarize_work(work: dict) -> dict:
    """Reduce an OpenAlex work object to the fields the UI and LLM need."""
    authors = [
        a.get("author", {}).get("display_name")
        for a in (work.get("authorships") or [])
        if a.get("author", {}).get("display_name")
    ]
    source = ((work.get("primary_location") or {}).get("source") or {})
    biblio = work.get("biblio") or {}
    first_page = biblio.get("first_page") or None
    last_page = biblio.get("last_page") or None
    if first_page and last_page and first_page != last_page:
        pages = f"{first_page}-{last_page}"
    else:
        pages = first_page or last_page or None
    return {
        "openalex_id": work.get("id"),
        "doi": clean_doi(work.get("doi") or "") or None,
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year"),
        "authors": authors[:6],       # for compact display
        "authors_full": authors,      # full list, for author verification
        "venue": source.get("display_name"),
        "volume": biblio.get("volume") or None,
        "issue": biblio.get("issue") or None,
        "pages": pages,
        "cited_by_count": work.get("cited_by_count"),
        "url": work.get("doi") or work.get("id"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
    }


class OpenAlexLookupError(Exception):
    """A lookup could not be completed (network error or non-auth 4xx/5xx after
    retries). Distinct from 'no results found' — the caller must NOT treat this
    as a hallucination, since a transient failure is not evidence of absence.

    The message is status-only on purpose: httpx exception text embeds the full
    request URL (api_key included), so it is never copied into a message."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _check_auth(resp: httpx.Response, key_sent: bool) -> None:
    """A 401/403 is an API-key rejection ONLY if a key was actually sent. Without
    a key it is a CDN/abuse-protection/transient refusal — a lookup failure for
    this reference, never a reason to abort the whole run."""
    if resp.status_code in (401, 403):
        if key_sent:
            raise OpenAlexAuthError("OpenAlex rejected the API key. Remove it or check that it is valid.")
        raise OpenAlexLookupError(f"OpenAlex returned {resp.status_code} (no API key was sent)",
                                  status=resp.status_code)


def _retry_after(resp: Any) -> Optional[float]:
    """Seconds OpenAlex asked us to wait, capped so one slow reference cannot
    stall a run; None when the header is absent or not a plain integer."""
    value = (getattr(resp, "headers", None) or {}).get("Retry-After")
    try:
        return min(float(int(value)), RETRY_AFTER_CAP_S) if value is not None else None
    except (TypeError, ValueError):
        return None


_BUDGET_RE = re.compile(r"insufficient budget|daily (?:limit|quota|budget)|resets at midnight", re.I)


def _budget_exhausted(resp: Any) -> bool:
    """OpenAlex's 429 for an exhausted DAILY budget — retrying is pointless."""
    return bool(_BUDGET_RE.search(getattr(resp, "text", "") or ""))


def _sleep(delay: float, cancel: Optional[threading.Event]) -> None:
    """Retry back-off that a cancelled run can interrupt."""
    if cancel is None:
        time.sleep(delay)
    elif cancel.wait(delay):
        raise OpenAlexLookupError("cancelled")


def _get(client: httpx.Client, path: str, params: Optional[dict] = None,
         attempts: int = 3, *, cancel: Optional[threading.Event] = None) -> httpx.Response:
    """GET with retry on transient failures (network errors, 429, 5xx).

    A transient failure here would otherwise masquerade as 'reference not found'
    and produce a false hallucination flag — so we retry, then raise loudly.
    Requests are paced through the process-wide token bucket; a 429 honours
    Retry-After; an exhausted daily budget is not retried; `cancel` aborts.
    """
    key_sent = "api_key" in (getattr(client, "params", None) or {})
    last_status: Optional[int] = None
    last_error: Optional[str] = None
    for i in range(attempts):
        if cancel is not None and cancel.is_set():
            raise OpenAlexLookupError("cancelled")
        _BUCKET.acquire(cancel)
        delay = 0.5 * (i + 1)
        try:
            resp = client.get(path, params=params)
        except httpx.HTTPError as exc:
            last_status, last_error = None, type(exc).__name__
            log.warning("OpenAlex request failed (%s), attempt %d/%d", last_error, i + 1, attempts)
        else:
            _check_auth(resp, key_sent)  # never retry an auth rejection
            if resp.status_code < 400 or resp.status_code == 404:
                return resp
            if resp.status_code not in (429,) and resp.status_code < 500:
                # Non-transient client error (bad query etc.) — don't spin.
                return resp
            last_status, last_error = resp.status_code, None
            if resp.status_code == 429 and _budget_exhausted(resp):
                log.warning("OpenAlex daily budget exhausted — not retrying")
                raise OpenAlexLookupError("OpenAlex's daily request budget is exhausted", status=429)
            delay = _retry_after(resp) or delay
            log.warning("OpenAlex returned %d, attempt %d/%d", last_status, i + 1, attempts)
        if i < attempts - 1:
            _sleep(delay, cancel)
    if last_status is not None:
        raise OpenAlexLookupError(f"OpenAlex returned {last_status}", status=last_status)
    raise OpenAlexLookupError(f"OpenAlex could not be reached ({last_error or 'unknown error'})")


def _lookup_note(exc: Exception, what: str) -> str:
    """User-facing note for a failed lookup — status or error class only, never
    the exception text (which may embed the request URL and the API key)."""
    status = getattr(exc, "status", None)
    if status:
        return f"OpenAlex returned {status} for the {what}."
    return f"OpenAlex could not be reached for the {what}."


def _get_work_by_doi(client: httpx.Client, doi: str, *,
                     cancel: Optional[threading.Event] = None) -> Optional[dict]:
    resp = _get(client, f"/works/https://doi.org/{doi}", cancel=cancel)
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        raise OpenAlexLookupError(f"OpenAlex returned {resp.status_code}", status=resp.status_code)
    return resp.json()


# Word-internal punctuation stays: the search index treats "don't" as ONE
# token, so sending "don t"/"dont" matches nothing and a real work gets
# falsely flagged as a potential hallucination. Everything else becomes a
# space — commas/colons/pipes/ampersands are OpenAlex filter syntax, and a
# stray "?" makes the API return a 400.
_SEARCH_PUNCT_TABLE = str.maketrans(
    {c: " " for c in string.punctuation if c not in "'-"})


def _search_query(title: str) -> str:
    return " ".join((title or "").translate(_SEARCH_PUNCT_TABLE).split())


def _search_works_by_title(client: httpx.Client, title: str, per_page: int = 6, *,
                           cancel: Optional[threading.Event] = None) -> list[dict]:
    # Try the apostrophe-preserving query first; fall back to the fully
    # normalized form on ANY failure (error status or zero hits).
    queries = [q for q in dict.fromkeys([_search_query(title), normalize_title(title)]) if q]
    if not queries:
        raise OpenAlexLookupError("The title contains no searchable text")
    errors: list[int] = []
    for q in queries:
        resp = _get(client, "/works", params={"filter": f"title.search:{q}", "per-page": per_page},
                    cancel=cancel)
        if resp.status_code >= 400:
            errors.append(resp.status_code)
            continue
        results = resp.json().get("results", [])
        if results:
            return results
    if len(errors) == len(queries):
        # Every query was REJECTED — that is "could not search", not "no such
        # work"; reporting it as not_found would be a false hallucination flag.
        raise OpenAlexLookupError(f"OpenAlex returned {errors[-1]} for the title search",
                                  status=errors[-1])
    return []


def score_candidate(ref: dict, work_summary: dict) -> float:
    """Score how well an OpenAlex work matches an extracted reference (0..~1.2)."""
    score = title_similarity(ref.get("title") or "", work_summary.get("title") or "")
    ref_year = ref.get("year")
    if ref_year and work_summary.get("year"):
        diff = abs(int(ref_year) - int(work_summary["year"]))
        if diff == 0:
            score += 0.1
        elif diff <= 1:
            score += 0.05
        else:
            score -= 0.1
    surname = (ref.get("first_author_surname") or "").strip().lower()
    if surname and work_summary.get("authors"):
        joined = " ".join(work_summary["authors"]).lower()
        score += 0.1 if surname in joined else -0.1
    return score


def _found(ref: dict, work: dict, notes: list[str]) -> dict:
    """Build a 'found' result, attaching the deterministic field-level check
    (title/authors/year/journal/DOI/volume/issue/pages)."""
    from .fieldcheck import compare_fields, field_mismatches

    fields = compare_fields(ref, work)
    mism = field_mismatches(fields)
    if mism:
        labels = ", ".join(f["field"] for f in mism)
        notes = notes + [f"Metadata mismatch on: {labels}. The work exists but the citation details differ."]
    return {"status": "found", "work": work, "candidates": [], "notes": notes,
            "field_check": fields, "field_mismatch_count": len(mism)}


def resolve_reference(ref: dict, api_key: Optional[str] = None, *,
                      client: Optional[httpx.Client] = None,
                      cancel: Optional[threading.Event] = None) -> dict:
    """Resolve one extracted reference against OpenAlex.

    Returns {"status": "found"|"fuzzy"|"not_found"|"lookup_failed", "work": ...,
             "candidates": [...], "notes": [...], and for 'found':
             "field_check": [...], "field_mismatch_count": N}

    `client`: reuse one connection pool across a whole bibliography (a fresh
    TLS handshake per reference otherwise). `cancel`: a stopped run aborts at
    the next request or back-off.
    """
    notes: list[str] = []
    candidates: list[dict] = []

    # Author + year alone cannot identify a work — without a title or DOI there
    # is nothing to search on, and "we couldn't look" must never be reported as
    # "this is fabricated".
    if not (ref.get("title") or clean_doi(ref.get("doi") or "")):
        return {"status": "lookup_failed", "work": None, "candidates": [],
                "notes": ["This reference has no title or DOI — author and year alone "
                          "are not enough to identify a work in OpenAlex, so it was "
                          "not checked. Provide the full reference entry to verify it."]}

    lookup_failed = False
    with (nullcontext(client) if client is not None else _client(api_key)) as client:
        doi = clean_doi(ref.get("doi") or "")
        doi_work = None
        if doi:
            try:
                raw = _get_work_by_doi(client, doi, cancel=cancel)
            except (httpx.HTTPError, OpenAlexLookupError) as exc:
                raw = None
                lookup_failed = True
                notes.append(_lookup_note(exc, "DOI lookup"))
            if raw:
                doi_work = summarize_work(raw)
                sim = title_similarity(ref.get("title") or "", doi_work["title"] or "")
                if sim >= 0.75 or not ref.get("title"):
                    return _found(ref, doi_work, notes)
                notes.append(
                    "The DOI in the reference resolves to a work with a different title "
                    f"(similarity {sim:.2f}). Possible fuzzy merge: correct DOI, wrong title (or vice versa)."
                )
                candidates.append({**doi_work, "match_reason": "DOI match, different title"})
            else:
                notes.append("The DOI given in the reference was not found in OpenAlex.")

        title = ref.get("title") or ""
        if title:
            try:
                results = _search_works_by_title(client, title, cancel=cancel)
            except (httpx.HTTPError, OpenAlexLookupError) as exc:
                results = []
                lookup_failed = True
                notes.append(_lookup_note(exc, "title search"))
            seen = {c.get("openalex_id") for c in candidates}
            scored = []
            for raw in results:
                summary = summarize_work(raw)
                if summary["openalex_id"] in seen:
                    continue
                scored.append((score_candidate(ref, summary), summary))
            scored.sort(key=lambda t: t[0], reverse=True)

            if scored:
                best_score, best = scored[0]
                best_sim = title_similarity(title, best["title"] or "")
                # A strong TITLE match means it is the same work — treat it as
                # found and let the field check report any wrong author/year/etc.
                # (don't let a wrong year or author demote it to 'fuzzy', since
                # those discrepancies are precisely what we want to surface).
                if best_sim >= FOUND_THRESHOLD and not candidates:
                    return _found(ref, best, notes)
                for sc, summary in scored[:4]:
                    if title_similarity(title, summary["title"] or "") >= FUZZY_THRESHOLD or sc >= FUZZY_THRESHOLD:
                        candidates.append({**summary, "match_reason": f"Title search (score {sc:.2f})"})

    if candidates:
        notes.append("No exact match, but close candidates exist — review them on the fuzzy-matches screen.")
        return {"status": "fuzzy", "work": None, "candidates": candidates, "notes": notes}

    if lookup_failed:
        # A lookup errored and produced no results — do NOT accuse the
        # reference of being fabricated on the strength of a failed request.
        notes.append("OpenAlex could not be reached to verify this reference — retry before treating it as unverified.")
        return {"status": "lookup_failed", "work": None, "candidates": [], "notes": notes}

    notes.append("No matching work found in OpenAlex — potential hallucinated reference.")
    return {"status": "not_found", "work": None, "candidates": [], "notes": notes}

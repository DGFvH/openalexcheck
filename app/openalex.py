"""OpenAlex lookups: resolve extracted references to works and abstracts.

No API key is needed for OpenAlex. Set OPENALEX_MAILTO to join the polite pool.
"""

from __future__ import annotations

import os
import re
import string
import time
from difflib import SequenceMatcher
from typing import Any, Optional

import httpx

from .keysafety import redact

OPENALEX_BASE = "https://api.openalex.org"

class OpenAlexAuthError(Exception):
    """Raised when OpenAlex rejects the supplied API key."""


# Similarity thresholds on normalized titles
FOUND_THRESHOLD = 0.88
FUZZY_THRESHOLD = 0.55

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _client(api_key: Optional[str] = None) -> httpx.Client:
    """OpenAlex client. `api_key` is the user's optional Premium key — one-time
    use, request-scoped, never stored. Sent as the documented `api_key` query
    parameter; any error text derived from these requests must be redact()ed
    because httpx embeds the full URL (query string included) in exceptions."""
    params = {}
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
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
    as a hallucination, since a transient failure is not evidence of absence."""


def _check_auth(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise OpenAlexAuthError("OpenAlex rejected the API key. Remove it or check that it is valid.")


def _get(client: httpx.Client, path: str, params: Optional[dict] = None,
         attempts: int = 3) -> httpx.Response:
    """GET with retry on transient failures (network errors, 429, 5xx).

    A transient failure here would otherwise masquerade as 'reference not found'
    and produce a false hallucination flag — so we retry, then raise loudly.
    """
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            resp = client.get(path, params=params)
        except httpx.HTTPError as exc:
            last_exc = exc
        else:
            _check_auth(resp)  # never retry an auth rejection
            if resp.status_code < 400 or resp.status_code == 404:
                return resp
            if resp.status_code not in (429,) and resp.status_code < 500:
                # Non-transient client error (bad query etc.) — don't spin.
                return resp
            last_exc = httpx.HTTPStatusError(
                f"OpenAlex returned {resp.status_code}", request=resp.request, response=resp)
        if i < attempts - 1:
            time.sleep(0.5 * (i + 1))
    raise OpenAlexLookupError(str(last_exc) if last_exc else "OpenAlex request failed")


def _get_work_by_doi(client: httpx.Client, doi: str) -> Optional[dict]:
    resp = _get(client, f"/works/https://doi.org/{doi}")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _search_works_by_title(client: httpx.Client, title: str, per_page: int = 6) -> list[dict]:
    # Commas and colons are OpenAlex filter syntax — strip them from the value.
    safe = normalize_title(title)
    if not safe:
        return []
    resp = _get(client, "/works", params={"filter": f"title.search:{safe}", "per-page": per_page})
    if resp.status_code >= 400:
        return []
    return resp.json().get("results", [])


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


def resolve_reference(ref: dict, api_key: Optional[str] = None) -> dict:
    """Resolve one extracted reference against OpenAlex.

    Returns {"status": "found"|"fuzzy"|"not_found"|"lookup_failed", "work": ...,
             "candidates": [...], "notes": [...], and for 'found':
             "field_check": [...], "field_mismatch_count": N}
    """
    notes: list[str] = []
    candidates: list[dict] = []

    lookup_failed = False
    with _client(api_key) as client:
        doi = clean_doi(ref.get("doi") or "")
        doi_work = None
        if doi:
            try:
                raw = _get_work_by_doi(client, doi)
            except (httpx.HTTPError, OpenAlexLookupError) as exc:
                raw = None
                lookup_failed = True
                notes.append(redact(f"OpenAlex DOI lookup failed: {exc}", api_key))
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
                results = _search_works_by_title(client, title)
            except (httpx.HTTPError, OpenAlexLookupError) as exc:
                results = []
                lookup_failed = True
                notes.append(redact(f"OpenAlex title search failed: {exc}", api_key))
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

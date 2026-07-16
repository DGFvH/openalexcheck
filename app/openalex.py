"""OpenAlex lookups: resolve extracted references to works and abstracts.

No API key is needed for OpenAlex. Set OPENALEX_MAILTO to join the polite pool.
"""

from __future__ import annotations

import os
import re
import string
from difflib import SequenceMatcher
from typing import Any, Optional

import httpx

OPENALEX_BASE = "https://api.openalex.org"

# Similarity thresholds on normalized titles
FOUND_THRESHOLD = 0.88
FUZZY_THRESHOLD = 0.55

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _client() -> httpx.Client:
    params = {}
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
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
    return {
        "openalex_id": work.get("id"),
        "doi": clean_doi(work.get("doi") or "") or None,
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year"),
        "authors": authors[:6],
        "venue": source.get("display_name"),
        "cited_by_count": work.get("cited_by_count"),
        "url": work.get("doi") or work.get("id"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
    }


def _get_work_by_doi(client: httpx.Client, doi: str) -> Optional[dict]:
    resp = client.get(f"/works/https://doi.org/{doi}")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _search_works_by_title(client: httpx.Client, title: str, per_page: int = 6) -> list[dict]:
    # Commas and colons are OpenAlex filter syntax — strip them from the value.
    safe = normalize_title(title)
    if not safe:
        return []
    resp = client.get("/works", params={"filter": f"title.search:{safe}", "per-page": per_page})
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


def resolve_reference(ref: dict) -> dict:
    """Resolve one extracted reference against OpenAlex.

    Returns {"status": "found"|"fuzzy"|"not_found", "work": ..., "candidates": [...], "notes": [...]}
    """
    notes: list[str] = []
    candidates: list[dict] = []

    with _client() as client:
        doi = clean_doi(ref.get("doi") or "")
        doi_work = None
        if doi:
            try:
                raw = _get_work_by_doi(client, doi)
            except httpx.HTTPError as exc:
                raw = None
                notes.append(f"OpenAlex DOI lookup failed: {exc}")
            if raw:
                doi_work = summarize_work(raw)
                sim = title_similarity(ref.get("title") or "", doi_work["title"] or "")
                if sim >= 0.75 or not ref.get("title"):
                    return {"status": "found", "work": doi_work, "candidates": [], "notes": notes}
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
            except httpx.HTTPError as exc:
                results = []
                notes.append(f"OpenAlex title search failed: {exc}")
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
                if best_sim >= FOUND_THRESHOLD and best_score >= FOUND_THRESHOLD and not candidates:
                    return {"status": "found", "work": best, "candidates": [], "notes": notes}
                for sc, summary in scored[:4]:
                    if title_similarity(title, summary["title"] or "") >= FUZZY_THRESHOLD or sc >= FUZZY_THRESHOLD:
                        candidates.append({**summary, "match_reason": f"Title search (score {sc:.2f})"})

    if candidates:
        notes.append("No exact match, but close candidates exist — review them on the fuzzy-matches screen.")
        return {"status": "fuzzy", "work": None, "candidates": candidates, "notes": notes}

    notes.append("No matching work found in OpenAlex — potential hallucinated reference.")
    return {"status": "not_found", "work": None, "candidates": [], "notes": notes}

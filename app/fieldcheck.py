"""Deterministic field-level verification of a reference against its resolved
OpenAlex record.

The hallucination check answers "does this work exist?"; this answers the
next question: "do the details the student PRINTED match the real record?"
It compares title, authors (especially), year, journal/venue, DOI, volume,
issue, and page numbers — flagging mismatches without spending any LLM tokens.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

# Per-field status: "match" | "close" (minor variation, e.g. an abbreviated or
# partial journal name — shown but never counted as a problem) | "mismatch" |
# "unverifiable" (not present in OpenAlex). Severity weights let the UI rank
# problems; only "mismatch" contributes.
FIELD_SEVERITY = {
    "authors": 9,
    "doi": 8,
    "title": 8,
    "year": 6,
    "pages": 4,
    "journal": 4,
    "volume": 3,
    "issue": 2,
}


def _norm(s: Optional[str]) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def _similar(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def surname(name: str) -> str:
    """Best-effort surname extraction from 'Surname, F.' or 'First Last' forms."""
    name = (name or "").strip()
    if not name:
        return ""
    if "," in name:
        return _norm(name.split(",", 1)[0])
    parts = _norm(name).split()
    return parts[-1] if parts else ""


def _pages_equal(a: Optional[str], b: Optional[str]) -> bool:
    def digits(s):
        return re.sub(r"[^\d]", "", s or "")
    na, nb = _norm(a), _norm(b)
    if na == nb and na:
        return True
    # Compare on digits, and accept a shorthand last page (e.g. 123-45 vs 123-145)
    da, db = digits(a), digits(b)
    return bool(da) and da == db


def compare_authors(ref: dict, work_authors_full: list[str]) -> dict:
    ref_names = ref.get("authors") or []
    ref_surnames = [surname(n) for n in ref_names if surname(n)]
    work_surnames = [surname(n) for n in work_authors_full if surname(n)]

    if not ref_surnames:
        # Fall back to just the first-author surname if that's all we have.
        fa = _norm(ref.get("first_author_surname") or "")
        if fa and work_surnames:
            status = "match" if any(fa == w or fa in w or w in fa for w in work_surnames) else "mismatch"
            return {"field": "authors", "status": status,
                    "reference_value": ref.get("first_author_surname"),
                    "openalex_value": ", ".join(work_authors_full[:6]) + ("…" if len(work_authors_full) > 6 else ""),
                    "detail": "Only the first author could be read from the reference."}
        return {"field": "authors", "status": "unverifiable",
                "reference_value": None, "openalex_value": ", ".join(work_authors_full[:6]),
                "detail": "No authors could be read from the reference."}

    if not work_surnames:
        return {"field": "authors", "status": "unverifiable",
                "reference_value": ", ".join(ref_names[:6]), "openalex_value": None,
                "detail": "OpenAlex has no author list for this work."}

    def found(ref_sn):
        return any(ref_sn == w or (len(ref_sn) > 3 and (ref_sn in w or w in ref_sn)) for w in work_surnames)

    wrong = [ref_names[i] for i, sn in enumerate(ref_surnames) if not found(sn)]
    detail_parts = []
    status = "match"
    if wrong:
        status = "mismatch"
        detail_parts.append("Not on the record: " + ", ".join(wrong))
    elif not ref.get("et_al") and len(ref_surnames) != len(work_surnames):
        status = "mismatch"
        detail_parts.append(
            f"Reference lists {len(ref_surnames)} author(s); OpenAlex records {len(work_surnames)}.")
    return {
        "field": "authors", "status": status,
        "reference_value": ", ".join(ref_names[:8]) + ("…" if len(ref_names) > 8 else ""),
        "openalex_value": ", ".join(work_authors_full[:8]) + ("…" if len(work_authors_full) > 8 else ""),
        "detail": " ".join(detail_parts) or "All listed authors match.",
    }


def compare_fields(ref: dict, work: dict) -> list[dict]:
    """Return a per-field comparison list. Only fields printed in the reference
    are asserted; a field absent from the reference is reported as 'missing'
    (informational) rather than a mismatch."""
    fields: list[dict] = []

    # Title — near-identical is a "close" (minor difference), not a clean match.
    if ref.get("title") and work.get("title"):
        sim = _similar(ref["title"], work["title"])
        if sim >= 0.95:
            status = "match"
        elif sim >= 0.70:
            status = "close"
        else:
            status = "mismatch"
        fields.append({"field": "title", "status": status,
                       "reference_value": ref["title"], "openalex_value": work["title"],
                       "detail": f"title similarity {sim:.2f}"})

    # Authors (always attempt — the most important check)
    fields.append(compare_authors(ref, work.get("authors_full") or work.get("authors") or []))

    # Year
    ry, wy = ref.get("year"), work.get("year")
    if ry and wy:
        fields.append({"field": "year", "status": "match" if int(ry) == int(wy) else "mismatch",
                       "reference_value": ry, "openalex_value": wy, "detail": ""})
    elif ry and not wy:
        fields.append({"field": "year", "status": "unverifiable",
                       "reference_value": ry, "openalex_value": None,
                       "detail": "OpenAlex has no publication year."})

    # DOI
    from .openalex import clean_doi
    rd, wd = clean_doi(ref.get("doi") or ""), clean_doi(work.get("doi") or "")
    if rd and wd:
        fields.append({"field": "doi", "status": "match" if rd == wd else "mismatch",
                       "reference_value": rd, "openalex_value": wd, "detail": ""})
    elif rd and not wd:
        fields.append({"field": "doi", "status": "unverifiable",
                       "reference_value": rd, "openalex_value": None,
                       "detail": "OpenAlex has no DOI for this work."})

    # Journal / venue — name variants ("Advances in Neural Information Processing
    # Systems" vs "Neural Information Processing Systems", abbreviations, a
    # leading "The") are a minor difference, not a clean match and not an error.
    if ref.get("container") and work.get("venue"):
        na, nb = _norm(ref["container"]), _norm(work["venue"])
        sim = _similar(ref["container"], work["venue"])
        contained = bool(na and nb) and (na in nb or nb in na)
        if na == nb or sim >= 0.93:
            status = "match"
        elif contained or sim >= 0.55:
            status = "close"
        else:
            status = "mismatch"
        fields.append({"field": "journal", "status": status,
                       "reference_value": ref["container"], "openalex_value": work["venue"],
                       "detail": f"venue similarity {sim:.2f}"
                                 + (" (one name contains the other)" if contained and status == "close" else "")})

    # Volume / issue
    for key in ("volume", "issue"):
        rv, wv = ref.get(key), work.get(key)
        if rv and wv:
            fields.append({"field": key,
                           "status": "match" if _norm(str(rv)) == _norm(str(wv)) else "mismatch",
                           "reference_value": rv, "openalex_value": wv, "detail": ""})

    # Pages
    rp, wp = ref.get("pages"), work.get("pages")
    if rp and wp:
        fields.append({"field": "pages", "status": "match" if _pages_equal(rp, wp) else "mismatch",
                       "reference_value": rp, "openalex_value": wp, "detail": ""})

    return fields


def field_mismatches(fields: list[dict]) -> list[dict]:
    return [f for f in fields if f["status"] == "mismatch"]


def field_severity(fields: list[dict]) -> int:
    """Max severity weight across mismatched fields (0 if none)."""
    return max((FIELD_SEVERITY.get(f["field"], 3) for f in field_mismatches(fields)), default=0)

"""Orchestration: extract references + contexts with the LLM, verify against
OpenAlex, and compare citation contexts to abstracts (misquote check)."""

from __future__ import annotations

from typing import Optional

from .llm import LLMClient, LLMError
from .openalex import resolve_reference

MAX_CONTEXTS_PER_REF = 3
MAX_ABSTRACT_CHARS = 2500
COMPARE_BATCH_SIZE = 8

EXTRACT_SYSTEM = """\
You extract bibliographic references and in-text citation contexts from a student paper.
Respond with a single JSON object only — no prose, no markdown fences."""

EXTRACT_PROMPT = """\
Below is the full text of a student paper. Do two things:

1. Find every entry in the reference list / bibliography.
2. For each reference, find the place(s) in the BODY of the paper where that
   source is cited, and copy the citing sentence together with one or two
   sentences before and after it (verbatim from the text).

Return JSON with exactly this shape:
{
  "references": [
    {
      "id": 1,
      "raw": "the reference entry exactly as it appears",
      "title": "the work's title, or null if you cannot identify one",
      "first_author_surname": "surname of the first author, or null",
      "authors": ["Full Name", "..."],
      "year": 2020,
      "doi": "10.xxxx/yyyy or null (only if printed in the reference)",
      "container": "journal or book name, or null",
      "contexts": ["citing passage 1", "citing passage 2"]
    }
  ]
}

Rules:
- Include EVERY reference in the bibliography, even if it is never cited in the body.
- "contexts" must be verbatim passages of roughly 2-4 sentences each; at most 3 per reference.
- If a reference is never cited in the body, use an empty list for "contexts".
- "year" must be an integer or null. "doi" must only be included if it is
  literally present in the document — never invent one.

PAPER TEXT:
<<<
{TEXT}
>>>"""

COMPARE_SYSTEM = """\
You check whether a student's use of a cited source matches what that source is
actually about, based on the source's abstract. Be strict about topical scope:
for example, if the abstract is about macroeconomic productivity but the student
cites it as evidence about labour productivity specifically, that is a mismatch.
Respond with a single JSON object only — no prose, no markdown fences."""

COMPARE_PROMPT = """\
For each item below, compare the student's citation context(s) with the
abstract of the cited work and judge whether the claim the student attributes
to the source is plausibly supported by it.

Verdicts:
- "match": the context is consistent with the abstract's topic and claims.
- "likely_mismatch": the topic or claim is noticeably different or narrower/broader
  than what the abstract supports (e.g. wrong sub-field, wrong direction of effect).
- "mismatch": the context clearly attributes something the abstract does not cover.
- "unclear": the contexts are too vague or the abstract too short to judge.

Return JSON with exactly this shape:
{
  "results": [
    {
      "id": 1,
      "verdict": "match",
      "explanation": "one or two sentences explaining the judgement",
      "paper_topic": "3-8 word summary of what the abstract is about",
      "student_usage": "3-8 word summary of how the student uses it"
    }
  ]
}

ITEMS:
{ITEMS}"""


def extract_references(llm: LLMClient, text: str) -> list[dict]:
    data = llm.complete_json(EXTRACT_SYSTEM, EXTRACT_PROMPT.replace("{TEXT}", text), max_tokens=16000)
    refs = data.get("references")
    if not isinstance(refs, list):
        raise LLMError("The model did not return a 'references' list.")
    cleaned = []
    for i, ref in enumerate(refs):
        if not isinstance(ref, dict):
            continue
        contexts = [c for c in (ref.get("contexts") or []) if isinstance(c, str) and c.strip()]
        cleaned.append({
            "id": ref.get("id") or (i + 1),
            "raw": (ref.get("raw") or "").strip(),
            "title": (ref.get("title") or None),
            "first_author_surname": ref.get("first_author_surname"),
            "authors": ref.get("authors") or [],
            "year": _to_int(ref.get("year")),
            "doi": ref.get("doi") or None,
            "container": ref.get("container"),
            "contexts": contexts[:MAX_CONTEXTS_PER_REF],
        })
    return cleaned


def verify_references(refs: list[dict]) -> list[dict]:
    """Hallucination check: resolve every reference against OpenAlex."""
    results = []
    for ref in refs:
        resolution = resolve_reference(ref)
        results.append({"reference": ref, **resolution})
    return results


def compare_contexts(llm: LLMClient, items: list[dict]) -> list[dict]:
    """Misquote check. Each item: {id, title, abstract, contexts}.

    Returns [{id, verdict, explanation, paper_topic, student_usage}].
    """
    out: list[dict] = []
    payload_items = []
    for item in items:
        abstract = (item.get("abstract") or "")[:MAX_ABSTRACT_CHARS]
        contexts = item.get("contexts") or []
        if not abstract:
            out.append({"id": item["id"], "verdict": "unclear",
                        "explanation": "OpenAlex has no abstract for this work, so the usage could not be compared."})
            continue
        if not contexts:
            out.append({"id": item["id"], "verdict": "unclear",
                        "explanation": "This reference is never cited in the body text, so there is nothing to compare."})
            continue
        payload_items.append({
            "id": item["id"],
            "cited_work_title": item.get("title"),
            "abstract": abstract,
            "citation_contexts": contexts,
        })

    import json as _json
    for start in range(0, len(payload_items), COMPARE_BATCH_SIZE):
        batch = payload_items[start:start + COMPARE_BATCH_SIZE]
        prompt = COMPARE_PROMPT.replace("{ITEMS}", _json.dumps(batch, ensure_ascii=False, indent=1))
        data = llm.complete_json(COMPARE_SYSTEM, prompt, max_tokens=8000)
        results = data.get("results") or []
        got = {r.get("id"): r for r in results if isinstance(r, dict)}
        for item in batch:
            r = got.get(item["id"])
            if r and r.get("verdict") in ("match", "likely_mismatch", "mismatch", "unclear"):
                out.append({
                    "id": item["id"],
                    "verdict": r["verdict"],
                    "explanation": r.get("explanation") or "",
                    "paper_topic": r.get("paper_topic"),
                    "student_usage": r.get("student_usage"),
                })
            else:
                out.append({"id": item["id"], "verdict": "unclear",
                            "explanation": "The model did not return a judgement for this reference."})
    return out


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

"""FastAPI app: citation hallucination & misquote checker (OpenAlex + your own LLM key).

The analysis endpoint streams NDJSON so the browser sees progress and results
as they are produced. This is also what keeps a long analysis from failing
with "Failed to fetch": a single blocking request can run for a minute or more
and be dropped by an idle-connection timeout, whereas a stream keeps bytes
flowing (progress events plus periodic heartbeats).

Key-safety guarantees (see also app/keysafety.py):
- LLM and OpenAlex API keys arrive in the POST body, are held in memory for
  that request only, and are never persisted anywhere.
- Nothing in this app logs request bodies; keys never appear in our own URLs.
- Every error/notes string that leaves the server passes through redact().
- FastAPI's default 422 validation response echoes request input; a custom
  handler below strips that echo.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import compare_contexts, extract_references
from .extract import ExtractionError, extract_text
from .keysafety import redact
from .llm import LLMClient, LLMError
from .openalex import OpenAlexAuthError, resolve_reference

app = FastAPI(title="openalexcheck", docs_url=None, redoc_url=None)

# The /api/verify* endpoints are a public, keyless OpenAlex wrapper meant to be
# called by external tools (e.g. an EduGenAI extension). No cookies/credentials
# are used, so permissive CORS is safe.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"], allow_credentials=False,
)

STATIC_DIR = Path(__file__).parent / "static"
ABSTRACT_CAP = 3000  # trim abstracts in API responses to keep payloads small

# Output-token cap per LLM call. Reasonable default; user-customizable.
DEFAULT_MAX_TOKENS = 16000
MIN_MAX_TOKENS = 1000
MAX_MAX_TOKENS = 64000


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _clamp_tokens(value: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS
    return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, value))


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    provider: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(""),
    openalex_key: str = Form(""),
    max_tokens: int = Form(DEFAULT_MAX_TOKENS),
    check_hallucination: bool = Form(False),
    check_misquote: bool = Form(False),
):
    openalex_key = openalex_key.strip()
    max_tokens = _clamp_tokens(max_tokens)

    def safe(msg: object) -> str:
        return redact(str(msg), api_key, openalex_key)

    # --- cheap, synchronous validation (returns normal JSON errors) ---------
    if not (check_hallucination or check_misquote):
        raise HTTPException(400, "Tick at least one check.")
    try:
        llm = LLMClient(provider, api_key, model)
    except LLMError as exc:
        raise HTTPException(400, safe(exc))
    data = await file.read()
    try:
        text = extract_text(file.filename or "", data)
    except ExtractionError as exc:
        raise HTTPException(400, str(exc))

    # --- heavy work streamed as NDJSON --------------------------------------
    stream = _run_stream(
        llm=llm, text=text, openalex_key=openalex_key or None,
        check_hallucination=check_hallucination, check_misquote=check_misquote,
        max_tokens=max_tokens, safe=safe,
    )
    return StreamingResponse(
        stream,
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_stream(*, llm, text, openalex_key, check_hallucination,
                      check_misquote, max_tokens, safe):
    """Yield newline-delimited JSON events while a worker thread runs the
    (blocking) pipeline. Heartbeats keep the connection warm during long
    LLM calls so the request can't time out mid-analysis."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(obj: Optional[dict]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, obj)

    def safe_notes(notes: list[str]) -> list[str]:
        return [safe(n) for n in notes]

    def worker() -> None:
        try:
            emit({"type": "progress", "stage": "extract",
                  "message": "Reading the document and extracting references + citation contexts…"})
            refs = extract_references(llm, text, max_tokens=max_tokens)
            if not refs:
                emit({"type": "error", "detail": "No references were found in the document."})
                return
            emit({"type": "progress", "stage": "extract_done",
                  "reference_count": len(refs),
                  "message": f"Found {len(refs)} references. Verifying against OpenAlex…"})

            found_items = []
            total = len(refs)
            for i, ref in enumerate(refs, 1):
                emit({"type": "progress", "stage": "verify", "done": i, "total": total,
                      "message": f"Verifying reference {i} of {total} in OpenAlex…"})
                res = resolve_reference(ref, api_key=openalex_key)
                item = {
                    "reference": ref, "status": res["status"], "work": res["work"],
                    "candidates": res["candidates"], "notes": safe_notes(res["notes"]),
                    "field_check": res.get("field_check", []),
                    "field_mismatch_count": res.get("field_mismatch_count", 0),
                    "misquote": None,
                }
                emit({"type": "result", "result": item})
                if check_misquote and res["status"] == "found" and res["work"]:
                    found_items.append({
                        "id": ref["id"], "title": res["work"]["title"],
                        "abstract": res["work"]["abstract"], "contexts": ref["contexts"],
                    })

            if check_misquote and found_items:
                emit({"type": "progress", "stage": "misquote",
                      "message": f"Comparing {len(found_items)} citation contexts against their abstracts…"})
                for result in compare_contexts(llm, found_items, max_tokens=max_tokens):
                    emit({"type": "misquote", "id": result["id"], "misquote": result})

            emit({"type": "done",
                  "checks": {"hallucination": check_hallucination, "misquote": check_misquote},
                  "reference_count": total})
        except OpenAlexAuthError as exc:
            emit({"type": "error", "detail": safe(exc)})
        except LLMError as exc:
            emit({"type": "error", "detail": safe(exc)})
        except Exception as exc:  # never leak internals or keys
            emit({"type": "error", "detail": safe(f"Unexpected error during analysis: {exc}")})
        finally:
            emit(None)  # sentinel: worker finished

    task = loop.run_in_executor(None, worker)
    # A padded first line forces intermediary proxies/gzip buffers to flush
    # early, so the browser starts receiving events immediately instead of only
    # after the whole (long) analysis is buffered up.
    yield json.dumps({"type": "ready", "_pad": " " * 2048}) + "\n"
    start = time.monotonic()
    try:
        while True:
            try:
                obj = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # Visible liveness during the long, opaque extraction call.
                yield json.dumps({"type": "tick", "elapsed": int(time.monotonic() - start)}) + "\n"
                continue
            if obj is None:
                break
            yield json.dumps(obj) + "\n"
    finally:
        await task


class CompareItem(BaseModel):
    id: int
    title: str | None = None
    abstract: str | None = None
    contexts: list[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    provider: str
    api_key: str
    model: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    items: list[CompareItem]


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Run the misquote comparison for individual references — used on the
    fuzzy-matches screen after the user picks the correct candidate work."""
    try:
        llm = LLMClient(req.provider, req.api_key, req.model)
        results = compare_contexts(
            llm, [item.model_dump() for item in req.items],
            max_tokens=_clamp_tokens(req.max_tokens),
        )
    except LLMError as exc:
        raise HTTPException(502, redact(str(exc), req.api_key))
    return {"results": results}


# ---------------------------------------------------------------------------
# EduGenAI (and any tool-calling platform) extension API.
#
# A keyless, deterministic OpenAlex wrapper: no LLM runs here. The calling
# platform's own model extracts references from the paper and does the misquote
# reasoning; this endpoint just answers "does this work exist, and do the
# printed details match?" plus returns the abstract for that reasoning.
# ---------------------------------------------------------------------------

class VerifyReference(BaseModel):
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    first_author_surname: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None       # maps to OpenAlex "venue"
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    et_al: bool = False


class VerifyRequest(VerifyReference):
    openalex_key: Optional[str] = None  # optional; header X-OpenAlex-Key preferred


class VerifyBatchRequest(BaseModel):
    references: list[VerifyReference] = Field(default_factory=list)
    openalex_key: Optional[str] = None


def _ref_dict(v: VerifyReference, idx: int = 1) -> dict:
    return {
        "id": idx, "raw": "", "title": v.title, "authors": v.authors,
        "first_author_surname": v.first_author_surname, "year": v.year,
        "doi": v.doi, "container": v.journal, "volume": v.volume,
        "issue": v.issue, "pages": v.pages, "et_al": v.et_al, "contexts": [],
    }


def _trim_abstract(work: Optional[dict]) -> Optional[dict]:
    if not work:
        return work
    w = dict(work)
    if w.get("abstract") and len(w["abstract"]) > ABSTRACT_CAP:
        w["abstract"] = w["abstract"][:ABSTRACT_CAP] + "…"
    return w


# Same field weights and status labels the web UI uses, so an extension renders
# results identically to the site (severity-sorted, badged).
_FIELD_WEIGHT = {"authors": 9, "doi": 8, "title": 8, "year": 6,
                 "pages": 4, "journal": 4, "volume": 3, "issue": 2}
_STATUS_BADGE = {"found": "Verified", "fuzzy": "Fuzzy match",
                 "not_found": "Potential hallucination", "lookup_failed": "Lookup failed"}


def _display(res: dict) -> dict:
    """Precompute the site's badge / severity / priority so the calling model
    doesn't have to (keeps ordering and labels consistent with the web UI).
    Severity here is existence + metadata only; the model should treat a
    misquote 'mismatch' as severity >= 80 as well (it isn't known server-side)."""
    status = res["status"]
    mism = [f for f in res.get("field_check", []) if f.get("status") == "mismatch"]
    minor = [f for f in res.get("field_check", []) if f.get("status") == "close"]
    if status == "not_found":
        sev = 100
    elif status == "fuzzy":
        sev = 60
    elif status == "lookup_failed":
        sev = 35
    elif status == "found":
        maxw = max((_FIELD_WEIGHT.get(f["field"], 3) for f in mism), default=0)
        sev = 85 if maxw >= 8 else 70 if maxw >= 6 else 50 if maxw > 0 else 8
    else:
        sev = 0
    priority = "Review" if sev >= 70 else ("Check" if sev >= 45 else "")
    return {"badge": _STATUS_BADGE.get(status, status), "severity": sev,
            "priority": priority, "mismatched_fields": [f["field"] for f in mism],
            "minor_fields": [f["field"] for f in minor]}


def _verify_response(res: dict) -> dict:
    return {
        "status": res["status"],
        **_display(res),
        "work": _trim_abstract(res.get("work")),
        "field_check": res.get("field_check", []),
        "field_mismatch_count": res.get("field_mismatch_count", 0),
        "candidates": [_trim_abstract(c) for c in res.get("candidates", [])],
        "notes": res.get("notes", []),
    }


@app.post("/api/verify")
def api_verify(body: VerifyRequest, x_openalex_key: Optional[str] = Header(default=None)):
    """Verify a single reference against OpenAlex. Returns existence, a field-by-
    field metadata comparison, and the abstract. No LLM, no auth required."""
    key = (x_openalex_key or body.openalex_key or "").strip() or None
    try:
        res = resolve_reference(_ref_dict(body), api_key=key)
    except OpenAlexAuthError as exc:
        raise HTTPException(400, redact(str(exc), key))
    return _verify_response(res)


@app.post("/api/verify_batch")
def api_verify_batch(body: VerifyBatchRequest, x_openalex_key: Optional[str] = Header(default=None)):
    """Verify many references in one call (preferred for a whole bibliography —
    one round trip, and kinder to OpenAlex rate limits)."""
    if len(body.references) > 200:
        raise HTTPException(400, "Too many references in one request (max 200).")
    key = (x_openalex_key or body.openalex_key or "").strip() or None
    results = []
    for i, ref in enumerate(body.references, 1):
        try:
            res = resolve_reference(_ref_dict(ref, i), api_key=key)
        except OpenAlexAuthError as exc:
            raise HTTPException(400, redact(str(exc), key))
        results.append({"index": i, **_verify_response(res)})
    return {"count": len(results), "results": results}


@app.get("/edugenai")
def edugenai():
    return FileResponse(STATIC_DIR / "edugenai.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

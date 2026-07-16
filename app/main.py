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
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import compare_contexts, extract_references
from .extract import ExtractionError, extract_text
from .keysafety import redact
from .llm import LLMClient, LLMError
from .openalex import OpenAlexAuthError, resolve_reference

app = FastAPI(title="openalexcheck", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"

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
    try:
        while True:
            try:
                obj = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                yield json.dumps({"type": "heartbeat"}) + "\n"
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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

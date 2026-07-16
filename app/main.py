"""FastAPI app: citation hallucination & misquote checker (OpenAlex + your own LLM key).

Key-safety guarantees (see also app/keysafety.py):
- LLM and OpenAlex API keys arrive in the POST body, are held in memory for
  that request only, and are never persisted anywhere.
- Nothing in this app logs request bodies; uvicorn's access log records only
  method/path/status, and keys are never placed in our own URLs.
- Every error detail that leaves the server passes through redact(), so a key
  can never surface via provider error bodies or httpx exception text.
- FastAPI's default 422 validation response echoes request input back; a
  custom handler below strips that echo.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import compare_contexts, extract_references, verify_references
from .extract import ExtractionError, extract_text
from .keysafety import redact
from .llm import LLMClient, LLMError
from .openalex import OpenAlexAuthError

app = FastAPI(title="openalexcheck", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # Default handler echoes the offending input (which may include an API
    # key) back in the response body — return field locations only.
    errors = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    provider: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(""),
    openalex_key: str = Form(""),
    check_hallucination: bool = Form(False),
    check_misquote: bool = Form(False),
):
    openalex_key = openalex_key.strip()

    def safe(msg: object) -> str:
        return redact(str(msg), api_key, openalex_key)

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

    # 1. LLM: extract references + citation contexts
    try:
        refs = extract_references(llm, text)
    except LLMError as exc:
        raise HTTPException(502, safe(f"Reference extraction failed: {exc}"))
    if not refs:
        raise HTTPException(422, "No references were found in the document.")

    # 2. OpenAlex: resolve each reference (needed for both checks — the
    #    misquote check compares against the OpenAlex abstract).
    try:
        verified = verify_references(refs, openalex_key or None)
    except OpenAlexAuthError as exc:
        raise HTTPException(400, safe(exc))

    # 3. Misquote check on resolved references
    misquote_results: dict = {}
    if check_misquote:
        items = [
            {
                "id": v["reference"]["id"],
                "title": v["work"]["title"],
                "abstract": v["work"]["abstract"],
                "contexts": v["reference"]["contexts"],
            }
            for v in verified
            if v["status"] == "found" and v["work"]
        ]
        try:
            for result in compare_contexts(llm, items):
                misquote_results[result["id"]] = result
        except LLMError as exc:
            raise HTTPException(502, safe(f"Misquote check failed: {exc}"))

    results = []
    for v in verified:
        ref = v["reference"]
        results.append({
            "reference": ref,
            "status": v["status"],
            "work": v["work"],
            "candidates": v["candidates"],
            "notes": [safe(n) for n in v["notes"]],
            "misquote": misquote_results.get(ref["id"]),
        })

    return {
        "checks": {"hallucination": check_hallucination, "misquote": check_misquote},
        "reference_count": len(refs),
        "results": results,
    }


class CompareItem(BaseModel):
    id: int
    title: str | None = None
    abstract: str | None = None
    contexts: list[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    provider: str
    api_key: str
    model: str = ""
    items: list[CompareItem]


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Run the misquote comparison for individual references — used on the
    fuzzy-matches screen after the user picks the correct candidate work."""
    try:
        llm = LLMClient(req.provider, req.api_key, req.model)
        results = compare_contexts(llm, [item.model_dump() for item in req.items])
    except LLMError as exc:
        raise HTTPException(502, redact(str(exc), req.api_key))
    return {"results": results}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

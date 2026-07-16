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
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

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
            refs, orphans = extract_references(llm, text, max_tokens=max_tokens)
            if not refs and not orphans:
                emit({"type": "error", "detail": "No references were found in the document."})
                return
            emit({"type": "progress", "stage": "extract_done",
                  "reference_count": len(refs),
                  "message": f"Found {len(refs)} references. Verifying against OpenAlex…"})
            if orphans:
                # In-text citations with no bibliography entry — the reader can
                # never look these up, so they are flagged in their own right.
                emit({"type": "orphans", "items": [
                    {"label": safe(o["label"]), "year": o["year"],
                     "context": safe(o["context"])} for o in orphans]})

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

# The /api/verify* endpoints face an LLM (an EduGenAI extension or any
# function-caller), so the request body arrives in whatever shape the model
# decided to emit: a stringified JSON array, a bare top-level array, a single
# reference object, args nested under "body"/"arguments", a non-string key,
# messy field types. Validating with Pydantic would 422 and fail the whole
# call — instead we read the raw body and normalize it ourselves, never
# rejecting on shape; _coerce_ref() then normalizes each field leniently.

async def _read_json(request: Request) -> Any:
    """Best-effort parse of the request body as JSON, tolerating a wrong or
    missing Content-Type (some callers POST JSON as text/plain) and a
    form-encoded body (key=value&… with JSON inside the values)."""
    try:
        return await request.json()
    except Exception:
        raw = (await request.body()).decode("utf-8", "replace").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass
        if "=" in raw and raw[:1] not in ("[", "{"):
            from urllib.parse import parse_qs
            try:
                pairs = parse_qs(raw, keep_blank_values=True)
                if pairs:
                    return {k: _loads_maybe(v[-1]) for k, v in pairs.items()}
            except Exception:
                pass
        return raw


def _loads_maybe(v: Any) -> Any:
    """If v is a string that looks like JSON, parse it; otherwise return as-is.
    LLM function-callers frequently serialize array/object arguments as strings."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in ("[", "{"):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


_REF_KEYS = {"title", "doi", "authors", "author", "first_author_surname",
             "year", "journal", "container", "volume", "issue", "pages", "et_al"}


def _looks_like_ref(d: Any) -> bool:
    return isinstance(d, dict) and any(k in d for k in _REF_KEYS)


def _as_key(v: Any) -> Optional[str]:
    """An OpenAlex key is always a string; anything else is caller junk -> ignore."""
    return (v.strip() or None) if isinstance(v, str) else None


def _normalize_batch(payload: Any, _depth: int = 0) -> tuple[list, Optional[str]]:
    """Reduce any reasonable request shape to (references_list, openalex_key).
    Handles: stringified JSON, a bare array, {"references": [...]}, a single
    reference object, and args nested under ANY wrapper key ({"parameters":
    {...}} etc. — EduGenAI wraps the function arguments, and the wrapper name
    is the platform's choice, so we descend rather than match a fixed list)."""
    payload = _loads_maybe(payload)
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        return [], None
    key = _as_key(payload.get("openalex_key"))
    refs = _loads_maybe(payload.get("references"))
    if isinstance(refs, dict):
        return [refs], key
    if isinstance(refs, list):
        return refs, key
    # The object itself may BE a single reference…
    if _looks_like_ref(payload):
        return [payload], key
    # …otherwise descend into nested values looking for something that yields
    # actual reference OBJECTS (so a stray list of scalars is never mistaken
    # for a bibliography).
    if _depth < 4:
        for v in payload.values():
            inner_refs, inner_key = _normalize_batch(v, _depth + 1)
            if inner_refs and any(isinstance(_loads_maybe(r), dict) for r in inner_refs):
                return inner_refs, (key or inner_key)
    return [], key


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_year(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"\b(1[5-9]\d\d|20\d\d|21\d\d)\b", str(v))  # a plausible 4-digit year
    return int(m.group(1)) if m else None


def _as_authors(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        # Split on separators that don't collide with "Surname, F." — semicolons,
        # ampersands, and the word "and". Commas inside names are left intact.
        parts = re.split(r"\s*;\s*|\s*&\s*|\s+and\s+", v)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(v, (list, tuple)):
        out = []
        for a in v:
            if isinstance(a, str):
                if a.strip():
                    out.append(a.strip())
            elif isinstance(a, dict):
                name = a.get("name") or a.get("display_name") or a.get("family") or a.get("last")
                if name:
                    out.append(str(name).strip())
            elif a is not None:
                out.append(str(a).strip())
        return out
    return [str(v).strip()]


def _coerce_ref(raw: dict, idx: int = 1) -> dict:
    """Normalize a raw reference object (whatever shape the caller sent) into the
    dict resolve_reference expects. Never raises."""
    journal = raw.get("journal")
    if journal is None:
        journal = raw.get("container")
    return {
        "id": idx, "raw": "", "title": _as_str(raw.get("title")),
        "authors": _as_authors(raw.get("authors")),
        "first_author_surname": _as_str(raw.get("first_author_surname")),
        "year": _as_year(raw.get("year")), "doi": _as_str(raw.get("doi")),
        "container": _as_str(journal), "volume": _as_str(raw.get("volume")),
        "issue": _as_str(raw.get("issue")), "pages": _as_str(raw.get("pages")),
        "et_al": bool(raw.get("et_al")), "contexts": [],
    }


def _safe_resolve(ref: dict, key: Optional[str]) -> dict:
    """resolve_reference that never raises (except on a bad API key, which is a
    global config error worth surfacing) — a single odd reference must not sink
    the whole batch."""
    try:
        return resolve_reference(ref, api_key=key)
    except OpenAlexAuthError:
        raise
    except Exception as exc:  # network oddities, unexpected data shapes, etc.
        return {"status": "lookup_failed", "work": None, "candidates": [],
                "notes": [redact(f"This reference could not be verified: {exc}", key)]}


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


@app.get("/api/verify")
@app.post("/api/verify")
async def api_verify(request: Request, x_openalex_key: Optional[str] = Header(default=None)):
    """Verify a single reference against OpenAlex. Returns existence, a field-by-
    field metadata comparison, and the abstract. No LLM, no auth required.
    Accepts any body shape an LLM might send — never rejects on shape or type.
    GET works too: the reference fields ride the query string (for platforms
    that never put function arguments in a POST body)."""
    payload = await _read_json(request)
    references, body_key = _normalize_batch(payload)
    if not references:
        references, body_key = _from_query(request)
    key = (x_openalex_key or body_key or "").strip() or None
    first = references[0] if references and isinstance(references[0], dict) else {}
    try:
        res = _safe_resolve(_coerce_ref(first, 1), key)
    except OpenAlexAuthError as exc:
        raise HTTPException(400, redact(str(exc), key))
    resp = {**_verify_response(res), "api_version": API_VERSION}
    if not references:
        resp["hint"] = _shape_hint(payload, request)
    return resp


def _batch_item(idx: int, raw: Any, key: Optional[str]) -> dict:
    raw = _loads_maybe(raw)  # a reference item may itself be stringified JSON
    if not isinstance(raw, dict):
        res = {"status": "lookup_failed", "work": None, "candidates": [],
               "notes": ["This entry could not be read as a reference object."]}
    else:
        res = _safe_resolve(_coerce_ref(raw, idx), key)  # may raise OpenAlexAuthError
    return {"index": idx, **_verify_response(res)}


# Included in every verify response so a pasted extension transcript shows
# WHICH build answered — a count:0 from a stale deployment is otherwise
# indistinguishable from a parsing failure on the current one.
# BUMP THIS on every change to these endpoints' behavior.
API_VERSION = "2026-07-16.8"


def _from_query(request: Request) -> tuple[list, Optional[str]]:
    """Fallback when the body yields no references: some platforms bind the
    function arguments to the URL query string instead of the body (an empty
    POST body with args in the query is the signature of that)."""
    if not request.query_params:
        return [], None
    return _normalize_batch({k: _loads_maybe(v) for k, v in request.query_params.items()})


def _shape_hint(payload: Any, request: Optional[Request] = None) -> str:
    """Human/LLM-readable description of a request body no references could be
    read from. Structure only — key names, types, sizes — NEVER values, which
    could contain keys or document text."""
    if payload is None:
        desc = "empty"
    elif isinstance(payload, str):
        desc = f"an unparseable string of {len(payload)} characters"
    elif isinstance(payload, dict):
        desc = "a JSON object with keys " + str(sorted(str(k) for k in payload)[:10])
    elif isinstance(payload, list):
        desc = f"a JSON array of {len(payload)} items, none of which is a reference object"
    else:
        desc = f"of type {type(payload).__name__}"
    if request is not None:
        ct = request.headers.get("content-type") or "none"
        cl = request.headers.get("content-length") or "0"
        qkeys = sorted(request.query_params.keys())
        # Non-infrastructure header KEYS (never values) — reveals arguments
        # riding in a custom header.
        boring = ("host", "user-agent", "accept", "content-", "connection",
                  "x-forwarded", "x-real-ip", "forwarded", "x-vercel", "cf-",
                  "sec-", "via", "cdn-", "traceparent", "x-request-id", "priority")
        extra_headers = sorted(k for k in {h.lower() for h in request.headers.keys()}
                               if not any(k.startswith(b) for b in boring))
        desc += (f" (method: {request.method}; Content-Type: {ct}; Content-Length: {cl}; "
                 f"query-string keys: {qkeys}; other header keys: {extra_headers})")
    return ("No references could be read from the request body, which was " + desc +
            '. Expected {"references": [{"title": "...", "authors": [...], "year": ...}, ...]}. '
            "If the arguments appear in NEITHER the body nor the query string (empty on both "
            "POST and GET), the platform's gateway is failing to serialize the nested "
            "'references' array parameter — redefine the function with a SINGLE STRING "
            "parameter named references_json and pass the reference array serialized as a "
            "JSON string (see 'Plan B' on /edugenai); this endpoint accepts that too. "
            "If you are an assistant relaying this to a user: report this hint verbatim. "
            "See /edugenai for the format and a test command.")


@app.get("/api/verify_batch")
@app.post("/api/verify_batch")
async def api_verify_batch(request: Request, x_openalex_key: Optional[str] = Header(default=None)):
    """Verify many references in one call (preferred for a whole bibliography —
    one round trip). Accepts any body shape an LLM might send; malformed entries
    are reported per-reference, never failing the whole batch; lookups run in
    parallel to keep a big bibliography fast. GET works too: the arguments ride
    the query string (for platforms that never put them in a POST body)."""
    payload = await _read_json(request)
    references, body_key = _normalize_batch(payload)
    if not references:
        references, body_key = _from_query(request)
    if len(references) > 200:
        raise HTTPException(400, "Too many references in one request (max 200).")
    key = (x_openalex_key or body_key or "").strip() or None
    items = list(enumerate(references, 1))
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda it: _batch_item(it[0], it[1], key), items))
    except OpenAlexAuthError as exc:
        raise HTTPException(400, redact(str(exc), key))
    resp: dict = {"count": len(results), "results": results, "api_version": API_VERSION}
    if not results:
        resp["hint"] = _shape_hint(payload, request)
    return resp


@app.get("/edugenai")
def edugenai():
    return FileResponse(STATIC_DIR / "edugenai.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

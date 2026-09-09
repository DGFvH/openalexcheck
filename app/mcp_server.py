"""MCP server (Streamable HTTP) exposing the reference check as one tool.

eduGenAI 2 (LibreChat build 2026.8.x) has no OpenAPI action import: its Personas
carry no tools section, and its Extensions panel accepts only a "URL of MCP
server". So the tool published at /openapi/edugenai.json has no consumer there,
and the same operation is served here over MCP instead. The schema is imported
from `toolspec`, so the two descriptions cannot drift.

Design notes
------------
* Stateless. No session id is issued, so it survives serverless cold starts and
  per-request isolation. A session-less server is fine for the official client.
* Plain `application/json`, not SSE. Streamable HTTP permits it and it sidesteps
  streaming quirks on the host. GET returns 405, which the client reads as "no
  server-initiated stream available".
* No `mcp` SDK dependency — the wire protocol needed here is a few dozen lines
  of JSON-RPC, and pinning the SDK on the serverless runtime buys nothing.

Wiring: `app.include_router(mcp_server.build_router(verify=...))`. The verifier
is injected rather than imported so this module never imports `main` (which
imports this one).
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from .toolspec import MAX_REFERENCES, OPERATION_DESCRIPTION, REFERENCE_PROPERTIES

SERVER_NAME = "phantocite"
SERVER_VERSION = "1.0.0"
LATEST_PROTOCOL = "2025-06-18"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
TOOL_NAME = "verify_references"

# The same operation the OpenAPI document describes, in MCP's shape.
VERIFY_REFERENCES_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "title": "Verify a paper's references against OpenAlex",
    "description": OPERATION_DESCRIPTION,
    "inputSchema": {
        "type": "object",
        "required": ["references"],
        "properties": {
            "references": {
                "type": "array",
                "description": ("Every entry in the paper's reference list. "
                                f"At most {MAX_REFERENCES} per call."),
                "maxItems": MAX_REFERENCES,
                "items": {
                    "type": "object",
                    "required": ["title"],
                    "properties": REFERENCE_PROPERTIES,
                },
            }
        },
    },
}

TOOLS = [VERIFY_REFERENCES_TOOL]

# Takes {"references": [...]} plus the request, returns the verify_batch body.
Verifier = Callable[[list, Request], Awaitable[dict] | dict]


def _tool_error(message: str) -> dict[str, Any]:
    """A failed tool call is a RESULT, not a protocol error: the model has to
    see the reason and react, not receive a transport failure."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _result(mid: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": payload}


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def build_router(verify: Verifier) -> APIRouter:
    router = APIRouter()

    async def _run_tool(name: str, arguments: dict[str, Any], request: Request) -> dict[str, Any]:
        if name != TOOL_NAME:
            return _tool_error(f"Unknown tool: {name}")

        references = (arguments or {}).get("references")
        # Defensive: some gateways flatten a nested array into a JSON string.
        # The HTTP endpoint already tolerates that, so accept it here too.
        if isinstance(references, str):
            try:
                references = json.loads(references)
            except json.JSONDecodeError:
                return _tool_error("The 'references' argument arrived as a string that is "
                                   "not valid JSON.")

        if not isinstance(references, list) or not references:
            return _tool_error(
                "No references were supplied. Extract the paper's reference list first, "
                "then call verify_references once with every entry that has a title.")
        if len(references) > MAX_REFERENCES:
            return _tool_error(
                f"{len(references)} references supplied; the batch limit is "
                f"{MAX_REFERENCES}. Split the bibliography across calls.")

        try:
            data = verify(references, request)
            if inspect.isawaitable(data):
                data = await data
        except Exception as exc:  # surfaced to the model, never swallowed
            return _tool_error(f"The verification request failed: {type(exc).__name__}: {exc}")

        return {
            "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
            "structuredContent": data,
            "isError": False,
        }

    async def _handle(message: Any, request: Request) -> dict[str, Any] | None:
        """One JSON-RPC message in, one reply out — None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, -32600, "Invalid Request")

        method = message.get("method")
        mid = message.get("id")
        # Notifications and client responses get no reply. `id: 0` is a request.
        if "id" not in message or mid is None:
            return None

        if method == "initialize":
            requested = (message.get("params") or {}).get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOLS else LATEST_PROTOCOL
            return _result(mid, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": ("Use verify_references once per paper, with the whole "
                                 "reference list. Extract the references from the document "
                                 "yourself first."),
            })
        if method == "ping":
            return _result(mid, {})
        if method == "tools/list":
            return _result(mid, {"tools": TOOLS})
        if method == "tools/call":
            params = message.get("params") or {}
            return _result(mid, await _run_tool(params.get("name"),
                                                params.get("arguments") or {}, request))
        # Neither resources nor prompts are advertised, but clients probe anyway;
        # empty lists are friendlier than -32601.
        if method == "resources/list":
            return _result(mid, {"resources": []})
        if method == "resources/templates/list":
            return _result(mid, {"resourceTemplates": []})
        if method == "prompts/list":
            return _result(mid, {"prompts": []})
        return _error(mid, -32601, f"Method not found: {method}")

    @router.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(_error(None, -32700, "Parse error"), status_code=400)

        # 2025-03-26 permitted JSON-RPC batches; 2025-06-18 removed them. Accept both.
        if isinstance(body, list):
            replies = [r for r in [await _handle(m, request) for m in body] if r is not None]
            return JSONResponse(replies) if replies else Response(status_code=202)

        reply = await _handle(body, request)
        return JSONResponse(reply) if reply is not None else Response(status_code=202)

    @router.get("/mcp")
    async def mcp_no_server_stream() -> Response:
        """No server-initiated SSE stream; the client falls back to POST-only."""
        return JSONResponse(
            _error(None, -32000, "Method Not Allowed: this server does not open SSE streams."),
            status_code=405)

    @router.delete("/mcp")
    async def mcp_terminate() -> Response:
        """Stateless — there is no session to tear down."""
        return Response(status_code=204)

    return router

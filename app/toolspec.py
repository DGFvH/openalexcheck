"""One shared description of the reference-checking tool.

`app/mcp_server.py` renders it as an MCP tool — the only kind eduGenAI 2's
Extensions panel accepts — and `openapi_document()` renders the same operation
as an OpenAPI document for platforms that import one instead (ChatGPT actions,
other LibreChat builds). Keeping both from one source means the two can never
describe different arguments.

Deliberately minimal: one operation. The app's own FastAPI-generated schema
describes every route, including the browser-facing ones, which would give an
agent tools it must never call.

OpenAPI 3.0.3 rather than 3.1: it is what LibreChat's own example uses, and it
avoids the 3.1 type-array syntax that older parsers reject. Schemas are inlined
rather than referenced through `components`, so no $ref resolution is needed.
"""

from __future__ import annotations

# Cap on one batch, shared by the HTTP route, the OpenAPI schema and the MCP tool.
MAX_REFERENCES = 200

REFERENCE_PROPERTIES = {
    "title": {"type": "string", "description": "The work's title, exactly as printed."},
    "authors": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Every author name printed in the reference, in order, e.g. 'Smith, J.'.",
    },
    "et_al": {
        "type": "boolean",
        "description": "True if the reference abbreviates the author list with 'et al.'.",
    },
    "year": {"type": "integer", "description": "Publication year as printed."},
    "doi": {"type": "string", "description": "Only if a DOI is printed in the reference."},
    "journal": {"type": "string", "description": "Journal, book or venue name as printed."},
    "volume": {"type": "string"},
    "issue": {"type": "string"},
    "pages": {"type": "string", "description": "Page range as printed, e.g. '123-145'."},
}

OPERATION_DESCRIPTION = (
    "Verify a list of bibliographic references against OpenAlex, the open index of "
    "scholarly works. For each reference this returns whether the work exists "
    "(found / fuzzy / not_found / lookup_failed), a field-by-field comparison of the "
    "printed metadata (title, authors, year, journal, DOI, volume, issue, pages) "
    "against the real record, a ready-made badge and severity score, and the abstract "
    "of the matched work so its use in the paper can be judged. Call this once with "
    "the whole bibliography, after extracting the references from the document "
    "yourself. Never call it with an empty or placeholder list."
)

_WORK = {
    "type": "object",
    "description": "The OpenAlex record this reference matched, or null when it did not match one.",
    "nullable": True,
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "year": {"type": "integer"},
        "venue": {"type": "string"},
        "doi": {"type": "string"},
        "url": {"type": "string"},
        "pages": {"type": "string"},
        "abstract": {"type": "string", "description": "Use this to judge whether the citation's claim is supported."},
    },
}

_FIELD_CHECK = {
    "type": "array",
    "description": "One entry per printed detail that could be compared with the OpenAlex record.",
    "items": {
        "type": "object",
        "properties": {
            "field": {"type": "string", "description": "title, authors, year, doi, journal, volume, issue or pages."},
            "status": {
                "type": "string",
                "description": "match; close (a minor naming variation, e.g. an abbreviated journal); or mismatch.",
            },
            "reference_value": {"type": "string", "description": "What the paper printed."},
            "openalex_value": {"type": "string", "description": "What the record says."},
        },
    },
}

_RESULT = {
    "type": "object",
    "properties": {
        "index": {"type": "integer", "description": "1-based position in the list you sent."},
        "status": {
            "type": "string",
            "description": ("found = the work exists and was matched; fuzzy = close candidates "
                            "but no confident match; not_found = no such work in OpenAlex, treat as "
                            "a probable fabrication; lookup_failed = it could not be checked, which "
                            "is NOT evidence of fabrication."),
        },
        "badge": {"type": "string", "description": "Verified, Fuzzy match, Potential hallucination or Lookup failed."},
        "severity": {"type": "integer", "description": "0-100. Sort your report by this, descending."},
        "priority": {"type": "string", "description": "Review, Check, or empty."},
        "mismatched_fields": {"type": "array", "items": {"type": "string"}},
        "minor_fields": {"type": "array", "items": {"type": "string"}},
        "field_mismatch_count": {"type": "integer"},
        "work": _WORK,
        "field_check": _FIELD_CHECK,
        "candidates": {
            "type": "array",
            "description": "For fuzzy matches only: the closest works, so the user can pick the right one.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": "integer"},
                    "venue": {"type": "string"},
                    "doi": {"type": "string"},
                    "url": {"type": "string"},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}


def openapi_document(base_url: str, version: str) -> dict:
    """The importable schema. `base_url` must be the origin the action calls:
    LibreChat rejects an action whose domain differs from servers[0].url."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Phantocite citation verifier",
            "version": version,
            "description": (
                "Checks a paper's references against OpenAlex: whether each cited work "
                "actually exists, whether the printed details match the real record, and "
                "what the source is really about. No API key is needed."
            ),
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/api/verify_batch": {
                "post": {
                    "operationId": "verify_references",
                    "summary": "Verify a paper's references against OpenAlex",
                    "description": OPERATION_DESCRIPTION,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
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
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "One result per reference, in the order sent.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "count": {"type": "integer"},
                                            "api_version": {"type": "string"},
                                            "results": {"type": "array", "items": _RESULT},
                                            "hint": {
                                                "type": "string",
                                                "description": ("Only present when nothing could be read from "
                                                                "the request. Report it verbatim to the user."),
                                            },
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            }
        },
    }

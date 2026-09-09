"""The description of the reference-checking tool, in one place.

`app/mcp_server.py` renders it as an MCP tool — the only kind eduGenAI 2's
Extensions panel accepts. It is deliberately one operation: the app's own
FastAPI-generated schema describes every route, including the browser-facing
ones, which would offer an agent tools it must never call. That schema is not
served either (see `openapi_url=None` in main).
"""

from __future__ import annotations

# Cap on one batch, shared by the HTTP route and the MCP tool.
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

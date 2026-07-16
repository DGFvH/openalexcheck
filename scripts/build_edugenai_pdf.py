#!/usr/bin/env python3
"""Generate app/static/edugenai-instructions.pdf.

Build-time only (reportlab is not a runtime dependency of the app). Re-run this
after editing the EduGenAI instructions to refresh the committed PDF:

    pip install reportlab
    python scripts/build_edugenai_pdf.py
"""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer,
)

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "edugenai-instructions.pdf"

ACCENT = HexColor("#1f6f5c")
INK = HexColor("#26241f")
MUTED = HexColor("#6f6a5f")
CODE_BG = HexColor("#f0ede4")
CODE_BORDER = HexColor("#d9d3c6")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], fontName="Helvetica-Bold",
                    fontSize=20, textColor=INK, spaceAfter=4, alignment=TA_LEFT)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontName="Helvetica",
                     fontSize=10, textColor=MUTED, spaceAfter=14, leading=14)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=14, textColor=ACCENT, spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Helvetica-Bold",
                    fontSize=11, textColor=INK, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName="Helvetica",
                      fontSize=10, textColor=INK, leading=15, spaceAfter=6)
LABEL = ParagraphStyle("LABEL", parent=styles["Normal"], fontName="Helvetica-Bold",
                       fontSize=8, textColor=MUTED, spaceBefore=6, spaceAfter=2)
CODE = ParagraphStyle("CODE", parent=styles["Code"], fontName="Courier", fontSize=7.6,
                      textColor=INK, backColor=CODE_BG, borderColor=CODE_BORDER,
                      borderWidth=0.5, borderPadding=6, leading=10, spaceAfter=8)
NOTE = ParagraphStyle("NOTE", parent=BODY, backColor=HexColor("#fbf7ec"),
                      borderColor=CODE_BORDER, borderWidth=0.5, borderPadding=7,
                      leftIndent=2, spaceBefore=4, spaceAfter=8)

HOST = "https://YOUR-DEPLOYMENT-HOST"

DETAIL = """Use this extension to fact-check the reference list of an uploaded paper.

Workflow:
1. Read the document and extract every entry in the reference list. For each, pull
   out: title, the full list of author names, year, DOI (only if printed),
   journal/venue, volume, issue, and page range. Also find the sentence(s) in the
   body where that source is cited, with one sentence of context on each side.
2. Call the "verify_references" function with all references at once.
3. For every reference the function returns:
   - If status is "not_found", flag it as a POTENTIAL HALLUCINATION.
   - If status is "fuzzy", tell the user the closest candidate(s) it returned.
   - If "field_check" contains any field with status "mismatch", report exactly
     which printed details are wrong (especially authors and year), showing the
     reference value vs the OpenAlex value.
   - Using the returned "abstract", judge whether the student's citing sentence
     matches what the paper is actually about. If the paper is on a different
     topic than the student implies, flag it as a MISQUOTE and explain in one
     sentence.
4. Present the results as a table sorted worst-first: hallucinations, then wrong
   authors/DOI/title, then misquotes, then wrong year, then minor field
   mismatches, then clean references. Never invent a DOI, author, or verdict that
   the function did not return."""

SCHEMA = """{
  "name": "verify_references",
  "description": "Verify a list of bibliographic references against OpenAlex.
    For each reference, returns whether the work exists (found / fuzzy /
    not_found), a field-by-field comparison of the printed metadata (title,
    authors, year, journal, DOI, volume, issue, pages) against the real record,
    and the abstract of the matched work.",
  "parameters": {
    "type": "object",
    "properties": {
      "references": {
        "type": "array",
        "description": "Every reference in the paper's bibliography.",
        "items": {
          "type": "object",
          "properties": {
            "title":   { "type": "string" },
            "authors": { "type": "array", "items": { "type": "string" },
                         "description": "Every author name printed, in order." },
            "et_al":   { "type": "boolean" },
            "year":    { "type": "integer" },
            "doi":     { "type": "string" },
            "journal": { "type": "string" },
            "volume":  { "type": "string" },
            "issue":   { "type": "string" },
            "pages":   { "type": "string" }
          },
          "required": ["title"]
        }
      }
    },
    "required": ["references"]
  }
}"""

REQ = """{
  "references": [
    {
      "title": "Highly accurate protein structure prediction with AlphaFold",
      "authors": ["Smith, J.", "Jones, B."],
      "et_al": false,
      "year": 2019,
      "journal": "Science",
      "pages": "100-110"
    }
  ]
}"""

RESP = """{
  "count": 1,
  "results": [{
    "index": 1,
    "status": "found",
    "work": {
      "title": "Highly accurate protein structure prediction with AlphaFold",
      "authors": ["John Jumper", "Richard Evans", "..."],
      "year": 2021, "venue": "Nature",
      "doi": "10.1038/s41586-021-03819-2",
      "abstract": "Proteins are essential to life ... (used for the misquote check)"
    },
    "field_mismatch_count": 3,
    "field_check": [
      { "field": "title",   "status": "match" },
      { "field": "authors", "status": "mismatch",
        "reference_value": "Smith, J., Jones, B.",
        "openalex_value": "John Jumper, Richard Evans, ..." },
      { "field": "year",    "status": "mismatch",
        "reference_value": 2019, "openalex_value": 2021 },
      { "field": "journal", "status": "mismatch",
        "reference_value": "Science", "openalex_value": "Nature" }
    ]
  }]
}"""


def code(text):
    return Preformatted(text, CODE)


def build():
    story = []
    story.append(Paragraph("Use the Citation Checker in EduGenAI", H1))
    story.append(Paragraph(
        "Register this tool as an EduGenAI Extension so the assistant can verify a "
        "paper's references against OpenAlex — inside your own EduGenAI chat, "
        "with no separate LLM key.", SUB))

    story.append(Paragraph("How it works", H2))
    story.append(Paragraph(
        "An EduGenAI Extension is a function-calling tool: you give EduGenAI an HTTP "
        "endpoint and a function definition, and its assistant calls that endpoint "
        "as a JSON request whenever it needs to. The work splits across two sides:", BODY))
    story.append(ListFlowable([
        ListItem(Paragraph("<b>EduGenAI's assistant</b> reads the paper and extracts the "
                           "reference list plus the sentences that cite each source.", BODY)),
        ListItem(Paragraph("<b>This extension endpoint</b> looks each reference up in "
                           "OpenAlex and checks title, authors, year, journal, DOI, volume, "
                           "issue and pages — deterministically, with no LLM — and "
                           "returns the abstract.", BODY)),
        ListItem(Paragraph("<b>EduGenAI's assistant</b> compares the citing sentence to that "
                           "abstract and flags misquotes.", BODY)),
    ], bulletType="bullet", start="disc"))
    story.append(Paragraph(
        "Because the reasoning stays on EduGenAI's side, the extension needs no LLM API "
        "key. It only wraps OpenAlex, which is free. An optional OpenAlex Premium key can "
        "be stored securely in the Headers / Azure Key Vault section (Step 3).", NOTE))

    story.append(Paragraph("Before you start", H2))
    story.append(Paragraph(
        "Deploy this app at a public HTTPS URL — EduGenAI calls it server-to-server. "
        f"Everywhere below, replace <font face='Courier'>{HOST}</font> with your "
        "deployment's address.", BODY))

    story.append(Paragraph("Step 1 — Open the Extension builder", H2))
    story.append(Paragraph("In EduGenAI, create a new Extension (Name, Short description, "
                           "Headers, Functions).", BODY))

    story.append(Paragraph("Step 2 — Name and describe it", H2))
    story.append(Paragraph("Name", LABEL))
    story.append(code("OpenAlex Citation Verifier"))
    story.append(Paragraph("Short description", LABEL))
    story.append(code("Verifies a paper's references against OpenAlex: checks that each\n"
                       "work exists and that the printed title, authors, year, journal,\n"
                       "DOI and pages match, and returns the abstract."))
    story.append(Paragraph("Detail description (paste verbatim — this is the assistant's "
                           "instruction)", LABEL))
    story.append(code(DETAIL))

    story.append(PageBreak())
    story.append(Paragraph("Step 3 — Add the Header", H2))
    story.append(Paragraph("In the Headers section add: <b>Key</b> "
                           "<font face='Courier'>Content-Type</font> &nbsp; <b>Value</b> "
                           "<font face='Courier'>application/json</font>.", BODY))
    story.append(Paragraph("Optional — OpenAlex Premium: add a second header "
                           "<font face='Courier'>X-OpenAlex-Key</font> with your key as the "
                           "value, using the Secure header values option so it is stored in "
                           "Azure Key Vault.", BODY))

    story.append(Paragraph("Step 4 — Add the function", H2))
    story.append(Paragraph("Method &amp; URL", LABEL))
    story.append(code(f"POST  {HOST}/api/verify_batch"))
    story.append(Paragraph("Function definition", LABEL))
    story.append(code(SCHEMA))

    story.append(Paragraph("Step 5 — Submit and test", H2))
    story.append(Paragraph("Click Submit. In a chat, upload a paper and ask: “Check the "
                           "citations in this paper against OpenAlex.”", BODY))

    story.append(PageBreak())
    story.append(Paragraph("What the endpoint returns", H2))
    story.append(Paragraph("Request EduGenAI sends to /api/verify_batch:", LABEL))
    story.append(code(REQ))
    story.append(Paragraph("Response (abbreviated) — wrong authors, year and journal are "
                           "caught, and the abstract is returned for the misquote step:", LABEL))
    story.append(code(RESP))
    story.append(Paragraph("A single-reference endpoint is also available at "
                           "<font face='Courier'>POST /api/verify</font> (same fields, no outer "
                           "“references” array). Prefer the batch endpoint for a whole "
                           "bibliography — one round trip, kinder to OpenAlex rate limits.", NOTE))

    story.append(Paragraph("Notes &amp; limits", H2))
    story.append(ListFlowable([
        ListItem(Paragraph("No LLM key is stored or used by the extension; OpenAlex is free "
                           "and needs no key.", BODY)),
        ListItem(Paragraph("OpenAlex's canonical year can differ from a printed year "
                           "(online-first vs issue year) — treat a lone year mismatch as a "
                           "prompt to double-check, not a verdict.", BODY)),
        ListItem(Paragraph("The misquote judgement uses the abstract only, so a claim "
                           "supported by the full text but not the abstract may read as "
                           "uncertain. Treat results as leads for a human reviewer.", BODY)),
        ListItem(Paragraph("Batch requests are capped at 200 references.", BODY)),
    ], bulletType="bullet", start="disc"))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Use the Citation Checker in EduGenAI")
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()

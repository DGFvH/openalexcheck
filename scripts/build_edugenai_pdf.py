#!/usr/bin/env python3
"""Generate app/static/edugenai-instructions.pdf.

A printable copy of /edugenai — the page is the authoritative version.

Build-time only (reportlab is not a runtime dependency of the app). Re-run this
after editing the EduGenAI instructions to refresh the committed PDF:

    pip install reportlab
    python scripts/build_edugenai_pdf.py
"""

import textwrap
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

HOST = "https://www.phantocite.com"
MCP_URL = f"{HOST}/mcp"
MCP_CURL = ("curl -s -X POST " + MCP_URL + " \\\n"
            '  -H "Content-Type: application/json" \\\n'
            '  -d \'{"jsonrpc":"2.0","id":1,"method":"tools/list"}\'')

# Pasted verbatim into the agent's Instructions field. Kept identical to the
# block on /edugenai (id="c-detail") — that page is the authoritative copy.
INSTRUCTIONS = """You check the reference list of a paper the user uploads. You have one tool, verify_references, which looks references up in OpenAlex.

WHEN TO RUN. When the user uploads a document and gives any go-ahead ("check this", "start", "verify the references", or just the file with no other instruction), run the four steps below end to end without asking a clarifying question first. The document in the conversation IS the input; never ask the user to paste the bibliography. Only if no document has been provided at all, ask for one, then run.

STEP 1 - EXTRACT (do this yourself, before calling the tool). Read the document. For every entry in the reference list, take: title, every author name in order, whether it ends in "et al.", year, DOI (only if printed), journal or venue, volume, issue, and pages. Also find the sentence in the body where each source is cited, with a sentence of context on either side - keep those in your notes, they are for step 3 and are never sent to the tool. Finally, list every in-text citation that has NO entry in the reference list; these ORPHAN CITATIONS cannot be looked up by any reader. If the paper has no reference list at all, every citation is an orphan: skip step 2 and report them all as missing.

STEP 2 - VERIFY. Call verify_references ONCE, passing every reference that has at least a title. Never call it with an empty or placeholder list. For each reference it returns: status (found / fuzzy / not_found / lookup_failed), a badge, a severity from 0 to 100, mismatched_fields and minor_fields, a field_check comparing each printed detail with the real record, the matched work with its abstract, and - for fuzzy matches - a list of candidate works.

STEP 3 - JUDGE THE USE. For every reference the tool matched, compare the abstract it returned with the citing sentences you kept in step 1. Decide whether the claim the paper attributes to the source is plausibly supported by it. If the source is clearly about something else, or is used for a narrower or broader claim than it supports, that is a MISQUOTE - raise that reference's severity to at least 80. Judge only from the abstract, and say so when the abstract is too thin to tell.

STEP 4 - REPORT. Never show the user the raw JSON; it is a machine interface. Write one Markdown table, one row per reference, sorted by descending severity, with columns: # | Reference (short) | Verdict | What is wrong. Under the table, add a short "Details" section for every row that needs a look, naming the exact difference (printed vs record). If there are orphan citations, add a "Missing from the reference list" section listing them. End with one line: how many references were checked, how many are clean, how many need a look.

REPORT ONLY THIS. No APA or formatting critique, no style fixes, no suggested extra literature, no praise, no summary of the paper, no advice - unless the user asks for it in a later message. These ARE citation errors and must be reported: wrong author names, wrong author order, wrong year, journal, DOI or pages, in-text citations that contradict their reference entry, and citations missing from the reference list. Punctuation, capitalisation and italics are style - stay silent on those.

NEVER invent a DOI, an author, a year or a verdict the tool did not return. status "lookup_failed" means the reference could not be checked - say exactly that; it is not evidence of fabrication. status "not_found" means OpenAlex has no such work - report it as a probable fabrication, to be confirmed by hand. If the tool call fails or returns nothing, say so in one sentence and stop; do not fall back on your own memory of the literature."""

CURL = ("curl -s -X POST " + HOST + "/api/verify_batch \\\n"
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"references":[{"title":"Attention is all you need","year":2017}]}\'')


def code(text, width=96):
    """Preformatted does not wrap, so hard-wrap long prose to the page width."""
    lines = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line, width) or [""])
    return Preformatted("\n".join(lines), CODE)


def bullets(*items):
    return ListFlowable([ListItem(Paragraph(t, BODY)) for t in items],
                        bulletType="bullet", start="disc")


def build():
    story = []
    story.append(Paragraph("Use Phantocite in eduGenAI 2", H1))
    story.append(Paragraph(
        "Give an eduGenAI 2 agent an Action that verifies a paper's references against "
        "OpenAlex — inside your own chat, with no API key of your own.", SUB))
    story.append(Paragraph(
        "eduGenAI 1 was withdrawn after vulnerabilities were found in an audit, and its "
        "Extension builder went with it. eduGenAI 2 (edugenai2.npuls.nl, sign in with SRAM) "
        "takes external tools one way only: Extensions &rarr; URL of MCP server. Its Personas "
        "have no tools section at all. So this app runs an MCP server, and you register "
        "its URL.", NOTE))
    story.append(Paragraph(
        "<b>Read this first: the domain must be whitelisted.</b> The extension form says so "
        "outright — an MCP URL on a domain Npuls has not allowed cannot be registered at all. "
        "Nothing on this side works around it.", NOTE))

    story.append(Paragraph("How it works", H2))
    story.append(Paragraph(
        "Two separate objects on the platform. A Persona is a saved assistant with its own "
        "instructions and model. An Extension is a connection to an MCP server. You create "
        "both, then switch the extension on in a chat.", BODY))
    story.append(bullets(
        "<b>The persona</b> reads the paper, extracts the reference list and the sentences "
        "that cite each source, judges whether each citation matches the source, and "
        "writes the report.",
        "<b>This MCP server</b> looks each reference up in OpenAlex and checks "
        "title, authors, year, journal, DOI, volume, issue and pages — deterministically, "
        "with no LLM — and returns the abstract.",
    ))
    story.append(Paragraph(
        "Because the reasoning stays on eduGenAI's side, the extension needs no LLM API key. "
        "It only wraps OpenAlex, which is free.", NOTE))

    story.append(Paragraph("Before you start", H2))
    story.append(bullets(
        "An eduGenAI 2 account (edugenai2.npuls.nl, SRAM / SURFconext).",
        "The whitelist above. Without it the extension cannot be added at all.",
        "A model that supports tool calling — the GPT models do. The open willma-* models "
        "are the ones most likely to answer from memory and never call the tool.",
    ))

    story.append(Paragraph("Step 1 — Create the persona", H2))
    story.append(Paragraph("Create a new Persona, give it a name, pick a GPT model, and set "
                           "the conversation style to Precise.", BODY))
    story.append(Paragraph("Name", LABEL))
    story.append(code("Citation checker (OpenAlex)"))

    story.append(Paragraph("Step 2 — Paste the instructions", H2))
    story.append(Paragraph("Into the persona's Instructions field, verbatim. This is the "
                           "whole workflow: what to extract, when to call the tool, how to "
                           "report.", BODY))
    story.append(code(INSTRUCTIONS))

    story.append(PageBreak())
    story.append(Paragraph("Step 3 — Add the extension", H2))
    story.append(Paragraph("Go to Extensions &rarr; Add extension and fill it in: name "
                           "<b>Phantocite</b>, description <b>Checks a paper's references "
                           "against OpenAlex</b>, transport <b>Streamable HTTP</b>, "
                           "authentication <b>No authentication</b>, and tick the trust "
                           "checkbox. The URL of the MCP server is:", BODY))
    story.append(code(MCP_URL))
    story.append(Paragraph("Before registering it, you can confirm the server answers from "
                           "any terminal — it should list verify_references:", BODY))
    story.append(code(MCP_CURL))
    story.append(Paragraph(
        "The server is stateless and answers in plain JSON rather than opening a stream, "
        "which is what makes it reliable on serverless hosting. If your build offers SSE as "
        "the transport, choose Streamable HTTP anyway.", NOTE))
    story.append(Paragraph(
        "Optional — OpenAlex Premium: for higher OpenAlex rate limits, set the extension's "
        "authentication to an API key sent as the header X-OpenAlex-Key. It is used per "
        "request and never stored. The tool works fine without one.", BODY))

    story.append(Paragraph("Step 4 — Test it", H2))
    story.append(Paragraph("Personas and extensions are separate objects, so the extension is "
                           "switched on per chat: start a chat with the persona, enable "
                           "Phantocite from the tools button in the composer, attach a paper, "
                           "and ask:", BODY))
    story.append(code("Check the references in the attached paper."))
    story.append(Paragraph("The persona should extract the references itself, call "
                           "verify_references once, and answer with a table. The tool call is "
                           "shown in the message, so you can confirm it ran.", BODY))

    story.append(Paragraph("Troubleshooting", H2))
    story.append(bullets(
        "<b>The extension will not save, or the domain is refused.</b> The domain has to be "
        "whitelisted by Npuls first — see the top of this page. Nothing on this side works "
        "around it.",
        "<b>The extension saves but no tool appears in the chat.</b> Enable Phantocite from "
        "the tools button in the composer for that conversation. If the list is empty, check "
        "the transport is Streamable HTTP and confirm the server answers the tools/list "
        "command in Step 3.",
        "<b>The persona never calls the tool.</b> Either the model has no tool support "
        "(switch to a GPT model, not an open willma-* one) or the instructions were not "
        "saved. Naming the tool in the prompt forces it: \u201cUse the verify_references tool "
        "to check the references in the attached paper.\u201d",
        "<b>Raw JSON in the chat.</b> The instructions were truncated or not saved; re-paste "
        "Step 2.",
        "<b>count: 0.</b> Every response carries api_version. If it is missing or old, the "
        "extension points at a stale deployment.",
        "<b>lookup_failed.</b> A failed lookup, not a fabrication — usually a reference with "
        "no usable title.",
    ))
    story.append(Paragraph("To check the endpoint independently of eduGenAI, run this from "
                           "any terminal — a healthy deployment answers in a couple of "
                           "seconds with count: 1 and a Verified result:", BODY))
    story.append(code(CURL))

    story.append(Paragraph("Notes & limits", H2))
    story.append(bullets(
        "No LLM key is stored or used by the extension; OpenAlex is free and needs no key. "
        "The persona's own model runs on eduGenAI's side, under its terms.",
        "Only bibliographic metadata reaches this endpoint. The paper's full text and the "
        "citing sentences stay inside eduGenAI.",
        "OpenAlex's canonical year can differ from a printed year (online-first vs issue "
        "year) — treat a lone year mismatch as a prompt to double-check, not a verdict.",
        "The misquote judgement uses the abstract only, so a claim supported by the full "
        "text but not the abstract may read as uncertain. Treat results as leads for a "
        "human reviewer.",
        "Batch requests are capped at 200 references.",
    ))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Use Phantocite in eduGenAI 2")
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()

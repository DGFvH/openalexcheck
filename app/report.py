"""PDF export of a finished citation check.

The browser already builds one record per reference (plus one per orphan
citation) for the CSV/Markdown/HTML downloads; this renders those same records
as a readable report. Values come from the caller's document, so every string
is truncated and XML-escaped before it reaches the PDF engine.

Nothing is stored: the bytes are returned to the caller and dropped.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape

# Caps — a report is a page per few references, not a database dump.
MAX_ROWS = 500
MAX_FIELDS = 40
MAX_VALUE_CHARS = 4000
MAX_LABEL_CHARS = 60

# The site palette (app/static/ui.css), so a printed report matches the screen.
INK = "#26241f"
MUTED = "#6f6a5f"
ACCENT = "#1f6f5c"
BORDER = "#e2ddd2"
BAD = "#b3392f"
WARN = "#8a5a12"

# The built-in PDF fonts are Latin-1 only, so a name like "Kaiser, Ł." comes
# out as a black box. Bitstream Vera ships inside reportlab (so it is always
# present in the deployed function) and covers Latin Extended-A.
FONT = "Vera"
FONT_BOLD = "Vera-Bold"
_SUPPORTED: frozenset | None = None


def _register_fonts() -> None:
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if FONT in pdfmetrics.getRegisteredFontNames():
        return
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont(FONT, fonts / "Vera.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, fonts / "VeraBd.ttf"))


def _supported() -> frozenset:
    """Code points the report font can actually draw."""
    global _SUPPORTED
    if _SUPPORTED is None:
        from reportlab.pdfbase import pdfmetrics
        try:
            # Register first: the answer is cached for the process, so asking
            # before the font exists would freeze a wrong (Latin-1) answer.
            _register_fonts()
            _SUPPORTED = frozenset(pdfmetrics.getFont(FONT).face.charToGlyph)
        except Exception:                      # unknown build — assume Latin-1
            _SUPPORTED = frozenset(range(0x20, 0x100))
    return _SUPPORTED


def _portable(text: str) -> str:
    """Replace characters the font cannot draw (Greek, CJK, rare diacritics)
    with the closest plain-Latin form, so nothing prints as a black box."""
    ok = _supported()
    out = []
    for ch in text:
        if ch in "\n\t" or ord(ch) in ok:
            out.append(ch)
            continue
        folded = "".join(c for c in unicodedata.normalize("NFKD", ch)
                         if not unicodedata.combining(c) and ord(c) in ok)
        out.append(folded or "?")
    return "".join(out)


# Shown in the block heading rather than the field list.
HEADING_FIELDS = ("#", "Status", "Priority")
PRIORITY_COLOUR = {"Review": BAD, "Check": WARN}
# Columns holding "<what the paper printed> / <what OpenAlex has>".
PAIR_SUFFIX = " ref / OA"


def needs_checking(row: dict) -> bool:
    """A record belongs in the first section unless it came back clean."""
    priority = str(row.get("Priority") or "").strip()
    status = str(row.get("Status") or "").strip()
    return priority in PRIORITY_COLOUR or status != "Verified"


def _collapse(name: str, value: str) -> tuple[str, str]:
    """'Year ref / OA' + '1948 / 1948' → ('Year', '1948').

    The two sides agree for most fields of most references, and printing them
    twice is what makes the report long. Only a real difference is spelled out.

    Text added here skips _portable, so it must stay within the font's
    repertoire — an arrow glyph, for one, would print as an empty box.
    """
    if not name.endswith(PAIR_SUFFIX):
        return name, value
    short = name[: -len(PAIR_SUFFIX)].strip()
    # The value arrives already stripped, so an empty side leaves a bare slash
    # at one end. A DOI can contain "/" but never " / ", so the padded
    # separator is the only safe place to split.
    if value == "/":
        left = right = ""
    elif " / " in value:
        left, right = value.split(" / ", 1)
    elif value.endswith(" /"):
        left, right = value[:-2], ""
    elif value.startswith("/ "):
        left, right = "", value[2:]
    else:
        return short, value
    left, right = left.strip(), right.strip()
    if not left and not right:
        return short, ""
    if left == right:
        return short, left
    if not left:
        return short, f"{right} <font color=\"{MUTED}\">(OpenAlex only)</font>"
    if not right:
        return short, f"{left} <font color=\"{MUTED}\">(not in OpenAlex)</font>"
    return short, f"{left} <font color=\"{MUTED}\">(OpenAlex: {right})</font>"


def _text(value: Any, limit: int = MAX_VALUE_CHARS) -> str:
    """One untrusted value → a string safe to hand to reportlab's mini-HTML."""
    if value is None:
        return ""
    s = str(value)
    # Control characters would corrupt the PDF text stream; keep newlines/tabs.
    s = "".join(c for c in s if c >= " " or c in "\n\t")
    s = _portable(s.strip())
    if len(s) > limit:
        s = s[:limit].rstrip() + " …"
    return escape(s).replace("\n", "<br/>")


def build_pdf(rows: Sequence[dict], summary: Iterable[str] = (),
              title: str = "Citation check") -> bytes:
    """Render the report records as PDF bytes."""
    # Imported here so the cold start of every other route stays light.
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    _register_fonts()
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName=FONT_BOLD,
                        fontSize=18, textColor=HexColor(INK), alignment=TA_LEFT,
                        spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontName=FONT,
                         fontSize=9, textColor=HexColor(MUTED), leading=13, spaceAfter=2)
    section = ParagraphStyle("section", parent=styles["Normal"], fontName=FONT_BOLD,
                             fontSize=10.5, textColor=HexColor(INK), leading=14,
                             spaceBefore=16, spaceAfter=2)
    head = ParagraphStyle("head", parent=styles["Normal"], fontName=FONT_BOLD,
                          fontSize=9.5, textColor=HexColor(INK), leading=13,
                          spaceBefore=8, spaceAfter=2)
    label = ParagraphStyle("label", parent=styles["Normal"], fontName=FONT_BOLD,
                           fontSize=7, textColor=HexColor(MUTED), leading=10)
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=FONT,
                          fontSize=8, textColor=HexColor(INK), leading=11)

    rows = [r for r in list(rows)[:MAX_ROWS] if isinstance(r, dict)]
    checking = [r for r in rows if needs_checking(r)]
    clean = [r for r in rows if not needs_checking(r)]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story: list = [Paragraph(_text(title, 120), h1)]
    line = " · ".join(_text(s, 120) for s in list(summary)[:12] if str(s).strip())
    if line:
        story.append(Paragraph(line, sub))
    story.append(Paragraph(
        f"{len(checking)} to check · {len(clean)} verified · generated {stamp}", sub))

    table_style = TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, HexColor(BORDER)),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ])
    list_style = TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, HexColor(BORDER)),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])

    def section_head(text: str, count: int, note: str, colour: str) -> list:
        return [
            Paragraph(f'<font color="{colour}">{_text(text, 60)}</font> '
                      f'<font color="{MUTED}">({count})</font>', section),
            HRFlowable(width="100%", thickness=0.8, color=HexColor(colour),
                       spaceBefore=1, spaceAfter=3),
            Paragraph(_text(note, 300), sub),
        ]

    def detail(row: dict) -> list:
        """One full block: heading plus every field that says something."""
        items = list(row.items())[:MAX_FIELDS]
        get = {str(k): v for k, v in items}
        number = _text(get.get("#"), 12)
        status = _text(get.get("Status"), 80)
        priority = str(get.get("Priority") or "").strip()
        # One verdict in the heading, as on screen: the priority when something
        # needs a look, otherwise the status. The status is never lost — it
        # drops into the field list whenever the priority takes its place.
        heading = f"#{number}" if number else "Orphan citation"
        if priority in PRIORITY_COLOUR:
            heading += f' · <font color="{PRIORITY_COLOUR[priority]}">{_text(priority, 20)}</font>'
        elif status:
            heading += f" · {status}"
        # "Status: Verified" under a "Review" heading reads as a contradiction —
        # and the matched title below already shows the work was found. Only a
        # status that says something ("Potential hallucination") earns a row.
        skip = ["#", "Priority"]
        if priority not in PRIORITY_COLOUR or status == "Verified":
            skip.append("Status")
        block: list = [Paragraph(heading, head)]

        cells = []
        for key, value in items:
            name = str(key)
            if name in skip:
                continue
            text = _text(value)
            if not text or text.strip("/-— ") == "":
                continue                      # e.g. "year ref / OA" with both sides empty
            name, text = _collapse(name, text)
            if not text:
                continue
            cells.append([Paragraph(_text(name, MAX_LABEL_CHARS), label),
                          Paragraph(text, body)])
        if cells:
            table = Table(cells, colWidths=[3.4 * cm, None], hAlign="LEFT")
            table.setStyle(table_style)
            block.append(table)
        # Keep a short block whole; a long one may split across pages.
        return [KeepTogether(block)] if len(cells) <= 8 else block

    def compact(rows_: list) -> list:
        """The clean references as a numbered list — one line each."""
        cells = []
        for row in rows_:
            number = _text(row.get("#"), 12)
            ref = _text(row.get("Reference (as printed)") or row.get("Matched title"), 400)
            cells.append([Paragraph(f"#{number}" if number else "—", label),
                          Paragraph(ref or "(unlabelled reference)", body)])
        if not cells:
            return []
        table = Table(cells, colWidths=[1.3 * cm, None], hAlign="LEFT")
        table.setStyle(list_style)
        return [table]

    if checking:
        story += section_head(
            "To check", len(checking),
            "Missing from OpenAlex, or the printed details or the use of the source "
            "differ from the record. Each one is set out in full below.", BAD)
        for row in checking:
            story += detail(row)
    if clean:
        story += section_head(
            "Verified", len(clean),
            "Found in OpenAlex with matching details, and cited consistently with the "
            "abstract. Listed for completeness.", ACCENT)
        story += compact(clean)
    if not rows:
        story.append(Paragraph("No references to report.", body))

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(HexColor(MUTED))
        canvas.drawString(1.6 * cm, 1.1 * cm, "Phantocite · citation check")
        canvas.drawRightString(A4[0] - 1.6 * cm, 1.1 * cm, str(canvas.getPageNumber()))
        canvas.setStrokeColor(HexColor(BORDER))
        canvas.line(1.6 * cm, 1.5 * cm, A4[0] - 1.6 * cm, 1.5 * cm)
        canvas.restoreState()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, title=title, author="Phantocite",
        leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=2 * cm,
    )
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    return buffer.getvalue()


def filename(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"citation-check-{now.strftime('%Y-%m-%d-%H-%M-%S')}.pdf"

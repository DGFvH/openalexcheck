"""PDF export of a finished citation check.

The browser already builds one record per reference (plus one per orphan
citation) for the CSV/Markdown/HTML downloads; this renders those same records
as a readable report. Values come from the caller's document, so every string
is truncated and XML-escaped before it reaches the PDF engine.

Nothing is stored: the bytes are returned to the caller and dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
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

# Shown in the block heading rather than the field list.
HEADING_FIELDS = ("#", "Status", "Priority")
PRIORITY_COLOUR = {"Review": BAD, "Check": WARN}


def _text(value: Any, limit: int = MAX_VALUE_CHARS) -> str:
    """One untrusted value → a string safe to hand to reportlab's mini-HTML."""
    if value is None:
        return ""
    s = str(value)
    # Control characters would corrupt the PDF text stream; keep newlines/tabs.
    s = "".join(c for c in s if c >= " " or c in "\n\t")
    s = s.strip()
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
        KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName="Helvetica-Bold",
                        fontSize=18, textColor=HexColor(INK), alignment=TA_LEFT,
                        spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontName="Helvetica",
                         fontSize=9, textColor=HexColor(MUTED), leading=13, spaceAfter=2)
    head = ParagraphStyle("head", parent=styles["Normal"], fontName="Helvetica-Bold",
                          fontSize=11, textColor=HexColor(INK), leading=15,
                          spaceBefore=10, spaceAfter=4)
    label = ParagraphStyle("label", parent=styles["Normal"], fontName="Helvetica-Bold",
                           fontSize=7.5, textColor=HexColor(MUTED), leading=11)
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=8.5, textColor=HexColor(INK), leading=12)

    rows = list(rows)[:MAX_ROWS]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story: list = [Paragraph(escape(title), h1)]
    line = " · ".join(_text(s, 120) for s in list(summary)[:12] if str(s).strip())
    if line:
        story.append(Paragraph(line, sub))
    story.append(Paragraph(f"{len(rows)} entries · generated {stamp}", sub))
    story.append(Spacer(1, 6))

    table_style = TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, HexColor(BORDER)),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])

    for row in rows:
        if not isinstance(row, dict):
            continue
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
            heading += f' · <font color="{PRIORITY_COLOUR[priority]}">{escape(priority)}</font>'
        elif status:
            heading += f" · {status}"
        skip = HEADING_FIELDS if priority not in PRIORITY_COLOUR else ("#", "Priority")
        block: list = [Paragraph(heading, head)]

        cells = []
        for key, value in items:
            name = str(key)
            if name in skip:
                continue
            text = _text(value)
            if not text or text in ("/", "-"):   # "year ref / OA" with both sides empty
                continue
            cells.append([Paragraph(_text(name, MAX_LABEL_CHARS), label),
                          Paragraph(text, body)])
        if cells:
            table = Table(cells, colWidths=[3.9 * cm, None], hAlign="LEFT")
            table.setStyle(table_style)
            block.append(table)
        # Keep a short block whole; a long one may split across pages.
        if len(cells) <= 8:
            story.append(KeepTogether(block))
        else:
            story.extend(block)

    if len(story) <= 3:
        story.append(Paragraph("No references to report.", body))

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
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

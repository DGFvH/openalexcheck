"""Design-system guard: every font, size, colour and radius comes from
app/static/ui.css; page stylesheets may only add layout. This is what keeps the
UI at one font family, five sizes, one palette and two radii."""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
HEX_OR_RGB = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\(")


def _style_blocks(html: str) -> str:
    # The page's own stylesheet is the first <style>; later ones are templates
    # for exported standalone files (which cannot use CSS variables).
    m = re.search(r"<style>(.*?)</style>", html, re.S)
    return m.group(1) if m else ""


def _page_styles():
    yield "index.html", _style_blocks((STATIC / "index.html").read_text())
    yield "edugenai.html", _style_blocks((STATIC / "edugenai.html").read_text())
    from app import auth
    yield "login", _style_blocks(auth.login_page("/").body.decode())


def test_ui_css_defines_the_whole_vocabulary():
    css = (STATIC / "ui.css").read_text()
    root = re.search(r":root\s*{(.*?)}", css, re.S).group(1)
    body = css.replace(root, "")
    assert re.findall(r"--fs-\w+:", root) and len(re.findall(r"--fs-\w+:", root)) == 5
    assert set(re.findall(r"font-family:\s*([^;]+);", body)) <= {"var(--font)", "var(--mono)"}
    sizes = set(re.findall(r"font-size:\s*([^;]+);", body))
    assert sizes <= {"var(--fs-xs)", "var(--fs-sm)", "var(--fs-md)", "var(--fs-lg)", "var(--fs-xl)", ".9em"}, sizes
    radii = set(re.findall(r"border-radius:\s*([^;]+);", body))
    assert radii <= {"var(--r)", "var(--r-pill)"}, radii
    # Colours outside :root: only tokens and white text on solid tones.
    raw = [m for m in HEX_OR_RGB.findall(body) if m.lower() != "#fff"]
    assert not raw, raw
    weights = set(re.findall(r"font-weight:\s*(\d+)", body))
    assert weights <= {"400", "600"}, weights


def test_page_styles_are_layout_only():
    for name, css in _page_styles():
        assert "font-family" not in css, f"{name}: font-family must come from ui.css"
        for size in re.findall(r"font-size:\s*([^;]+);", css):
            assert size.startswith("var(--fs-"), f"{name}: font-size {size!r} not on the scale"
        for radius in re.findall(r"border-radius:\s*([^;]+);", css):
            assert radius in ("var(--r)", "var(--r-pill)"), f"{name}: radius {radius!r}"
        raw = [m for m in HEX_OR_RGB.findall(css) if m.lower() != "#fff"]
        assert not raw, f"{name}: raw colours {raw}"
        weights = set(re.findall(r"font-weight:\s*(\d+)", css))
        assert weights <= {"400", "600"}, f"{name}: weights {weights}"
        assert "Georgia" not in css


def test_pages_link_the_shared_stylesheet():
    for name in ("index.html", "edugenai.html"):
        assert '<link rel="stylesheet" href="/static/ui.css">' in (STATIC / name).read_text()
    from app import auth
    assert "/static/ui.css" in auth.login_page("/").body.decode()


# Results screen: colour means verdict and lives only in chips and the field
# table's left rule — never on prose; the only glyph is the ▸ disclosure marker.
def test_results_prose_carries_no_verdict_colour_or_glyphs():
    html = (STATIC / "index.html").read_text()
    css = _style_blocks(html)
    for rule in re.findall(r"([^{}]+){([^{}]*)}", css):
        selector, body = rule[0].strip(), rule[1]
        if re.search(r"color:\s*var\(--(ok|warn|bad)\)", body):
            assert selector.startswith((".chip", ".btn", ".table", ".error", ".field-note-never")), \
                f"verdict colour on prose: {selector}"
    js = html[html.rindex("<script"):]
    for glyph in "⚠✓⏳🧪⬇":
        assert glyph not in js, f"glyph {glyph} in a results template"
    assert "chip soft" in js and "overallChip" in js

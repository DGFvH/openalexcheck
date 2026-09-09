#!/usr/bin/env python3
"""Generate app/static/og-image.png — the 1200x630 card shown when a link to
the site is posted on LinkedIn, Slack, WhatsApp, X and the rest.

Build-time only (Playwright is not a runtime dependency). Re-run this after
changing the wordmark or the descriptor:

    pip install playwright && playwright install chromium
    python scripts/build_og_image.py
"""

from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
OUT = STATIC / "og-image.png"
CHROMIUM = "/opt/pw-browsers/chromium"       # preinstalled in the dev container

# The site palette (app/static/ui.css).
BG, CARD, INK, MUTED = "#f6f4ef", "#ffffff", "#26241f", "#6f6a5f"
ACCENT, OK_BG, BAD, BAD_BG, BORDER = "#1f6f5c", "#dfeee8", "#b3392f", "#f6dcd8", "#e2ddd2"

# The favicon ghost, with the media query dropped so it renders light-mode.
MARK = """
<svg viewBox="0 0 100 100" width="300" height="300" aria-hidden="true">
  <style>
    .ln { fill: none; stroke: #26241f; stroke-width: 5; stroke-linejoin: round; stroke-linecap: round; }
    .body { fill: #ffffff; stroke: #26241f; stroke-width: 5; stroke-linejoin: round; }
    .ink { fill: #26241f; }
  </style>
  <path class="body" d="M22,74 L22,54 C22,37 34,27 50,27 C66,27 78,37 78,54 L78,74
    Q70,86 62,74 Q54,86 46,74 Q38,86 30,74 Q26,80 22,74 Z"/>
  <ellipse class="ink" cx="42" cy="50" rx="4" ry="6"/>
  <ellipse class="ink" cx="58" cy="50" rx="4" ry="6"/>
  <ellipse class="ink" cx="50" cy="63" rx="3.5" ry="5"/>
  <path class="body" d="M16,60 L34,54 L38,70 L20,76 Z"/>
  <path class="ln" d="M25,57 L29,71"/>
  <path class="body" d="M37,25 L63,25 L64,32 Q50,37 36,32 Z"/>
  <path class="body" d="M50,8 L77,19 L50,30 L23,19 Z"/>
  <path class="ln" d="M77,19 C82,25 81,31 79,36"/>
  <rect class="ink" x="75.5" y="35" width="7" height="10" rx="2.5"/>
</svg>
"""

HTML = f"""<!doctype html><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; width: 1200px; height: 630px; background: {BG};
         font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: {INK};
         display: flex; align-items: center; gap: 64px; padding: 0 84px; }}
  .text {{ flex: 1; }}
  h1 {{ margin: 0 0 6px; font-size: 104px; font-weight: 700; letter-spacing: -.03em;
        color: {ACCENT}; line-height: 1; text-shadow: 0 4px 0 {OK_BG}; }}
  .bar {{ width: 132px; height: 10px; border-radius: 999px; background: {ACCENT}; margin: 18px 0 30px; }}
  h2 {{ margin: 0 0 18px; font-size: 42px; font-weight: 600; line-height: 1.2; }}
  p {{ margin: 0 0 34px; font-size: 27px; line-height: 1.45; color: {MUTED}; max-width: 34ch; }}
  .chips {{ display: flex; gap: 12px; }}
  .chip {{ font-size: 22px; font-weight: 600; padding: 10px 20px; border-radius: 999px; }}
  .chip.ok {{ color: {ACCENT}; background: {OK_BG}; }}
  .chip.bad {{ color: {BAD}; background: {BAD_BG}; }}
  .mark {{ width: 340px; height: 340px; display: flex; align-items: center; justify-content: center;
           background: {CARD}; border: 2px solid {BORDER}; border-radius: 40px; flex: none; }}
</style>
<div class="text">
  <h1>Phantocite</h1>
  <div class="bar"></div>
  <h2>Citation checker for student papers</h2>
  <p>Does every reference actually exist &mdash; and does each citation match what the source says?</p>
  <div class="chips">
    <span class="chip ok">Checked against OpenAlex</span>
    <span class="chip bad">Finds misquotes</span>
  </div>
</div>
<div class="mark">{MARK}</div>
"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page(viewport={"width": 1200, "height": 630},
                                device_scale_factor=1)
        page.set_content(HTML, wait_until="load")
        page.screenshot(path=str(OUT))
        browser.close()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

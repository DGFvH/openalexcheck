"""Browser tests for the run lifecycle: an in-process uvicorn serves the real
page while the analysis pipeline is replaced with fakes, so the frontend is
exercised end to end without a network or an API key."""

import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

from app import main  # noqa: E402

CHROMIUM = "/opt/pw-browsers/chromium"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    import uvicorn
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(5)


@pytest.fixture(scope="module")
def browser():
    import os
    if not os.path.exists(CHROMIUM):
        pytest.skip("chromium not installed")
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROMIUM)
        yield b
        b.close()


def _ref(i, **over):
    r = {"id": i, "raw": f"Ref {i}", "title": f"Title {i}", "authors": ["A, B."], "year": 2020,
         "doi": None, "container": None, "volume": None, "issue": None, "pages": None,
         "et_al": False, "contexts": ["ctx"], "first_author_surname": "A"}
    r.update(over)
    return r


def _found(ref, url="https://openalex.org/W1"):
    return {"status": "found", "work": {"title": ref["title"], "abstract": "abs", "url": url,
                                        "authors": ["B A"], "year": 2020, "venue": "J"},
            "candidates": [], "notes": [], "field_check": [], "field_mismatch_count": 0}


def _install(monkeypatch, refs, orphans=(), resolve=None, compare=None):
    monkeypatch.setattr(main, "extract_text", lambda name, data: "body " * 100)
    monkeypatch.setattr(main, "extract_references", lambda llm, text, max_tokens=0: (refs, list(orphans)))
    monkeypatch.setattr(main, "resolve_reference", resolve or (lambda ref, api_key=None, **kw: _found(ref)))
    if compare is None:
        def compare(llm, items, max_tokens=0, **kw):
            for it in items:
                yield {"id": it["id"], "verdict": "match", "explanation": "ok"}
    monkeypatch.setattr(main, "compare_contexts", compare)


def _login(page, base_url, password="test-pw"):
    page.goto(base_url + "/login", wait_until="domcontentloaded")
    page.fill("input[name=password]", password)
    page.click("button[type=submit]")
    page.wait_for_url(base_url + "/")


def _run(page, base_url):
    page.goto(base_url + "/", wait_until="networkidle")
    if page.url.endswith("/login?next=/"):
        _login(page, base_url)
    page.set_input_files("#file", {"name": "paper.pdf", "mimeType": "application/pdf", "buffer": b"%PDF-1.4 x"})
    page.click("#run")
    page.wait_for_function("() => ['Done.','Stopped — partial results.','Failed.'].includes(document.getElementById('progress-status').textContent) || !document.getElementById('run').classList.contains('hidden')", timeout=15000)


def test_xss_payload_is_inert(browser, base_url, monkeypatch):
    payload = '<img src=x onerror="window.__pwned=1">'
    _install(monkeypatch, [_ref(1, raw=payload, title=payload)],
             resolve=lambda ref, api_key=None, **kw: _found(ref, url="javascript:window.__pwned=2"))
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    _run(page, base_url)
    page.wait_for_timeout(300)
    assert page.evaluate("window.__pwned") is None
    assert page.evaluate("document.querySelectorAll('#screen-results img').length") == 0
    hrefs = page.evaluate("[...document.querySelectorAll('#screen-results a')].map(a => a.getAttribute('href'))")
    assert all(not h.lower().startswith("javascript:") for h in hrefs)
    assert payload in page.inner_text("#screen-results")   # rendered as text
    assert errors == []
    page.close()


def test_error_midway_keeps_partial_results_and_exports(browser, base_url, monkeypatch):
    from app.openalex import OpenAlexAuthError

    def resolve(ref, api_key=None, **kw):
        if ref["id"] == 2:
            raise OpenAlexAuthError("OpenAlex rejected the API key.")
        return _found(ref)

    _install(monkeypatch, [_ref(1), _ref(2), _ref(3)], resolve=resolve)
    page = browser.new_page()
    _run(page, base_url)
    assert page.is_visible(".partial-banner")
    assert page.evaluate("getComputedStyle(document.getElementById('downloads')).display") == "flex"
    assert "Checking…" not in page.inner_text("#screen-results")
    assert "rejected the API key" in page.inner_text("#error")
    page.close()


def test_eof_without_done_is_reported_as_partial(browser, base_url, monkeypatch):
    def cut_pipeline(*, emit, control, **kw):
        emit({"type": "progress", "stage": "extract_done", "reference_count": 3, "message": "Found 3"})
        emit({"type": "result", "result": {"reference": _ref(1), **_found(_ref(1)), "misquote": None}})
        emit(None)   # stream ends: no done, no error

    monkeypatch.setattr(main, "_pipeline", cut_pipeline)
    page = browser.new_page()
    _run(page, base_url)
    assert "connection ended" in page.inner_text("#error")
    assert "2 of 3 references were not verified" in page.inner_text(".partial-banner")
    page.close()


def test_second_run_clears_previous_paper(browser, base_url, monkeypatch):
    _install(monkeypatch, [_ref(1)], orphans=[{"label": "Doe (2021)", "year": 2021, "context": "As Doe (2021) says"}])
    page = browser.new_page()
    _run(page, base_url)
    assert "Doe (2021)" in page.inner_text("#orphans")
    _install(monkeypatch, [_ref(1), _ref(2)])
    _run(page, base_url)
    assert page.inner_text("#orphans").strip() == ""
    assert "2 references" in page.inner_text("#summary")
    assert "Doe (2021)" not in page.inner_text("#results-area")
    page.close()


def test_csv_export_guards_formula_cells(browser, base_url, monkeypatch):
    _install(monkeypatch, [_ref(1, raw='=HYPERLINK("https://evil.example","x")')])
    page = browser.new_page()
    _run(page, base_url)
    with page.expect_download() as dl:
        page.click("#downloads button[data-fmt=csv]")
    text = open(dl.value.path(), encoding="utf-8-sig").read()
    assert "\"'=HYPERLINK" in text and '"=HYPERLINK' not in text
    page.close()


def test_password_gate_in_browser(browser, base_url):
    page = browser.new_page()
    page.goto(base_url + "/", wait_until="domcontentloaded")
    assert page.url.endswith("/login?next=/")
    page.fill("input[name=password]", "wrong")
    page.click("button[type=submit]")
    page.wait_for_selector(".err")
    _login(page, base_url)
    assert page.locator("#llm-note").count() == 0 and page.locator("#model").count() == 0
    assert page.locator("#api_key").count() == 0 and page.locator("#provider").count() == 0
    assert "Custom" not in page.locator("#model_select").inner_text()
    assert page.locator("#max_tokens").count() == 0 and page.locator("#openalex_key").count() == 0
    assert "Advanced options" not in page.inner_text("#form-card")
    assert "pages" in page.inner_text("#capacity-note") and "references" in page.inner_text("#capacity-note")
    assert page.is_visible("text=Log out")
    page.click("#sample")
    page.wait_for_timeout(500)
    assert page.evaluate("document.querySelectorAll('#screen-results .ref').length") == 5
    page.close()


def test_cookie_bar_acceptance_is_remembered(browser, base_url):
    page = browser.new_page()
    _login(page, base_url)
    assert page.is_visible("#cookie-bar")
    page.reload(wait_until="domcontentloaded")
    assert page.is_visible("#cookie-bar")          # stays until accepted
    page.click("#cookie-accept")
    assert page.is_hidden("#cookie-bar")
    assert page.evaluate("localStorage.getItem('phantocite_cookies')") == "all"
    page.reload(wait_until="domcontentloaded")
    assert page.is_hidden("#cookie-bar")
    page.close()

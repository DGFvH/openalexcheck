import json

import pytest

from app.llm import LLMError, _parse_json
from app.openalex import (
    clean_doi,
    normalize_title,
    reconstruct_abstract,
    score_candidate,
    summarize_work,
    title_similarity,
)


def test_reconstruct_abstract():
    inv = {"Labour": [0], "productivity": [1, 4], "drives": [2], "aggregate": [3]}
    assert reconstruct_abstract(inv) == "Labour productivity drives aggregate productivity"


def test_reconstruct_abstract_empty():
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_clean_doi():
    assert clean_doi("https://doi.org/10.1000/XYZ") == "10.1000/xyz"
    assert clean_doi("doi: 10.1000/xyz") == "10.1000/xyz"
    assert clean_doi("10.1000/xyz") == "10.1000/xyz"


def test_normalize_and_similarity():
    a = "Macroeconomic Productivity: A Review!"
    b = "macroeconomic productivity — a review"
    assert normalize_title(a) == "macroeconomic productivity a review"
    assert title_similarity(a, b) > 0.95
    assert title_similarity("completely different thing", a) < 0.5


def test_resolve_without_title_or_doi_is_unverifiable_not_hallucinated():
    """Author + year alone cannot identify a work. EduGenAI sends such entries
    when a paper has in-text citations but no bibliography — they must come back
    lookup_failed ('not checked'), never not_found ('potential hallucination')."""
    from app.openalex import resolve_reference
    res = resolve_reference({"id": 1, "title": None, "doi": None,
                             "authors": ["Dosumu"], "first_author_surname": "Dosumu",
                             "year": 2023, "container": None, "volume": None,
                             "issue": None, "pages": None, "et_al": True, "contexts": []})
    assert res["status"] == "lookup_failed"
    assert "no title or DOI" in res["notes"][0]


def test_extract_references_returns_orphan_citations(monkeypatch):
    """Extraction returns (references, orphans); orphans are in-text citations
    with no bibliography entry, tolerated in messy shapes."""
    from app import analysis

    class FakeLLM:
        def complete_json(self, system, user, max_tokens=0, thinking=True):
            return {"references": [
                        {"id": 1, "raw": "Smith, J. (2020). A real title. Journal.",
                         "title": "A real title", "first_author_surname": "Smith",
                         "authors": ["Smith, J."], "year": 2020, "contexts": []}],
                    "orphan_citations": [
                        {"label": "Jones et al. (2019)", "year": 2019,
                         "context": "As Jones et al. (2019) argue…"},
                        {"label": "", "year": None, "context": "no label -> dropped"},
                        "not a dict -> dropped"]}

    refs, orphans = analysis.extract_references(FakeLLM(), "text")
    assert len(refs) == 1 and refs[0]["title"] == "A real title"
    assert orphans == [{"label": "Jones et al. (2019)", "year": 2019,
                        "context": "As Jones et al. (2019) argue…"}]


def test_search_query_keeps_apostrophes():
    """OpenAlex title.search indexes "don't" as one token — stripping the
    apostrophe ("don t"/"dont") matches nothing, so a real work would be
    falsely flagged as a potential hallucination. Only actual filter-syntax
    characters (comma, colon, pipe, ampersand) may be removed."""
    from app.openalex import _search_query
    t = "If you don't want to be late, enumerate: Unpacking reduces the planning fallacy"
    q = _search_query(t)
    assert "don't" in q                      # apostrophe survives
    assert "," not in q and ":" not in q     # filter syntax stripped
    assert _search_query("A & B | C") == "A B C"
    assert _search_query("Memory bias?") == "Memory bias"   # '?' causes an API 400
    assert _search_query("Self-attention models") == "Self-attention models"
    assert _search_query("") == ""


def test_score_candidate_rewards_year_and_author():
    ref = {"title": "Macroeconomic productivity trends", "year": 2019,
           "first_author_surname": "Smith"}
    work = {"title": "Macroeconomic productivity trends", "year": 2019,
            "authors": ["Jane Smith", "Bob Jones"]}
    good = score_candidate(ref, work)
    bad = score_candidate(ref, {**work, "year": 2005, "authors": ["Someone Else"]})
    assert good > 1.0
    assert good > bad


def test_summarize_work():
    work = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/a",
        "title": "T",
        "publication_year": 2020,
        "authorships": [{"author": {"display_name": "A B"}}],
        "primary_location": {"source": {"display_name": "Journal"}},
        "cited_by_count": 3,
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
    }
    s = summarize_work(work)
    assert s["doi"] == "10.1/a"
    assert s["abstract"] == "Hello world"
    assert s["authors"] == ["A B"]
    assert s["venue"] == "Journal"


def test_parse_json_plain_and_fenced_and_prose():
    assert _parse_json('{"a": 1}') == {"a": 1}
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('Here you go:\n{"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(LLMError):
        _parse_json("no json here")


def test_redact_removes_all_keys():
    from app.keysafety import REDACTED, redact

    msg = "error for url 'https://api.openalex.org/works?api_key=SECRET1&x=1' with sk-ant-SECRET2"
    out = redact(msg, "SECRET1", "sk-ant-SECRET2", None, "  ")
    assert "SECRET1" not in out
    assert "SECRET2" not in out
    assert out.count(REDACTED) == 2


def test_openalex_client_includes_key_as_param():
    from app.openalex import _client

    with _client("premium-key") as c:
        assert c.params.get("api_key") == "premium-key"
    with _client(None) as c:
        assert "api_key" not in c.params


def test_lookup_failure_is_not_a_hallucination(monkeypatch):
    """A transient OpenAlex failure must surface as 'lookup_failed', never as a
    'not_found' hallucination accusation."""
    import httpx
    from app import openalex

    def boom(*a, **k):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(openalex, "_get_work_by_doi", boom)
    monkeypatch.setattr(openalex, "_search_works_by_title", boom)
    res = openalex.resolve_reference(
        {"title": "Some real paper", "year": 2020, "doi": "10.1/x"})
    assert res["status"] == "lookup_failed"
    assert res["status"] != "not_found"


def test_get_retries_then_raises(monkeypatch):
    import httpx
    from app import openalex

    calls = {"n": 0}

    class C:
        def get(self, path, params=None):
            calls["n"] += 1
            raise httpx.ConnectError("down")

    monkeypatch.setattr(openalex.time, "sleep", lambda *_: None)
    with pytest.raises(openalex.OpenAlexLookupError):
        openalex._get(C(), "/works", attempts=3)
    assert calls["n"] == 3


def test_compare_authors_detects_wrong_author():
    from app.fieldcheck import compare_authors
    ref = {"authors": ["Smith, J.", "Jones, B."], "et_al": False,
           "first_author_surname": "Smith"}
    r = compare_authors(ref, ["John Jumper", "Richard Evans", "Alex Pritzel"])
    assert r["status"] == "mismatch"
    assert "Smith" in r["detail"]


def test_compare_authors_flags_wrong_order():
    """Swapped author order is a real citation error (it reassigns first
    authorship) — 'Kahneman & Tversky (1974)' must not pass as a clean match
    for Tversky & Kahneman."""
    from app.fieldcheck import compare_authors
    res = compare_authors({"authors": ["Kahneman, D.", "Tversky, A."], "et_al": False},
                          ["Amos Tversky", "Daniel Kahneman"])
    assert res["status"] == "mismatch"
    assert "order" in res["detail"].lower()
    # Correct order still passes.
    ok = compare_authors({"authors": ["Tversky, A.", "Kahneman, D."], "et_al": False},
                         ["Amos Tversky", "Daniel Kahneman"])
    assert ok["status"] == "match"


def test_compare_authors_matches_surnames_across_formats():
    from app.fieldcheck import compare_authors
    ref = {"authors": ["Jumper, J.", "Evans, R."], "et_al": True}
    r = compare_authors(ref, ["John Jumper", "Richard Evans", "Alex Pritzel"])
    assert r["status"] == "match"


def test_compare_fields_flags_year_and_pages():
    from app.fieldcheck import compare_fields, field_mismatches, field_severity
    ref = {"title": "A study of things", "authors": ["Doe, J."], "et_al": True,
           "year": 2019, "doi": None, "container": "Journal of Things",
           "volume": "5", "issue": None, "pages": "10-20"}
    work = {"title": "A study of things", "authors_full": ["Jane Doe"],
            "year": 2021, "doi": None, "venue": "Journal of Things",
            "volume": "5", "issue": None, "pages": "10-20"}
    fields = compare_fields(ref, work)
    mism = {f["field"] for f in field_mismatches(fields)}
    assert "year" in mism
    assert "pages" not in mism   # pages actually match
    assert field_severity(fields) >= 6  # year weight


def test_pages_equal_shorthand():
    from app.fieldcheck import _pages_equal
    assert _pages_equal("123-145", "123-145")
    assert not _pages_equal("100-110", "583-589")
    assert _pages_equal("e0234", "e0234")


def test_surname_extraction():
    from app.fieldcheck import surname
    assert surname("Smith, John A.") == "smith"
    assert surname("John A. Smith") == "smith"
    assert surname("") == ""


def test_verify_endpoints(monkeypatch):
    """The keyless extension endpoints wrap resolve_reference without an LLM."""
    from fastapi.testclient import TestClient
    from app import main

    def fake_resolve(ref, api_key=None, **kw):
        if "cheese" in (ref.get("title") or "").lower():
            return {"status": "not_found", "work": None, "candidates": [], "notes": []}
        return {"status": "found",
                "work": {"title": ref["title"], "authors": ["Real Author"], "year": 2021,
                         "venue": "Nature", "doi": None, "abstract": "x" * 5000, "url": "u"},
                "candidates": [], "notes": ["ok"],
                "field_check": [{"field": "year", "status": "mismatch",
                                 "reference_value": ref.get("year"), "openalex_value": 2021}],
                "field_mismatch_count": 1}

    monkeypatch.setattr(main, "resolve_reference", fake_resolve)
    client = TestClient(main.app)

    r = client.post("/api/verify", json={"title": "A real paper", "year": 2019})
    d = r.json()
    assert r.status_code == 200
    assert d["status"] == "found"
    assert d["field_mismatch_count"] == 1
    assert len(d["work"]["abstract"]) <= main.ABSTRACT_CAP + 1  # trimmed

    rb = client.post("/api/verify_batch", json={"references": [
        {"title": "A real paper", "year": 2019},
        {"title": "Quantum cheese networks", "year": 2021},
    ]})
    b = rb.json()
    assert b["count"] == 2
    assert b["results"][0]["status"] == "found"
    assert b["results"][1]["status"] == "not_found"


def test_verify_batch_cap():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    r = client.post("/api/verify_batch", json={"references": [{"title": "x"}] * 201})
    assert r.status_code == 400


def test_verify_accepts_messy_llm_input(monkeypatch):
    """Real LLM-extracted references are messy: a non-numeric year, a numeric
    volume/issue/pages, authors as one string, a dict-shaped author, journal
    under either 'journal' or 'container', and even a non-object entry. None of
    these may 422 or sink the batch — they must all resolve to a 200 result."""
    from fastapi.testclient import TestClient
    from app import main

    captured = {}

    def fake_resolve(ref, api_key=None, **kw):
        captured["ref"] = ref
        return {"status": "found",
                "work": {"title": ref.get("title"), "authors": [], "year": None,
                         "venue": None, "doi": None, "abstract": None, "url": "u"},
                "candidates": [], "notes": [], "field_check": [], "field_mismatch_count": 0}

    monkeypatch.setattr(main, "resolve_reference", fake_resolve)
    client = TestClient(main.app)

    # Single verify with the shapes that previously triggered a strict-Pydantic 422.
    r = client.post("/api/verify", json={
        "title": "Motivation through the design of work",
        "authors": "Hackman, J. R. & Oldham, G. R.",
        "year": "n.d.", "volume": 16, "issue": 2, "pages": 250,
        "journal": "Organizational Behavior",
    })
    assert r.status_code == 200
    assert r.json()["status"] == "found"
    # Coercion normalized the messy fields before hitting resolve_reference.
    ref = captured["ref"]
    assert ref["authors"] == ["Hackman, J. R.", "Oldham, G. R."]
    assert ref["year"] is None            # "n.d." has no plausible 4-digit year
    assert ref["volume"] == "16" and ref["pages"] == "250"
    assert ref["container"] == "Organizational Behavior"

    # A year buried in prose is still recovered; 'container' is accepted too.
    r2 = client.post("/api/verify", json={"title": "T", "year": "forthcoming 2023",
                                          "container": "Journal X",
                                          "authors": [{"family": "Vaswani"}, "Shazeer"]})
    assert r2.status_code == 200
    assert captured["ref"]["year"] == 2023
    assert captured["ref"]["authors"] == ["Vaswani", "Shazeer"]
    assert captured["ref"]["container"] == "Journal X"

    # Batch tolerates a non-object entry (reported per-reference, batch survives).
    rb = client.post("/api/verify_batch", json={"references": [
        {"title": "Good", "year": "2020"},
        "just a string, not an object",
        {"title": "Also good", "volume": 3},
    ]})
    assert rb.status_code == 200
    b = rb.json()
    assert b["count"] == 3
    assert b["results"][0]["status"] == "found"
    assert b["results"][1]["status"] == "lookup_failed"
    assert b["results"][2]["status"] == "found"


def test_verify_tolerates_any_request_shape(monkeypatch):
    """An LLM function-caller may send the body in shapes that a strict schema
    rejects with 422: a stringified references array, a bare top-level array, the
    whole body as a JSON string, a non-string openalex_key, reference items that
    are themselves stringified JSON, or args nested under a wrapper key. Every one
    of these must be normalized to a 200 result, never a 422."""
    import json as _json
    from fastapi.testclient import TestClient
    from app import main

    seen = {"key": "sentinel"}

    def fake_resolve(ref, api_key=None, **kw):
        seen["key"] = api_key
        return {"status": "found",
                "work": {"title": ref.get("title"), "authors": [], "year": None,
                         "venue": None, "doi": None, "abstract": None, "url": "u"},
                "candidates": [], "notes": [], "field_check": [], "field_mismatch_count": 0}

    monkeypatch.setattr(main, "resolve_reference", fake_resolve)
    client = TestClient(main.app)
    H = {"Content-Type": "application/json"}

    def post(path, raw_body):
        return client.post(path, content=raw_body, headers=H)

    # H1: references as a STRINGIFIED JSON array (the most common EduGenAI shape).
    r = post("/api/verify_batch", '{"references":"[{\\"title\\":\\"A\\"},{\\"title\\":\\"B\\"}]"}')
    assert r.status_code == 200 and r.json()["count"] == 2

    # H2: a bare top-level array as the whole body.
    r = post("/api/verify_batch", '[{"title":"A"},{"title":"B"},{"title":"C"}]')
    assert r.status_code == 200 and r.json()["count"] == 3

    # H6: the ENTIRE body serialized as a JSON string.
    r = post("/api/verify_batch", _json.dumps(_json.dumps({"references": [{"title": "A"}]})))
    assert r.status_code == 200 and r.json()["count"] == 1

    # H5: a non-string openalex_key must be ignored, not 422 (and never used as a key).
    r = post("/api/verify_batch", '{"references":[{"title":"A"}],"openalex_key":{}}')
    assert r.status_code == 200 and r.json()["count"] == 1
    assert seen["key"] is None  # junk key dropped

    # Reference item that is itself a stringified JSON object.
    r = post("/api/verify_batch", '{"references":["{\\"title\\":\\"A\\"}"]}')
    assert r.status_code == 200
    assert r.json()["results"][0]["status"] == "found"  # parsed, not lookup_failed

    # H7: args nested under a function-call wrapper key.
    r = post("/api/verify_batch", '{"body":{"references":[{"title":"A"},{"title":"B"}]}}')
    assert r.status_code == 200 and r.json()["count"] == 2

    # A single reference object POSTed to the batch endpoint (no 'references' wrapper).
    r = post("/api/verify_batch", '{"title":"A single ref","year":"2020"}')
    assert r.status_code == 200 and r.json()["count"] == 1

    # Single endpoint: a bare reference object, and a string openalex_key that IS used.
    r = post("/api/verify", '{"title":"Solo","openalex_key":"real-key-123"}')
    assert r.status_code == 200 and r.json()["status"] == "found"
    assert seen["key"] == "real-key-123"

    # Wrong Content-Type (JSON posted as text/plain) is still parsed.
    r = client.post("/api/verify_batch", content='{"references":[{"title":"A"}]}',
                    headers={"Content-Type": "text/plain"})
    assert r.status_code == 200 and r.json()["count"] == 1

    # EduGenAI wraps the function arguments under a platform-chosen key
    # ("parameters"); any wrapper name must be descended into, not just a
    # known list. This was the real-world count:0 failure.
    for wrapper in ("parameters", "params", "properties", "anything_else"):
        r = post("/api/verify_batch",
                 _json.dumps({wrapper: {"references": [{"title": "A"}, {"title": "B"}]}}))
        assert r.status_code == 200 and r.json()["count"] == 2, wrapper

    # Wrapper alongside other keys (e.g. the function name).
    r = post("/api/verify_batch",
             '{"name":"verify_references","parameters":{"references":[{"title":"A"}]}}')
    assert r.status_code == 200 and r.json()["count"] == 1

    # A list of scalars nested under a random key is NOT mistaken for a
    # bibliography, and unrelated payloads still yield an empty result.
    r = post("/api/verify_batch", '{"tags":["alpha","beta"],"query":"Start"}')
    assert r.status_code == 200 and r.json()["count"] == 0

    # A form-encoded body with the JSON inside a value is unwrapped too.
    from urllib.parse import quote
    r = client.post("/api/verify_batch",
                    content="references=" + quote('[{"title":"A"},{"title":"B"}]'),
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 200 and r.json()["count"] == 2

    # Responses are self-diagnosing: every response carries api_version, and a
    # zero-result response carries a hint describing the received shape by its
    # KEY NAMES only — never the values (which could be keys/document text).
    r = post("/api/verify_batch", '{"references":[{"title":"A"}]}')
    assert r.json()["api_version"] == main.API_VERSION and "hint" not in r.json()
    r = post("/api/verify_batch", '{"query":"SECRETVALUE","tags":["SECRETVALUE"]}')
    d = r.json()
    assert d["count"] == 0 and "api_version" in d
    assert "query" in d["hint"] and "tags" in d["hint"]     # key names shown
    assert "SECRETVALUE" not in d["hint"]                   # values never echoed

    # Some platforms bind the function arguments to the URL query string and
    # POST an empty body (the observed EduGenAI failure) — args in the query
    # must work, and an empty-body hint must name the transport facts.
    from urllib.parse import quote as _q
    r = client.post("/api/verify_batch?references=" + _q('[{"title":"A"}]'), content=b"")
    assert r.status_code == 200 and r.json()["count"] == 1
    r = client.post("/api/verify_batch?unrelated=1", content=b"")
    d = r.json()
    assert d["count"] == 0
    assert "empty" in d["hint"] and "unrelated" in d["hint"]  # query keys surfaced
    assert "GET" in d["hint"]  # the empty-body escape hatch is spelled out

    # GET works outright — for platforms that never fill a POST body, the
    # function method can simply be switched to GET.
    r = client.get("/api/verify_batch?references=" + _q('[{"title":"A"},{"title":"B"}]'))
    assert r.status_code == 200 and r.json()["count"] == 2
    r = client.get("/api/verify?title=" + _q("A real paper") + "&year=2019")
    assert r.status_code == 200 and r.json()["status"] == "found"

    # Plan B: gateways that cannot serialize a nested array parameter get a
    # schema with ONE STRING parameter — the array as a JSON string, under
    # 'references_json', via POST body or GET query string.
    r = post("/api/verify_batch", '{"references_json":"[{\\"title\\":\\"A\\"},{\\"title\\":\\"B\\"}]"}')
    assert r.status_code == 200 and r.json()["count"] == 2
    r = client.get("/api/verify_batch?references_json=" + _q('[{"title":"A"}]'))
    assert r.status_code == 200 and r.json()["count"] == 1
    # Empty-body hint points at Plan B.
    r = client.post("/api/verify_batch", content=b"")
    assert "references_json" in r.json()["hint"]


def test_echo_endpoint_mirrors_transport():
    """/api/echo is the ground-truth transport test: it reflects method, query,
    headers, and raw body byte-for-byte (key-like values redacted), so 'does the
    platform send anything?' can be settled without trusting any parsing."""
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)

    r = client.post("/api/echo?probe=XYZ", content=b'{"message":"HELLO-42"}',
                    headers={"Content-Type": "application/json",
                             "X-OpenAlex-Key": "supersecret"})
    d = r.json()
    rec = d["received"]
    assert rec["method"] == "POST"
    assert "HELLO-42" in rec["body_first_2000_chars"]
    assert rec["query_params"]["probe"] == "XYZ"
    assert rec["headers"]["x-openalex-key"] == "•••redacted•••"   # never echo keys
    assert "supersecret" not in r.text
    assert d["api_version"] == main.API_VERSION

    g = client.get("/api/echo?message=VIA-QUERY").json()["received"]
    assert g["method"] == "GET" and g["query_params"]["message"] == "VIA-QUERY"
    assert g["body_total_chars"] == 0

    # Body echo is capped so the endpoint can't be used as an amplifier.
    big = client.post("/api/echo", content=b"A" * 10000).json()["received"]
    assert len(big["body_first_2000_chars"]) == 2000 and big["body_total_chars"] == 10000


def test_parse_json_handles_braces_inside_strings():
    from app.llm import _parse_json
    # A citation context containing math braces must not break brace matching.
    txt = 'Sure:\n{"references":[{"id":1,"title":"On {a,b}","note":"f(x)={1}/{2}"}]} thanks'
    d = _parse_json(txt)
    assert d["references"][0]["title"] == "On {a,b}"


def test_complete_json_truncation_message(monkeypatch):
    from app.llm import LLMClient, LLMError
    c = LLMClient("anthropic", "sk-test")
    # Simulate a provider reply that hit the token ceiling and is unparseable.
    monkeypatch.setattr(c, "_complete", lambda *a, **k: ('{"references":[{"id":1', True))
    try:
        c.complete_json("s", "u", max_tokens=100)
        assert False, "should raise"
    except LLMError as e:
        assert "cut off" in str(e).lower() and "token" in str(e).lower()


def test_anthropic_streams_to_avoid_10min_limit(monkeypatch):
    """The Anthropic call must STREAM (client.messages.stream), not use the
    blocking .create — the SDK rejects a non-streaming request whose max_tokens
    could take >10 min ('Streaming is required...'), which a large token limit
    trips. Verify the streamed final message is accumulated."""
    import sys, types

    class FakeBlock:
        type = "text"
        text = '{"references": []}'

    class FakeMessage:
        stop_reason = "end_turn"
        content = [FakeBlock()]

    class FakeStreamCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return FakeMessage()

    used = {"stream": False, "create": False}

    class FakeMessages:
        def stream(self, **kw): used["stream"] = True; return FakeStreamCtx()
        def create(self, **kw): used["create"] = True; raise AssertionError("must not call create")

    class FakeAnthropic:
        def __init__(self, api_key=None): self.messages = FakeMessages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = FakeAnthropic
    fake.AuthenticationError = type("AuthenticationError", (Exception,), {})
    fake.APIStatusError = type("APIStatusError", (Exception,), {})
    fake.APIConnectionError = type("APIConnectionError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from app.llm import LLMClient
    c = LLMClient("anthropic", "sk-test")
    out = c.complete_json("s", "u", max_tokens=64000)   # a value that would trip the guard
    assert out == {"references": []}
    assert used["stream"] and not used["create"]


def test_complete_json_generic_error_when_not_truncated(monkeypatch):
    from app.llm import LLMClient, LLMError
    c = LLMClient("anthropic", "sk-test")
    monkeypatch.setattr(c, "_complete", lambda *a, **k: ("not json at all", False))
    try:
        c.complete_json("s", "u")
        assert False, "should raise"
    except LLMError as e:
        assert "did not return valid JSON" in str(e)


def test_display_matches_site_severity():
    from app.main import _display
    # hallucination
    assert _display({"status": "not_found"})["severity"] == 100
    # found + author mismatch (weight 9) -> 85 / Review
    d = _display({"status": "found", "field_check": [
        {"field": "authors", "status": "mismatch"}, {"field": "year", "status": "mismatch"}]})
    assert d["severity"] == 85 and d["priority"] == "Review"
    assert set(d["mismatched_fields"]) == {"authors", "year"}
    assert d["badge"] == "Verified"
    # found + only a minor field (volume, weight 3) -> 50 / Check
    d2 = _display({"status": "found", "field_check": [{"field": "volume", "status": "mismatch"}]})
    assert d2["severity"] == 50 and d2["priority"] == "Check"
    # found clean -> 8 / no priority
    d3 = _display({"status": "found", "field_check": []})
    assert d3["severity"] == 8 and d3["priority"] == ""
    # lookup_failed -> 35, never treated as hallucination
    assert _display({"status": "lookup_failed"})["severity"] == 35


def test_journal_variants_are_close_not_match():
    from app.fieldcheck import compare_fields, field_mismatches
    ref = {"title": "Attention is all you need", "authors": ["Vaswani, A."], "et_al": True,
           "year": 2017, "doi": None,
           "container": "Advances in Neural Information Processing Systems"}
    work = {"title": "Attention Is All You Need", "authors_full": ["Ashish Vaswani"],
            "year": 2017, "doi": None, "venue": "Neural Information Processing Systems"}
    fields = {f["field"]: f for f in compare_fields(ref, work)}
    # containment/high-similarity venue variant -> "close", never a clean match
    assert fields["journal"]["status"] == "close"
    # and never counted as a mismatch (no severity impact)
    assert "journal" not in {f["field"] for f in field_mismatches(compare_fields(ref, work))}
    # a leading "The" is trivial enough to stay a full match
    ref2 = dict(ref, container="The Review of Economics and Statistics")
    work2 = dict(work, venue="Review of Economics and Statistics")
    fields2 = {f["field"]: f for f in compare_fields(ref2, work2)}
    assert fields2["journal"]["status"] == "match"
    # a genuinely different venue is still a mismatch
    ref3 = dict(ref, container="Science")
    work3 = dict(work, venue="IEEE Conference on Computer Vision and Pattern Recognition")
    fields3 = {f["field"]: f for f in compare_fields(ref3, work3)}
    assert fields3["journal"]["status"] == "mismatch"


def test_display_exposes_minor_fields():
    from app.main import _display
    d = _display({"status": "found", "field_check": [
        {"field": "journal", "status": "close"},
        {"field": "title", "status": "match"}]})
    assert d["minor_fields"] == ["journal"]
    assert d["mismatched_fields"] == []
    assert d["severity"] == 8  # minor variations don't raise severity


# ---------------------------------------------------------------------------
# Commit 2: OpenAlex error semantics, redaction coverage, logging, health
# ---------------------------------------------------------------------------

def test_redact_strips_percent_encoded_key():
    from app.keysafety import redact
    key = "ab/cd e+f"
    msg = "GET https://api.openalex.org/works?api_key=ab%2Fcd%20e%2Bf and ab%2Fcd+e%2Bf and " + key
    out = redact(msg, key)
    assert key not in out and "ab%2Fcd" not in out


class _StubResp:
    def __init__(self, status, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class _StubClient:
    """Minimal OpenAlex client double: a scripted list of responses/exceptions,
    usable as a context manager (resolve_reference does `with _client() as c`)."""

    def __init__(self, script, params=None):
        self.script = list(script)
        self.params = params or {}
        self.calls = 0

    def get(self, path, params=None):
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script_default()
        if isinstance(item, Exception):
            raise item
        return item

    def script_default(self):
        return _StubResp(200, {"results": []})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_notes_never_contain_exception_text_or_url(monkeypatch):
    """httpx exception text embeds the request URL (api_key included); notes
    must carry only a status/class, never that text."""
    import httpx
    from app import openalex
    stub = _StubClient([httpx.ConnectError("boom https://x?api_key=SECRET")] * 9)
    monkeypatch.setattr(openalex, "_client", lambda key=None: stub)
    monkeypatch.setattr(openalex.time, "sleep", lambda s: None)
    res = openalex.resolve_reference({"title": "Some paper", "doi": "10.1/x", "year": 2020})
    joined = " ".join(res["notes"])
    assert res["status"] == "lookup_failed"
    assert "SECRET" not in joined and "http" not in joined
    assert "could not be reached" in joined


def test_403_without_key_is_lookup_failure_not_auth_error():
    from app import openalex
    from app.openalex import OpenAlexAuthError, OpenAlexLookupError
    with pytest.raises(OpenAlexLookupError) as ei:
        openalex._get(_StubClient([_StubResp(403)]), "/works")
    assert ei.value.status == 403
    with pytest.raises(OpenAlexAuthError):
        openalex._get(_StubClient([_StubResp(403)], params={"api_key": "k"}), "/works")


def test_403_without_key_resolves_to_lookup_failed(monkeypatch):
    from app import openalex
    stub = _StubClient([_StubResp(403)] * 9)
    monkeypatch.setattr(openalex, "_client", lambda key=None: stub)
    res = openalex.resolve_reference({"title": "Some paper", "year": 2020})
    assert res["status"] == "lookup_failed"
    assert any("403" in n for n in res["notes"])


def test_title_search_rejected_queries_raise_not_empty(monkeypatch):
    """Every query rejected (4xx) = 'could not search', which must become
    lookup_failed — not 'no such work' (a false hallucination flag)."""
    from app import openalex
    from app.openalex import OpenAlexLookupError
    with pytest.raises(OpenAlexLookupError) as ei:
        openalex._search_works_by_title(_StubClient([_StubResp(400), _StubResp(400)]), "Some title?")
    assert ei.value.status == 400
    # One rejected + one genuine empty answer -> a real "no hits".
    assert openalex._search_works_by_title(
        _StubClient([_StubResp(400), _StubResp(200, {"results": []})]), "Some title?") == []
    with pytest.raises(OpenAlexLookupError):
        openalex._search_works_by_title(_StubClient([]), "???")


def test_doi_lookup_error_status_only():
    from app import openalex
    from app.openalex import OpenAlexLookupError
    with pytest.raises(OpenAlexLookupError) as ei:
        openalex._get_work_by_doi(_StubClient([_StubResp(400)]), "10.1/x")
    assert ei.value.status == 400 and "http" not in str(ei.value)


def test_health_get_and_head():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "api_version", "git_sha", "polite_pool"}
    assert r.json()["api_version"] == main.API_VERSION
    assert client.head("/api/health").status_code == 200


def test_httpx_logger_is_quiet():
    import logging
    import app.main  # noqa: F401  (import side effect)
    assert logging.getLogger("httpx").level == logging.WARNING


# ---------------------------------------------------------------------------
# Commit 3: OpenAlex pacing, Retry-After, daily budget, cancel + client reuse
# ---------------------------------------------------------------------------

def test_token_bucket_paces_requests():
    from app.openalex import TokenBucket
    clock = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    b = TokenBucket(rate=2, capacity=2, clock=lambda: clock["t"], sleep=sleep)
    for _ in range(4):          # 2 immediate, then ~0.5 s per token at 2/s
        b.acquire()
    assert abs(sum(slept) - 1.0) < 1e-6
    assert TokenBucket(rate=0, capacity=1).acquire() is None   # disabled = no wait


def test_get_honours_retry_after_capped(monkeypatch):
    from app import openalex
    slept = []
    monkeypatch.setattr(openalex.time, "sleep", lambda s: slept.append(s))
    stub = _StubClient([_StubResp(429, headers={"Retry-After": "3"}), _StubResp(200, {"ok": 1})])
    assert openalex._get(stub, "/works").status_code == 200
    assert slept == [3.0]
    slept.clear()
    stub = _StubClient([_StubResp(503, headers={"Retry-After": "999"}), _StubResp(200)])
    openalex._get(stub, "/works")
    assert slept == [openalex.RETRY_AFTER_CAP_S]


def test_get_stops_on_exhausted_daily_budget(monkeypatch):
    from app import openalex
    monkeypatch.setattr(openalex.time, "sleep", lambda s: None)
    body = ('{"error":"Rate limit exceeded","message":"Insufficient budget. This request '
            'costs $0.001 but you only have $0 remaining. Resets at midnight UTC."}')
    stub = _StubClient([_StubResp(429, text=body)] * 3)
    with pytest.raises(openalex.OpenAlexLookupError) as ei:
        openalex._get(stub, "/works")
    assert stub.calls == 1 and ei.value.status == 429


def test_get_aborts_when_cancelled(monkeypatch):
    import threading
    from app import openalex
    cancel = threading.Event()
    cancel.set()
    stub = _StubClient([_StubResp(200)])
    with pytest.raises(openalex.OpenAlexLookupError):
        openalex._get(stub, "/works", cancel=cancel)
    assert stub.calls == 0
    # Cancel raised during the retry back-off, not after all attempts.
    cancel2 = threading.Event()
    calls = {"n": 0}

    class Flaky:
        def get(self, path, params=None):
            calls["n"] += 1
            cancel2.set()
            return _StubResp(503)

    with pytest.raises(openalex.OpenAlexLookupError):
        openalex._get(Flaky(), "/works", cancel=cancel2)
    assert calls["n"] == 1


def test_resolve_uses_passed_client(monkeypatch):
    from app import openalex
    monkeypatch.setattr(openalex, "_client", lambda key=None: (_ for _ in ()).throw(AssertionError("must reuse client")))
    stub = _StubClient([_StubResp(200, {"results": []})] * 4)
    res = openalex.resolve_reference({"title": "Some paper", "year": 2020}, client=stub)
    assert res["status"] == "not_found" and stub.calls >= 1


def test_mailto_warning_logged_once(monkeypatch, caplog):
    import logging
    from app import openalex
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    monkeypatch.setitem(openalex._WARNED, "mailto", False)
    with caplog.at_level(logging.WARNING, logger="phantocite.openalex"):
        openalex._client().close()
        openalex._client().close()
    assert sum("OPENALEX_MAILTO" in r.message for r in caplog.records) == 1


# ---------------------------------------------------------------------------
# Commit 4: verify endpoints run off the event loop with a shared client
# ---------------------------------------------------------------------------

def test_verify_endpoints_run_off_event_loop(monkeypatch):
    """The lookups are blocking I/O; if they ran on the loop thread, every
    in-flight /api/analyze stream would stop ticking for the duration."""
    import asyncio
    from fastapi.testclient import TestClient
    from app import main
    seen = []

    def fake_resolve(ref, api_key=None, **kw):
        try:
            asyncio.get_running_loop()
            seen.append("on-loop")
        except RuntimeError:
            seen.append("off-loop")
        assert kw.get("client") is None or hasattr(kw["client"], "get")
        return {"status": "not_found", "work": None, "candidates": [], "notes": []}

    monkeypatch.setattr(main, "resolve_reference", fake_resolve)
    client = TestClient(main.app)
    assert client.post("/api/verify", json={"title": "A"}).status_code == 200
    assert client.post("/api/verify_batch", json={"references": [{"title": "A"}, {"title": "B"}]}).json()["count"] == 2
    assert seen == ["off-loop"] * 3


def test_verify_single_accepts_stringified_first_reference(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    captured = {}

    def fake_resolve(ref, api_key=None, **kw):
        captured["title"] = ref["title"]
        return {"status": "not_found", "work": None, "candidates": [], "notes": []}

    monkeypatch.setattr(main, "resolve_reference", fake_resolve)
    r = TestClient(main.app).post("/api/verify", json={"references": ['{"title":"Stringy"}']})
    assert r.status_code == 200 and captured["title"] == "Stringy"


# ---------------------------------------------------------------------------
# Commit 5: provider HTTP timeouts scale with max_tokens; timeouts are named
# ---------------------------------------------------------------------------

def test_post_json_timeout_scales_and_is_named(monkeypatch):
    import httpx
    from app import llm
    captured = {}

    class R:
        status_code = 200

        def json(self):
            return {"ok": 1}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["timeout"] = timeout
        return R()

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    llm._post_json("https://x", {}, {}, "OpenAI", max_tokens=16000)
    assert captured["timeout"].read == pytest.approx(min(llm.LLM_READ_TIMEOUT_CAP_S, 60 + 16000 / 20))
    llm._post_json("https://x", {}, {}, "OpenAI", max_tokens=64000)
    assert captured["timeout"].read == llm.LLM_READ_TIMEOUT_CAP_S

    def slow(url, json=None, headers=None, timeout=None):
        raise httpx.ReadTimeout("x")

    monkeypatch.setattr(llm.httpx, "post", slow)
    with pytest.raises(llm.LLMError, match="did not answer"):
        llm._post_json("https://x", {}, {}, "Gemini", max_tokens=8000)

    def down(url, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("x")

    monkeypatch.setattr(llm.httpx, "post", down)
    with pytest.raises(llm.LLMError, match="Could not reach"):
        llm._post_json("https://x", {}, {}, "Gemini")


# ---------------------------------------------------------------------------
# Commit 6: compare_contexts is a resilient generator
# ---------------------------------------------------------------------------

class _CompareLLM:
    """Scripted LLM: `fail_calls` is the set of call numbers that raise."""

    def __init__(self, fail_calls=()):
        self.calls = 0
        self.fail_calls = set(fail_calls)
        self._api_key = "sk-secret"

    def redact(self, text):
        return text.replace(self._api_key, "•••")

    def complete_json(self, system, user, max_tokens=0, thinking=True):
        import json as _j
        self.calls += 1
        if self.calls in self.fail_calls:
            raise LLMError(f"overloaded (key sk-secret) call {self.calls}")
        # Echo a 'match' verdict for every id in the batch.
        ids = [it["id"] for it in _j.loads(user.split("ITEMS:", 1)[1])]
        return {"results": [{"id": i, "verdict": "match", "explanation": "ok"} for i in ids]}


def _compare_items(n):
    return [{"id": i, "title": f"T{i}", "abstract": "An abstract.", "contexts": ["ctx"]}
            for i in range(1, n + 1)]


def test_compare_contexts_survives_a_failed_batch():
    from app.analysis import compare_contexts
    llm = _CompareLLM(fail_calls={2, 3})          # batch 2 fails twice (call + retry)
    out = list(compare_contexts(llm, _compare_items(10)))
    assert llm.calls == 3
    assert [o["verdict"] for o in out[:8]] == ["match"] * 8
    assert [o["verdict"] for o in out[8:]] == ["unclear", "unclear"]
    assert "sk-secret" not in out[9]["explanation"] and "failed" in out[9]["explanation"]


def test_compare_contexts_retries_once_then_succeeds():
    from app.analysis import compare_contexts
    llm = _CompareLLM(fail_calls={2})
    out = list(compare_contexts(llm, _compare_items(10)))
    assert llm.calls == 3 and all(o["verdict"] == "match" for o in out)


def test_compare_contexts_strict_raises_and_cancel_stops():
    import threading
    from app.analysis import compare_contexts
    with pytest.raises(LLMError):
        list(compare_contexts(_CompareLLM(fail_calls={1, 2}), _compare_items(2), strict=True))
    cancel = threading.Event()
    llm = _CompareLLM()
    gen = compare_contexts(llm, _compare_items(10), cancel=cancel)
    first = [next(gen) for _ in range(8)]
    cancel.set()
    assert list(gen) == [] and len(first) == 8 and llm.calls == 1


def test_compare_contexts_shortcuts_without_abstract_or_context():
    from app.analysis import compare_contexts
    llm = _CompareLLM()
    items = [{"id": 1, "title": "A", "abstract": "", "contexts": ["c"]},
             {"id": 2, "title": "B", "abstract": "abs", "contexts": []}]
    out = list(compare_contexts(llm, items))
    assert [o["verdict"] for o in out] == ["unclear", "unclear"] and llm.calls == 0


# ---------------------------------------------------------------------------
# Commit 7: positional ids, echo redaction
# ---------------------------------------------------------------------------

def test_extract_references_assigns_positional_ids():
    """Model-supplied ids are untrusted (a paper can prompt-inject them) and
    were used unescaped as DOM ids; ids are now positional and unique."""
    from app import analysis

    class FakeLLM:
        def complete_json(self, *a, **k):
            return {"references": [
                {"id": 0, "raw": "a", "title": "A"},
                {"id": 0, "raw": "b", "title": "B"},
                {"id": '<img src=x onerror=alert(1)>', "raw": "c", "title": "C"}]}

    refs, _ = analysis.extract_references(FakeLLM(), "text")
    assert [r["id"] for r in refs] == [1, 2, 3]


def test_echo_redacts_keys_in_body_query_and_infra_header_values():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    body = b'{"references":[{"title":"A"}],"openalex_key":"supersecret","message":"HELLO-42","x":"sk-abcdefghijklmnop"}'
    r = client.post("/api/echo?api_key=qsecret&probe=XYZ", content=body,
                    headers={"Content-Type": "application/json", "X-Forwarded-For": "1.2.3.4"})
    text = r.text
    rec = r.json()["received"]
    assert "supersecret" not in text and "qsecret" not in text and "sk-abcdefghijklmnop" not in text
    assert "HELLO-42" in rec["body_first_2000_chars"] and rec["query_params"]["probe"] == "XYZ"
    assert "x-forwarded-for" in rec["headers"] and "1.2.3.4" not in text
    # form-encoded body
    r = client.post("/api/echo", content=b"openalex_key=formsecret&references=%5B%5D",
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert "formsecret" not in r.text and "references=" in r.json()["received"]["body_first_2000_chars"]


# ---------------------------------------------------------------------------
# Commit 8: the analysis stream — extraction off-loop, isolation, order, budget, cancel
# ---------------------------------------------------------------------------

def _stream_events(client, monkeypatch, refs, resolve=None, compare=None, budget=None,
                   filename="paper.pdf"):
    import time as _t
    from app import main
    if budget is not None:
        monkeypatch.setattr(main, "ANALYSIS_BUDGET_S", budget)
    monkeypatch.setattr(main, "extract_text", lambda name, data: "body text " * 50)
    monkeypatch.setattr(main, "extract_references", lambda llm, text, max_tokens=0: (refs, []))
    if resolve is None:
        def resolve(ref, api_key=None, **kw):
            return {"status": "found", "work": {"title": ref["title"], "abstract": "abs"},
                    "candidates": [], "notes": [], "field_check": [], "field_mismatch_count": 0}
    monkeypatch.setattr(main, "resolve_reference", resolve)
    if compare is None:
        def compare(llm, items, max_tokens=0, **kw):
            for it in items:
                yield {"id": it["id"], "verdict": "match", "explanation": "ok"}
    monkeypatch.setattr(main, "compare_contexts", compare)
    data = {"check_hallucination": "true", "check_misquote": "true"}
    with client.stream("POST", "/api/analyze", files={"file": (filename, b"%PDF-1.4 x")}, data=data) as r:
        assert r.status_code == 200
        return [json.loads(l) for l in r.iter_lines() if l.strip()]


def _refs(n):
    return [{"id": i, "raw": f"R{i}", "title": f"Title {i}", "authors": [], "year": 2020,
             "doi": None, "container": None, "volume": None, "issue": None, "pages": None,
             "et_al": False, "contexts": ["ctx"], "first_author_surname": None}
            for i in range(1, n + 1)]


def test_analyze_extraction_error_is_a_stream_event_not_http_400(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    data = {"check_hallucination": "true"}
    with client.stream("POST", "/api/analyze", files={"file": ("paper.txt", b"hello")}, data=data) as r:
        assert r.status_code == 200
        events = [json.loads(l) for l in r.iter_lines() if l.strip()]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "ready" and "error" in kinds
    assert "Unsupported file type" in next(e for e in events if e["type"] == "error")["detail"]
    # Server-side LLM configuration problems are plain HTTP 503s.
    monkeypatch.delenv("LLM_API_KEY")
    r = client.post("/api/analyze", files={"file": ("p.pdf", b"x")}, data=data)
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]


def test_analyze_results_stay_ordered_with_parallel_lookups(monkeypatch):
    import time as _t
    from fastapi.testclient import TestClient
    from app import main
    n = 6

    def slow_reverse(ref, api_key=None, **kw):
        _t.sleep(0.02 * (n - ref["id"]))   # later references finish FIRST
        return {"status": "found", "work": {"title": ref["title"], "abstract": "abs"},
                "candidates": [], "notes": [], "field_check": [], "field_mismatch_count": 0}

    events = _stream_events(TestClient(main.app), monkeypatch, _refs(n), resolve=slow_reverse)
    ids = [e["result"]["reference"]["id"] for e in events if e["type"] == "result"]
    assert ids == list(range(1, n + 1))
    assert [e["done"] for e in events if e["type"] == "progress" and e["stage"] == "verify"] == list(range(1, n + 1))
    done = events[-1]
    assert done["type"] == "done" and done["incomplete"] is False and done["unprocessed"] == 0
    assert [e["type"] for e in events][:2] == ["ready", "progress"] and events[1]["stage"] == "read"


def test_analyze_one_bad_reference_does_not_kill_the_run(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    def flaky(ref, api_key=None, **kw):
        if ref["id"] == 2:
            raise ValueError("Expecting value: line 1 column 1")
        return {"status": "found", "work": {"title": ref["title"], "abstract": "abs"},
                "candidates": [], "notes": [], "field_check": [], "field_mismatch_count": 0}

    events = _stream_events(TestClient(main.app), monkeypatch, _refs(3), resolve=flaky)
    results = {e["result"]["reference"]["id"]: e["result"] for e in events if e["type"] == "result"}
    assert set(results) == {1, 2, 3} and results[2]["status"] == "lookup_failed"
    assert "Expecting value" not in json.dumps(results[2]["notes"])
    assert events[-1]["type"] == "done"


def test_analyze_time_budget_stops_before_lookups(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    events = _stream_events(TestClient(main.app), monkeypatch, _refs(4), budget=0)
    assert not [e for e in events if e["type"] == "result"]
    done = events[-1]
    assert done["type"] == "done" and done["incomplete"] is True
    assert done["unprocessed"] == 4 and done["reason"] == "time_budget"


def test_pipeline_cancel_stops_the_run(monkeypatch):
    import threading, time as _t
    from app import main
    from app.llm import LLMClient
    emitted = []
    control = main.RunControl(cancel=threading.Event(), deadline=_t.monotonic() + 100)
    control.cancel.set()
    monkeypatch.setattr(main, "extract_text", lambda name, data: "text")
    monkeypatch.setattr(main, "extract_references", lambda llm, text, max_tokens=0: (_refs(3), []))
    monkeypatch.setattr(main, "resolve_reference", lambda ref, api_key=None, **kw: (_ for _ in ()).throw(AssertionError("must not look up")))
    main._pipeline(emit=emitted.append, control=control, llm=LLMClient("openai", "sk-test"),
                   filename="p.pdf", data=b"x", openalex_key=None, check_hallucination=True,
                   check_misquote=True, max_tokens=1000, safe=str)
    done = [e for e in emitted if e and e["type"] == "done"][0]
    assert done["incomplete"] and done["reason"] == "cancelled" and done["unprocessed"] == 3
    assert emitted[-1] is None


def test_stream_close_sets_cancel(monkeypatch):
    import asyncio
    from app import main
    from app.llm import LLMClient

    def slow_pipeline(*, emit, control, **kw):
        control.cancel.wait(5)   # simulate a long stage that observes the flag
        emit(None)

    monkeypatch.setattr(main, "_pipeline", slow_pipeline)

    async def go():
        gen = main._run_stream(llm=LLMClient("openai", "sk-test"), filename="p.pdf", data=b"x",
                               openalex_key=None, check_hallucination=True,
                               check_misquote=False, max_tokens=1000, safe=str)
        first = await gen.__anext__()
        assert '"ready"' in first
        await gen.aclose()
        return main._LAST_CONTROL["control"].cancel.is_set()

    assert asyncio.run(go()) is True


# ---------------------------------------------------------------------------
# Commit 10: nonce CSP on pages, per-IP rate limiting
# ---------------------------------------------------------------------------

def test_pages_send_nonce_csp_matching_their_scripts():
    import re as _re
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    for path in ("/", "/edugenai"):
        r = client.get(path)
        assert r.status_code == 200
        csp = r.headers["content-security-policy"]
        nonce = _re.search(r"script-src 'nonce-([^']+)'", csp).group(1)
        assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
        assert "frame-ancestors 'none'" in csp and "googletagmanager.com" in csp
        scripts = _re.findall(r"<script([^>]*)>", r.text)
        assert scripts and all(f'nonce="{nonce}"' in s for s in scripts)
        assert r.headers["x-content-type-options"] == "nosniff"
    # Nonces are per request.
    assert client.get("/").headers["content-security-policy"] != client.get("/").headers["content-security-policy"]


def test_rate_limiter_returns_429_per_ip(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    limiter = main.RateLimiter({"/api/verify": (2, 60), "*": (100, 60)}, enabled=True)
    monkeypatch.setattr(main, "_LIMITER", limiter)
    monkeypatch.setattr(main, "resolve_reference",
                        lambda ref, api_key=None, **kw: {"status": "not_found", "work": None, "candidates": [], "notes": []})
    client = TestClient(main.app)
    hdr = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
    assert client.post("/api/verify", json={"title": "A"}, headers=hdr).status_code == 200
    assert client.post("/api/verify", json={"title": "A"}, headers=hdr).status_code == 200
    r = client.post("/api/verify", json={"title": "A"}, headers=hdr)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert client.post("/api/verify", json={"title": "A"}, headers={"X-Forwarded-For": "198.51.100.9"}).status_code == 200
    assert client.get("/api/health", headers=hdr).status_code == 200   # exempt


def test_rate_limiter_window_expires():
    from app import main
    lim = main.RateLimiter({"*": (2, 10)})
    assert lim.hit("ip", "/api/x", now=0) is None and lim.hit("ip", "/api/x", now=1) is None
    assert lim.hit("ip", "/api/x", now=2) == 9
    assert lim.hit("ip", "/api/x", now=11) is None


def test_unsearchable_title_note_is_accurate(monkeypatch):
    from app import openalex
    monkeypatch.setattr(openalex, "_client", lambda key=None: _StubClient([]))
    res = openalex.resolve_reference({"title": "???", "year": 2020})
    assert res["status"] == "lookup_failed" and "no searchable text" in " ".join(res["notes"])


# ---------------------------------------------------------------------------
# Server-side LLM key
# ---------------------------------------------------------------------------

def test_missing_server_key_is_a_clear_503(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.delenv("LLM_API_KEY")
    client = TestClient(main.app)
    assert client.get("/").status_code == 200
    r = client.post("/api/analyze", files={"file": ("p.pdf", b"x")}, data={"check_hallucination": "true"})
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]
    assert client.post("/api/compare", json={"items": []}).status_code == 503


def test_every_page_and_endpoint_is_open():
    """No sign-in anywhere: pages, the keyless API and the key-spending
    endpoints all answer directly."""
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    for path in ("/", "/edugenai", "/api/health", "/static/favicon.svg"):
        assert client.get(path).status_code == 200, path
    assert client.post("/api/verify_batch", json={"references": []}).status_code == 200
    assert client.post("/api/echo", json={"m": 1}).status_code == 200
    assert client.post("/api/report.pdf", json={"rows": [{"#": 1}]}).status_code == 200
    home = client.get("/").text
    assert "Log out" not in home and "sign in" not in home.lower()


def test_compare_uses_server_key(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    seen = {}

    def fake_compare(llm, items, max_tokens=0, **kw):
        seen["provider"], seen["key"] = llm.provider, llm._api_key
        return iter([{"id": 1, "verdict": "match", "explanation": "ok"}])

    monkeypatch.setattr(main, "compare_contexts", fake_compare)
    client = TestClient(main.app)
    r = client.post("/api/compare", json={"items": [{"id": 1, "title": "T", "abstract": "a", "contexts": ["c"]}],
                                           "provider": "gemini", "api_key": "sk-client"})   # ignored
    assert r.status_code == 200 and seen == {"provider": "openai", "key": "sk-test"}


# The output-token cap is fixed server-side: a posted 'max_tokens' (old cached
# page) is ignored and /api/compare is clamped to the same limit. The page
# carries the capacity note with the server's numbers.
def test_token_limit_is_fixed_server_side(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    seen = {}

    async def fake_stream(**kw):
        seen.update(kw)
        yield '{"type":"ready"}\n'

    monkeypatch.setattr(main, "_run_stream", fake_stream)
    client = TestClient(main.app)
    r = client.post("/api/analyze", files={"file": ("p.pdf", b"x")},
                    data={"check_hallucination": "true", "max_tokens": "64000",
                          "openalex_key": "  "})
    assert r.status_code == 200
    assert seen["max_tokens"] == main.LLM_MAX_TOKENS == 12000
    assert seen["openalex_key"] is None
    assert main._clamp_tokens(64000) == main.LLM_MAX_TOKENS
    assert main._clamp_tokens(10) == main.MIN_MAX_TOKENS
    assert main._clamp_tokens("junk") == main.LLM_MAX_TOKENS

    html = client.get("/").text
    assert "Advanced options" not in html and 'id="max_tokens"' not in html
    assert 'id="openalex_key"' not in html and "{{CAP_" not in html
    note = html[html.index('id="capacity-note"'):html.index("</div>", html.index('id="capacity-note"'))]
    assert f"{main.CAPACITY_NOTE_PAGES} pages" in note and f"{main.CAPACITY_NOTE_REFS} references" in note


def test_long_document_warns_in_progress_log(monkeypatch):
    from app import main
    from app.llm import LLMClient
    events = []
    monkeypatch.setattr(main, "extract_text", lambda f, d: "x" * (main.LONG_DOC_CHARS + 1))
    monkeypatch.setattr(main, "extract_references", lambda llm, text, max_tokens=0: ([], []))
    import threading, time
    control = main.RunControl(cancel=threading.Event(), deadline=time.monotonic() + 60)
    main._pipeline(emit=lambda e: events.append(e), control=control,
                   llm=LLMClient("openai", "sk-test"), filename="p.pdf", data=b"x",
                   openalex_key=None, check_hallucination=True, check_misquote=False,
                   max_tokens=1000, safe=str)
    msgs = [e["message"] for e in events if e and e.get("type") == "progress"]
    assert any("Long document" in m and "may be partial" in m for m in msgs)


def test_cookie_bar_and_consent_mode_on_both_pages():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    for path in ("/", "/edugenai"):
        html = client.get(path).text
        # GA starts with analytics cookies denied; the bar's buttons flip it.
        assert "gtag('consent', 'default', {analytics_storage: 'denied'" in html
        assert 'id="cookie-bar" hidden' in html
        assert 'id="cookie-accept"' in html and "Accept cookies" in html
        assert html.index("gtag('consent', 'default'") < html.index("gtag('config'")


# PDF export: the browser posts the rows it already built for the other
# downloads and gets a typeset report back. Values come from a student's
# document, so hostile text must not reach the PDF engine unescaped.
def test_report_pdf_export():
    from fastapi.testclient import TestClient
    from app import main, report

    rows = [{"#": 1, "Status": "Verified", "Priority": "Review",
             "Reference (as printed)": "Shannon & Weaver (1948) <b>x</b> & y",
             "Year ref / OA": " / ", "OpenAlex URL": "https://openalex.org/W1"}]
    client = TestClient(main.app)
    r = client.post("/api/report.pdf", json={"rows": rows, "summary": ["1 reference"]})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF") and len(r.content) > 800
    assert ".pdf" in r.headers["content-disposition"]

    assert client.post("/api/report.pdf", json={"rows": []}).status_code == 400
    too_many = [{"#": i} for i in range(report.MAX_ROWS + 1)]
    assert client.post("/api/report.pdf", json={"rows": too_many}).status_code == 400


def test_report_values_are_escaped_and_capped():
    from app import report
    assert report._text("a & b <script>") == "a &amp; b &lt;script&gt;"
    assert report._text("line\nbreak") == "line<br/>break"
    assert report._text("\x00\x07clean") == "clean"
    assert report._text(None) == "" and report._text(7) == "7"
    long = report._text("x" * (report.MAX_VALUE_CHARS + 500))
    assert len(long) < report.MAX_VALUE_CHARS + 10 and long.endswith("…")
    # A row of hostile values still produces a valid document.
    pdf = report.build_pdf([{"#": "1", "Notes": "</para><b>x", "Title": "&amp;"}])
    assert pdf.startswith(b"%PDF")


# The report is split into "to check" and "verified", and paired values are
# collapsed to one side when both agree — that is what keeps it short.
def test_report_groups_and_collapses():
    from app import report

    review = {"#": 2, "Status": "Verified", "Priority": "Review"}
    fuzzy = {"#": 3, "Status": "Fuzzy match", "Priority": ""}
    orphan = {"#": "", "Status": "Cited, missing from reference list", "Priority": "Review"}
    clean = {"#": 1, "Status": "Verified", "Priority": ""}
    assert [report.needs_checking(r) for r in (review, fuzzy, orphan, clean)] == [True, True, True, False]

    assert report._collapse("Year ref / OA", "1948 / 1948") == ("Year", "1948")
    assert report._collapse("Year ref / OA", "/") == ("Year", "")
    name, value = report._collapse("Year ref / OA", "2015 / 2016")
    assert name == "Year" and "2015" in value and "2016" in value
    name, value = report._collapse("Year ref / OA", "2021 /")
    assert name == "Year" and "not in OpenAlex" in value
    name, value = report._collapse("DOI ref / OA", "/ 10.1109/CVPR.2016.90")
    assert name == "DOI" and value.startswith("10.1109/CVPR.2016.90")     # split on " / ", not "/"
    assert report._collapse("Notes", "a / b") == ("Notes", "a / b")       # only paired columns

    # A clean reference costs one line; one needing a look keeps its detail.
    rows = [clean | {"Reference (as printed)": "Clean", "Year ref / OA": "1948 / 1948"},
            review | {"Reference (as printed)": "Suspect", "Year ref / OA": "2015 / 2016"}]
    assert report.build_pdf(rows).startswith(b"%PDF")


def test_report_font_draws_european_names():
    """Built-in PDF fonts are Latin-1: 'Kaiser, Ł.' would print as a black box."""
    from app import report
    assert report._portable("Kaiser, Ł. · Čapek — “x”") == "Kaiser, Ł. · Čapek — “x”"
    assert "α" not in report._portable("α")          # no glyph → never a black box
    assert report._portable("Erdős") in ("Erdős", "Erdos")
    assert report.FONT in report.build_pdf([{"#": 1}]).decode("latin-1")


# Link previews (LinkedIn, Slack, X) need absolute URLs — a relative og:image
# is ignored — so the pages carry {{SITE_URL}} and _page() fills it in.
def test_link_preview_metadata_is_absolute():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    for path, expected_url in (("/", f"{main.SITE_URL}/"), ("/edugenai", f"{main.SITE_URL}/edugenai")):
        html = client.get(path).text
        assert "{{" not in html                      # every token substituted
        assert f'<meta property="og:url" content="{expected_url}">' in html
        assert f'<meta property="og:image" content="{main.SITE_URL}/static/og-image.png">' in html
        assert f'<link rel="canonical" href="{expected_url}">' in html
        assert '<meta name="twitter:card" content="summary_large_image">' in html
        assert '<meta name="description" content="' in html
        assert 'og:image:width" content="1200"' in html and 'og:image:height" content="630"' in html
    r = client.get("/static/og-image.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


# Contract: the argument names the tool advertises are the ones the endpoint
# accepts. Holds whether or not any schema document is published.
def test_documented_reference_fields_are_accepted(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main, toolspec

    monkeypatch.setattr(main, "_run_batch",
                        lambda items, key: [{"index": i, "status": "found"} for i, _ in items])
    client = TestClient(main.app)
    reference = {name: {"title": "T", "authors": ["A, B"], "et_al": False, "year": 2020,
                        "doi": "10.1/x", "journal": "J", "volume": "1", "issue": "2",
                        "pages": "1-2"}[name] for name in toolspec.REFERENCE_PROPERTIES}
    r = client.post("/api/verify_batch", json={"references": [reference]})
    assert r.status_code == 200 and r.json()["count"] == 1


def test_edugenai_page_documents_the_new_flow():
    from fastapi.testclient import TestClient
    from app import main
    html = TestClient(main.app).get("/edugenai").text
    assert f"{main.SITE_URL}/mcp" in html          # the URL you register
    assert "Add extension" in html and "Streamable HTTP" in html
    assert "whitelist" in html and "edugenai@npuls.nl" in html   # the blocker, up front
    assert "Temporarily offline" not in html and "Add Action" not in html
    assert "verify_references" in html
    assert "openapi" not in html.lower()          # no schema is published


# Terms of use: reachable, linked from where a document is uploaded, and the
# app no longer publishes an auto-generated inventory of every route.
def test_terms_page_and_consent_line():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    terms = client.get("/terms")
    assert terms.status_code == 200
    for heading in ("What you are responsible for", "no warranty", "Liability",
                    "What happens to your document"):
        assert heading in terms.text, heading
    assert "{{" not in terms.text                       # tokens substituted

    home = client.get("/").text
    consent = home[home.index('id="consent"'):home.index("</p>", home.index('id="consent"'))]
    assert "confirm that you may share it" in consent and 'href="/terms"' in consent
    assert "personal data" in consent
    for path in ("/", "/edugenai"):          # the terms page does not link to itself
        assert 'href="/terms"' in client.get(path).text, path


def test_no_schema_document_is_published():
    from fastapi.testclient import TestClient
    from app import main
    client = TestClient(main.app)
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi/edugenai.json").status_code == 404
    # The one tool meant to be called from outside is described over MCP.
    assert client.post("/mcp", json={"jsonrpc": "2.0", "id": 1,
                                     "method": "tools/list"}).status_code == 200


# ---------------------------------------------------------------------------
# MCP server — eduGenAI 2's Extensions panel accepts only an MCP server URL,
# so this is the integration that has a consumer. Protocol details matter:
# a client that trips on one of them silently shows no tool at all.
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setattr(main, "_run_batch",
                        lambda items, key: [{"index": i, "status": "found", "badge": "Verified"}
                                            for i, _ in items])
    client = TestClient(main.app)

    def rpc(**message):
        message.setdefault("jsonrpc", "2.0")
        return client.post("/mcp", json=message)

    rpc.client = client
    return rpc


def test_mcp_handshake_and_tool_listing(mcp):
    from app import toolspec
    r = mcp(id=1, method="initialize", params={"protocolVersion": "2025-03-26"}).json()["result"]
    assert r["protocolVersion"] == "2025-03-26"          # a supported version is echoed back
    assert r["serverInfo"]["name"] == "phantocite"
    assert "tools" in r["capabilities"] and r["instructions"]
    unknown = mcp(id=1, method="initialize", params={"protocolVersion": "1999-01-01"}).json()
    assert unknown["result"]["protocolVersion"] == "2025-06-18"   # falls back to the latest

    tools = mcp(id=2, method="tools/list").json()["result"]["tools"]
    assert len(tools) == 1 and tools[0]["name"] == "verify_references"
    schema = tools[0]["inputSchema"]
    # Drift guard: the MCP tool and the OpenAPI operation share one description.
    assert schema["properties"]["references"]["items"]["properties"] == toolspec.REFERENCE_PROPERTIES
    assert schema["properties"]["references"]["maxItems"] == toolspec.MAX_REFERENCES
    assert tools[0]["description"] == toolspec.OPERATION_DESCRIPTION
    assert mcp(id=3, method="ping").json()["result"] == {}


def test_mcp_tool_call_returns_results(mcp):
    r = mcp(id=1, method="tools/call",
            params={"name": "verify_references",
                    "arguments": {"references": [{"title": "A"}, {"title": "B"}]}}).json()["result"]
    assert r["isError"] is False
    assert r["structuredContent"]["count"] == 2
    assert json.loads(r["content"][0]["text"]) == r["structuredContent"]
    # A gateway that flattens the array into a JSON string is still understood.
    s = mcp(id=2, method="tools/call",
            params={"name": "verify_references",
                    "arguments": {"references": '[{"title": "A"}]'}}).json()["result"]
    assert s["isError"] is False and s["structuredContent"]["count"] == 1


def test_mcp_tool_failures_are_results_not_protocol_errors(mcp):
    """The model has to see why a call failed, so failures come back as tool
    results with isError, never as JSON-RPC errors or 500s."""
    from app import toolspec
    for arguments in ({"references": []}, {}, {"references": "not json"},
                      {"references": [{"title": "x"}] * (toolspec.MAX_REFERENCES + 1)}):
        body = mcp(id=1, method="tools/call",
                   params={"name": "verify_references", "arguments": arguments}).json()
        assert "error" not in body, arguments
        assert body["result"]["isError"] is True, arguments
    unknown = mcp(id=2, method="tools/call", params={"name": "nope", "arguments": {}}).json()
    assert unknown["result"]["isError"] is True


def test_mcp_transport_details(mcp):
    # Notifications get no reply; `id: 0` is a request, not a notification.
    assert mcp(method="notifications/initialized").status_code == 202
    assert mcp(id=0, method="ping").json()["id"] == 0
    # Batches (allowed in 2025-03-26) and malformed input.
    batch = mcp.client.post("/mcp", json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                          {"jsonrpc": "2.0", "method": "notifications/x"}])
    assert batch.status_code == 200 and len(batch.json()) == 1
    assert mcp.client.post("/mcp", content=b"{not json").json()["error"]["code"] == -32700
    assert mcp(id=1, method="does/not/exist").json()["error"]["code"] == -32601
    # No server-initiated stream; nothing to tear down.
    assert mcp.client.get("/mcp").status_code == 405
    assert mcp.client.delete("/mcp").status_code == 204
    # Empty lists rather than errors for the capabilities we do not advertise.
    for method, key in (("resources/list", "resources"), ("prompts/list", "prompts")):
        assert mcp(id=1, method=method).json()["result"][key] == []


def test_mcp_endpoint_is_rate_limited():
    """/mcp sits outside /api/, but reaches the same OpenAlex budget."""
    from fastapi.testclient import TestClient
    from app import main
    assert "/mcp" in main.RATE_LIMITS
    limiter = main.RateLimiter({"/mcp": (2, 60), "*": (100, 60)}, enabled=True)
    client = TestClient(main.app)
    main._LIMITER, saved = limiter, main._LIMITER
    try:
        codes = [client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code
                 for _ in range(3)]
    finally:
        main._LIMITER = saved
    assert codes == [200, 200, 429]

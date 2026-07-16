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

    def fake_resolve(ref, api_key=None):
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

    def fake_resolve(ref, api_key=None):
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

    def fake_resolve(ref, api_key=None):
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

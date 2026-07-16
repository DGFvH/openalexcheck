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

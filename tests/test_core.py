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

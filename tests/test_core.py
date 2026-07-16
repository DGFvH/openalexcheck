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

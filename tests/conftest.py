"""Shared fixtures: the suite must never be throttled by the process-wide
OpenAlex token bucket (rate 0 disables pacing)."""

import pytest


@pytest.fixture(autouse=True)
def _unthrottled_openalex(monkeypatch):
    from app import openalex
    monkeypatch.setattr(openalex, "_BUCKET", openalex.TokenBucket(rate=0, capacity=1))


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    from app import main
    monkeypatch.setattr(main._LIMITER, "enabled", False)


@pytest.fixture(autouse=True)
def _server_llm_env(monkeypatch):
    """Default test state: server key configured, gate on with a known
    password. Tests that need the open/unconfigured states delenv explicitly."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("SITE_PASSWORD", "test-pw")


def login(client, password="test-pw"):
    r = client.post("/login", data={"password": password, "next": "/"}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return client

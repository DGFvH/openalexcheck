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
    """Default test state: the server's LLM key is configured. Tests that need
    the unconfigured state delenv explicitly."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

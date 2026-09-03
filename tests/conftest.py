"""Shared fixtures: the suite must never be throttled by the process-wide
OpenAlex token bucket (rate 0 disables pacing)."""

import pytest


@pytest.fixture(autouse=True)
def _unthrottled_openalex(monkeypatch):
    from app import openalex
    monkeypatch.setattr(openalex, "_BUCKET", openalex.TokenBucket(rate=0, capacity=1))

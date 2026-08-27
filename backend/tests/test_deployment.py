"""What the hosted deployment depends on that local development does not.

Two things only bite once this runs in a container: `/api/health` has to name
the model actually in use (it is the first thing checked after a deploy, and a
wrong answer there sends you looking in the wrong place), and the API has to be
able to serve the built frontend from its own origin, because there is no Vite
proxy in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from onemind.api.main import app, resolve_active_model
from onemind.config import settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# -- health names the live model --------------------------------------------


def test_health_reports_the_groq_model_when_groq_is_selected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_model", "qwen/qwen3.8-27b")
    assert resolve_active_model() == "qwen/qwen3.8-27b"


def test_health_reports_the_ollama_model_when_ollama_is_selected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "qwen3.5:4b")
    assert resolve_active_model() == "qwen3.5:4b"


def test_health_reports_the_bedrock_model_when_bedrock_is_selected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "bedrock")
    monkeypatch.setattr(settings, "bedrock_model_id", "anthropic.claude-x")
    assert resolve_active_model() == "anthropic.claude-x"


def test_health_endpoint_carries_the_resolved_model(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_model", "qwen/qwen3.8-27b")
    body = client.get("/api/health").json()
    assert body["provider"] == "groq"
    assert body["model"] == "qwen/qwen3.8-27b"


def test_unknown_provider_does_not_crash_health(monkeypatch) -> None:
    """Health must stay answerable even when the provider name is wrong -
    that is precisely the situation you need it to report."""
    monkeypatch.setattr(settings, "llm_provider", "nonsense")
    assert resolve_active_model() == "unknown"


# -- serving the built frontend ---------------------------------------------


def test_api_routes_still_win_when_the_frontend_is_mounted(client: TestClient) -> None:
    """The static mount sits at '/', so it must be registered after /api/*."""
    assert client.get("/api/health").status_code == 200


def test_frontend_is_not_mounted_when_static_dir_is_unset() -> None:
    """Local development serves the UI from Vite; the API must not guess."""
    from onemind.api.main import _frontend_directory

    assert _frontend_directory("") is None


def test_frontend_is_not_mounted_when_the_directory_is_missing(tmp_path: Path) -> None:
    assert _missing(tmp_path) is None


def _missing(tmp_path: Path):
    from onemind.api.main import _frontend_directory

    return _frontend_directory(str(tmp_path / "does-not-exist"))


def test_frontend_directory_resolves_when_index_html_is_present(tmp_path: Path) -> None:
    from onemind.api.main import _frontend_directory

    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert _frontend_directory(str(tmp_path)) == tmp_path


def test_frontend_directory_rejects_a_build_with_no_index(tmp_path: Path) -> None:
    """An empty dist means the frontend build silently failed; mounting it
    would serve 404s that look like a routing bug."""
    from onemind.api.main import _frontend_directory

    assert _frontend_directory(str(tmp_path)) is None

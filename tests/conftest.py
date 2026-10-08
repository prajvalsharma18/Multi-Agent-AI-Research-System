from types import SimpleNamespace

import pytest

import cache
import metrics


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / ".cache")
    monkeypatch.setattr(cache, "SCRAPE_CACHE_DIR", tmp_path / ".cache" / "scrapes")
    monkeypatch.setattr(cache, "SEARCH_CACHE_DIR", tmp_path / ".cache" / "searches")
    monkeypatch.setattr(cache, "STAGE_CACHE_DIR", tmp_path / ".cache" / "stages")
    monkeypatch.setattr(cache, "PIPELINE_CACHE_DIR", tmp_path / ".cache" / "pipeline")
    monkeypatch.setenv("PIPELINE_CACHE_ENABLED", "true")
    metrics.reset_metrics()
    yield
    metrics.reset_metrics()


@pytest.fixture
def article_metadata():
    return SimpleNamespace(title="Research article", author="A. Author", date="2026-01-02")

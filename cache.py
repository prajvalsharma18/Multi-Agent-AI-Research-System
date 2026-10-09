"""File-based cache for search, scrapes, LLM stages, and pipeline checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_ROOT = Path(__file__).parent / ".cache"
SCRAPE_CACHE_DIR = CACHE_ROOT / "scrapes"
SEARCH_CACHE_DIR = CACHE_ROOT / "searches"
STAGE_CACHE_DIR = CACHE_ROOT / "stages"
PIPELINE_CACHE_DIR = CACHE_ROOT / "pipeline"


def cache_enabled() -> bool:
    return os.getenv("PIPELINE_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")


def _hash_key(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _topic_key(topic: str) -> str:
    return _hash_key(topic.strip().lower())


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Scrape cache
# ---------------------------------------------------------------------


def get_cached_scrape(url: str) -> str | None:
    if not cache_enabled():
        return None
    path = SCRAPE_CACHE_DIR / f"{_hash_key(url)}.json"
    data = _read_json(path)
    return data.get("content") if data else None


def set_cached_scrape(url: str, content: str) -> None:
    if not cache_enabled():
        return
    path = SCRAPE_CACHE_DIR / f"{_hash_key(url)}.json"
    _write_json(path, {"url": url, "content": content})


# ---------------------------------------------------------------------
# Search cache
# ---------------------------------------------------------------------


def get_cached_search(query: str) -> str | None:
    if not cache_enabled():
        return None
    path = SEARCH_CACHE_DIR / f"{_hash_key(query.strip().lower())}.json"
    data = _read_json(path)
    return data.get("results") if data else None


def set_cached_search(query: str, results: str) -> None:
    if not cache_enabled():
        return
    path = SEARCH_CACHE_DIR / f"{_hash_key(query.strip().lower())}.json"
    _write_json(path, {"query": query, "results": results})


# ---------------------------------------------------------------------
# LLM / node stage cache
# ---------------------------------------------------------------------


def get_stage_cache(stage: str, cache_key: str) -> Any | None:
    if not cache_enabled():
        return None
    path = STAGE_CACHE_DIR / stage / f"{_hash_key(cache_key)}.json"
    data = _read_json(path)
    return data.get("value") if data else None


def set_stage_cache(stage: str, cache_key: str, value: Any) -> None:
    if not cache_enabled():
        return
    path = STAGE_CACHE_DIR / stage / f"{_hash_key(cache_key)}.json"
    _write_json(path, {"stage": stage, "key": cache_key, "value": value})


def get_node_cache(stage: str, topic: str) -> dict | None:
    cached = get_stage_cache(stage, topic.strip().lower())
    return cached if isinstance(cached, dict) else None


def set_node_cache(stage: str, topic: str, data: dict) -> None:
    set_stage_cache(stage, topic.strip().lower(), data)


# ---------------------------------------------------------------------
# Pipeline checkpoint (resume after API errors)
# ---------------------------------------------------------------------


def get_checkpoint(topic: str) -> dict | None:
    if not cache_enabled():
        return None
    path = PIPELINE_CACHE_DIR / f"{_topic_key(topic)}.json"
    data = _read_json(path)
    return data.get("state") if data else None


def merge_checkpoint(topic: str, partial: dict) -> None:
    if not cache_enabled():
        return
    from agents import get_active_model
    from research_quality import RESEARCH_DATA_VERSION, RESEARCH_PROMPT_VERSION
    existing = get_checkpoint(topic) or {"topic": topic.strip()}
    if existing.get("_research_version") != RESEARCH_DATA_VERSION:
        existing = {"topic": topic.strip()}
    models_used = set(existing.get("_llm_models", []))
    if existing.get("_llm_model"):
        models_used.add(existing["_llm_model"])
    models_used.add(get_active_model())
    existing["_llm_models"] = sorted(models_used)
    existing["_research_version"] = RESEARCH_DATA_VERSION
    existing["_research_prompt_version"] = RESEARCH_PROMPT_VERSION
    existing.pop("_llm_model", None)
    existing.update(partial)
    set_checkpoint(topic, existing)


def set_checkpoint(topic: str, state: dict) -> None:
    if not cache_enabled():
        return
    path = PIPELINE_CACHE_DIR / f"{_topic_key(topic)}.json"
    _write_json(path, {"topic": topic.strip(), "state": state})


def clear_checkpoint(topic: str) -> None:
    path = PIPELINE_CACHE_DIR / f"{_topic_key(topic)}.json"
    if path.exists():
        path.unlink()


def load_initial_state(topic: str) -> dict:
    """Merge topic with any saved checkpoint for resumable runs."""
    topic = topic.strip()
    state: dict = {"topic": topic}
    checkpoint = get_checkpoint(topic)
    if checkpoint:
        from research_quality import RESEARCH_DATA_VERSION, RESEARCH_PROMPT_VERSION
        from agents import get_model_candidates
        candidates = set(get_model_candidates())
        models_used = set(checkpoint.get("_llm_models", []))
        if checkpoint.get("_llm_model"):
            models_used.add(checkpoint["_llm_model"])
        if (
            checkpoint.get("_research_version") == RESEARCH_DATA_VERSION
            and checkpoint.get("_research_prompt_version") == RESEARCH_PROMPT_VERSION
            and models_used and models_used.issubset(candidates)
        ):
            state.update(checkpoint)
        else:
            clear_checkpoint(topic)
    return state

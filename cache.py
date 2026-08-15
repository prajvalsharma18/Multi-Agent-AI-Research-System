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
    path = SCRAPE_CACHE_DIR / f"{_hash_key(url)}.json"
    data = _read_json(path)
    return data.get("content") if data else None


def set_cached_scrape(url: str, content: str) -> None:
    path = SCRAPE_CACHE_DIR / f"{_hash_key(url)}.json"
    _write_json(path, {"url": url, "content": content})


# ---------------------------------------------------------------------
# Search cache
# ---------------------------------------------------------------------


def get_cached_search(query: str) -> str | None:
    path = SEARCH_CACHE_DIR / f"{_hash_key(query.strip().lower())}.json"
    data = _read_json(path)
    return data.get("results") if data else None


def set_cached_search(query: str, results: str) -> None:
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
# Pipeline checkpoint (resume after 429)
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
    existing = get_checkpoint(topic) or {"topic": topic.strip()}
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


USAGE_FILE = CACHE_ROOT / "gemini_daily_usage.json"


def get_daily_gemini_limit() -> int:
    return int(os.getenv("GEMINI_DAILY_LIMIT", "500"))


def record_gemini_request() -> int:
    """Increment today's Gemini request counter; return new total."""
    from datetime import date

    today = date.today().isoformat()
    data = _read_json(USAGE_FILE) or {}
    if data.get("date") != today:
        data = {"date": today, "count": 0}
    data["count"] = int(data.get("count", 0)) + 1
    _write_json(USAGE_FILE, data)
    return data["count"]


def get_daily_gemini_count() -> int:
    from datetime import date

    data = _read_json(USAGE_FILE)
    if not data or data.get("date") != date.today().isoformat():
        return 0
    return int(data.get("count", 0))


def load_initial_state(topic: str) -> dict:
    """Merge topic with any saved checkpoint for resumable runs."""
    topic = topic.strip()
    state: dict = {"topic": topic}
    checkpoint = get_checkpoint(topic)
    if checkpoint:
        state.update(checkpoint)
    return state

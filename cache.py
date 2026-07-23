"""Simple file-based cache for scraped URL content."""

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache" / "scrapes"


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def get_cached_scrape(url: str) -> str | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("content")
    except Exception:
        return None


def set_cached_scrape(url: str, content: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url)
    path.write_text(
        json.dumps({"url": url, "content": content}, ensure_ascii=False),
        encoding="utf-8",
    )

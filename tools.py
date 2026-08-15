"""Knowledge retrieval tools: Tavily search, Trafilatura scraping, parallel fetch."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import trafilatura
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

from cache import get_cached_scrape, get_cached_search, set_cached_scrape, set_cached_search
from metrics import get_metrics
from source_scoring import MIN_SOURCE_SCORE, score_label, score_url

load_dotenv()

MAX_SCRAPE_CHARS = int(os.getenv("MAX_SCRAPE_CHARS", "3000"))
MAX_PARALLEL_SCRAPES = int(os.getenv("MAX_URLS_TO_SCRAPE", "3"))


def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. Add it to a .env file in the project root."
        )
    return TavilyClient(api_key=api_key)


def dedupe_urls(urls: list[str]) -> list[str]:
    """Remove duplicate URLs while preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        normalized = url.strip().rstrip("/")
        if not normalized.startswith("http"):
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            unique.append(url.strip())
    return unique


def _parse_search_results(text: str) -> list[dict]:
    sources: list[dict] = []
    blocks = re.split(r"-{10,}", text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title = _field(block, "Title") or "N/A"
        url = _field(block, "URL") or ""
        snippet = _field(block, "Snippet") or ""
        score_str = _field(block, "Quality Score")

        if not url.startswith("http"):
            continue

        score = int(score_str.split("/")[0]) if score_str else score_url(url)
        sources.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "score": score,
        })

    return sources


def _field(block: str, name: str) -> str | None:
    match = re.search(rf"{name}:\s*(.+?)(?:\n[A-Z]|\Z)", block, re.DOTALL)
    return match.group(1).strip() if match else None


def _format_tavily_results(results: dict) -> str:
    output = []
    for i, r in enumerate(results.get("results", []), start=1):
        url = r.get("url", "N/A")
        score = score_url(url) if url.startswith("http") else 0
        snippet = (r.get("content") or "").strip()[:300]

        output.append(
            f"""Result {i}
Title: {r.get('title', 'N/A')}
URL: {url}
Quality Score: {score}/10 ({score_label(score)})
Snippet:
{snippet}"""
        )

    return ("\n" + "-" * 70 + "\n").join(output)


def _clean_extracted_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    seen: set[str] = set()
    skip_patterns = (
        "cookie", "subscribe", "newsletter", "all rights reserved",
        "privacy policy", "terms of use", "sign up", "advertisement",
    )
    for line in lines:
        if not line or len(line) < 20:
            continue
        lower = line.lower()
        if any(pat in lower for pat in skip_patterns):
            continue
        key = lower[:80]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return "\n".join(cleaned)


def _scrape_single_url(url: str) -> str:
    cached = get_cached_scrape(url)
    if cached:
        return cached

    get_metrics().log_scrape()

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Title: N/A\nURL: {url}\nContent:\nCould not fetch page."

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            deduplicate=True,
        )
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else "N/A"

        if not text or len(text.strip()) < 50:
            return f"Title: {title}\nURL: {url}\nContent:\nCould not extract article body."

        content = _clean_extracted_text(text.strip())[:MAX_SCRAPE_CHARS]
        result = f"Title: {title}\nURL: {url}\nContent:\n{content}"
        set_cached_scrape(url, result)
        return result

    except Exception as e:
        return f"Title: N/A\nURL: {url}\nContent:\nCould not scrape URL.\nReason: {e}"


@tool(return_direct=True)
def web_search(query: str) -> str:
    """Search the web. Returns titles, URLs, quality scores and snippets."""
    cached = get_cached_search(query)
    if cached:
        get_metrics().log_cache_hit("tavily_search")
        return cached

    get_metrics().log_cache_miss("tavily_search")

    try:
        tavily = _get_tavily_client()
        results = tavily.search(
            query=query,
            max_results=5,
            search_depth="basic",
            include_raw_content=False,
        )
        formatted = _format_tavily_results(results)
        set_cached_search(query, formatted)
        return formatted

    except Exception as e:
        return f"Search failed: {e}"


@tool
def scrape_url(url: str) -> str:
    """Fetch and extract the main article body from one webpage URL."""
    return _scrape_single_url(url)


@tool(return_direct=True)
def scrape_urls_batch(urls: str) -> str:
    """Scrape multiple URLs in parallel. Pass comma-separated URLs."""
    url_list = dedupe_urls([
        u.strip()
        for u in urls.split(",")
        if u.strip().startswith("http")
    ])[:MAX_PARALLEL_SCRAPES]
    if not url_list:
        return "No valid URLs provided."
    scraped = scrape_urls_parallel(url_list)
    return ("\n" + "=" * 70 + "\n").join(scraped)


def scrape_urls_parallel(urls: list[str], max_workers: int = MAX_PARALLEL_SCRAPES) -> list[str]:
    url_list = dedupe_urls(urls)[:MAX_PARALLEL_SCRAPES]
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scrape_single_url, url): url for url in url_list}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def rank_sources_from_search(search_text: str, min_score: int = MIN_SOURCE_SCORE) -> list[dict]:
    sources = _parse_search_results(search_text)
    if not sources:
        urls = re.findall(r"https?://[^\s\)\]>\"']+", search_text)
        seen: set[str] = set()
        for url in urls:
            url = url.rstrip(".,;")
            if url not in seen:
                seen.add(url)
                sources.append({
                    "title": "N/A",
                    "url": url,
                    "snippet": "",
                    "score": score_url(url),
                })

    sources.sort(key=lambda s: s["score"], reverse=True)
    return [s for s in sources if s["score"] >= min_score]

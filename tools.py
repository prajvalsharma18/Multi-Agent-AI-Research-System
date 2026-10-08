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
from source_scoring import MIN_SOURCE_SCORE, canonicalize_url, domain_for_url, score_label, score_source, score_url

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
        normalized = canonicalize_url(url)
        if not normalized.startswith("http"):
            continue
        key = normalized
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
        published_date = _field(block, "Published Date")

        if not url.startswith("http"):
            continue

        sources.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "published_date": published_date,
        })

    return sources


def _field(block: str, name: str) -> str | None:
    match = re.search(rf"{name}:\s*(.+?)(?:\n[A-Z]|\Z)", block, re.DOTALL)
    return match.group(1).strip() if match else None


def _format_tavily_results(results: dict) -> str:
    output = []
    records = results.get("results", []) if isinstance(results, dict) else []
    if not isinstance(records, list):
        return ""
    for r in records:
        if not isinstance(r, dict):
            continue
        url = r.get("url") or "N/A"
        if not isinstance(url, str):
            url = "N/A"
        score = score_url(url) if url.startswith("http") else 0
        snippet = str(r.get("content") or "").strip()[:300]

        output.append(
            f"""Result {len(output) + 1}
Title: {r.get('title') or 'N/A'}
URL: {url}
Published Date: {r.get('published_date') or 'Unknown'}
Authority Score: {score}/10 ({score_label(score)})
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
            url = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    f"Title: N/A\nURL: {url}\nContent:\nCould not scrape URL.\nReason: {exc}"
                )
    return results


def rank_sources_from_search(search_text: str, min_score: int = MIN_SOURCE_SCORE, query: str = "") -> list[dict]:
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

    unique = {}
    for source in sources:
        key = canonicalize_url(source["url"])
        if key and key not in unique:
            source["url"] = key
            unique[key] = source
    ranked = []
    seen_domains = set()
    for source in sorted(unique.values(), key=lambda s: score_url(s["url"]), reverse=True):
        domain = domain_for_url(source["url"])
        score, breakdown = score_source(source, query, domain not in seen_domains)
        authority_score = score_url(source["url"])
        source.update(domain=domain, score=score, source_score=score, authority_score=authority_score, source_score_breakdown=breakdown, query_used=query)
        # Authority is the acceptance floor. The composite score ranks sources
        # without accidentally excluding independent pages from the same domain.
        if authority_score >= min_score:
            ranked.append(source)
            seen_domains.add(domain)
    return sorted(ranked, key=lambda item: item["score"], reverse=True)

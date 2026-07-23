import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import trafilatura
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

from cache import get_cached_scrape, set_cached_scrape
from source_scoring import MIN_SOURCE_SCORE, score_label, score_url

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_SCRAPE_CHARS = 6000
MAX_PARALLEL_SCRAPES = 5


def _parse_search_results(text: str) -> list[dict]:
    """Parse web_search tool output into structured source dicts."""
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


def _scrape_single_url(url: str) -> str:
    """Scrape one URL with trafilatura + file cache."""
    cached = get_cached_scrape(url)
    if cached:
        return cached

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Title: N/A\nURL: {url}\nContent:\nCould not fetch page."

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else "N/A"

        if not text or len(text.strip()) < 50:
            return f"Title: {title}\nURL: {url}\nContent:\nCould not extract article body."

        content = text.strip()[:MAX_SCRAPE_CHARS]
        result = f"Title: {title}\nURL: {url}\nContent:\n{content}"
        set_cached_scrape(url, result)
        return result

    except Exception as e:
        return f"Title: N/A\nURL: {url}\nContent:\nCould not scrape URL.\nReason: {e}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Returns titles, URLs, quality scores and snippets from top search results.
    """

    try:
        results = tavily.search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_raw_content=False,
        )

        output = []

        for i, r in enumerate(results["results"], start=1):
            url = r.get("url", "N/A")
            score = score_url(url) if url.startswith("http") else 0
            snippet = (r.get("content") or "").strip()

            output.append(
                f"""Result {i}
Title: {r.get('title', 'N/A')}
URL: {url}
Quality Score: {score}/10 ({score_label(score)})
Snippet:
{snippet}"""
            )

        return ("\n" + "-" * 70 + "\n").join(output)

    except Exception as e:
        return f"Search failed: {e}"


@tool
def scrape_url(url: str) -> str:
    """
    Fetch and extract the main article body from a webpage URL.
    Uses trafilatura for clean article text (no nav/ads).
    Call this tool once per URL you want to read.
    """
    return _scrape_single_url(url)


def scrape_urls_parallel(urls: list[str], max_workers: int = MAX_PARALLEL_SCRAPES) -> list[str]:
    """Scrape multiple URLs in parallel (used as reader-agent fallback)."""
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scrape_single_url, url): url for url in urls}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def rank_sources_from_search(search_text: str, min_score: int = MIN_SOURCE_SCORE) -> list[dict]:
    """Parse and rank sources; filter by minimum quality score."""
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

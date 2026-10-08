from types import SimpleNamespace

import tools


def test_successful_article_extraction_and_cache(monkeypatch, article_metadata):
    monkeypatch.setattr(tools, "get_cached_scrape", lambda url: None)
    saved = {}
    monkeypatch.setattr(tools, "set_cached_scrape", lambda url, body: saved.update(url=url, body=body))
    monkeypatch.setattr(tools.trafilatura, "fetch_url", lambda url: "<article>page</article>")
    monkeypatch.setattr(tools.trafilatura, "extract", lambda *a, **k: "A detailed article with more than fifty characters of useful research content.")
    monkeypatch.setattr(tools.trafilatura, "extract_metadata", lambda page: article_metadata)
    result = tools._scrape_single_url("https://example.org/a")
    assert "Title: Research article" in result
    assert "useful research content" in result
    assert saved["body"] == result


def test_empty_fetch_and_empty_extraction_are_graceful(monkeypatch):
    monkeypatch.setattr(tools, "get_cached_scrape", lambda url: None)
    monkeypatch.setattr(tools.trafilatura, "fetch_url", lambda url: None)
    assert "Could not fetch page" in tools._scrape_single_url("https://example.org/empty")
    monkeypatch.setattr(tools.trafilatura, "fetch_url", lambda url: "<html/>")
    monkeypatch.setattr(tools.trafilatura, "extract", lambda *a, **k: "")
    monkeypatch.setattr(tools.trafilatura, "extract_metadata", lambda page: None)
    assert "Could not extract article body" in tools._scrape_single_url("https://example.org/blank")


def test_extraction_failure_is_contained(monkeypatch):
    monkeypatch.setattr(tools, "get_cached_scrape", lambda url: None)
    monkeypatch.setattr(tools.trafilatura, "fetch_url", lambda url: (_ for _ in ()).throw(RuntimeError("offline")))
    assert "Could not scrape URL" in tools._scrape_single_url("https://example.org/fail")


def test_cleaner_deduplicates_and_long_text_is_limited(monkeypatch, article_metadata):
    monkeypatch.setattr(tools, "MAX_SCRAPE_CHARS", 75)
    monkeypatch.setattr(tools, "get_cached_scrape", lambda url: None)
    monkeypatch.setattr(tools, "set_cached_scrape", lambda *args: None)
    monkeypatch.setattr(tools.trafilatura, "fetch_url", lambda url: "page")
    line = "Useful article information repeated for evidence and context."
    monkeypatch.setattr(tools.trafilatura, "extract", lambda *a, **k: f"{line}\n{line}\n" + "x" * 100)
    monkeypatch.setattr(tools.trafilatura, "extract_metadata", lambda page: article_metadata)
    result = tools._scrape_single_url("https://example.org/long")
    assert result.count(line) == 1
    assert len(result.split("Content:\n", 1)[1]) <= 75


def test_parallel_scrapes_keep_success_when_another_fails(monkeypatch):
    monkeypatch.setattr(tools, "MAX_PARALLEL_SCRAPES", 3)
    def scrape(url):
        if url.endswith("bad"):
            raise RuntimeError("worker failure")
        return f"article {url}"
    monkeypatch.setattr(tools, "_scrape_single_url", scrape)
    results = tools.scrape_urls_parallel(["https://example.org/good", "https://example.org/bad"])
    assert any("article https://example.org/good" in result for result in results)
    assert any("Could not scrape URL" in result for result in results)

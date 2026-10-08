from types import SimpleNamespace

import tools


def test_search_formats_normalized_tavily_response(monkeypatch):
    client = SimpleNamespace(search=lambda **kwargs: {"results": [
        {"title": "WHO report", "url": "https://who.int/report", "content": "Evidence"}
    ]})
    monkeypatch.setattr(tools, "_get_tavily_client", lambda: client)
    monkeypatch.setattr(tools, "get_cached_search", lambda query: None)
    stored = {}
    monkeypatch.setattr(tools, "set_cached_search", lambda query, result: stored.update(query=query, result=result))
    result = tools.web_search.invoke({"query": "health"})
    assert "Title: WHO report" in result
    assert "Authority Score: 10/10" in result
    assert stored["query"] == "health"
    assert "https://who.int/report" in stored["result"]


def test_search_uses_cache_without_calling_tavily(monkeypatch):
    monkeypatch.setattr(tools, "get_cached_search", lambda query: "cached")
    monkeypatch.setattr(tools, "_get_tavily_client", lambda: (_ for _ in ()).throw(AssertionError("network path")))
    assert tools.web_search.invoke({"query": "cached query"}) == "cached"


def test_empty_results_return_empty_text(monkeypatch):
    monkeypatch.setattr(tools, "_get_tavily_client", lambda: SimpleNamespace(search=lambda **kwargs: {"results": []}))
    monkeypatch.setattr(tools, "get_cached_search", lambda query: None)
    monkeypatch.setattr(tools, "set_cached_search", lambda *args: None)
    assert tools.web_search.invoke({"query": "nothing"}) == ""


def test_search_failure_is_reported(monkeypatch):
    monkeypatch.setattr(tools, "_get_tavily_client", lambda: SimpleNamespace(search=lambda **kwargs: 1 / 0))
    monkeypatch.setattr(tools, "get_cached_search", lambda query: None)
    assert "Search failed:" in tools.web_search.invoke({"query": "failure"})


def test_result_parser_discards_invalid_urls_and_ranks_valid_sources():
    text = "Title: Social\nURL: https://facebook.com/post\nSnippet: low\n\n" + "-" * 70 + "\n" + \
        "Title: Official\nURL: https://www.who.int/report\nSnippet: high"
    results = tools.rank_sources_from_search(text)
    assert [item["title"] for item in results] == ["Official"]
    assert results[0]["score"] == 8.8
    assert results[0]["source_score_breakdown"]["authority"] == 10


def test_composite_score_does_not_filter_relevant_pages_from_authoritative_domain():
    text = ("\n" + "-" * 70 + "\n").join(
        f"Title: Redis guide {n}\nURL: https://redis.io/guide/{n}\nSnippet: Redis caching read-through latency"
        for n in range(3)
    )
    results = tools.rank_sources_from_search(text, query="Redis caching read-through latency")
    assert len(results) == 3
    assert {item["domain"] for item in results} == {"redis.io"}
    assert all(item["score"] < 9 and item["authority_score"] == 9 for item in results)


def test_ranked_search_urls_are_normalized_after_deduplication():
    text = "Title: A\nURL: https://www.who.int/report/?utm_source=x#section\nSnippet: evidence"
    results = tools.rank_sources_from_search(text)
    assert results[0]["url"] == "https://who.int/report"


def test_url_deduplication_preserves_order_and_drops_invalid_urls():
    assert tools.dedupe_urls([
        "https://example.org/a", "https://example.org/a/", "bad", "https://other.org"
    ]) == ["https://example.org/a", "https://other.org"]


def test_malformed_tavily_result_does_not_crash_search(monkeypatch):
    monkeypatch.setattr(tools, "_get_tavily_client", lambda: SimpleNamespace(search=lambda **kwargs: {"results": [
        None, {"title": "Valid", "url": "https://who.int/article", "content": "Useful content"}
    ]}))
    monkeypatch.setattr(tools, "get_cached_search", lambda query: None)
    monkeypatch.setattr(tools, "set_cached_search", lambda *args: None)
    result = tools.web_search.invoke({"query": "malformed"})
    assert "Search failed:" not in result
    assert "Title: Valid" in result

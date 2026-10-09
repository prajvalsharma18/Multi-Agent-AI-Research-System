import cache


def test_stage_cache_miss_write_hit_and_round_trip():
    assert cache.get_stage_cache("writer", "key") is None
    value = {"report": "A report", "count": 2}
    cache.set_stage_cache("writer", "key", value)
    assert cache.get_stage_cache("writer", "key") == value


def test_stage_cache_is_model_specific_when_model_is_in_key():
    cache.set_stage_cache("writer", "gpt-a|same prompt", "A")
    cache.set_stage_cache("writer", "gpt-b|same prompt", "B")
    assert cache.get_stage_cache("writer", "gpt-a|same prompt") == "A"
    assert cache.get_stage_cache("writer", "gpt-b|same prompt") == "B"


def test_disabled_stage_cache_is_noop(monkeypatch):
    monkeypatch.setenv("PIPELINE_CACHE_ENABLED", "false")
    cache.set_stage_cache("writer", "key", "value")
    assert cache.get_stage_cache("writer", "key") is None


def test_disabled_search_and_scrape_caches_are_noops(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_CACHE_ENABLED", "false")
    monkeypatch.setattr(cache, "SEARCH_CACHE_DIR", tmp_path / "search")
    monkeypatch.setattr(cache, "SCRAPE_CACHE_DIR", tmp_path / "scrape")

    cache.set_cached_search("query", "cached search")
    cache.set_cached_scrape("https://example.org/page", "cached scrape")

    assert cache.get_cached_search("query") is None
    assert cache.get_cached_scrape("https://example.org/page") is None
    assert not (tmp_path / "search").exists()
    assert not (tmp_path / "scrape").exists()


def test_search_and_scrape_cache_round_trip_when_enabled():
    cache.set_cached_search("focused query", "search output")
    cache.set_cached_scrape("https://example.org/page", "fetched document")

    assert cache.get_cached_search("focused query") == "search output"
    assert cache.get_cached_scrape("https://example.org/page") == "fetched document"


def test_corrupted_json_is_a_cache_miss():
    path = cache.STAGE_CACHE_DIR / "writer" / f"{cache._hash_key('key')}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert cache.get_stage_cache("writer", "key") is None


def test_checkpoint_round_trip_and_clear():
    cache.set_checkpoint("Topic", {"topic": "Topic", "report": "draft"})
    assert cache.get_checkpoint(" topic ") == {"topic": "Topic", "report": "draft"}
    cache.clear_checkpoint("Topic")
    assert cache.get_checkpoint("Topic") is None


def test_checkpoint_rejected_when_saved_model_is_not_configured(monkeypatch):
    import agents

    monkeypatch.setenv("OPENAI_MODEL", "gpt-new")
    monkeypatch.delenv("OPENAI_FALLBACK_MODEL", raising=False)
    cache.set_checkpoint("Topic", {"topic": "Topic", "_llm_models": ["gpt-old"], "report": "old"})
    assert cache.load_initial_state("Topic") == {"topic": "Topic"}
    assert cache.get_checkpoint("Topic") is None


def test_checkpoint_accepts_configured_primary_or_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-new")
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "gpt-old")
    from research_quality import RESEARCH_DATA_VERSION, RESEARCH_PROMPT_VERSION
    cache.set_checkpoint("Topic", {"topic": "Topic", "_research_version": RESEARCH_DATA_VERSION, "_research_prompt_version": RESEARCH_PROMPT_VERSION, "_llm_models": ["gpt-old"], "report": "old"})
    assert cache.load_initial_state("Topic")["report"] == "old"


def test_checkpoint_rejected_when_research_prompt_version_is_stale(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-new")
    from research_quality import RESEARCH_DATA_VERSION
    cache.set_checkpoint("Topic", {"topic": "Topic", "_research_version": RESEARCH_DATA_VERSION, "_llm_models": ["gpt-new"], "report": "stale"})
    assert cache.load_initial_state("Topic") == {"topic": "Topic"}
    assert cache.get_checkpoint("Topic") is None

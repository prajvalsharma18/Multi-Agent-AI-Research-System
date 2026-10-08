import metrics


def test_initial_metrics_and_summary():
    value = metrics.PipelineMetrics()
    assert value.openai_calls == value.retries == value.cache_hits == 0
    assert "OpenAI calls this run: 0" in value.summary()
    assert value.elapsed_seconds >= 0


def test_metrics_count_calls_retries_cache_and_scrapes():
    value = metrics.PipelineMetrics()
    value.log_openai_call("Writer", "gpt-test")
    value.retries += 1
    value.log_cache_hit("search")
    value.log_cache_miss("reader")
    value.log_scrape(2)
    assert (value.openai_calls, value.retries, value.cache_hits, value.cache_misses, value.scrape_calls) == (1, 1, 1, 1, 2)
    assert "gpt-test" in value.summary()


def test_reset_metrics_returns_new_zeroed_instance():
    before = metrics.get_metrics()
    before.log_scrape()
    after = metrics.reset_metrics()
    assert after is metrics.get_metrics()
    assert after.scrape_calls == 0
    assert before is not after

import pytest

import pipeline


def test_empty_topic_is_rejected_before_graph_execution():
    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.run_research_pipeline("  ")


def test_pipeline_returns_graph_result(monkeypatch):
    import graph
    monkeypatch.setattr(graph, "research_graph", type("FakeGraph", (), {"invoke": lambda self, state: {**state, "final_report": "done"}})())
    monkeypatch.setattr("cache.load_initial_state", lambda topic: {"topic": topic})
    result = pipeline.run_research_pipeline(" topic ")
    assert result["topic"] == "topic"
    assert result["final_report"] == "done"


def test_fresh_run_clears_checkpoint_and_starts_clean(monkeypatch):
    import graph
    called = {}
    monkeypatch.setattr(graph, "research_graph", type("FakeGraph", (), {"invoke": lambda self, state: called.update(state) or state})())
    monkeypatch.setattr("cache.clear_checkpoint", lambda topic: called.update(cleared=topic))
    result = pipeline.run_research_pipeline("topic", resume=False)
    assert called == {"cleared": "topic", "topic": "topic"}
    assert result == {"topic": "topic"}


def test_pipeline_contains_graph_failure(monkeypatch):
    import graph
    def fail(state):
        raise RuntimeError("node failed")
    monkeypatch.setattr(graph, "research_graph", type("FakeGraph", (), {"invoke": staticmethod(fail)})())
    monkeypatch.setattr("cache.load_initial_state", lambda topic: {"topic": topic})
    result = pipeline.run_research_pipeline("topic")
    assert result["error"] == "node failed"

import pytest

import agents


@pytest.mark.parametrize("feedback,expected", [
    ("Revision Required: NO", False),
    ("Revision Required: YES", True),
    ("Score: 9/10", False),
    ("Key Findings Check: PASS\nSources Check: PASS\nEvidence Check: PASS", False),
])
def test_critic_revision_decision(feedback, expected):
    assert agents.critic_needs_revision(feedback) is expected


def test_empty_or_malformed_critic_feedback_requests_review():
    assert agents.critic_needs_revision("") is True
    assert agents.critic_needs_revision("garbled output") is True


def test_model_candidates_include_explicit_fallback_only(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "primary")
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "fallback")
    assert agents.get_model_candidates() == ["primary", "fallback"]
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "primary")
    assert agents.get_model_candidates() == ["primary"]


def test_missing_api_key_fails_before_model_creation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        agents._require_openai_api_key()


def test_chat_model_disables_reasoning_for_chat_completions_tool_calls(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(agents, "ChatOpenAI", FakeChatOpenAI)
    agents.get_llm.cache_clear()
    try:
        agents.get_llm()
        assert captured["model"] == "gpt-5.6-luna"
        assert captured["reasoning_effort"] == "none"
    finally:
        agents.get_llm.cache_clear()


def test_agent_prompts_require_evidence_grounding():
    assert "Never invent facts or URLs" in agents.WRITER_SYSTEM_PROMPT
    assert "use only validated claim/evidence context" in agents.REVISION_SYSTEM_PROMPT
    assert "Revision Required" in agents.CRITIC_SYSTEM_PROMPT


def test_search_agent_node_normalizes_search_output(monkeypatch):
    import graph

    class FakeAgent:
        def invoke(self, *args, **kwargs):
            return {"messages": []}

    monkeypatch.setattr(graph, "build_search_agent", lambda: FakeAgent())
    monkeypatch.setattr(graph, "invoke_with_llm_retry", lambda fn, **kwargs: {
        "messages": [agents_message("Title: Official\nURL: https://who.int/report\nSnippet: evidence")]
    })
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    monkeypatch.setattr(graph, "set_node_cache", lambda *args: None)
    result = graph.search_node({"topic": "health"})
    assert result["ranked_sources"][0]["url"] == "https://who.int/report"
    assert result["search_results"].startswith("Title: Official")


def agents_message(content):
    from langchain_core.messages import AIMessage
    return AIMessage(content=content)


def test_reader_node_handles_empty_source_list_without_llm(monkeypatch):
    import graph

    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    result = graph.reader_node({"topic": "empty", "ranked_sources": []})
    assert "No reliable sources" in result["reader_summary"]


def test_reader_node_summarizes_scraped_context(monkeypatch):
    import graph

    monkeypatch.setattr(graph, "_ranked_urls", lambda sources: ["https://who.int/report"])
    monkeypatch.setattr(graph, "_get_tool_output", lambda *args: "scraped article text")
    monkeypatch.setattr(graph, "invoke_with_llm_retry", lambda fn, **kwargs: {})
    monkeypatch.setattr(graph, "_cached_llm", lambda stage, key, fn: "summary output")
    cached_writes = []
    cached_reads = []
    monkeypatch.setattr(graph, "get_stage_cache", lambda stage, key: cached_reads.append((stage, key)) or None)
    monkeypatch.setattr(graph, "set_stage_cache", lambda *args: cached_writes.append(args))
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    result = graph.reader_node({"topic": "health", "ranked_sources": [{"url": "https://who.int/report"}]})
    assert "No claims passed evidence validation" in result["reader_summary"]
    assert result["documents"] == []
    assert cached_writes[0][1].startswith("evidence-v2|")
    assert cached_reads[0][1] == cached_writes[0][1]


def test_writer_node_uses_topic_and_research_state(monkeypatch):
    import graph

    monkeypatch.setattr(graph, "_cached_llm", lambda stage, key, fn: "draft report")
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    result = graph.writer_node({"topic": "health", "reader_summary": "evidence notes"})
    assert result["report"].startswith("draft report")
    assert result["citation_map"] == {}


def test_critic_node_approves_report_and_sets_final_report(monkeypatch):
    import graph

    feedback = "Score: 9/10\nRevision Required: NO"
    monkeypatch.setattr(graph, "_cached_llm", lambda stage, key, fn: feedback)
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    result = graph.critic_node({"topic": "health", "report": "report body"})
    assert result["feedback"] == feedback
    assert result["needs_revision"] is False
    assert result["final_report"] == "report body"


def test_critic_deterministic_quality_gate_overrides_model_approval(monkeypatch):
    import graph

    monkeypatch.setattr(graph, "_cached_llm", lambda stage, key, fn: "Score: 9/10\nRevision Required: NO")
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    result = graph.critic_node({
        "topic": "health", "report": "polished but unsupported",
        "quality_metrics": {
            "sources": {"classification": "single_low_authority_source", "total_sources": 1, "unique_domains": 1, "average_authority_score": 5},
            "evidence": {"total_claims": 2, "evidence_coverage": .5},
            "citations": {"total_claims": 2, "citation_coverage": 0, "invalid_citations": 1},
        },
    })
    assert result["needs_revision"] is True
    assert result["quality_route"] == "research"
    assert result["quality_approved"] is False
    assert "final_report" not in result
    assert result["critique"]["overall_score"] <= 5


def test_revision_node_advances_count_and_preserves_output(monkeypatch):
    import graph

    monkeypatch.setattr(graph, "_cached_llm", lambda stage, key, fn: "revised report")
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    result = graph.revision_node({
        "topic": "health", "reader_summary": "notes", "report": "draft",
        "feedback": "add evidence", "revision_count": 1,
    })
    assert result["report"].startswith("revised report")
    assert result["final_report"] == ""
    assert result["revision_count"] == 2
    assert result["needs_revision"] is False

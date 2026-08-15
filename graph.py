"""LangGraph research pipeline with reflection loop and state management."""

from __future__ import annotations

import hashlib
import os
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agents import (
    build_reader_agent,
    build_search_agent,
    critic_needs_revision,
    get_critic_chain,
    get_revision_chain,
    get_summarizer_chain,
    get_writer_chain,
)
from cache import (
    get_node_cache,
    get_stage_cache,
    merge_checkpoint,
    set_node_cache,
    set_stage_cache,
)
from export import save_report
from gemini_retry import invoke_with_gemini_retry
from metrics import get_metrics
from source_scoring import MIN_SOURCE_SCORE
from tools import dedupe_urls, rank_sources_from_search, scrape_urls_parallel

SEPARATOR = "=" * 70
MAX_URLS_TO_SCRAPE = int(os.getenv("MAX_URLS_TO_SCRAPE", "3"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "3"))
MAX_WRITER_INPUT_CHARS = int(os.getenv("MAX_WRITER_INPUT_CHARS", "7000"))


class ResearchState(TypedDict, total=False):
    topic: str
    search_results: str
    ranked_sources: list[dict]
    reader_summary: str
    report: str
    feedback: str
    needs_revision: bool
    final_report: str
    output_paths: dict[str, str]


def _log(step: str) -> None:
    print(f"\n{SEPARATOR}\n{step}\n{SEPARATOR}")


def _content_to_str(content) -> str:
    """Normalize LangChain/Gemini message content (may be str or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            elif block is not None:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _hash_text(text: str) -> str:
    text = _content_to_str(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _get_tool_output(result: dict, tool_name: str) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, ToolMessage) and msg.name == tool_name and msg.content:
            return _content_to_str(msg.content)
    return ""


def _count_tool_calls(result: dict, tool_name: str) -> int:
    return sum(
        1 for msg in result.get("messages", [])
        if isinstance(msg, ToolMessage) and msg.name == tool_name
    )


def _get_agent_final_text(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return _content_to_str(msg.content)
    return ""


def _format_source_list(sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources[:MAX_URLS_TO_SCRAPE], start=1):
        lines.append(
            f"Result {i}\n"
            f"Title: {s['title']}\n"
            f"URL: {s['url']}\n"
            f"Quality Score: {s['score']}/10\n"
            f"Snippet: {s.get('snippet', '')[:200]}"
        )
    return ("\n" + "-" * 70 + "\n").join(lines)


def _format_scraped_for_summarizer(scraped: list[str]) -> str:
    return ("\n" + "=" * 70 + "\n").join(scraped)


def _save_stage(topic: str, partial: dict) -> None:
    merge_checkpoint(topic, partial)


def _cached_llm(stage: str, cache_key: str, invoke_fn) -> str:
    cached = get_stage_cache(stage, cache_key)
    if cached is not None:
        get_metrics().log_cache_hit(stage)
        return _content_to_str(cached)

    get_metrics().log_cache_miss(stage)
    result = invoke_with_gemini_retry(invoke_fn, step=stage)
    result = _content_to_str(result)
    set_stage_cache(stage, cache_key, result)
    return result


def _ranked_urls(ranked: list[dict]) -> list[str]:
    return dedupe_urls([s["url"] for s in ranked[:MAX_URLS_TO_SCRAPE]])


def _trim_for_writer(text: str) -> str:
    text = _content_to_str(text)
    if len(text) <= MAX_WRITER_INPUT_CHARS:
        return text
    return text[:MAX_WRITER_INPUT_CHARS] + "\n\n[... truncated for token budget ...]"


def _agent_config() -> dict:
    return {"recursion_limit": AGENT_RECURSION_LIMIT}


def search_node(state: ResearchState) -> ResearchState:
    _log("STEP 1 : SEARCH AGENT")
    topic = state["topic"]

    if state.get("search_results") and state.get("ranked_sources"):
        get_metrics().log_cache_hit("search")
        print("Using checkpointed search results.")
        return {}

    cached = get_node_cache("search", topic)
    if cached:
        get_metrics().log_cache_hit("search")
        print("Using cached search results.")
        _save_stage(topic, cached)
        return cached

    get_metrics().log_cache_miss("search")
    search_agent = build_search_agent()
    result = invoke_with_gemini_retry(
        lambda: search_agent.invoke({
            "messages": [(
                "user",
                f"Search the web about: {topic}\nCall web_search once.",
            )],
        }, config=_agent_config()),
        step="Search Agent",
    )

    search_results = _content_to_str(
        _get_tool_output(result, "web_search") or _get_agent_final_text(result)
    )
    ranked = rank_sources_from_search(search_results)

    print("\nSearch Results:\n")
    print(search_results[:3000])
    print(f"\nRanked sources (score >= {MIN_SOURCE_SCORE}): {len(ranked)}")
    for s in ranked[:MAX_URLS_TO_SCRAPE]:
        print(f"  [{s['score']}/10] {s['url']}")

    output = {"search_results": search_results, "ranked_sources": ranked}
    set_node_cache("search", topic, output)
    _save_stage(topic, output)
    return output


def reader_node(state: ResearchState) -> ResearchState:
    _log("STEP 2 : READER AGENT (scrape_urls_batch tool call)")
    topic = state["topic"]

    if state.get("reader_summary"):
        get_metrics().log_cache_hit("reader")
        print("Using checkpointed reader summary.")
        return {}

    ranked = state.get("ranked_sources", [])
    if not ranked:
        summary = "No reliable sources found (all below quality threshold)."
        _save_stage(topic, {"reader_summary": summary})
        return {"reader_summary": summary}

    urls = _ranked_urls(ranked)
    reader_cache_key = f"{topic}|{'|'.join(urls)}"
    cached_summary = get_stage_cache("reader_summary", reader_cache_key)
    if cached_summary:
        get_metrics().log_cache_hit("reader")
        print("Using cached reader summary.")
        partial = {"reader_summary": _content_to_str(cached_summary)}
        _save_stage(topic, partial)
        return partial

    get_metrics().log_cache_miss("reader")
    reader_agent = build_reader_agent()

    result = invoke_with_gemini_retry(
        lambda: reader_agent.invoke({
            "messages": [(
                "user",
                f"""Research topic: "{topic}"

Call scrape_urls_batch ONCE with these comma-separated URLs:
{", ".join(urls)}""",
            )],
        }, config=_agent_config()),
        step="Reader Agent",
    )

    scraped_text = _get_tool_output(result, "scrape_urls_batch")
    if not scraped_text or "No valid URLs" in scraped_text:
        print("Reader agent skipped batch scrape — parallel fallback...")
        for url in urls:
            print(f"  Scraping: {url}")
        scraped = scrape_urls_parallel(urls)
        print(f"  -> {len(scraped)} page(s) scraped in parallel")
        scraped_text = _format_scraped_for_summarizer(scraped)

    summ_key = f"{topic}|{_hash_text(scraped_text)}"
    summary = _cached_llm(
        "summarizer",
        summ_key,
        lambda: get_summarizer_chain().invoke({
            "topic": topic,
            "count": len(urls),
            "scraped_content": scraped_text,
        }),
    )

    print("\nReader Summary:\n")
    print(summary[:3000])
    if len(summary) > 3000:
        print(f"\n... ({len(summary)} chars total)")

    set_stage_cache("reader_summary", reader_cache_key, summary)
    partial = {"reader_summary": summary}
    _save_stage(topic, partial)
    return partial


def writer_node(state: ResearchState) -> ResearchState:
    _log("STEP 3 : REPORT WRITER")
    topic = state["topic"]

    if state.get("report"):
        get_metrics().log_cache_hit("writer")
        print("Using checkpointed draft report.")
        return {}

    research = _trim_for_writer(state["reader_summary"])
    cache_key = f"{topic}|{_hash_text(research)}"
    report = _cached_llm(
        "writer",
        cache_key,
        lambda: get_writer_chain().invoke({"topic": topic, "research": research}),
    )

    print("\nDraft Report:\n")
    print(report[:2000])

    partial = {"report": report}
    _save_stage(topic, partial)
    return partial


def critic_node(state: ResearchState) -> ResearchState:
    _log("STEP 4 : RESEARCH CRITIC")
    topic = state["topic"]

    if state.get("feedback"):
        get_metrics().log_cache_hit("critic")
        print("Using checkpointed critic feedback.")
        partial: ResearchState = {}
        if "needs_revision" not in state:
            needs_revision = critic_needs_revision(state["feedback"])
            partial["needs_revision"] = needs_revision
            if not needs_revision and not state.get("final_report"):
                partial["final_report"] = state["report"]
            _save_stage(topic, partial)
        return partial

    report = state["report"]
    cache_key = _hash_text(report)
    feedback = _cached_llm(
        "critic",
        cache_key,
        lambda: get_critic_chain().invoke({"report": report}),
    )

    needs_revision = critic_needs_revision(feedback)
    print("\nCritic Feedback:\n")
    print(feedback)
    print(f"\nRevision needed: {needs_revision}")

    partial = {"feedback": feedback, "needs_revision": needs_revision}
    if not needs_revision:
        partial["final_report"] = report
        print("\nDraft approved by critic — skipping Revision agent.")

    _save_stage(topic, partial)
    return partial


def revision_node(state: ResearchState) -> ResearchState:
    _log("STEP 5 : WRITER REVISION (reflection loop)")
    topic = state["topic"]

    if state.get("final_report"):
        get_metrics().log_cache_hit("revision")
        print("Using checkpointed final report.")
        return {}

    research = _trim_for_writer(state["reader_summary"])
    report = state["report"]
    feedback = state.get("feedback", "")
    cache_key = f"{topic}|{_hash_text(report)}|{_hash_text(feedback)}"

    final_report = _cached_llm(
        "revision",
        cache_key,
        lambda: get_revision_chain().invoke({
            "topic": topic,
            "research": research,
            "report": report,
            "feedback": feedback,
        }),
    )

    print("\nFinal Report:\n")
    print(final_report[:2000])

    partial = {"final_report": final_report}
    _save_stage(topic, partial)
    return partial


def export_node(state: ResearchState) -> ResearchState:
    _log("STEP 6 : EXPORT (Markdown + PDF)")

    final_report = state.get("final_report") or state.get("report", "")
    paths = save_report(
        topic=state["topic"],
        report=final_report,
        feedback=state.get("feedback"),
    )

    print(f"\nMarkdown saved: {paths['markdown']}")
    print(f"PDF saved:      {paths['pdf']}")
    print(f"\n{get_metrics().summary()}")

    partial = {"output_paths": paths, "final_report": final_report}
    _save_stage(state["topic"], partial)
    return partial


def _route_after_critic(state: ResearchState) -> Literal["revision", "export"]:
    return "revision" if state.get("needs_revision", False) else "export"


def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revision", revision_node)
    graph.add_node("export", export_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "reader")
    graph.add_edge("reader", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"revision": "revision", "export": "export"},
    )
    graph.add_edge("revision", "export")
    graph.add_edge("export", END)

    return graph.compile()


research_graph = build_research_graph()

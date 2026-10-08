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
    get_active_model,
)
from cache import (
    get_node_cache,
    get_stage_cache,
    merge_checkpoint,
    set_node_cache,
    set_stage_cache,
)
from export import save_report
from llm_retry import invoke_with_llm_retry
from metrics import get_metrics
from source_scoring import MIN_SOURCE_SCORE
from research_models import Claim, Critique, Document, Evidence, ResearchNote, SearchResult
from research_quality import (RESEARCH_DATA_VERSION, RESEARCH_PROMPT_VERSION, build_citation_context, calculate_domain_diversity, calculate_evidence_metrics, materialize_citations, parse_documents, parse_extraction, quality_gate, validate_extraction)
from tools import dedupe_urls, rank_sources_from_search, scrape_urls_parallel

SEPARATOR = "=" * 70
MAX_URLS_TO_SCRAPE = int(os.getenv("MAX_URLS_TO_SCRAPE", "3"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "3"))
MAX_REVISION_ITERATIONS = 2
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
    revision_count: int
    documents: list[dict]
    evidence: list[dict]
    claims: list[dict]
    research_notes: list[dict]
    quality_metrics: dict
    citation_map: dict
    critique: dict


def _log(step: str) -> None:
    print(f"\n{SEPARATOR}\n{step}\n{SEPARATOR}")


def _safe_print(value: str) -> None:
    """Print external text safely on Windows consoles with legacy encodings."""
    import sys
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(str(value).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _content_to_str(content) -> str:
    """Normalize LangChain message content (may be str or list of blocks)."""
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
    cache_key = f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{cache_key}"
    cached = get_stage_cache(stage, cache_key)
    if cached is not None:
        get_metrics().log_cache_hit(stage)
        return _content_to_str(cached)
    get_metrics().log_cache_miss(stage)
    result = invoke_with_llm_retry(invoke_fn, step=stage)
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

    cached = get_node_cache("search", f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{topic}")
    if cached:
        get_metrics().log_cache_hit("search")
        print("Using cached search results.")
        _save_stage(topic, cached)
        return cached

    get_metrics().log_cache_miss("search")
    result = invoke_with_llm_retry(
        lambda: build_search_agent().invoke({
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
    ranked = [SearchResult.model_validate(source).model_dump() for source in rank_sources_from_search(search_results, query=topic)]

    print("\nSearch Results:\n")
    _safe_print(search_results[:3000])
    print(f"\nAccepted sources (authority >= {MIN_SOURCE_SCORE}/10; sorted by composite score): {len(ranked)}")
    for s in ranked[:MAX_URLS_TO_SCRAPE]:
        print(f"  [{s['score']}/10] {s['url']}")

    search_metrics = {
        "queries": 1,
        "results": sum(1 for line in search_results.splitlines() if line.startswith("Result ")),
        "accepted_results": len(ranked),
        "unique_domains": len({source.get("domain") for source in ranked if source.get("domain")}),
    }
    get_metrics().set_quality(search=search_metrics)
    output = {"search_results": search_results, "ranked_sources": ranked, "quality_metrics": {"search": search_metrics}}
    set_node_cache("search", f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{topic}", output)
    _save_stage(topic, output)
    return output


def reader_node(state: ResearchState) -> ResearchState:
    _log("STEP 2 : READER AGENT (scrape_urls_batch tool call)")
    topic = state["topic"]

    if state.get("reader_summary") and state.get("documents") is not None:
        get_metrics().log_cache_hit("reader")
        print("Using checkpointed reader summary.")
        return {}

    ranked = state.get("ranked_sources", [])
    if not ranked:
        summary = "No reliable sources found (all below quality threshold)."
        quality = {**state.get("quality_metrics", {}), "sources": calculate_domain_diversity([]), "evidence": calculate_evidence_metrics([], [], {"rejected": 0})}
        partial = {"reader_summary": summary, "documents": [], "evidence": [], "claims": [], "research_notes": [], "quality_metrics": quality}
        _save_stage(topic, partial)
        return partial

    urls = _ranked_urls(ranked)
    reader_cache_key = f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{topic}|{'|'.join(urls)}"
    cached_summary = get_stage_cache("reader_summary", reader_cache_key)
    if cached_summary:
        get_metrics().log_cache_hit("reader")
        print("Using cached reader summary.")
        partial = cached_summary if isinstance(cached_summary, dict) else {"reader_summary": _content_to_str(cached_summary)}
        _save_stage(topic, partial)
        return partial # type: ignore

    get_metrics().log_cache_miss("reader")
    result = invoke_with_llm_retry(
        lambda: build_reader_agent().invoke({
            "messages": [(
                "user",
                f"""Research topic: "{topic}"

Call scrape_urls_batch ONCE with these comma-separated URLs:
{", ".join(urls)}""",
            )],
        }, config=_agent_config()), # type: ignore
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
    documents = parse_documents(scraped_text, ranked)
    extraction_raw = _cached_llm(
        "summarizer",
        summ_key,
        lambda: get_summarizer_chain().invoke({
            "topic": topic,
            "count": len(urls),
            "scraped_content": scraped_text,
        }),
    )

    evidence, claims, notes, evidence_stats = validate_extraction(topic, documents, parse_extraction(extraction_raw))
    summary = build_citation_context(claims, evidence, ranked)
    source_metrics = calculate_domain_diversity(ranked, claims)
    evidence_metrics = calculate_evidence_metrics(evidence, claims, evidence_stats)
    quality_metrics = {**state.get("quality_metrics", {}), "sources": source_metrics, "evidence": evidence_metrics, "extraction": evidence_stats}
    get_metrics().set_quality(**quality_metrics)

    reader_cache_key = f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{topic}|{'|'.join(urls)}"
    set_stage_cache("reader_summary", reader_cache_key, {"reader_summary": summary, "documents": [x.model_dump() for x in documents], "evidence": [x.model_dump() for x in evidence], "claims": [x.model_dump() for x in claims], "research_notes": [x.model_dump() for x in notes], "quality_metrics": quality_metrics})
    partial = {"reader_summary": summary, "documents": [x.model_dump() for x in documents], "evidence": [x.model_dump() for x in evidence], "claims": [x.model_dump() for x in claims], "research_notes": [x.model_dump() for x in notes], "quality_metrics": quality_metrics}
    _save_stage(topic, partial)
    return partial


def writer_node(state: ResearchState) -> ResearchState:
    _log("STEP 3 : REPORT WRITER")
    topic = state["topic"]

    if state.get("report"):
        get_metrics().log_cache_hit("writer")
        print("Using checkpointed draft report.")
        return {}

    research = _trim_for_writer(build_citation_context(
        [Claim.model_validate(x) for x in state.get("claims", [])],
        [Evidence.model_validate(x) for x in state.get("evidence", [])],
        state.get("ranked_sources", []),
    ) if state.get("claims") else "No validated evidence; state evidence is insufficient.")
    cache_key = f"{topic}|{_hash_text(research)}"
    raw_report = _cached_llm(
        "writer",
        cache_key,
        lambda: get_writer_chain().invoke({"topic": topic, "research": research}),
    )

    claims = [Claim.model_validate(x) for x in state.get("claims", [])]
    report, citation_map, citation_metrics = materialize_citations(raw_report, claims, state.get("ranked_sources", []))
    quality = dict(state.get("quality_metrics", {})); quality["citations"] = citation_metrics
    print("\nDraft Report:\n")
    _safe_print(report[:2000])

    partial = {"report": report, "citation_map": citation_map, "quality_metrics": quality}
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
        lambda: get_critic_chain().invoke({"report": report + "\n\nValidated research context:\n" + (state.get("reader_summary", "No validated evidence."))}),
    )

    quality = state.get("quality_metrics", {})
    gate = quality_gate(quality.get("sources", {}), quality.get("evidence", {}), quality.get("citations", {}))
    needs_revision = critic_needs_revision(feedback) or gate["revision_required"]
    if gate["issues"]:
        feedback += "\n\nDeterministic research quality issues:\n" + "\n".join(f"- {issue}" for issue in gate["issues"])
    print("\nCritic Feedback:\n")
    _safe_print(feedback)
    print(f"\nRevision needed: {needs_revision}")

    score_match = __import__("re").search(r"Score:\s*(\d+(?:\.\d+)?)/10", feedback)
    model_critic_score = float(score_match.group(1)) if score_match else 0.0
    critic_score = model_critic_score
    if gate["revision_required"]:
        critic_score = min(critic_score, 5.0)
    critique = Critique(
        overall_score=max(0, min(10, critic_score)),
        factual_accuracy=max(0, min(10, critic_score)),
        citation_correctness=max(0, min(10, critic_score)),
        source_quality=max(0, min(10, min(critic_score, quality.get("sources", {}).get("average_source_quality", critic_score)))),
        source_diversity=max(0, min(10, critic_score)),
        evidence_coverage=max(0, min(10, critic_score)),
        completeness=max(0, min(10, critic_score)),
        revision_required=needs_revision,
        actionable_feedback=gate["issues"],
    ).model_dump()
    quality = {**quality, "critic": {"model_score": model_critic_score, "effective_score": critic_score, "revision_requested": needs_revision}}
    partial = {"feedback": feedback, "needs_revision": needs_revision, "critique": critique, "quality_metrics": quality}
    get_metrics().set_quality(model_critic_score=model_critic_score, effective_critic_score=critic_score, revision_requested=needs_revision)
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

    research = _trim_for_writer(build_citation_context(
        [Claim.model_validate(x) for x in state.get("claims", [])],
        [Evidence.model_validate(x) for x in state.get("evidence", [])],
        state.get("ranked_sources", []),
    ) if state.get("claims") else state.get("reader_summary", "No validated evidence."))
    report = state["report"]
    feedback = state.get("feedback", "")
    cache_key = f"{topic}|{_hash_text(report)}|{_hash_text(feedback)}"

    raw_final_report = _cached_llm(
        "revision",
        cache_key,
        lambda: get_revision_chain().invoke({
            "topic": topic,
            "research": research,
            "report": report,
            "feedback": feedback,
        }),
    )
    final_report, citation_map, citation_metrics = materialize_citations(
        raw_final_report, [Claim.model_validate(x) for x in state.get("claims", [])], state.get("ranked_sources", [])
    )

    print("\nFinal Report:\n")
    _safe_print(final_report[:2000])

    revision_count = state.get("revision_count", 0) + 1
    partial = {
        "report": final_report,
        # Only a passing critic should mark the report final. Clearing this
        # field lets the next critic review the revised draft and keeps the
        # revision counter advancing when another revision is still required.
        "final_report": "",
        "feedback": "",
        "needs_revision": False,
        "revision_count": revision_count,
        "citation_map": citation_map,
        "quality_metrics": {**state.get("quality_metrics", {}), "citations": citation_metrics},
    }
    _save_stage(topic, partial)
    return partial


def export_node(state: ResearchState) -> ResearchState:
    _log("STEP 6 : EXPORT (Markdown + PDF)")

    final_report = state.get("final_report") or state.get("report", "")
    quality = state.get("quality_metrics", {})
    source = quality.get("sources", {}); evidence = quality.get("evidence", {}); citations = quality.get("citations", {})
    critic = state.get("critique", {})
    critic_quality = quality.get("critic", {})
    quality_section = (
        "\n\n## Research Quality\n\n"
        f"- Sources: {source.get('unique_domains', 0)} domains / {source.get('unique_sources', 0)} sources\n"
        f"- Source diversity: {source.get('diversity_ratio', 0):.0%}\n"
        f"- Average source quality: {source.get('average_source_quality', 0):.1f}/10\n"
        f"- Evidence coverage: {evidence.get('evidence_coverage', 0):.0%}\n"
        f"- Supported claims: {evidence.get('supported_claims', 0)} / {evidence.get('total_claims', 0)}; unsupported: {evidence.get('unsupported_claims', 0)}\n"
        f"- Citation coverage: {citations.get('citation_coverage', 0):.0%}\n"
        f"- Average claim confidence: {evidence.get('average_claim_confidence', 0):.2f}\n"
        f"- Critic score: model {critic_quality.get('model_score', critic.get('overall_score', 0))}/10; effective {critic_quality.get('effective_score', critic.get('overall_score', 0))}/10\n"
    )
    final_report = final_report.rstrip() + quality_section
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
    can_revise = state.get("revision_count", 0) < MAX_REVISION_ITERATIONS
    return "revision" if state.get("needs_revision", False) and can_revise else "export"


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
    graph.add_edge("revision", "critic")
    graph.add_edge("export", END)

    return graph.compile()


research_graph = build_research_graph()

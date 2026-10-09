"""LangGraph research pipeline with reflection loop and state management."""

from __future__ import annotations

import hashlib
import os
import re
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
    cache_enabled,
    get_node_cache,
    get_stage_cache,
    merge_checkpoint,
    set_node_cache,
    set_stage_cache,
)
from export import save_report
from llm_retry import invoke_with_llm_retry
from metrics import get_metrics
from research_models import Claim, Critique, Document, Evidence, ResearchNote, SearchResult
from research_quality import (RESEARCH_DATA_VERSION, RESEARCH_PROMPT_VERSION, build_citation_context, calculate_domain_diversity, calculate_evidence_metrics, materialize_citations, normalize_text, parse_documents, parse_extraction, quality_gate, validate_extraction)
from source_scoring import MIN_SOURCE_SCORE, canonicalize_url
from tools import dedupe_urls, rank_sources_from_search, scrape_urls_parallel, web_search

SEPARATOR = "=" * 70
MAX_URLS_TO_SCRAPE = int(os.getenv("MAX_URLS_TO_SCRAPE", "3"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "3"))


def _configured_limit(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid {name}; using safe default {default}.")
        return default
    if value < 0 or value > 10:
        print(f"Out-of-range {name}; using safe default {default}.")
        return default
    return value


MAX_RESEARCH_ROUNDS = _configured_limit("MAX_RESEARCH_ROUNDS", 2)
MAX_REVISION_ITERATIONS = _configured_limit("MAX_REVISION_ITERATIONS", 2)
MAX_RECOVERY_QUERIES = 3
MAX_RECOVERY_QUERY_CHARS = 240
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
    research_rounds: int
    targeted_queries: list[str]
    processed_urls: list[str]
    recovery_progress: bool
    recovery_metrics: dict
    targeted_queries_attempted: list[str]
    quality_route: str
    critic_evaluations: list[dict]
    termination_reason: str
    unresolved_issues: list[str]
    quality_approved: bool


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


def _document_pipeline_diagnostics(documents: list[Document], attempted: int) -> dict:
    outcomes = [item.metadata.get("pipeline_outcome", "unknown") for item in documents]
    fetch_successes = sum(outcome in {"success", "extraction_failed"} for outcome in outcomes)
    fetch_failures = sum(outcome == "fetch_failed" for outcome in outcomes)
    extraction_failures = sum(outcome == "extraction_failed" for outcome in outcomes)
    known_outcomes = sum(
        outcome in {
            "success", "fetch_failed", "extraction_failed", "scrape_failed",
            "scrape_failure_unknown",
        }
        for outcome in outcomes
    )
    return {
        "scrape_attempts": attempted,
        "documents_parsed": len(documents),
        "fetch_successes": fetch_successes,
        "fetch_failures": fetch_failures,
        "extraction_attempts": fetch_successes,
        "extraction_failures": extraction_failures,
        "scrape_failure_unknown": sum(
            outcome in {"scrape_failed", "scrape_failure_unknown"}
            for outcome in outcomes
        ),
        "unreported_outcomes": max(0, attempted - known_outcomes),
        "outcomes": {
            outcome: outcomes.count(outcome)
            for outcome in sorted(set(outcomes))
        },
    }


def _save_stage(topic: str, partial: dict) -> None:
    merge_checkpoint(topic, partial)


def _cached_llm(stage: str, cache_key: str, invoke_fn) -> str:
    cache_key = f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{cache_key}"
    cached = get_stage_cache(stage, cache_key)
    if cached is not None:
        get_metrics().log_cache_hit(stage)
        return _content_to_str(cached)
    if cache_enabled():
        get_metrics().log_cache_miss(stage)
    result = invoke_with_llm_retry(invoke_fn, step=stage)
    result = _content_to_str(result)
    set_stage_cache(stage, cache_key, result)
    return result


def _ranked_urls(ranked: list[dict]) -> list[str]:
    return dedupe_urls([s["url"] for s in ranked[:MAX_URLS_TO_SCRAPE]])


def _critic_section(feedback: str, section: str) -> list[str]:
    match = re.search(
        rf"(?ims)^\s*{re.escape(section)}\s*:\s*(.*?)(?=^\s*[A-Z][A-Za-z ]+\s*:\s*|\Z)",
        feedback or "",
    )
    if not match:
        return []
    return [
        item[:MAX_RECOVERY_QUERY_CHARS].strip()
        for line in match.group(1).splitlines()
        if (item := re.sub(r"^\s*[-*]\s*", "", line).strip())
        and item.casefold() not in {"none", "n/a"}
    ]


def _targeted_queries(
    topic: str,
    gaps: list[str],
    claims: list[Claim],
    suggestions: list[str],
) -> list[str]:
    candidates = [*suggestions]
    unsupported = [claim.claim_text for claim in claims if not claim.supported and claim.claim_text.strip()]
    candidates.extend(f"reliable source evidence for {claim}" for claim in unsupported)
    if not candidates:
        if any("domain" in gap.lower() or "corroborat" in gap.lower() for gap in gaps):
            candidates.append(f"independent authoritative technical sources about {topic}")
        candidates.extend(f"source evidence for {gap} in {topic}" for gap in gaps)
        if not candidates and gaps:
            candidates.append(f"authoritative evidence about {topic}")

    queries: list[str] = []
    seen: set[str] = set()
    normalized_topic = normalize_text(topic)
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        query = re.sub(r"\s+", " ", candidate).strip(" \t\r\n-•")
        if not query or len(query) > MAX_RECOVERY_QUERY_CHARS or normalize_text(query) == normalized_topic:
            continue
        key = normalize_text(query)
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) == MAX_RECOVERY_QUERIES:
            break
    return queries


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
    defaults = {
        "research_rounds": state.get("research_rounds", 0),
        "revision_count": state.get("revision_count", 0),
        "targeted_queries": state.get("targeted_queries", []),
        "processed_urls": state.get("processed_urls", []),
    }

    if state.get("search_results") and state.get("ranked_sources"):
        get_metrics().log_cache_hit("search")
        print("Using checkpointed search results.")
        return defaults

    cached = get_node_cache("search", f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{topic}")
    if cached:
        get_metrics().log_cache_hit("search")
        print("Using cached search results.")
        _save_stage(topic, cached)
        cached.update(defaults)
        cached["processed_urls"] = list(dict.fromkeys([
            *cached.get("processed_urls", []),
            *(canonicalize_url(item.get("url", "")) for item in cached.get("ranked_sources", [])),
        ]))
        return cached

    if cache_enabled():
        get_metrics().log_cache_miss("search")
    get_metrics().log_search(recovery=False)
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
    search_diagnostics: dict = {}
    ranked = [
        SearchResult.model_validate(source).model_dump()
        for source in rank_sources_from_search(
            search_results,
            query=topic,
            diagnostics=search_diagnostics,
        )
    ]
    if search_results.startswith("Search failed:"):
        search_diagnostics["outcome"] = "search_failed"

    print("\nSearch Results:\n")
    _safe_print(search_results[:3000])
    print(f"\nAccepted sources (authority >= {MIN_SOURCE_SCORE}/10; sorted by composite score): {len(ranked)}")
    for s in ranked[:MAX_URLS_TO_SCRAPE]:
        print(f"  [{s['score']}/10] {s['url']}")

    search_metrics = {
        "queries": 1,
        "results": search_diagnostics.get("results_returned", 0),
        "accepted_results": len(ranked),
        "unique_domains": len({source.get("domain") for source in ranked if source.get("domain")}),
        "outcome": search_diagnostics.get("outcome"),
    }
    get_metrics().set_quality(search=search_metrics)
    output = {
        "search_results": search_results,
        "ranked_sources": ranked,
        "quality_metrics": {
            "search": search_metrics,
            "diagnostics": {"initial_search": search_diagnostics},
        },
    }
    output.update(defaults)
    output["processed_urls"] = list(dict.fromkeys([
        *output["processed_urls"],
        *(canonicalize_url(item.get("url", "")) for item in ranked),
    ]))
    get_metrics().set_quality(search=search_metrics)
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
        diagnostics = quality.get("diagnostics", {})
        quality["diagnostics"] = {
            **diagnostics,
            "initial_fetch": _document_pipeline_diagnostics([], attempted=0),
            "extraction": {"status": "not_run_no_accepted_sources"},
            "evidence_validation": {
                "proposals": 0,
                "accepted": 0,
                "rejected": 0,
                "rejection_reasons": {},
            },
            "claim_support": {
                "total": 0,
                "supported": 0,
                "unsupported": 0,
            },
        }
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

    if cache_enabled():
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
    extraction_diagnostics: dict = {}
    extraction_raw = _cached_llm(
        "summarizer",
        summ_key,
        lambda: get_summarizer_chain().invoke({
            "topic": topic,
            "count": len(urls),
            "scraped_content": scraped_text,
        }),
    )

    extraction = parse_extraction(extraction_raw, extraction_diagnostics)
    evidence, claims, notes, evidence_stats = validate_extraction(topic, documents, extraction)
    summary = build_citation_context(claims, evidence, ranked)
    source_metrics = calculate_domain_diversity(ranked, claims)
    evidence_metrics = calculate_evidence_metrics(evidence, claims, evidence_stats)
    diagnostics = dict(state.get("quality_metrics", {}).get("diagnostics", {}))
    diagnostics.update({
        "initial_fetch": _document_pipeline_diagnostics(documents, attempted=len(urls)),
        "extraction": extraction_diagnostics,
        "evidence_validation": {
            "proposals": evidence_stats["proposals"],
            "accepted": evidence_stats["valid"],
            "rejected": evidence_stats["rejected"],
            "rejection_reasons": evidence_stats["rejection_reasons"],
        },
        "claim_support": {
            "total": len(claims),
            "supported": sum(claim.supported for claim in claims),
            "unsupported": sum(not claim.supported for claim in claims),
            "mappings": [
                {
                    "claim_id": claim.claim_id,
                    "evidence_ids": claim.evidence_ids,
                    "source_urls": claim.source_urls,
                    "supported": claim.supported,
                }
                for claim in claims
            ],
        },
    })
    quality_metrics = {
        **state.get("quality_metrics", {}),
        "sources": source_metrics,
        "evidence": evidence_metrics,
        "extraction": evidence_stats,
        "diagnostics": diagnostics,
    }
    get_metrics().set_quality(**quality_metrics)

    reader_cache_key = f"{RESEARCH_DATA_VERSION}|{RESEARCH_PROMPT_VERSION}|{get_active_model()}|{topic}|{'|'.join(urls)}"
    set_stage_cache("reader_summary", reader_cache_key, {"reader_summary": summary, "documents": [x.model_dump() for x in documents], "evidence": [x.model_dump() for x in evidence], "claims": [x.model_dump() for x in claims], "research_notes": [x.model_dump() for x in notes], "quality_metrics": quality_metrics})
    partial = {"reader_summary": summary, "documents": [x.model_dump() for x in documents], "evidence": [x.model_dump() for x in evidence], "claims": [x.model_dump() for x in claims], "research_notes": [x.model_dump() for x in notes], "quality_metrics": quality_metrics}
    _save_stage(topic, partial)
    return partial


def _merge_recovery_evidence(
    existing_evidence: list[Evidence],
    existing_claims: list[Claim],
    new_evidence: list[Evidence],
    new_claims: list[Claim],
) -> tuple[list[Evidence], list[Claim], int]:
    evidence = list(existing_evidence)
    evidence_ids = {item.evidence_id for item in evidence}
    next_evidence_number = 1
    for item in evidence:
        match = re.fullmatch(r"E(\d+)", item.evidence_id)
        if match:
            next_evidence_number = max(next_evidence_number, int(match.group(1)) + 1)

    evidence_id_map: dict[str, str] = {}
    for item in new_evidence:
        while f"E{next_evidence_number}" in evidence_ids:
            next_evidence_number += 1
        new_id = f"E{next_evidence_number}"
        next_evidence_number += 1
        evidence_id_map[item.evidence_id] = new_id
        evidence_ids.add(new_id)
        evidence.append(item.model_copy(update={"evidence_id": new_id}))

    claims = list(existing_claims)
    by_text = {normalize_text(claim.claim_text): index for index, claim in enumerate(claims)}
    used_claim_ids = {claim.claim_id for claim in claims}
    next_claim_number = 1
    for claim in claims:
        match = re.fullmatch(r"C(\d+)", claim.claim_id)
        if match:
            next_claim_number = max(next_claim_number, int(match.group(1)) + 1)
    evidence_by_id = {item.evidence_id: item for item in evidence}

    for proposed in new_claims:
        mapped_ids = list(dict.fromkeys(
            evidence_id_map[item_id]
            for item_id in proposed.evidence_ids
            if item_id in evidence_id_map
        ))
        if not mapped_ids:
            continue
        key = normalize_text(proposed.claim_text)
        if key in by_text:
            index = by_text[key]
            current = claims[index]
            combined_ids = list(dict.fromkeys([*current.evidence_ids, *mapped_ids]))
            combined_urls = list(dict.fromkeys(
                evidence_by_id[item_id].source_url
                for item_id in combined_ids
                if item_id in evidence_by_id
            ))
            claims[index] = current.model_copy(update={
                "evidence_ids": combined_ids,
                "source_urls": combined_urls,
                "confidence": max(current.confidence, proposed.confidence),
            })
            continue
        while f"C{next_claim_number}" in used_claim_ids:
            next_claim_number += 1
        claim_id = f"C{next_claim_number}"
        next_claim_number += 1
        used_claim_ids.add(claim_id)
        urls = list(dict.fromkeys(
            evidence_by_id[item_id].source_url
            for item_id in mapped_ids
            if item_id in evidence_by_id
        ))
        claims.append(proposed.model_copy(update={
            "claim_id": claim_id,
            "evidence_ids": mapped_ids,
            "source_urls": urls,
        }))
        by_text[key] = len(claims) - 1
    return evidence, claims, len(evidence) - len(existing_evidence)


def research_recovery_node(state: ResearchState) -> ResearchState:
    """Search narrowly for research gaps and merge only validated new evidence."""
    _log("STEP 5 : TARGETED RESEARCH RECOVERY")
    topic = state["topic"]
    research_rounds = state.get("research_rounds", 0) + 1
    quality = state.get("quality_metrics", {})
    gate = quality_gate(
        quality.get("sources", {}),
        quality.get("evidence", {}),
        quality.get("citations", {}),
    )
    feedback = state.get("feedback", "")
    research_gaps = list(dict.fromkeys([
        *gate["research_gaps"],
        *_critic_section(feedback, "Research Gaps"),
    ]))
    claims = [Claim.model_validate(item) for item in state.get("claims", [])]
    suggestions = _critic_section(feedback, "Targeted Queries")
    queries = _targeted_queries(topic, research_gaps, claims, suggestions)
    query_history = list(state.get("targeted_queries", []))
    seen_queries = {normalize_text(query) for query in query_history}
    queries = [query for query in queries if normalize_text(query) not in seen_queries]
    query_history.extend(queries)

    processed_urls = {
        canonicalize_url(url)
        for url in state.get("processed_urls", [])
        if canonicalize_url(url)
    }
    candidates_by_url: dict[str, dict] = {}
    new_candidate_urls: set[str] = set()
    duplicate_urls = 0
    search_errors: list[str] = []
    recovery_search_diagnostics = {
        "search_calls": 0,
        "results_returned": 0,
        "parsed_candidates": 0,
        "invalid_url_candidates": 0,
        "duplicate_candidates": 0,
        "source_scoring_rejections": 0,
        "accepted_sources": 0,
        "search_outcomes": [],
        "candidate_decisions": [],
    }
    for query in queries:
        get_metrics().log_search(recovery=True)
        recovery_search_diagnostics["search_calls"] += 1
        try:
            search_result = _content_to_str(web_search.invoke({"query": query}))
        except Exception:
            search_errors.append("Search tool raised an exception during targeted search.")
            recovery_search_diagnostics["search_outcomes"].append("search_exception")
            continue
        if search_result.startswith("Search failed:"):
            search_errors.append("Tavily targeted search failed")
            recovery_search_diagnostics["search_outcomes"].append("search_failed")
            continue
        call_diagnostics: dict = {}
        candidates_from_search = rank_sources_from_search(
            search_result,
            min_score=0,
            query=query,
            diagnostics=call_diagnostics,
        )
        recovery_search_diagnostics["results_returned"] += call_diagnostics.get("results_returned", 0)
        recovery_search_diagnostics["parsed_candidates"] += call_diagnostics.get("parsed_candidates", 0)
        recovery_search_diagnostics["invalid_url_candidates"] += call_diagnostics.get("invalid_url_candidates", 0)
        recovery_search_diagnostics["duplicate_candidates"] += call_diagnostics.get("duplicate_candidates", 0)
        candidate_details = {
            item["url"]: item
            for item in call_diagnostics.get("candidate_decisions", [])
        }
        accepted_before_query = recovery_search_diagnostics["accepted_sources"]
        rejected_before_query = recovery_search_diagnostics["source_scoring_rejections"]
        duplicates_before_query = duplicate_urls
        for candidate in candidates_from_search:
            url = canonicalize_url(candidate.get("url", ""))
            if not url:
                continue
            detail = candidate_details.get(url, {})
            detail["query"] = query
            if url in processed_urls or url in candidates_by_url:
                duplicate_urls += 1
                detail["disposition"] = "duplicate_url"
                recovery_search_diagnostics["candidate_decisions"].append(detail)
                continue
            processed_urls.add(url)
            new_candidate_urls.add(url)
            if candidate.get("authority_score", 0) >= MIN_SOURCE_SCORE:
                candidates_by_url[url] = candidate
                detail.update(accepted=True, rejection_reason=None, disposition="accepted_for_fetch")
                recovery_search_diagnostics["accepted_sources"] += 1
            else:
                detail.update(
                    accepted=False,
                    rejection_reason="authority_below_minimum",
                    disposition="rejected_by_source_scoring",
                )
                recovery_search_diagnostics["source_scoring_rejections"] += 1
            recovery_search_diagnostics["candidate_decisions"].append(detail)
        accepted_this_query = recovery_search_diagnostics["accepted_sources"] - accepted_before_query
        rejected_this_query = recovery_search_diagnostics["source_scoring_rejections"] - rejected_before_query
        duplicates_this_query = duplicate_urls - duplicates_before_query
        call_diagnostics["accepted_sources"] = accepted_this_query
        call_diagnostics["source_scoring_rejections"] = rejected_this_query
        call_diagnostics["outcome"] = (
            "sources_accepted"
            if accepted_this_query
            else "all_candidates_rejected"
            if rejected_this_query
            else "duplicate_candidates_only"
            if duplicates_this_query
            else "no_parseable_candidates"
        )
        recovery_search_diagnostics["search_outcomes"].append(call_diagnostics["outcome"])

    candidates = sorted(
        candidates_by_url.values(),
        key=lambda source: source.get("score", 0),
        reverse=True,
    )[:MAX_URLS_TO_SCRAPE]
    new_urls = [canonicalize_url(source["url"]) for source in candidates]
    scraped = scrape_urls_parallel(new_urls) if new_urls else []
    documents = parse_documents(_format_scraped_for_summarizer(scraped), candidates)
    successful_documents = [item for item in documents if item.extraction_status == "success"]
    fetch_diagnostics = _document_pipeline_diagnostics(documents, attempted=len(new_urls))
    failed_fetches = fetch_diagnostics["fetch_failures"]
    new_evidence: list[Evidence] = []
    new_claims: list[Claim] = []
    extraction_stats = {
        "proposals": 0,
        "valid": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "claims": 0,
        "supported_claims": 0,
        "unsupported_claims": 0,
    }
    extraction_diagnostics = {
        "status": "not_run_no_successful_fetch",
        "evidence_proposals": 0,
        "claim_proposals": 0,
    }
    if successful_documents:
        scraped_text = _format_scraped_for_summarizer(scraped)
        cache_key = f"{topic}|recovery|{research_rounds}|{_hash_text(scraped_text)}"
        raw_extraction = _cached_llm(
            "recovery_summarizer",
            cache_key,
            lambda: get_summarizer_chain().invoke({
                "topic": topic,
                "count": len(successful_documents),
                "scraped_content": scraped_text,
            }),
        )
        recovery_extraction = parse_extraction(raw_extraction, extraction_diagnostics)
        new_evidence, new_claims, _, extraction_stats = validate_extraction(
            topic,
            successful_documents,
            recovery_extraction,
        )

    existing_evidence = [Evidence.model_validate(item) for item in state.get("evidence", [])]
    merged_evidence, merged_claims, evidence_added = _merge_recovery_evidence(
        existing_evidence,
        claims,
        new_evidence,
        new_claims,
    )
    has_progress = evidence_added > 0
    sources_by_url = {
        canonicalize_url(item.get("url", "")): item
        for item in state.get("ranked_sources", [])
        if canonicalize_url(item.get("url", ""))
    }
    for source in candidates:
        sources_by_url.setdefault(canonicalize_url(source["url"]), source)
    merged_sources = list(sources_by_url.values())
    source_metrics = calculate_domain_diversity(merged_sources, merged_claims)
    old_extraction = quality.get("extraction", {})
    combined_stats = {
        key: old_extraction.get(key, 0) + extraction_stats.get(key, 0)
        for key in {"proposals", "valid", "rejected"}
    }
    old_rejection_reasons = old_extraction.get("rejection_reasons", {})
    combined_stats["rejection_reasons"] = {
        key: old_rejection_reasons.get(key, 0) + extraction_stats["rejection_reasons"].get(key, 0)
        for key in set(old_rejection_reasons) | set(extraction_stats["rejection_reasons"])
    }
    evidence_metrics = calculate_evidence_metrics(merged_evidence, merged_claims, combined_stats)
    previous_recovery = state.get("recovery_metrics", {})
    previous_rounds = state.get("research_rounds", 0)
    prior_queries = state.get("targeted_queries_attempted")
    query_history_known = previous_recovery.get(
        "targeted_queries_known",
        prior_queries is not None or previous_rounds == 0,
    )
    recovery_queries = [*prior_queries, *queries] if query_history_known and prior_queries is not None else (
        [*queries] if previous_rounds == 0 else None
    )

    def accumulate_recovery_metric(key: str, current: int) -> int | None:
        previous_value = previous_recovery.get(key)
        if previous_rounds == 0:
            previous_value = 0
        if not isinstance(previous_value, int):
            return None
        return previous_value + current

    previous_diagnostic_rounds = previous_recovery.get("diagnostics_by_round", [])
    if not isinstance(previous_diagnostic_rounds, list):
        previous_diagnostic_rounds = []
    gate_after = quality_gate(
        source_metrics,
        evidence_metrics,
        {} if has_progress else quality.get("citations", {}),
    )
    prior_gate_issues = set(gate.get("issues", []))
    remaining_gate_issues = set(gate_after.get("issues", []))
    resolved_gate_issue_count = len(prior_gate_issues - remaining_gate_issues)
    new_gate_issue_count = len(remaining_gate_issues - prior_gate_issues)
    round_diagnostics = {
        "round": research_rounds,
        "queries_attempted": queries,
        "search": recovery_search_diagnostics,
        "candidate_urls_discovered": len(new_candidate_urls),
        "duplicate_urls": duplicate_urls,
        "sources_accepted_for_fetch": len(candidates),
        "fetch": fetch_diagnostics,
        "extraction": extraction_diagnostics,
        "evidence_validation": {
            "proposals": extraction_stats["proposals"],
            "accepted": extraction_stats["valid"],
            "rejected": extraction_stats["rejected"],
            "rejection_reasons": extraction_stats["rejection_reasons"],
        },
        "claim_support": {
            "claims_proposed": extraction_stats["claims"],
            "supported_claims": extraction_stats["supported_claims"],
            "unsupported_claims": extraction_stats["unsupported_claims"],
        },
        "new_validated_evidence": evidence_added,
        "quality_gate_issues_before": len(prior_gate_issues),
        "quality_gate_issues_after": len(remaining_gate_issues),
        "quality_gate_issues_resolved": resolved_gate_issue_count,
        "quality_gate_issues_added": new_gate_issue_count,
        "quality_gate_issues_improved": resolved_gate_issue_count > 0,
        "termination_reason": "progress" if evidence_added else "no_research_progress",
    }
    diagnostics = dict(quality.get("diagnostics", {}))
    diagnostics["recovery_rounds"] = [*previous_diagnostic_rounds, round_diagnostics]

    new_candidate_url_count = accumulate_recovery_metric("new_candidate_urls", len(new_candidate_urls))
    duplicate_url_count = accumulate_recovery_metric("duplicate_urls", duplicate_urls)
    newly_accepted_sources = accumulate_recovery_metric("newly_accepted_sources", len(candidates))
    successful_fetches = accumulate_recovery_metric("successful_fetches", fetch_diagnostics["fetch_successes"])
    failed_fetches_total = accumulate_recovery_metric("failed_fetches", failed_fetches)
    extraction_failures_total = accumulate_recovery_metric(
        "extraction_failures",
        fetch_diagnostics["extraction_failures"],
    )
    scrape_failure_unknown_total = accumulate_recovery_metric(
        "scrape_failure_unknown",
        fetch_diagnostics["scrape_failure_unknown"],
    )
    unreported_fetch_outcomes_total = accumulate_recovery_metric(
        "unreported_fetch_outcomes",
        fetch_diagnostics["unreported_outcomes"],
    )
    accepted_evidence = accumulate_recovery_metric("accepted_evidence", len(new_evidence))
    rejected_evidence = accumulate_recovery_metric("rejected_evidence", extraction_stats.get("rejected", 0))
    new_validated_evidence = accumulate_recovery_metric("new_validated_evidence", evidence_added)
    recovery_errors = [*previous_recovery.get("errors", []), *search_errors]
    recovery_queries_list = recovery_queries or []
    merged_quality = {
        **quality,
        "sources": source_metrics,
        "evidence": evidence_metrics,
        "extraction": combined_stats,
        "recovery": {
            "rounds_used": research_rounds,
            "queries_attempted": recovery_queries,
            "new_candidate_urls": new_candidate_url_count,
            "duplicate_urls": duplicate_url_count,
            "newly_accepted_sources": newly_accepted_sources,
            "successful_fetches": successful_fetches,
            "failed_fetches": failed_fetches_total,
            "extraction_failures": extraction_failures_total,
            "scrape_failure_unknown": scrape_failure_unknown_total,
            "unreported_fetch_outcomes": unreported_fetch_outcomes_total,
            "accepted_evidence": accepted_evidence,
            "rejected_evidence": rejected_evidence,
            "new_validated_evidence": new_validated_evidence,
            "diagnostics_by_round": [*previous_diagnostic_rounds, round_diagnostics],
            "errors": recovery_errors,
        },
        "diagnostics": diagnostics,
    }
    if has_progress:
        merged_quality["citations"] = {}
        merged_quality.pop("critic", None)
    recovery_metrics = {
        "rounds_used": research_rounds,
        "targeted_queries_attempted": recovery_queries_list,
        "targeted_queries_known": recovery_queries is not None,
        "new_candidate_urls": new_candidate_url_count,
        "duplicate_urls": duplicate_url_count,
        "newly_accepted_sources": newly_accepted_sources,
        "successful_fetches": successful_fetches,
        "failed_fetches": failed_fetches_total,
        "extraction_failures": extraction_failures_total,
        "scrape_failure_unknown": scrape_failure_unknown_total,
        "unreported_fetch_outcomes": unreported_fetch_outcomes_total,
        "accepted_evidence": accepted_evidence,
        "rejected_evidence": rejected_evidence,
        "new_validated_evidence": new_validated_evidence,
        "diagnostics_by_round": [*previous_diagnostic_rounds, round_diagnostics],
        "errors": recovery_errors,
    }
    get_metrics().set_quality(sources=source_metrics, evidence=evidence_metrics, recovery=recovery_metrics)
    if has_progress:
        get_metrics().quality.pop("critic", None)
        get_metrics().quality.pop("citations", None)
    termination_reason = "" if has_progress else "no_research_progress"
    unresolved = [] if has_progress else list(dict.fromkeys([
        *state.get("unresolved_issues", []),
        *research_gaps,
        *search_errors,
        *([] if queries else ["No safe, specific targeted query could be generated."]),
        *([] if evidence_added else ["Targeted research produced no new validated evidence."]),
    ]))
    summary = build_citation_context(merged_claims, merged_evidence, merged_sources)
    partial: ResearchState = {
        "research_rounds": research_rounds,
        "targeted_queries": query_history,
        "targeted_queries_attempted": recovery_metrics["targeted_queries_attempted"],
        "processed_urls": sorted(processed_urls),
        "recovery_metrics": recovery_metrics,
        "recovery_progress": has_progress,
        "termination_reason": termination_reason,
        "unresolved_issues": unresolved,
        "quality_approved": False,
        "ranked_sources": merged_sources,
        "documents": [*state.get("documents", []), *(item.model_dump() for item in documents)],
        "evidence": [item.model_dump() for item in merged_evidence],
        "claims": [item.model_dump() for item in merged_claims],
        "research_notes": [
            ResearchNote(
                topic=topic,
                claim=claim.claim_text,
                supporting_evidence=claim.evidence_ids,
                source_urls=claim.source_urls,
                confidence=claim.confidence,
            ).model_dump()
            for claim in merged_claims
        ],
        "reader_summary": summary,
        "quality_metrics": merged_quality,
    }
    if has_progress:
        partial.update({
            "report": "",
            "final_report": "",
            "feedback": "",
            "needs_revision": False,
            "critique": {},
            "citation_map": {},
            "output_paths": {},
        })
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


def _claim_mapping_gaps(state: ResearchState) -> list[str]:
    evidence_ids = {
        item.get("evidence_id")
        for item in state.get("evidence", [])
        if isinstance(item, dict)
    }
    gaps = []
    for raw_claim in state.get("claims", []):
        claim = Claim.model_validate(raw_claim)
        if any(item_id not in evidence_ids for item_id in claim.evidence_ids):
            gaps.append(f"Claim has missing evidence references: {claim.claim_text}")
        if claim.supported and not claim.source_urls:
            gaps.append(f"Supported claim has no validated source URL: {claim.claim_text}")
    return gaps


def _assessment(state: ResearchState, feedback: str, score_status: str = "valid") -> dict:
    quality = state.get("quality_metrics", {})
    gate = quality_gate(
        quality.get("sources", {}),
        quality.get("evidence", {}),
        quality.get("citations", {}),
    )
    mapping_gaps = _claim_mapping_gaps(state)
    research_gaps = list(dict.fromkeys([
        *gate["research_gaps"],
        *mapping_gaps,
        *_critic_section(feedback, "Research Gaps"),
    ]))
    writing_issues = _critic_section(feedback, "Writing Issues")
    targeted_queries = _critic_section(feedback, "Targeted Queries")
    critic_requests_revision = critic_needs_revision(feedback)
    if critic_requests_revision and not research_gaps and not writing_issues:
        writing_issues = [feedback.strip() or "Critic requested a revision."]
    mandatory_failures = list(dict.fromkeys([
        *gate["hard_failures"],
        *mapping_gaps,
        *(
            [f"Critic score is {score_status}; approval requires a valid score from 0 to 10."]
            if score_status != "valid" else []
        ),
    ]))
    if research_gaps:
        decision = "research"
    elif critic_requests_revision or writing_issues or mandatory_failures:
        decision = "revision"
    else:
        decision = "approved"
    return {
        "gate": gate,
        "research_gaps": research_gaps,
        "writing_issues": writing_issues,
        "targeted_queries": targeted_queries,
        "mandatory_failures": mandatory_failures,
        "decision": decision,
    }


def _parse_critic_score(feedback: str) -> tuple[float | None, float | None, str]:
    """Return (validated score, numeric raw score, parse status) without zero defaults."""
    score_line = re.search(r"(?im)^\s*Score\s*:\s*(.*?)\s*$", feedback or "")
    if not score_line:
        return None, None, "missing"
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*/\s*10", score_line.group(1))
    if not match:
        return None, None, "malformed"
    raw_score = float(match.group(1))
    if not 0 <= raw_score <= 10:
        return None, raw_score, "out_of_range"
    return raw_score, raw_score, "valid"


def critic_node(state: ResearchState) -> ResearchState:
    _log("STEP 4 : RESEARCH CRITIC")
    topic = state["topic"]
    report = state.get("report", "")

    if state.get("feedback"):
        get_metrics().log_cache_hit("critic")
        print("Using checkpointed critic feedback.")
        feedback = state["feedback"]
    else:
        report = state["report"]
        cache_key = _hash_text(report)
        feedback = _cached_llm(
            "critic",
            cache_key,
            lambda: get_critic_chain().invoke({"report": report + "\n\nValidated research context:\n" + (state.get("reader_summary", "No validated evidence."))}),
        )

    model_critic_score, raw_model_critic_score, score_status = _parse_critic_score(feedback)
    assessment = _assessment(state, feedback, score_status)
    gate = assessment["gate"]
    quality = dict(state.get("quality_metrics", {}))
    needs_revision = assessment["decision"] != "approved"
    if gate["issues"]:
        feedback += "\n\nDeterministic research quality issues:\n" + "\n".join(f"- {issue}" for issue in gate["issues"])
    if assessment["mandatory_failures"]:
        feedback += "\n\nMandatory validation failures:\n" + "\n".join(
            f"- {issue}" for issue in assessment["mandatory_failures"]
        )
    print("\nCritic Feedback:\n")
    _safe_print(feedback)

    critic_score = model_critic_score
    if critic_score is not None and (gate["revision_required"] or assessment["mandatory_failures"]):
        critic_score = min(critic_score, 5.0)
    research_rounds = state.get("research_rounds", 0)
    revision_count = state.get("revision_count", 0)
    route = assessment["decision"]
    termination_reason = ""
    if assessment["decision"] == "research" and research_rounds >= MAX_RESEARCH_ROUNDS:
        has_revision_work = bool(
            assessment["writing_issues"] or assessment["mandatory_failures"]
        )
        if has_revision_work and revision_count < MAX_REVISION_ITERATIONS:
            route = "revision"
        elif has_revision_work and revision_count >= MAX_REVISION_ITERATIONS:
            termination_reason = "research_and_revision_budgets_exhausted"
        else:
            termination_reason = "research_budget_exhausted"
    elif assessment["decision"] == "revision" and revision_count >= MAX_REVISION_ITERATIONS:
        termination_reason = (
            "hard_validation_failure"
            if assessment["mandatory_failures"]
            else "revision_budget_exhausted"
        )
    print(f"\nQuality route: {route}")
    can_approve = assessment["decision"] == "approved"
    unresolved = [] if can_approve else list(dict.fromkeys([
        *assessment["research_gaps"],
        *assessment["writing_issues"],
        *assessment["mandatory_failures"],
    ]))
    source_quality_score = quality.get("sources", {}).get("average_source_quality")
    critique = Critique(
        overall_score=critic_score,
        factual_accuracy=critic_score,
        citation_correctness=critic_score,
        source_quality=(
            min(critic_score, source_quality_score)
            if critic_score is not None and isinstance(source_quality_score, (int, float))
            else critic_score
        ),
        source_diversity=critic_score,
        evidence_coverage=critic_score,
        completeness=critic_score,
        research_gaps=assessment["research_gaps"],
        writing_issues=assessment["writing_issues"],
        targeted_queries=assessment["targeted_queries"],
        approval=can_approve,
        revision_required=needs_revision,
        actionable_feedback=list(dict.fromkeys([
            *gate["issues"],
            *assessment["writing_issues"],
            *assessment["mandatory_failures"],
        ])),
    ).model_dump()
    evaluation = {
        "model_score": model_critic_score,
        "raw_model_score": raw_model_critic_score,
        "score_status": score_status,
        "effective_score": critic_score,
        "approval": can_approve,
        "decision": route,
        "assessment": assessment["decision"],
        "gate_failures": gate["issues"],
        "research_gaps": assessment["research_gaps"],
        "writing_issues": assessment["writing_issues"],
        "research_round": research_rounds,
        "revision_count": revision_count,
    }
    evaluations = [*state.get("critic_evaluations", []), evaluation]
    quality = {**quality, "critic": {
        "model_score": model_critic_score,
        "raw_model_score": raw_model_critic_score,
        "score_status": score_status,
        "effective_score": critic_score,
        "revision_requested": needs_revision,
        "approval": can_approve,
        "decision": route,
        "gate_failures": gate["issues"],
    }, "evaluations": evaluations}
    partial: ResearchState = {
        "feedback": feedback,
        "needs_revision": needs_revision,
        "quality_route": route,
        "quality_approved": can_approve,
        "unresolved_issues": unresolved,
        "termination_reason": termination_reason,
        "critic_evaluations": evaluations,
        "critique": critique,
        "quality_metrics": quality,
    }
    get_metrics().set_quality(
        model_critic_score=model_critic_score,
        raw_model_critic_score=raw_model_critic_score,
        critic_score_status=score_status,
        effective_critic_score=critic_score,
        revision_requested=needs_revision,
        quality_route=route,
        final_gate_passed=not gate["revision_required"] and not assessment["mandatory_failures"],
    )
    if can_approve:
        partial["final_report"] = report
        print("\nDraft approved by critic — skipping Revision agent.")
        partial["termination_reason"] = "approved"
    elif termination_reason:
        partial["final_report"] = state.get("final_report") or state.get("report", "")
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
    recovery = quality.get("recovery", {})
    final_gate = quality_gate(source, evidence, citations)
    approved = bool(state.get("quality_approved")) and not final_gate["revision_required"]
    termination_reason = state.get("termination_reason") or (
        "approved" if approved else "quality_requirements_unresolved"
    )
    unresolved = list(dict.fromkeys([
        *state.get("unresolved_issues", []),
        *([] if approved else final_gate["issues"]),
    ]))
    evaluations = state.get("critic_evaluations", [])
    latest_evaluation = evaluations[-1] if evaluations else {}
    model_score = critic_quality.get("model_score", latest_evaluation.get("model_score"))
    effective_score = critic_quality.get(
        "effective_score",
        latest_evaluation.get("effective_score", critic.get("overall_score")),
    )

    def percentage(value) -> str:
        return f"{value:.0%}" if isinstance(value, (int, float)) else "not measured"

    def decimal(value, places: int = 1, suffix: str = "") -> str:
        return f"{value:.{places}f}{suffix}" if isinstance(value, (int, float)) else "not measured"

    def count(value, default: int | None = None) -> str:
        if isinstance(value, int):
            return str(value)
        return str(default) if default is not None else "not measured"

    recovery_rounds = state.get("research_rounds", 0)
    recovery_default = 0 if recovery_rounds == 0 else None
    recovery_query_count = (
        len(recovery["queries_attempted"])
        if isinstance(recovery.get("queries_attempted"), list)
        else recovery_default
    )
    quality_section = (
        "\n\n## Research Quality\n\n"
        f"- Quality status: {'APPROVED' if approved else 'NOT APPROVED'}\n"
        f"- Mandatory quality gates: {'passed' if not final_gate['revision_required'] else 'failed'}\n"
        f"- Termination reason: {termination_reason}\n"
        f"- Research recovery rounds: {recovery_rounds}\n"
        f"- Recovery queries / candidate URLs / new sources: {count(recovery_query_count)} / "
        f"{count(recovery.get('new_candidate_urls'), recovery_default)} / "
        f"{count(recovery.get('newly_accepted_sources'), recovery_default)}\n"
        f"- Duplicate recovery URLs / successful / failed fetches: "
        f"{count(recovery.get('duplicate_urls'), recovery_default)} / "
        f"{count(recovery.get('successful_fetches'), recovery_default)} / "
        f"{count(recovery.get('failed_fetches'), recovery_default)}\n"
        f"- Recovery evidence accepted / rejected: "
        f"{count(recovery.get('accepted_evidence'), recovery_default)} / "
        f"{count(recovery.get('rejected_evidence'), recovery_default)}\n"
        f"- Writer revisions: {state.get('revision_count', 0)}\n"
        f"- Sources: {count(source.get('unique_domains'))} domains / {count(source.get('unique_sources'))} sources\n"
        f"- Source diversity: {percentage(source.get('diversity_ratio'))}\n"
        f"- Average source quality: {decimal(source.get('average_source_quality'), suffix='/10')}\n"
        f"- Evidence coverage: {percentage(evidence.get('evidence_coverage'))}\n"
        f"- Supported claims: {count(evidence.get('supported_claims'))} / {count(evidence.get('total_claims'))}; "
        f"unsupported: {count(evidence.get('unsupported_claims'))}\n"
        f"- Citation coverage: {percentage(citations.get('citation_coverage'))}\n"
        f"- Invalid citations: {count(citations.get('invalid_citations'))}\n"
        f"- Average claim confidence: {decimal(evidence.get('average_claim_confidence'), 2)}\n"
        f"- Critic score: model {decimal(model_score, suffix='/10')}; "
        f"effective {decimal(effective_score, suffix='/10')}\n"
        f"- Unresolved issues: {'; '.join(unresolved) if unresolved else 'None'}\n"
    )
    final_report = final_report.rstrip() + quality_section
    quality = {
        **quality,
        "final_gate_passed": not final_gate["revision_required"],
        "final_quality_approved": approved,
        "termination_reason": termination_reason,
        "unresolved_issues": unresolved,
    }
    get_metrics().set_quality(
        final_gate_passed=not final_gate["revision_required"],
        final_quality_approved=approved,
        termination_reason=termination_reason,
        unresolved_issues=unresolved,
    )
    paths = save_report(
        topic=state["topic"],
        report=final_report,
        feedback=state.get("feedback"),
    )

    print(f"\nMarkdown saved: {paths['markdown']}")
    print(f"PDF saved:      {paths['pdf']}")
    print(f"\n{get_metrics().summary()}")

    partial = {
        "output_paths": paths,
        "final_report": final_report,
        "quality_metrics": quality,
        "quality_approved": approved,
        "termination_reason": termination_reason,
        "unresolved_issues": unresolved,
    }
    _save_stage(state["topic"], partial)
    return partial


def _route_after_critic(state: ResearchState) -> Literal["research_recovery", "revision", "export"]:
    decision = state.get("quality_route")
    if decision == "research":
        return "research_recovery" if state.get("research_rounds", 0) < MAX_RESEARCH_ROUNDS else "export"
    if decision == "approved":
        return "export"
    if decision == "revision":
        return "revision" if state.get("revision_count", 0) < MAX_REVISION_ITERATIONS else "export"
    can_revise = state.get("revision_count", 0) < MAX_REVISION_ITERATIONS
    return "revision" if state.get("needs_revision", False) and can_revise else "export"


def _route_after_recovery(state: ResearchState) -> Literal["writer", "export"]:
    return "writer" if state.get("recovery_progress", False) else "export"


def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revision", revision_node)
    graph.add_node("research_recovery", research_recovery_node)
    graph.add_node("export", export_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "reader")
    graph.add_edge("reader", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "research_recovery": "research_recovery",
            "revision": "revision",
            "export": "export",
        },
    )
    graph.add_edge("revision", "critic")
    graph.add_conditional_edges(
        "research_recovery",
        _route_after_recovery,
        {"writer": "writer", "export": "export"},
    )
    graph.add_edge("export", END)

    return graph.compile()


research_graph = build_research_graph()

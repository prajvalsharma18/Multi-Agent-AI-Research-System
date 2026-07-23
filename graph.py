"""LangGraph research pipeline with reflection loop and state management."""

from typing import TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    revision_chain,
    summarizer_chain,
    writer_chain,
)
from export import save_report
from source_scoring import MIN_SOURCE_SCORE
from tools import rank_sources_from_search, scrape_urls_parallel

SEPARATOR = "=" * 70
MAX_URLS_TO_SCRAPE = 5
READER_FALLBACK_THRESHOLD = 3  # min scrape_url calls before fallback kicks in


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    topic: str
    search_results: str
    ranked_sources: list[dict]
    reader_summary: str
    report: str
    feedback: str
    final_report: str
    output_paths: dict[str, str]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _log(step: str) -> None:
    print(f"\n{SEPARATOR}\n{step}\n{SEPARATOR}")


def _get_tool_output(result: dict, tool_name: str) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, ToolMessage) and msg.name == tool_name and msg.content:
            return msg.content
    return ""


def _count_tool_calls(result: dict, tool_name: str) -> int:
    return sum(
        1 for msg in result.get("messages", [])
        if isinstance(msg, ToolMessage) and msg.name == tool_name
    )


def _get_agent_final_text(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
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


# ---------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------

def search_node(state: ResearchState) -> ResearchState:
    _log("STEP 1 : SEARCH AGENT")

    search_agent = build_search_agent()
    result = search_agent.invoke({
        "messages": [(
            "user",
            f"""Search the web about: {state["topic"]}

Requirements:
- use official sources whenever possible
- include at least 5 sources
- prioritize government websites
- include recent news
- include Wikipedia only if official sources are unavailable

Return title, URL and summary for each result.""",
        )],
    })

    search_results = _get_tool_output(result, "web_search") or _get_agent_final_text(result)
    ranked = rank_sources_from_search(search_results)

    print("\nSearch Results:\n")
    print(search_results[:3000])

    print(f"\nRanked sources (score >= {MIN_SOURCE_SCORE}): {len(ranked)}")
    for s in ranked[:MAX_URLS_TO_SCRAPE]:
        print(f"  [{s['score']}/10] {s['url']}")

    return {
        "search_results": search_results,
        "ranked_sources": ranked,
    }


def reader_node(state: ResearchState) -> ResearchState:
    _log("STEP 2 : READER AGENT (scrape_url tool calls)")

    ranked = state.get("ranked_sources", [])
    if not ranked:
        return {"reader_summary": "No reliable sources found (all below quality threshold)."}

    source_list = _format_source_list(ranked)
    reader_agent = build_reader_agent()

    result = reader_agent.invoke({
        "messages": [(
            "user",
            f"""Research topic: "{state['topic']}"

Below are ranked search results. You MUST call scrape_url on the top 3–5 URLs
with Quality Score >= {MIN_SOURCE_SCORE}. Do NOT answer until you have scraped them.

{source_list}

After scraping, produce the structured multi-source summary.""",
        )],
    })

    scrape_count = _count_tool_calls(result, "scrape_url")
    print(f"\nscrape_url tool calls by agent: {scrape_count}")

    if scrape_count >= READER_FALLBACK_THRESHOLD:
        summary = _get_agent_final_text(result)
        if not summary:
            tool_outputs = [
                msg.content for msg in result["messages"]
                if isinstance(msg, ToolMessage) and msg.name == "scrape_url"
            ]
            summary = summarizer_chain.invoke({
                "topic": state["topic"],
                "count": len(tool_outputs),
                "scraped_content": _format_scraped_for_summarizer(tool_outputs),
            })
    else:
        # Agent skipped tools — parallel fallback scrape + summarizer
        print(f"Agent scraped fewer than {READER_FALLBACK_THRESHOLD} URLs — parallel fallback...")
        urls = [s["url"] for s in ranked[:MAX_URLS_TO_SCRAPE]]
        for url in urls:
            print(f"  Scraping: {url}")
        scraped = scrape_urls_parallel(urls)
        print(f"  -> {len(scraped)} pages scraped in parallel")

        summary = summarizer_chain.invoke({
            "topic": state["topic"],
            "count": len(scraped),
            "scraped_content": _format_scraped_for_summarizer(scraped),
        })

    print("\nReader Summary:\n")
    print(summary[:3000])
    if len(summary) > 3000:
        print(f"\n... ({len(summary)} chars total)")

    return {"reader_summary": summary}


def writer_node(state: ResearchState) -> ResearchState:
    _log("STEP 3 : REPORT WRITER")

    report = writer_chain.invoke({
        "topic": state["topic"],
        "research": state["reader_summary"],
    })

    print("\nDraft Report:\n")
    print(report[:2000])

    return {"report": report}


def critic_node(state: ResearchState) -> ResearchState:
    _log("STEP 4 : RESEARCH CRITIC")

    feedback = critic_chain.invoke({"report": state["report"]})

    print("\nCritic Feedback:\n")
    print(feedback)

    return {"feedback": feedback}


def revision_node(state: ResearchState) -> ResearchState:
    _log("STEP 5 : WRITER REVISION (reflection loop)")

    final_report = revision_chain.invoke({
        "topic": state["topic"],
        "research": state["reader_summary"],
        "report": state["report"],
        "feedback": state["feedback"],
    })

    print("\nFinal Report:\n")
    print(final_report[:2000])

    return {"final_report": final_report}


def export_node(state: ResearchState) -> ResearchState:
    _log("STEP 6 : EXPORT (Markdown + PDF)")

    paths = save_report(
        topic=state["topic"],
        report=state["final_report"],
        feedback=state.get("feedback"),
    )

    print(f"\nMarkdown saved: {paths['markdown']}")
    print(f"PDF saved:      {paths['pdf']}")

    return {"output_paths": paths}


# ---------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------

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
    graph.add_edge("critic", "revision")
    graph.add_edge("revision", "export")
    graph.add_edge("export", END)

    return graph.compile()


research_graph = build_research_graph()

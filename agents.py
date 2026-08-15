"""Gemini-powered agents and chains for the research pipeline."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from source_scoring import MIN_SOURCE_SCORE
from tools import scrape_url, scrape_urls_batch, web_search

load_dotenv()

# Free-tier defaults (user quotas): gemini-3.5-flash-lite → 15 RPM / 500 RPD
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)

_active_model: str | None = None


def _require_google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Add it to a .env file in the project root."
        )
    return key


def get_model_candidates() -> list[str]:
    """Preferred model first, then fallbacks (deduplicated)."""
    preferred = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    seen: set[str] = set()
    ordered: list[str] = []
    for model in (preferred, *FALLBACK_GEMINI_MODELS):
        if model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def set_active_model(model: str) -> None:
    global _active_model
    _active_model = model


def get_active_model() -> str:
    return _active_model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def clear_agent_caches() -> None:
    get_llm.cache_clear()
    get_writer_chain.cache_clear()
    get_revision_chain.cache_clear()
    get_summarizer_chain.cache_clear()
    get_critic_chain.cache_clear()
    build_search_agent.cache_clear()
    build_reader_agent.cache_clear()


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return a shared Gemini LLM instance (lazy-loaded)."""
    return ChatGoogleGenerativeAI(
        model=get_active_model(),
        temperature=0,
        google_api_key=_require_google_api_key(),
        max_retries=1,
    )


def get_search_llm() -> ChatGoogleGenerativeAI:
    """Gemini with forced web_search tool calling for the search agent."""
    return get_llm().bind_tools([web_search], tool_choice="any")


def get_reader_llm() -> ChatGoogleGenerativeAI:
    """Gemini with forced batch scrape tool calling for the reader agent."""
    return get_llm().bind_tools([scrape_urls_batch], tool_choice="any")


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

SEARCH_SYSTEM_PROMPT = """You are a web research agent. Call web_search once with a focused query.
The tool returns formatted results — return them as-is. Do not add extra commentary."""

READER_SYSTEM_PROMPT = f"""You are a research reader agent. Your ONLY tool is scrape_urls_batch.

Call scrape_urls_batch ONCE with a comma-separated list of the top URLs (Quality Score >= {MIN_SOURCE_SCORE}).
The tool returns scraped page text — return it as-is. Do not summarize."""

WRITER_SYSTEM_PROMPT = f"""You are an expert research analyst and technical writer.

Rules:
- Use ONLY the reader summary provided — no other sources
- Only cite sources with quality score >= {MIN_SOURCE_SCORE}
- Never invent facts
- Write professionally with clear structure"""

REVISION_SYSTEM_PROMPT = f"""You are an expert research analyst revising a report based on critic feedback.

Rules:
- Fix every issue raised by the critic
- Keep only sources with quality score >= {MIN_SOURCE_SCORE}
- Do not invent new facts — use only the original research summary
- Improve clarity, structure, and completeness"""

CRITIC_SYSTEM_PROMPT = """You are a senior research reviewer. Be efficient.

If the report has Introduction, Key Findings (3+), Conclusion, Sources with URLs:
- Score 8–10 and Verdict: approve as-is unless there is a clear factual gap.

Only request revision for missing sections, weak citations, or factual issues."""

SUMMARIZER_SYSTEM_PROMPT = """You synthesize scraped webpage content into brief structured research notes.

For EACH source, output:

### Source N: [Title]
URL: ...
Quality Score: .../10
Key Points:
- (2–3 bullets max)

Use ONLY the scraped content. Be concise."""

# ---------------------------------------------------------------------
# Agents (tool-calling)
# ---------------------------------------------------------------------


def get_revision_score_threshold() -> int:
    return int(os.getenv("REVISION_SCORE_THRESHOLD", "8"))


def critic_needs_revision(feedback: str) -> bool:
    """Return True when the Critic identifies issues requiring Revision."""
    score_match = re.search(r"Score:\s*(\d+)/10", feedback)
    score = int(score_match.group(1)) if score_match else 0
    if score >= get_revision_score_threshold():
        return False

    lower = feedback.lower()
    skip_phrases = (
        "no revision needed",
        "no changes needed",
        "ready to publish",
        "approve as-is",
        "approved as-is",
    )
    if any(phrase in lower for phrase in skip_phrases):
        return False

    improve_section = re.search(
        r"areas to improve:\s*(.+?)(?:\nverdict:|\Z)",
        feedback,
        re.IGNORECASE | re.DOTALL,
    )
    if improve_section:
        bullets = [
            line.strip()
            for line in improve_section.group(1).splitlines()
            if line.strip().startswith("-")
        ]
        trivial = {"-", "- none", "- n/a", "- none."}
        if bullets and all(b.lower() in trivial for b in bullets):
            return False

    return True


@lru_cache(maxsize=1)
def build_search_agent():
    """Search agent: Tavily web_search via Gemini tool calling."""
    return create_agent(
        model=get_search_llm(),
        tools=[web_search],
        system_prompt=SEARCH_SYSTEM_PROMPT,
    )


@lru_cache(maxsize=1)
def build_reader_agent():
    """Reader agent: batch scrape tool call + structured summarization."""
    return create_agent(
        model=get_reader_llm(),
        tools=[scrape_urls_batch],
        system_prompt=READER_SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------
# Chains (Writer, Critic, Revision, Summarizer)
# ---------------------------------------------------------------------

_writer_prompt = ChatPromptTemplate.from_messages([
    ("system", WRITER_SYSTEM_PROMPT),
    (
        "human",
        """
Topic: {topic}

Research Summary (from reader agent — use ONLY this):
{research}

Write a concise report (keep each section brief):

# Introduction

# Key Findings
(3 findings, each backed by a source)

# Conclusion

# Sources
(List URLs from the research summary with quality scores)
""",
    ),
])

_revision_prompt = ChatPromptTemplate.from_messages([
    ("system", REVISION_SYSTEM_PROMPT),
    (
        "human",
        """
Topic: {topic}

Original Research Summary:
{research}

Draft Report:
{report}

Critic Feedback:
{feedback}

Write the IMPROVED final report addressing all critic feedback.
Use the same format: Introduction, Key Findings, Conclusion, Sources.
""",
    ),
])

_summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZER_SYSTEM_PROMPT),
    (
        "human",
        """
Topic: {topic}

Scraped content from {count} sources:
{scraped_content}
""",
    ),
])

_critic_prompt = ChatPromptTemplate.from_messages([
    ("system", CRITIC_SYSTEM_PROMPT),
    (
        "human",
        """
Review the following report.

Report:
{report}

Return exactly:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ... (use "- None" if no issues)

Verdict:
(approve as-is OR list required fixes)
""",
    ),
])


@lru_cache(maxsize=1)
def get_writer_chain():
    return _writer_prompt | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_revision_chain():
    return _revision_prompt | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_summarizer_chain():
    return _summarizer_prompt | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_critic_chain():
    return _critic_prompt | get_llm() | StrOutputParser()

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
- Never invent facts or URLs
- Copy every URL EXACTLY as it appears in the research summary — do not modify, shorten, or replace
- In the Sources section, use clickable Markdown links:
  - [Source Title](EXACT_URL) — Quality Score: X/10
- Write professionally with clear structure"""

REVISION_SYSTEM_PROMPT = f"""You are an expert research analyst revising a report based on critic feedback.

Rules:
- Fix only genuine issues raised by the critic — do not rewrite content that already passes
- Keep only sources with quality score >= {MIN_SOURCE_SCORE}
- Do not invent new facts — use only the original research summary
- Preserve exact URLs from the research summary — never modify or invent URLs
- Sources must use clickable Markdown: [Source Title](EXACT_URL) — Quality Score: X/10
- Improve clarity, structure, and completeness only where the critic flagged problems"""

CRITIC_SYSTEM_PROMPT = """You are a senior research reviewer. Be accurate and conservative.

Before evaluating:
1. Read the ENTIRE report carefully — evaluate only what is visibly present.
2. Count Key Findings explicitly (each numbered/bulleted finding under Key Findings).
3. Check Sources for URLs — count plain https:// links AND Markdown links [Title](URL).
4. Never claim a section, finding, or URL is missing when it is clearly present.
5. Never invent content that is not in the report.

Revision policy (conservative):
- Set Revision Required: NO when Key Findings Check, Sources Check, and Evidence Check all PASS.
- Set Revision Required: YES only for genuine substantive problems (missing sections, <3 findings, no URLs, factual gaps).
- Do not request revision for style preferences when requirements are met."""

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


def _parse_revision_required(feedback: str) -> bool | None:
    """Parse structured Revision Required field. None if absent."""
    match = re.search(r"Revision Required:\s*(YES|NO)\b", feedback, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "YES"
    return None


def critic_needs_revision(feedback: str) -> bool:
    """Return True only when the Critic genuinely requires Revision."""
    explicit = _parse_revision_required(feedback)
    if explicit is not None:
        return explicit

    # Fallback when structured field is missing (legacy / malformed output)
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
        "revision required: no",
    )
    if any(phrase in lower for phrase in skip_phrases):
        return False

    # Structured PASS checks — if all pass, skip revision
    checks = re.findall(
        r"(Key Findings Check|Sources Check|Evidence Check):\s*(PASS|FAIL)",
        feedback,
        re.IGNORECASE,
    )
    if checks and all(result.upper() == "PASS" for _, result in checks):
        return False

    improve_section = re.search(
        r"areas to improve:\s*(.+?)(?:\nrevision required:|\nverdict:|\Z)",
        feedback,
        re.IGNORECASE | re.DOTALL,
    )
    if improve_section:
        bullets = [
            line.strip()
            for line in improve_section.group(1).splitlines()
            if line.strip().startswith("-")
        ]
        trivial = {"-", "- none", "- n/a", "- none.", "- no issues", "- no issues."}
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
(List every source using EXACT URLs from the research summary as clickable Markdown links)
- [Source Title](EXACT_URL) — Quality Score: X/10
(Do not modify URLs — copy them exactly from the research summary)
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

Write the IMPROVED final report addressing only genuine critic issues.
Use the same format: Introduction, Key Findings, Conclusion, Sources.
Preserve exact URLs as [Source Title](EXACT_URL) — Quality Score: X/10.
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
Review the following report carefully. Inspect what is actually present before judging.

Report:
{report}

Step 1 — Count Key Findings under the Key Findings section (each distinct finding).
Step 2 — Check Sources for URLs (plain https:// links AND Markdown [Title](URL) links).
Step 3 — Verify Introduction, Key Findings (3+), Conclusion, and Sources sections exist.

Return exactly this structure:

Score: X/10

Key Findings Check: PASS/FAIL
Sources Check: PASS/FAIL
Evidence Check: PASS/FAIL

Strengths:
- ...

Areas to Improve:
- ... (use "- None" if no genuine issues)

Revision Required: YES/NO

Verdict:
(approve as-is OR list only genuine substantive fixes needed)

Rules:
- If you count 3+ Key Findings, set Key Findings Check: PASS.
- If Sources contains any URL (plain or Markdown link), set Sources Check: PASS.
- Set Revision Required: NO when all three checks PASS unless there is a clear factual error.
- Never claim missing content that is visibly in the report.
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

"""OpenAI-powered agents and chains for the research pipeline."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from source_scoring import MIN_SOURCE_SCORE
from tools import scrape_url, scrape_urls_batch, web_search

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"

_active_model: str | None = None


def _require_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to a .env file in the project root."
        )
    return key


def get_model_candidates() -> list[str]:
    """Configured primary model and optional explicitly configured fallback."""
    models = [os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)]
    fallback = os.getenv("OPENAI_FALLBACK_MODEL", "").strip()
    if fallback and fallback not in models:
        models.append(fallback)
    return models


def set_active_model(model: str) -> None:
    global _active_model
    _active_model = model


def get_active_model() -> str:
    return _active_model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def clear_agent_caches() -> None:
    get_llm.cache_clear()
    get_writer_chain.cache_clear()
    get_revision_chain.cache_clear()
    get_summarizer_chain.cache_clear()
    get_critic_chain.cache_clear()
    build_search_agent.cache_clear()
    build_reader_agent.cache_clear()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return the shared OpenAI chat model (backed by the official SDK)."""
    return ChatOpenAI(
        model=get_active_model(),
        api_key=_require_openai_api_key(),
        timeout=float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60")),
        max_retries=0,
        reasoning_effort="none",
    )


def get_search_llm() -> ChatOpenAI:
    """OpenAI model with forced Tavily tool calling for the search agent."""
    return get_llm().bind_tools([web_search], tool_choice="required")


def get_reader_llm() -> ChatOpenAI:
    """OpenAI model with forced batch scrape tool calling for the reader agent."""
    return get_llm().bind_tools([scrape_urls_batch], tool_choice="required")


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

SEARCH_SYSTEM_PROMPT = """You are a web research agent. Call web_search once with a focused query.
The tool returns formatted results — return them as-is. Do not add extra commentary."""

READER_SYSTEM_PROMPT = f"""You are a research reader agent. Your ONLY tool is scrape_urls_batch.

Call scrape_urls_batch ONCE with the exact URLs supplied by the user. Python has already applied the authority threshold ({MIN_SOURCE_SCORE}/10) and ranked the accepted sources.
The tool returns scraped page text — return it as-is. Do not summarize."""

WRITER_SYSTEM_PROMPT = f"""You are an expert research analyst and technical writer.

Rules:
- Use ONLY validated claim/evidence context provided — no other sources
- Use only accepted sources in the validated research context. Their composite ranking score is descriptive, not a second acceptance threshold.
- Never invent facts or URLs
- Mark factual findings with the supplied [C#] claim IDs. Do not invent IDs or URLs.
- Do not write URLs, Markdown links, or source entries; cite claims only with [C#] markers.
- Every material factual sentence must end with one or more supplied [C#] markers.
- Write professionally with clear structure"""

REVISION_SYSTEM_PROMPT = f"""You are an expert research analyst revising a report based on critic feedback.

Rules:
- Fix only genuine issues raised by the critic — do not rewrite content that already passes
- Use only accepted sources in the validated research context. Their composite ranking score is descriptive, not a second acceptance threshold.
- Do not invent facts — use only validated claim/evidence context
- Keep supplied claim markers [C#] on factual statements; never invent claim IDs
- Do not create or alter source URLs; citations are materialized from validated mappings
- Improve clarity, structure, and completeness only where the critic flagged problems"""

CRITIC_SYSTEM_PROMPT = """You are a senior research reviewer. Be accurate and conservative.

Before evaluating:
1. Read the ENTIRE report carefully — evaluate only what is visibly present.
2. Count Key Findings explicitly (each numbered/bulleted finding under Key Findings).
3. Check Sources for URLs — count plain https:// links AND Markdown links [Title](URL).
4. Never claim a section, finding, or URL is missing when it is clearly present.
5. Never invent content that is not in the report.

Assess claim support, citation mapping, source quality, independent domain diversity, contradictions, completeness, then prose. A single lower-authority domain reused for claims is a substantive evidence defect; one genuinely authoritative source by itself is not.
Revision Required must be YES for important unsupported factual claims, missing or wrong citations, fabricated URLs, or material evidence gaps. Prose quality cannot compensate for weak evidence."""

SUMMARIZER_SYSTEM_PROMPT = """Extract source-grounded research evidence from the scraped pages.
Return ONLY valid JSON with keys evidence and claims. Evidence objects have source_url (copied exactly), excerpt (verbatim text from page), supporting_text, relevance_score and confidence from 0 to 1, and optional location. Claims have claim_text, evidence_indices (zero-based indices into evidence), confidence and importance from 0 to 1. Use only verifiable excerpts; do not invent URLs or facts. Include unsupported claims only with an empty evidence_indices list. Keep the evidence concise."""

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
    """Search agent: Tavily web_search via OpenAI tool calling."""
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
(findings must be supported and marked with supplied [C#] claim IDs)

# Conclusion

# Sources
(Do not add links or entries. Python materializes sources from the [C#] markers.)
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
Preserve valid [C#] claim markers. Python will materialize citations from validated mappings.
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

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from source_scoring import MIN_SOURCE_SCORE
from tools import scrape_url, web_search

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

# Force tool use for reader agent — Gemini must call scrape_url
reader_llm = llm.bind_tools([scrape_url], tool_choice="any")

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

SEARCH_SYSTEM_PROMPT = """You are a web research agent with the web_search tool.

Search the web thoroughly. Requirements:
- Use official sources whenever possible (government, institutional)
- Include at least 5 sources
- Prioritize government and institutional websites
- Include recent news from reputable outlets
- Include Wikipedia ONLY if official sources are unavailable

After searching, return a brief overview listing each result's title, URL, and one-line summary.
Do NOT paste full webpage text."""

READER_SYSTEM_PROMPT = f"""You are a research reader agent. Your ONLY tool is scrape_url.

Workflow (mandatory):
1. Review the ranked source list (with quality scores).
2. Pick the top 3–5 URLs with Quality Score >= {MIN_SOURCE_SCORE}.
3. Call scrape_url once for EACH selected URL (multiple tool calls).
4. After ALL scrapes return, write a structured multi-source summary.

NEVER use internal knowledge or snippets. ONLY use scrape_url output.

Output format (required):

### Source 1: [Title]
URL: ...
Quality Score: .../10
Key Points:
- ...
- ...

### Source 2: [Title]
URL: ...
Quality Score: .../10
Key Points:
- ...
- ...

(Repeat for each scraped source)"""

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

# ---------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=SEARCH_SYSTEM_PROMPT,
    )


def build_reader_agent():
    return create_agent(
        model=reader_llm,
        tools=[scrape_url],
        system_prompt=READER_SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", WRITER_SYSTEM_PROMPT),
    (
        "human",
        """
Topic: {topic}

Research Summary (from reader agent — use ONLY this):
{research}

Write a comprehensive report:

# Introduction

# Key Findings
(At least 3 detailed findings, each backed by a source)

# Conclusion

# Sources
(List every URL from the research summary with quality scores)
""",
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

revision_prompt = ChatPromptTemplate.from_messages([
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

revision_chain = revision_prompt | llm | StrOutputParser()

summarizer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You synthesize scraped webpage content into structured research notes.

Output format for EACH source:

### Source N: [Title]
URL: ...
Quality Score: .../10
Key Points:
- ...
- ...

Use ONLY the scraped content provided. No internal knowledge.""",
    ),
    (
        "human",
        """
Topic: {topic}

Scraped content from {count} sources:
{scraped_content}
""",
    ),
])

summarizer_chain = summarizer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior research reviewer.

Evaluate reports for accuracy, completeness, clarity, structure, and evidence quality.
Be constructive and specific.""",
    ),
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
- ...

Areas to Improve:
- ...
- ...

Verdict:
...
""",
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()

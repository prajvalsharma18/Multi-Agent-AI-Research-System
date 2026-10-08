# Multi-Research AI System

A verification-first **agentic research system built with LangGraph** that searches the web, retrieves and reads source documents, extracts structured evidence, derives claims, generates a cited research report, critiques its own output, and revises it using deterministic research-quality gates.

The system is designed around a simple principle:

> **LLMs generate and reason over research artifacts; deterministic validation decides whether those artifacts are trustworthy enough to export.**

---

## Overview

Traditional LLM research workflows often rely heavily on the model to:

* choose sources,
* interpret documents,
* generate claims,
* attach citations,
* and decide whether the final answer is reliable.

This system separates those responsibilities.

The research pipeline maintains explicit relationships between:

```text
Search Result
     ↓
Document
     ↓
Evidence
     ↓
Claim
     ↓
Citation
     ↓
Final Report
```

Every important research artifact can therefore be traced back to its source document.

The system also combines:

* **LangGraph** for workflow orchestration
* **LLMs** for research reasoning and generation
* **Tavily** for web search
* **structured Pydantic models** for evidence and claim tracking
* **deterministic validation gates** for evidence, citations, and source diversity
* **Critic + Revision** for iterative quality improvement
* **caching and checkpoint compatibility controls**
* **Markdown and PDF report exporters**

---

# Architecture

```text
                         ┌──────────────────┐
                         │   User Question  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Search      │
                         │     (Tavily)     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Reader      │
                         │ Fetch / Extract  │
                         │    Documents     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Evidence      │
                         │ Structured       │
                         │   extraction     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Claims      │
                         │ Claim ↔ Evidence │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Writer      │
                         │ Research Report  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Critic       │
                         │ Model + Gates    │
                         └────────┬─────────┘
                                  │
                       ┌──────────┴──────────┐
                       │                     │
                    Approved             Revision
                       │                     │
                       │              ┌──────▼──────┐
                       │              │   Revision  │
                       │              └──────┬──────┘
                       │                     │
                       │              Citation validation
                       │                     │
                       │              Critic again
                       │                     │
                       └──────────┬──────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Export      │
                         │ Markdown + PDF   │
                         └──────────────────┘
```

---

# Core Design

## 1. Search

The Search stage uses Tavily to retrieve candidate web sources for the research question.

Search results are normalized before entering the downstream pipeline.

Each source is evaluated using multiple signals including:

* authority
* relevance
* recency
* evidence quality
* source diversity

### Authority vs Composite Score

The system intentionally separates **source acceptance** from **source ranking**.

```text
Authority score
      ↓
Acceptance gate

Composite score
      ↓
Ordering among accepted sources
```

This prevents a source with high authority from being rejected simply because its composite score is lower due to other factors.

---

# 2. Reader

The Reader fetches the selected sources and converts them into documents that can be consumed by downstream research stages.

The system maintains a relationship between:

```text
URL → fetched document → evidence
```

Scraping is cached where appropriate to avoid unnecessary repeated requests.

---

# 3. Structured Evidence

Research evidence is represented explicitly instead of relying only on free-form model output.

The current evidence pipeline uses the `evidence-v2` schema.

Conceptually:

```text
SearchResult
     ↓
Document
     ↓
Evidence
```

Each evidence item maintains source traceability.

The system validates that accepted excerpts:

* originate from successfully fetched documents
* correspond to their source URL
* actually occur in the fetched document text after normalization

This makes evidence independently verifiable.

---

# 4. Claims

The Summarizer/claim stage converts evidence into explicit research claims.

Each claim maintains a relationship to supporting evidence.

```text
Claim C1
 ├── Evidence E1
 ├── Evidence E5
 └── Evidence E6
```

This enables the system to calculate:

* evidence coverage
* claim confidence
* multi-source support
* claim-to-source traceability

The system does not silently treat unsupported claims as supported.

---

# 5. Citation Validation

Citation generation is deliberately separated from citation trust.

The Writer can produce citation markers such as:

```text
[C1]
[C2]
[C3]
```

These markers are resolved through the validated claim/evidence/source graph.

Conceptually:

```text
[C1]
 ↓
Claim
 ↓
Evidence
 ↓
Document
 ↓
Validated URL
```

Model-authored URLs that are not present in the validated source map are treated as invalid rather than automatically trusted.

This prevents the LLM from introducing arbitrary or fabricated sources into the final report.

---

# 6. Research Quality Gates

The system uses deterministic gates alongside the model Critic.

Current validation includes:

### Evidence Gate

Checks whether claims are sufficiently supported by validated evidence.

### Citation Gate

Checks:

* citation coverage
* claim/source mapping
* invalid or unmapped URLs

### Source Diversity Gate

Checks whether accepted sources provide sufficient domain diversity.

This is intentionally deterministic.

A model cannot simply approve a report and bypass a failed deterministic gate.

---

# 7. Critic

The Critic evaluates the generated report.

The system combines:

```text
Model Critic
      +
Deterministic Research Quality Gates
```

For example, a model may produce:

```text
Critic score = 8/10
```

while a deterministic quality gate fails because source diversity is insufficient.

The deterministic result can therefore override the model's approval.

This is an important architectural property of the system:

> **The LLM is not the final authority over research validity.**

---

# 8. Revision Loop

When the Critic requires revision, the graph sends the report through the configured revision loop.

After each revision:

```text
Revision
   ↓
Citation validation
   ↓
Critic
   ↓
Deterministic gates
```

The system currently limits revision attempts to prevent an uncontrolled loop.

The latest validation run reached the configured maximum of **2 revisions**.

---

# 9. Caching

The system uses caching to avoid unnecessarily repeating expensive operations.

The validation work also identified and fixed a Reader cache-key mismatch where read and write keys were inconsistent.

A regression test was added for cache-key consistency.

The research cache/checkpoint compatibility also includes versioning information so stale state is not blindly reused.

---

# 10. Checkpoint Compatibility

The current compatibility checks include:

```text
Model identity
+
evidence-v2
+
citations-v2 prompt fingerprint
```

This prevents a checkpoint produced under an incompatible research schema or prompt configuration from being reused.

A stale/incompatible checkpoint is rejected instead of silently becoming part of a new research run.

---

# 11. Report Generation

The system generates:

```text
Markdown
PDF
```

The final report contains sections such as:

* Introduction
* Key Findings
* Conclusion
* Sources
* Research Quality
* Critic Review

The exporter also produces research-quality metrics including:

* supported claims
* unsupported claims
* evidence coverage
* citation coverage
* source diversity
* average claim confidence
* Critic score

The bibliography is generated from the validated source map rather than blindly copying model-generated URLs.

---

# Validation

## Automated Tests

The established baseline is:

```text
72 tests passed
```

Compilation also passed with:

```bash
python -m compileall -q .
```

The validation work additionally introduced regression coverage around citation materialization and cache behavior.

---

# Live Validation

A fresh live run was executed using:

```text
How does Redis caching reduce application latency?
```

The run completed:

```text
Search
  ↓
Reader
  ↓
Evidence
  ↓
Claims
  ↓
Writer
  ↓
Critic
  ↓
Revision × 2
  ↓
Export
```

### Latest Live Metrics

| Metric                        |            Result |
| ----------------------------- | ----------------: |
| OpenAI calls                  |                 9 |
| Cache hits                    |                 0 |
| Cache misses                  |                10 |
| Scrape calls                  |                 3 |
| Revisions                     |                 2 |
| Supported claims              |             5 / 5 |
| Unsupported claims            |                 0 |
| Evidence coverage             |              100% |
| Citation coverage             |              100% |
| Average claim confidence      |             0.978 |
| Accepted URLs                 |                 3 |
| Accepted domains              |                 1 |
| Source diversity ratio        |             33.3% |
| Model Critic score            |              8/10 |
| Effective deterministic score |              5/10 |
| Final decision                | Revision required |

---

# Live Evidence Results

The latest run fetched:

```text
3 / 3 documents successfully
```

All accepted evidence excerpts were traceable to successfully fetched documents.

```text
Accepted excerpts: 6
Rejected excerpts: 0
Evidence coverage: 100%
```

Five claims were extracted and all five had valid evidence mappings.

---

# Citation Results

Citation validation ran:

```text
Writer
   ↓
Revision 1
   ↓
Revision 2
```

for a total of **3 citation validations**.

The latest live run reported:

```text
Citation coverage: 100%
Invalid citations: 0
Unvalidated retained URLs: 0
```

The generated Markdown and PDF contained actual validated URLs, and the previous:

```text
[unmapped citation removed]
```

placeholder no longer appeared.

---

# Current Validation Status

The system is **not yet marked as fully validation-passed**.

The latest live run exposed a legitimate deterministic quality failure:

```text
3 accepted URLs
1 accepted domain
33.3% diversity ratio
```

All accepted sources came from `redis.io`, so the source-diversity gate failed.

The model Critic gave:

```text
8/10
```

but the deterministic gate reduced the effective result to:

```text
5/10
```

and required another revision.

This behavior is intentional: deterministic research-quality gates are not bypassed by model approval.

---

# Latest Export Status

## Markdown

The latest Markdown artifact:

* contains 3 deduplicated validated URLs
* contains no `[unmapped citation removed]` placeholder
* contains the requested Research Quality metrics

**Status: PASS for bibliography/export structure.**

There is still a redundant inline citation in the live artifact caused by adjacent claim markers.

A narrow exporter fix was subsequently added to merge adjacent claim markers before citation materialization.

That fix has **not yet been test-verified** because the configured project Python interpreter returned an access error in the validation environment.

---

## PDF

The latest PDF:

* is readable
* contains actual validated URLs
* contains Research Quality metrics
* contains no bibliography placeholder
* contains only validated URL annotations

**Status: PASS for bibliography/export structure.**

The PDF still reflects the redundant inline citation from the live run because it was generated before the narrow citation-cleanup fix.

---

# Current Limitations

### 1. Source diversity

The latest live run accepted sources from only one domain.

The source-diversity gate therefore failed.

The system does **not** bypass this failure simply because the model Critic approved the report.

### 2. Adjacent citation cleanup

A narrow exporter change was added to merge adjacent claim markers that reference the same source and prevent redundant inline links.

The corresponding regression test has not yet been executed because the configured Python interpreter was inaccessible.

### 3. Revision retrieval

The current graph performs revision on the existing research context. Revision does not automatically perform additional web retrieval to discover new sources.

Therefore, if source diversity is insufficient, revision alone may not necessarily introduce a new domain.

### 4. Heuristic source scoring

Source quality scoring is heuristic and combines multiple factors.

Missing publication dates receive a neutral recency treatment.

---

# Engineering Highlights

The project focuses on reliability mechanisms around LLM-based research rather than simply generating a web-search answer.

### Structured research state

```text
SearchResult
Document
Evidence
Claim
ResearchNote
```

are maintained as structured data rather than only raw model text.

### Evidence traceability

```text
Claim
 ↓
Evidence
 ↓
Document
 ↓
URL
```

makes claims auditable.

### Deterministic citation validation

The model does not have unrestricted authority to introduce URLs into the final bibliography.

### Deterministic quality gates

Model approval can be overridden when objective validation fails.

### Revision loop

Critic feedback is incorporated through bounded revision attempts.

### Cache/checkpoint safety

Schema and prompt fingerprints help prevent incompatible cached research state from being reused.

---

# Example Research Flow

For a query such as:

```text
How does Redis caching reduce application latency?
```

the system performs:

```text
1. Search for candidate sources
        ↓
2. Score and filter sources
        ↓
3. Fetch documents
        ↓
4. Extract evidence
        ↓
5. Generate claims
        ↓
6. Map claims to evidence
        ↓
7. Generate cited report
        ↓
8. Validate citations
        ↓
9. Run Critic
        ↓
10. Apply deterministic quality gates
        ↓
11. Revise if required
        ↓
12. Revalidate
        ↓
13. Export Markdown + PDF
```

---

# Tech Stack

* **Python**
* **LangGraph**
* **Pydantic**
* **Tavily**
* **OpenAI**
* **Pytest**
* **Markdown**
* **PDF generation**
* Web scraping / document extraction
* Caching
* Checkpointing

---

# Project Structure

A simplified structure:

```text
Multi_research_ai_system_langgraph/
│
├── agents.py
├── app.py
├── cache.py
├── graph.py
├── research_quality.py
├── tools.py
│
├── tests/
│   ├── test_agents.py
│   ├── test_cache.py
│   ├── test_graph.py
│   ├── test_research_quality.py
│   └── test_search.py
│
├── reports/
│   ├── *.md
│   └── *.pdf
│
└── ...
```

---

# Validation Philosophy

The system follows a layered validation model:

```text
                 LLM
                  │
        ┌─────────▼─────────┐
        │ Generate Research │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ Structured State  │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ Evidence Mapping  │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ Citation Mapping  │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │ Deterministic     │
        │ Quality Gates     │
        └─────────┬─────────┘
                  │
             ┌────▼────┐
             │ Critic  │
             └────┬────┘
                  │
             Revision
                  │
                  ▼
             Final Export
```

The objective is not to eliminate LLM uncertainty completely.

Instead, the system reduces the amount of trust placed directly in the model by introducing **structured intermediate representations, provenance tracking, deterministic validation, and bounded revision**.

---

# Current Status

**Core research pipeline:** Functional

**Evidence and claim traceability:** Implemented

**Citation validation:** Implemented

**Deterministic quality gates:** Implemented

**Markdown/PDF bibliography validation:** Passed in latest live artifact

**Automated baseline:** 72 tests passed

**Final post-fix validation:** In progress

The remaining validation work is focused on verifying the latest adjacent-citation exporter fix and determining whether the source-diversity gate can pass with the current retrieval behavior.

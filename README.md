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
                  Research gap          Writing issue
                       │                     │
              Targeted Tavily search      Revision
                       │                     │
              Fetch + validate evidence     │
                       │                     │
                 Merge evidence              │
                       └──────────┬──────────┘
                                  ▼
                                Writer
                                  │
                                  └──────────► Critic (bounded loop)
                                                 │
                                      Approved or budget exhausted
                                                 │
                                                 ▼
                                        Markdown + PDF Export
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

# 8. Bounded Research Recovery and Revision

The Critic and deterministic gates route deficiencies by type. Research gaps (unsupported claims, insufficient
evidence, weak authority, or domain concentration) trigger targeted Tavily searches before rewriting. Writing-only
issues use the existing revision path without unnecessary searches.

After either targeted recovery or a writing revision:

```text
Targeted research or Revision
   ↓
Evidence and citation validation
   ↓
Critic
   ↓
Deterministic gates and bounded routing
```

`MAX_RESEARCH_ROUNDS` and `MAX_REVISION_ITERATIONS` are independent budgets, both defaulting to 2. Values from 0
through 10 are accepted; invalid values use the safe default. Recovery terminates early when no new traceable
evidence is obtained. Budget exhaustion and unresolved quality issues remain visible in the final report.
Recovery candidates reuse existing URL normalization, source scoring, scraping, and evidence validation. New
evidence is merged without discarding prior accepted evidence or reusing IDs.

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
citations-v3-recovery-routing prompt fingerprint
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

The latest verified offline suite contains:

```text
98 tests passed
```

Compilation also passed with:

```bash
.venv\Scripts\python.exe -m compileall .
```

The suite includes bounded-recovery routing, evidence merging and validation, cache-policy, citation materialization,
and export regressions.

---

# Live Validation

The most recent pre-recovery live run used:

```text
How does Redis caching reduce application latency?
```

Its historical route was:

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

### Historical Metrics (Pre-Recovery)

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

# Historical Evidence and Citation Results

The historical pre-recovery run fetched:

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

That historical live run reported:

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

# Historical Phase 2 Live Recovery Run (Before Export-Metrics Fix)

A fresh uncached graph run was completed with the topic:

```text
What is Redis caching, how does it reduce application latency, and what are its limitations?
```

The graph was started from `{topic}` with `PIPELINE_CACHE_ENABLED=false`; it did not load or clear a saved
checkpoint. The configured model was `gpt-5.6-luna` with `reasoning_effort="none"`.

Observed node transitions:

```text
Search → Reader → Writer → Critic → Targeted Recovery → Writer → Critic → Targeted Recovery → Export
```

The Critic did not approve the initial report, so recovery ran twice. The second round obtained no new validated
evidence and correctly terminated without approval; no writing-only revision was requested.

| Metric | Live result |
|---|---:|
| Initial Tavily searches | 1 |
| Recovery searches / rounds | 2 / 2 |
| Accepted URLs / domains | 4 / 2 |
| Source diversity | 50% |
| Recovery new URLs / duplicate URLs | 2 / 1 |
| Successful / failed recovery fetches | 2 / 0 |
| Total successful fetches / scrapes | 4 / 4 |
| Accepted / rejected recovery evidence | 9 / 0 |
| Total validated evidence excerpts | 18 |
| Supported / unsupported claims | 15 / 2 |
| Evidence coverage | 88.2% |
| Model / effective Critic score | 8/10 / 5/10 |
| Writer revisions | 0 |
| OpenAI calls / retries | 8 / 0 |
| Cache hits | 0 (caches disabled) |
| Runtime | 66.8 seconds |
| Termination | `no_research_progress`; not approved |

The run summary did not preserve cache-miss totals or citation-coverage measurements. Do not infer those
values from the generated report: a no-progress recovery path had cleared those fields before export. The
source-diversity result is based on four accepted URLs across two domains, while all three cited bibliography URLs
in this run happened to be on `redis.io`.

## Historical Artifacts and Defect

The exact artifacts from this run were:

- `reports/what-is-redis-caching-how-does-it-reduce-application-latency_20261009_150550.md`
- `reports/what-is-redis-caching-how-does-it-reduce-application-latency_20261009_150550.pdf`

Both files exist and open. The Markdown has 18 inline links and three deduplicated bibliography URLs; every inline
URL is in that bibliography, no unmapped-citation placeholder or invalid link was found, and there were no
adjacent identical links. The PDF has 25 link annotations, all pointing to URLs in the validated bibliography.

However, the artifact’s Research Quality section is **not fully accurate**: it reports zero citation coverage and
zero Critic scores even though the two Critic evaluations scored 8/10 (effective 5/10). It also reports zero new
recovery sources because counters reflected only the last recovery round. The graph remained NOT APPROVED, which
is the correct gate outcome, and its evidence counts and termination reason match the run.

The run state was not persisted because caches were disabled. Therefore these live artifacts were not rewritten
after the fix; rewriting them would require inventing unrecorded citation-coverage data. This run predates the
offline-verified fixes below.

# Phase 2.1 Post-Fix Live Validation

One fresh uncached run was completed after the recovery-metrics and export fixes. It started from topic-only graph
state and used the configured `gpt-5.6-luna` model with `reasoning_effort="none"`.

Observed transitions:

```text
Search → Reader → Writer → Critic → Research Recovery → Writer → Critic → Export
```

The initial Critic routed to research recovery (model score 7/10; effective score 5/10). One targeted recovery
round added validated evidence; the final Critic approved the report (9/10 model and effective scores). No Writer
revision was requested.

| Metric | Live result |
|---|---:|
| Initial Tavily searches / results | 1 / 5 |
| Initially accepted search results | 0 |
| Recovery searches / rounds | 1 / 1 |
| Recovery candidates / accepted sources | 5 / 1 |
| Accepted URLs / domains | 1 / 1 |
| Source diversity ratio | 100% |
| Average source quality / authority | 6.95/10 / 9/10 |
| Successful / failed recovery fetches | 1 / 0 |
| Accepted / rejected recovery evidence | 13 / 0 |
| Total evidence excerpts | 13 |
| Supported / unsupported claims | 8 / 0 |
| Evidence / citation coverage | 100% / 100% |
| Invalid citations | 0 |
| Revisions | 0 |
| OpenAI calls | 6 |
| Cache hits / enabled-cache misses | 0 / 0 (cache disabled) |
| Runtime | 49.36 seconds |
| Termination | `approved` |

The one accepted URL was `https://redis.io/solutions/caching`. The 100% diversity ratio reflects one unique
domain among one accepted URL; it does **not** establish independent corroboration. The Critic noted that a second
source could strengthen neutrality. No distinct-source citation behavior was exercised by this live run.

Fresh artifacts:

- `reports/what-is-redis-caching-how-does-it-reduce-application-latency_20261009_154011.md`
- `reports/what-is-redis-caching-how-does-it-reduce-application-latency_20261009_154011.pdf`

Both artifacts opened successfully. The Research Quality values match the final graph state in both formats. The
Markdown contains 10 inline links and one deduplicated bibliography URL; no unmapped-citation placeholder or
adjacent duplicate-link syntax was present. All 10 PDF link annotations point to the same validated URL. Repeated
links in separate claim statements remain because those claims each require their citation.

# Current Validation Status

The bounded recovery implementation and the live-observed no-progress export defect have offline regression
coverage. The latest full offline suite contains 98 passing tests; `compileall` and `git diff --check` passed after
the changes. A post-fix live run exercised one recovery round and generated the verified artifacts above. The
earlier no-progress live artifact remains historical and predates the fixes.

Source diversity remains dependent on actual search results. Multiple validated URLs from one domain are reported
as a research-quality limitation and are not treated as a software defect by themselves.

## Current Limitations

### 1. Retrieval outcome

Targeted recovery is bounded and cannot guarantee that Tavily will return independent, authoritative sources or
that fetched pages will contain verifiable evidence.

### 2. Heuristic source scoring

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

The repository keeps the Python modules at the root because the CLI, Streamlit
app, benchmark runner, and tests import them as top-level modules. Benchmark
specifications and tests are versioned; reports, caches, credentials, and the
virtual environment are local runtime data and are excluded by `.gitignore`.

```text
Multi_research_ai_system_langgraph/
├── .env.example
├── README.md
├── requirements.txt
├── app.py                 # Streamlit UI
├── pipeline.py            # CLI entry point
├── benchmark.py           # Offline and opt-in live evaluation
├── agents.py
├── cache.py
├── export.py
├── graph.py               # LangGraph orchestration
├── llm_retry.py
├── list_model.py          # OpenAI model-listing utility
├── metrics.py
├── research_models.py
├── research_quality.py
├── source_scoring.py
├── tools.py
├── benchmarks/
│   ├── benchmark_cases.json
│   └── offline_scenarios.json
├── tests/
└── reports/               # Generated locally; Git-ignored
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

**Offline tests:** 130 passed in the latest local run (2026-10-09)

**Bounded research recovery:** Implemented and offline-tested

**Fresh live validation of recovery:** Completed; one recovery round was exercised

---

# Phase 3 — Evaluation and Benchmarking

Phase 3 adds a small evaluation entry point around the existing LangGraph application. The benchmark runner does
not implement a second research pipeline: live cases invoke `graph.research_graph` with topic-only initial state.
Run IDs are unique per invocation, and each live case writes report artifacts and, when enabled, cache data under
its run-specific directory. Existing application checkpoints, caches, and timestamped reports are not reused or
overwritten.

## Dataset

`benchmarks/benchmark_cases.json` is the versioned specification for 10 research questions. Each case has a stable
ID, question, category, expected source characteristics, minimum unique URL/domain counts, freshness requirements,
expected insufficient-evidence behavior, and evaluation notes. The loader validates required fields, types, unique
IDs, and the 8–12 case dataset limit before a run.

The evaluation date is recorded in the dataset (`2026-10-09`). Freshness expectations are relative to that date;
the current pipeline does not consistently validate publication dates, so live freshness measurements remain
unavailable unless the graph adds reliable date metadata.

## Offline fixture conformance

Run all deterministic fixture scenarios without credentials or external calls:

```powershell
& '.\.venv\Scripts\python.exe' benchmark.py --offline
```

Select fixture cases by dataset ID and choose an output directory:

```powershell
& '.\.venv\Scripts\python.exe' benchmark.py --offline --case redis-caching --case idempotency-subquestions --output-dir reports\benchmarks
```

`benchmarks/offline_scenarios.json` specifies 10 controlled conformance scenarios for approval routing, recovery
success/failure, duplicates, fetch failures, budget limits, evidence merging, and no-progress preservation. These
fixture values are synthetic test inputs. Offline output verifies fixture invariants and reports
`research_cases_executed: 0`; it does not answer the research questions or claim mocked answers as live results.
The production graph's deterministic recovery, evidence, metrics, citation, and export behavior is separately
tested by the offline pytest suite with mocked boundaries.

## Controlled live evaluation

Live evaluation requires the explicit `--live` flag, an explicit cache setting, and configured `OPENAI_API_KEY` and
`TAVILY_API_KEY` variables. The runner never prints or serializes key values. It uses the configured model and
Tavily provider, preserves `reasoning_effort="none"`, starts the actual graph from `{topic}`, and has no automatic
retry policy for whole benchmark cases.

The default live limit is two cases; at most three may be selected in one invocation. More selected cases than
the configured cap is rejected rather than silently truncated. A fresh run-scoped cache directory prevents cache
or checkpoint mixing:

```powershell
& '.\.venv\Scripts\python.exe' benchmark.py --live --case redis-caching --case http3-performance --max-live-cases 2 --cache disabled --output-dir reports\benchmarks
```

To explicitly evaluate the isolated cache-enabled behavior instead:

```powershell
& '.\.venv\Scripts\python.exe' benchmark.py --live --case redis-caching --max-live-cases 1 --cache enabled --output-dir reports\benchmarks
```

Use `--latest --output-dir reports\benchmarks` to locate the most recent aggregate Markdown and JSON files. Live
per-case Markdown/PDF exports are placed below `reports\benchmarks\artifacts\<run-id>\<case-id>`; isolated cache
files, if enabled, are under `reports\benchmarks\isolated-cache\<run-id>\<case-id>`.

## Metric definitions and interpretation

Every result preserves unavailable values as JSON `null` and reports measured denominators:

| Metric | Definition |
|---|---|
| Accepted unique URLs | Count of canonicalized accepted source URLs in final graph state. |
| Unique domains | Count of registrable-domain approximations derived by existing URL utilities; this does not establish independence. |
| Source diversity | Existing unique-domain/source measure; the report also exposes explicit URL/domain counts and each case's minimum-source requirement. |
| Authority | Existing domain-based heuristic score on a 0–10 scale; it is not an independent audit of a source. |
| Relevance | Existing keyword-overlap heuristic in `source_score_breakdown.relevance`; it is not human review. |
| Evidence coverage | Existing supported-claim count divided by total extracted claims. With no claims, coverage and average claim confidence are not measured, not zero. |
| Citation coverage | Claims with validated mapped citations divided by the citation metric's total-claim denominator. With no claims, coverage is not measured; it does not establish corroboration. |
| Recovery productive rate | Cases with at least one new validated recovery evidence item divided by cases where recovery was attempted. Quality-gate improvement and final approval are separate. |
| Quality approval rate | Approved completed graph runs divided by completed runs with a known approval decision. |
| Operational pass | Completed graph run with passing Markdown/PDF integrity checks; it is not a factual-correctness judgment. |
| Cache hits/misses | Benchmark values are reported only when cache is enabled. Hits include instrumented checkpoint, node/stage-cache, and Tavily-cache reuse. Misses count only enabled lookups explicitly instrumented by LLM/search stages; scraper-cache lookups are not currently counted. With cache disabled, both are not applicable (`null`), not cache misses. |
| Runtime/model calls | Instrumented execution time and OpenAI call counts. Token usage and monetary cost are unavailable and are not estimated. |

Source relevance and authority are heuristic assessments. Multiple domains are only a corroboration proxy; the
system does not detect copied/syndicated content, so `independent_corroboration_verified` remains unknown.
A 100% diversity ratio with one URL/domain must not be read as independent corroboration. Freshness is currently
reported as unavailable because publication dates are not consistently validated. Empty-source averages and
diversity ratios are also not measured because they have no denominator.

An unapproved report remains a completed evaluation outcome (`rejected_by_quality_gate`), distinct from a runner
or export failure. An approval is the application's deterministic gate result and Critic route, not a guarantee of
truth or readiness for production use.

## Aggregate artifacts and historical results

Each invocation writes:

- `<run-id>.json`: run metadata, configuration, per-case metrics, failures/incomplete cases, denominators, and limitations.
- `<run-id>.md`: a human-readable aggregate of the same run.

Offline fixture reports and live graph results are marked as separate modes and cannot be aggregated together.
Live cases with different cache settings cannot be mixed in one aggregate. Runtime medians/ranges are only included
when at least two values are measured.

The Phase 1/2/2.1 results above are historical records and are not overwritten by Phase 3 reports. Passing tests
or a small live sample is not evidence that the system is production-ready.

### Observed Phase 3 live smoke run

Run `live-20261009T103949Z-697a686f` executed two selected cases with cache disabled, Tavily, and the configured
`gpt-5.6-luna` model (`reasoning_effort="none"`). This is a two-case smoke sample, not a benchmark conclusion:

- Both graph runs completed, attempted one recovery round, made no evidence progress, and terminated with
  `no_research_progress`; neither was approved.
- The WebAuthn case had 3 accepted URLs across 2 domains, 6 accepted evidence excerpts, 4/4 supported claims,
  100% evidence/citation coverage, and Critic model/effective scores of 8/5.
- The HTTP/3 case had no accepted sources or validated claims, so its evidence/citation coverage and source quality
  are not measured. Its Critic model/effective scores were 10/5.
- The run made 9 OpenAI calls in 51.213 seconds. Two initial searches and two recovery searches were recorded.
  Cache hit/miss counts were not applicable because caching was disabled. Recovery did not add validated evidence.
- Three scrape calls were instrumented across the two cases; the available metrics do not separate initial fetches
  from recovery fetches. The WebAuthn recovery recorded 2 successful and 0 failed fetches.
- Across cases, 3 unique URLs and 2 domains were accepted in total; per-case source minimums were met in 1 of 2
  cases. These metrics do not verify independent corroboration.
- Both Markdown/PDF exports passed integrity checks. Operational completion and quality approval are separate:
  both cases were operationally complete but rejected by the quality gate.

The aggregate report files retained in the local, Git-ignored `reports/` directory are
`reports/benchmarks/live-20261009T103949Z-697a686f.json` and
`reports/benchmarks/live-20261009T103949Z-697a686f.md`. The original per-case exports and corrected copies of both cases are retained below
`reports/benchmarks/artifacts/live-20261009T103949Z-697a686f/`; the corrected copies mark zero-denominator
metrics as not measured. The original aggregate revealed that the benchmark exporter had mislabeled Critic gate
failures as fetch failures and counted empty-denominator coverage as zero. The aggregate was recalculated from the
saved run data and the reports were re-exported and revalidated without making additional live API calls.

### Phase 3.1 diagnostic semantics

New live benchmark JSON records separate execution status, deterministic quality-gate status, and export-integrity
status. `pipeline_diagnostics` records available initial-search, fetch, extraction, evidence-validation, claim-support,
and per-recovery-round outcomes. Search decision details include URL/domain and authority, relevance, and composite
scores, but omit search snippets. Recovery counters aggregate across rounds; historical runs are not retroactively
filled with diagnostic values that were not captured.

Critic model scores retain distinct raw, validated, and effective values. Missing, malformed, or out-of-range scores
are reported as unavailable with a score status and cannot approve a report. A valid high model score remains
separate from deterministic gate failures; where required quality checks fail, the effective score is capped for
display but does not override the gate. Source quality used by the gate is measured from sources mapped to supported
claims when those mappings exist, rather than from unused search candidates alone. Heuristic authority/relevance and
domain diversity remain proxies, not human verification or proof of independent corroboration.

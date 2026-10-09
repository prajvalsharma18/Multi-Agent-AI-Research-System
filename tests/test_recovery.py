import json
from types import SimpleNamespace

import graph
from research_models import Claim, Evidence
from research_quality import calculate_domain_diversity, calculate_evidence_metrics


def _source(url, title="Source", score=8, authority=9):
    from source_scoring import domain_for_url

    return {
        "url": url,
        "title": title,
        "domain": domain_for_url(url),
        "score": score,
        "source_score": score,
        "authority_score": authority,
        "snippet": "Source content about Redis caching and latency.",
    }


def _good_quality(sources=None, claims=None):
    if sources is None:
        sources = [
            _source("https://redis.io/docs/a"),
            _source("https://docs.python.org/guide"),
        ]
    if claims is None:
        claims = [
            Claim(
                claim_id="C1",
                claim_text="Redis stores frequently requested values in memory.",
                evidence_ids=["E1"],
                source_urls=[sources[0]["url"]],
                confidence=.9,
            )
        ]
    evidence = [
        Evidence(
            evidence_id="E1",
            source_url=sources[0]["url"],
            source_title=sources[0]["title"],
            excerpt="Redis stores frequently requested values in memory.",
            supporting_text="The cache stores frequently requested values.",
            relevance_score=.9,
            confidence=.9,
        )
    ]
    return {
        "sources": calculate_domain_diversity(sources, claims),
        "evidence": calculate_evidence_metrics(evidence, claims, {"rejected": 0}),
        "citations": {
            "total_claims": len(claims),
            "citation_coverage": 1,
            "invalid_citations": 0,
        },
    }


def _candidate(url="https://docs.python.org/guide"):
    return (
        "Result 1\n"
        "Title: Python caching guide\n"
        f"URL: {url}\n"
        "Snippet: Redis caching reduces application latency by avoiding repeated database reads."
    )


class _FakeSearch:
    def __init__(self, answers):
        self.answers = answers
        self.queries = []

    def invoke(self, request):
        query = request["query"]
        self.queries.append(query)
        return self.answers.get(query, "")


def _recovery_state(feedback=None, *, claims=None, evidence=None, processed=None):
    sources = [
        _source("https://redis.io/docs/first"),
        _source("https://redis.io/docs/second"),
    ]
    if claims is None:
        claims = [
            Claim(
                claim_id="C1",
                claim_text="Redis stores frequently requested values in memory.",
                evidence_ids=["E1"],
                source_urls=[sources[0]["url"]],
                confidence=.9,
            )
        ]
    if evidence is None:
        evidence = [
            Evidence(
                evidence_id="E1",
                source_url=sources[0]["url"],
                source_title=sources[0]["title"],
                excerpt="Redis stores frequently requested values in memory.",
                supporting_text="The cache stores frequently requested values.",
                relevance_score=.9,
                confidence=.9,
            )
        ]
    quality = _good_quality(sources, claims)
    quality["sources"] = calculate_domain_diversity(sources, claims)
    return {
        "topic": "Redis caching and application latency",
        "report": "Existing draft.",
        "final_report": "",
        "ranked_sources": sources,
        "documents": [],
        "evidence": [item.model_dump() for item in evidence],
        "claims": [item.model_dump() for item in claims],
        "quality_metrics": quality,
        "feedback": feedback or (
            "Score: 6/10\nRevision Required: YES\n"
            "Research Gaps:\n- Multiple sources from one domain need independent corroboration.\n"
            "Writing Issues:\n- None\n"
            "Targeted Queries:\n- independent authoritative Redis caching implementation sources"
        ),
        "research_rounds": 0,
        "revision_count": 0,
        "targeted_queries": [],
        "targeted_queries_attempted": [],
        "processed_urls": processed or [source["url"] for source in sources],
        "recovery_metrics": {},
    }


def _install_recovery_mocks(monkeypatch, search, *, scraped=None, extraction=None):
    monkeypatch.setattr(graph, "web_search", search)
    scrape_calls = []

    def scrape(urls):
        scrape_calls.extend(urls)
        return scraped or [
            "Title: Python caching guide\n"
            "URL: https://docs.python.org/guide\n"
            "Content:\n"
            "Redis stores frequently requested values in memory to avoid repeated database reads."
        ]

    monkeypatch.setattr(graph, "scrape_urls_parallel", scrape)
    monkeypatch.setattr(
        graph,
        "_cached_llm",
        lambda stage, key, invoke: extraction if extraction is not None else invoke(),
    )
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    return scrape_calls


def test_quality_routing_separates_research_writing_and_hard_failures():
    good = _good_quality()
    feedback = (
        "Score: 9/10\nRevision Required: YES\n"
        "Research Gaps:\n- None\n"
        "Writing Issues:\n- Clarify the cache-aside explanation.\n"
        "Targeted Queries:\n- None"
    )
    assert graph._assessment({"quality_metrics": good}, feedback)["decision"] == "revision"

    unsupported = _good_quality()
    unsupported["evidence"]["evidence_coverage"] = .5
    unsupported["evidence"]["unsupported_claims"] = 1
    assert graph._assessment({"quality_metrics": unsupported}, "Score: 10/10\nRevision Required: NO")["decision"] == "research"

    citation_failure = _good_quality()
    citation_failure["citations"]["invalid_citations"] = 1
    citation_assessment = graph._assessment(
        {"quality_metrics": citation_failure},
        "Score: 10/10\nRevision Required: NO\nResearch Gaps:\n- None\nWriting Issues:\n- None",
    )
    assert citation_assessment["decision"] == "revision"
    assert citation_assessment["mandatory_failures"]


def test_critic_scores_preserve_missing_malformed_and_out_of_range_states():
    assert graph._parse_critic_score("Score: 8/10") == (8.0, 8.0, "valid")
    assert graph._parse_critic_score("Score: 10/10") == (10.0, 10.0, "valid")
    assert graph._parse_critic_score("Revision Required: NO") == (None, None, "missing")
    assert graph._parse_critic_score("Score: excellent") == (None, None, "malformed")
    assert graph._parse_critic_score("Score: 11/10") == (None, 11.0, "out_of_range")

    assessment = graph._assessment(
        {"quality_metrics": _good_quality()},
        "Revision Required: NO",
        score_status="missing",
    )
    assert assessment["decision"] == "revision"
    assert any("score is missing" in item for item in assessment["mandatory_failures"])


def test_targeted_query_generation_is_bounded_deduplicated_and_not_broad():
    claim = Claim(
        claim_id="C1",
        claim_text="Caching lowers repeated database reads.",
        evidence_ids=[],
        source_urls=[],
    )
    queries = graph._targeted_queries(
        "Redis caching",
        ["Evidence coverage is insufficient."],
        [claim],
        [
            "  independent authoritative sources about Redis caching ",
            "INDEPENDENT authoritative sources about Redis caching",
            "x" * (graph.MAX_RECOVERY_QUERY_CHARS + 1),
            None,
        ],
    )
    assert len(queries) <= graph.MAX_RECOVERY_QUERIES
    assert len({query.casefold() for query in queries}) == len(queries)
    assert "Redis caching" not in queries
    assert any("Caching lowers repeated database reads" in query for query in queries)


def test_recovery_acquires_independent_source_and_merges_valid_evidence(monkeypatch):
    query_a = "independent authoritative Redis caching implementation sources"
    query_b = "Redis cache backend latency evidence"
    search = _FakeSearch({
        query_a: _candidate(),
        query_b: _candidate("https://docs.python.org/guide/?utm_source=search"),
    })
    scrape_calls = _install_recovery_mocks(
        monkeypatch,
        search,
        extraction=json.dumps({
            "evidence": [{
                "source_url": "https://docs.python.org/guide",
                "excerpt": "Redis stores frequently requested values in memory",
                "supporting_text": "The cache avoids repeated database reads.",
                "relevance_score": .9,
                "confidence": .9,
            }],
            "claims": [{
                "claim_text": "Redis stores frequently requested values in memory.",
                "evidence_indices": [0],
                "confidence": .95,
            }],
        }),
    )
    state = _recovery_state()
    state["feedback"] = (
        "Revision Required: YES\n"
        "Research Gaps:\n- Multiple sources from one domain need independent corroboration.\n"
        "Targeted Queries:\n"
        f"- {query_a}\n- {query_b}"
    )

    result = graph.research_recovery_node(state)

    assert search.queries == [query_a, query_b]
    assert scrape_calls == ["https://docs.python.org/guide"]
    assert result["recovery_progress"] is True
    assert result["research_rounds"] == 1
    assert len(result["evidence"]) == 2
    assert result["evidence"][0]["evidence_id"] == "E1"
    assert result["evidence"][1]["evidence_id"] == "E2"
    assert result["claims"][0]["claim_id"] == "C1"
    assert result["claims"][0]["evidence_ids"] == ["E1", "E2"]
    assert result["claims"][0]["source_urls"] == [
        "https://redis.io/docs/first",
        "https://docs.python.org/guide",
    ]
    assert result["quality_metrics"]["sources"]["unique_domains"] == 2
    assert result["recovery_metrics"]["duplicate_urls"] == 1
    assert result["report"] == ""


def test_two_recovery_rounds_accumulate_urls_fetches_and_evidence(monkeypatch):
    first_url = "https://docs.python.org/guide/a"
    second_url = "https://docs.python.org/guide/b"
    query_one = "independent Redis implementation evidence"
    query_two = "Redis caching corroboration details"
    query_three = "Redis latency independent evidence"
    search = _FakeSearch({
        query_one: _candidate(first_url),
        query_two: ("\n" + "-" * 70 + "\n").join((
            _candidate(first_url + "?utm_source=duplicate"),
            _candidate("https://redis.io/docs/first"),
        )),
        query_three: ("\n" + "-" * 70 + "\n").join((
            _candidate(second_url),
            _candidate(first_url),
        )),
    })
    scrape_calls = []
    extraction_calls = []

    def scrape(urls):
        scrape_calls.extend(urls)
        return [
            f"Title: Python caching guide\nURL: {url}\nContent:\n"
            "Redis stores frequently requested values in memory to avoid repeated database reads."
            for url in urls
        ]

    def extract(stage, key, invoke):
        url = scrape_calls[len(extraction_calls)]
        extraction_calls.append(url)
        return json.dumps({
            "evidence": [{
                "source_url": url,
                "excerpt": "Redis stores frequently requested values in memory",
                "supporting_text": "The cache avoids repeated database reads.",
                "relevance_score": .9,
                "confidence": .9,
            }],
            "claims": [{
                "claim_text": "Redis stores frequently requested values in memory.",
                "evidence_indices": [0],
                "confidence": .9,
            }],
        })

    monkeypatch.setattr(graph, "web_search", search)
    monkeypatch.setattr(graph, "scrape_urls_parallel", scrape)
    monkeypatch.setattr(graph, "_cached_llm", extract)
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    state = _recovery_state()
    state["feedback"] = (
        "Revision Required: YES\nResearch Gaps:\n- Independent corroboration needed.\n"
        f"Targeted Queries:\n- {query_one}\n- {query_two}"
    )

    first = graph.research_recovery_node(state)
    next_state = {**state, **first}
    next_state["feedback"] = (
        "Revision Required: YES\nResearch Gaps:\n- Independent corroboration needed.\n"
        f"Targeted Queries:\n- {query_three}"
    )
    second = graph.research_recovery_node(next_state)

    assert search.queries == [query_one, query_two, query_three]
    assert scrape_calls == [first_url, second_url]
    assert second["research_rounds"] == 2
    recovery_metrics = second["recovery_metrics"]
    assert recovery_metrics["rounds_used"] == 2
    assert recovery_metrics["targeted_queries_attempted"] == [query_one, query_two, query_three]
    assert recovery_metrics["targeted_queries_known"] is True
    assert recovery_metrics["new_candidate_urls"] == 2
    assert recovery_metrics["duplicate_urls"] == 3
    assert recovery_metrics["newly_accepted_sources"] == 2
    assert recovery_metrics["successful_fetches"] == 2
    assert recovery_metrics["failed_fetches"] == 0
    assert recovery_metrics["accepted_evidence"] == 2
    assert recovery_metrics["new_validated_evidence"] == 2
    assert recovery_metrics["rejected_evidence"] == 0
    assert recovery_metrics["errors"] == []
    assert [item["round"] for item in recovery_metrics["diagnostics_by_round"]] == [1, 2]
    assert [item["new_validated_evidence"] for item in recovery_metrics["diagnostics_by_round"]] == [1, 1]
    assert [item["fetch"]["fetch_successes"] for item in recovery_metrics["diagnostics_by_round"]] == [1, 1]
    assert recovery_metrics["diagnostics_by_round"][0]["duplicate_urls"] == 2
    assert recovery_metrics["diagnostics_by_round"][1]["duplicate_urls"] == 1
    assert len(second["evidence"]) == 3
    assert second["claims"][0]["evidence_ids"] == ["E1", "E2", "E3"]


def test_recovery_searches_for_unsupported_claim_instead_of_rewriting(monkeypatch):
    claim = Claim(
        claim_id="C1",
        claim_text="Redis cache invalidation is always automatic.",
        evidence_ids=[],
        source_urls=[],
    )
    query = f"reliable source evidence for {claim.claim_text}"
    search = _FakeSearch({query: _candidate()})
    _install_recovery_mocks(
        monkeypatch,
        search,
        extraction=json.dumps({
            "evidence": [{
                "source_url": "https://docs.python.org/guide",
                "excerpt": "Redis stores frequently requested values in memory",
                "supporting_text": "Caching stores frequently used values.",
                "relevance_score": .8,
                "confidence": .8,
            }],
            "claims": [{
                "claim_text": claim.claim_text,
                "evidence_indices": [0],
                "confidence": .8,
            }],
        }),
    )
    state = _recovery_state(claims=[claim], evidence=[])
    state["quality_metrics"]["sources"] = _good_quality()["sources"]
    state["quality_metrics"]["evidence"]["evidence_coverage"] = 0
    state["quality_metrics"]["evidence"]["unsupported_claims"] = 1
    state["feedback"] = "Revision Required: YES\nResearch Gaps:\n- None\nTargeted Queries:\n- None"

    result = graph.research_recovery_node(state)

    assert search.queries == [query]
    assert result["recovery_progress"] is True
    assert result["claims"][0]["evidence_ids"] == ["E1"]
    assert result["claims"][0]["source_urls"] == ["https://docs.python.org/guide"]
    assert result["claims"][0]["confidence"] == .8


def test_recovery_diagnostics_capture_source_scoring_rejection(monkeypatch):
    query = "authoritative independent Redis caching evidence"
    search = _FakeSearch({query: _candidate("https://example.com/redis-caching")})
    scrape_calls = _install_recovery_mocks(monkeypatch, search)
    state = _recovery_state()
    state["feedback"] = (
        "Revision Required: YES\nResearch Gaps:\n- More evidence needed.\n"
        f"Targeted Queries:\n- {query}"
    )

    result = graph.research_recovery_node(state)

    assert search.queries == [query]
    assert scrape_calls == []
    round_data = result["recovery_metrics"]["diagnostics_by_round"][0]
    assert round_data["search"]["results_returned"] == 1
    assert round_data["search"]["parsed_candidates"] == 1
    assert round_data["search"]["accepted_sources"] == 0
    assert round_data["search"]["source_scoring_rejections"] == 1
    assert round_data["search"]["candidate_decisions"][0]["authority_score"] < graph.MIN_SOURCE_SCORE
    assert round_data["search"]["candidate_decisions"][0]["rejection_reason"] == "authority_below_minimum"
    assert round_data["fetch"]["scrape_attempts"] == 0
    assert round_data["termination_reason"] == "no_research_progress"


def test_fetch_and_extraction_failures_are_distinguished_from_unreported_outcomes():
    from research_quality import parse_documents

    documents = parse_documents(
        "Title: Fetch failure\nURL: https://example.org/fetch\n"
        "Content:\nCould not fetch page.\nFailure category: fetch_empty.\n"
        + "=" * 70 + "\n"
        "Title: Extraction failure\nURL: https://example.org/extract\n"
        "Content:\nCould not extract article body.\nFailure category: extraction_empty_or_short."
    )

    diagnostics = graph._document_pipeline_diagnostics(documents, attempted=3)

    assert diagnostics["fetch_successes"] == 1
    assert diagnostics["fetch_failures"] == 1
    assert diagnostics["extraction_failures"] == 1
    assert diagnostics["unreported_outcomes"] == 1
    assert diagnostics["outcomes"] == {"extraction_failed": 1, "fetch_failed": 1}


def test_recovery_does_not_accept_failed_fetches_or_fabricate_evidence(monkeypatch):
    query = "independent authoritative Redis caching implementation sources"
    search = _FakeSearch({query: _candidate()})
    scrape_calls = _install_recovery_mocks(
        monkeypatch,
        search,
        scraped=[
            "Title: Python caching guide\n"
            "URL: https://docs.python.org/guide\n"
            "Content:\nCould not fetch page."
        ],
        extraction=json.dumps({"evidence": [], "claims": []}),
    )
    state = _recovery_state()
    state["quality_metrics"]["citations"] = {
        "total_claims": 3,
        "citation_coverage": .67,
        "invalid_citations": 0,
    }
    state["quality_metrics"]["critic"] = {
        "model_score": 8,
        "effective_score": 5,
        "approval": False,
    }
    state["critique"] = {"overall_score": 5, "approval": False}
    state["citation_map"] = {"C1": ["https://redis.io/docs/first"]}
    state["unresolved_issues"] = ["Previous unresolved writing issue."]
    state["recovery_metrics"] = {
        "new_candidate_urls": 2,
        "duplicate_urls": 4,
        "newly_accepted_sources": 1,
        "successful_fetches": 1,
        "failed_fetches": 2,
        "accepted_evidence": 3,
        "rejected_evidence": 1,
        "targeted_queries_attempted": ["previous recovery query"],
        "errors": ["Earlier search failed."],
    }
    state["research_rounds"] = 1
    state["targeted_queries_attempted"] = ["previous recovery query"]
    state["feedback"] = (
        "Revision Required: YES\nResearch Gaps:\n- Multiple sources from one domain.\n"
        f"Targeted Queries:\n- {query}"
    )

    result = graph.research_recovery_node(state)
    final_state = {**state, **result}

    assert scrape_calls == ["https://docs.python.org/guide"]
    assert result["recovery_progress"] is False
    assert result["evidence"] == state["evidence"]
    assert result["claims"] == state["claims"]
    assert result["termination_reason"] == "no_research_progress"
    recovery_metrics = result["recovery_metrics"]
    assert recovery_metrics["rounds_used"] == 2
    assert recovery_metrics["targeted_queries_attempted"] == ["previous recovery query", query]
    assert recovery_metrics["targeted_queries_known"] is True
    assert recovery_metrics["new_candidate_urls"] == 3
    assert recovery_metrics["duplicate_urls"] == 4
    assert recovery_metrics["newly_accepted_sources"] == 2
    assert recovery_metrics["successful_fetches"] == 1
    assert recovery_metrics["failed_fetches"] == 3
    assert recovery_metrics["accepted_evidence"] == 3
    assert recovery_metrics["rejected_evidence"] == 1
    assert recovery_metrics["new_validated_evidence"] is None
    assert recovery_metrics["extraction_failures"] is None
    assert recovery_metrics["errors"] == ["Earlier search failed."]
    assert recovery_metrics["diagnostics_by_round"][-1]["termination_reason"] == "no_research_progress"
    assert recovery_metrics["diagnostics_by_round"][-1]["fetch"]["fetch_failures"] == 1
    assert result["quality_metrics"]["citations"] == state["quality_metrics"]["citations"]
    assert result["quality_metrics"]["critic"] == state["quality_metrics"]["critic"]
    assert final_state["critique"] == state["critique"]
    assert final_state["citation_map"] == state["citation_map"]
    assert final_state["report"] == state["report"]
    assert final_state["feedback"] == state["feedback"]
    assert final_state["quality_metrics"]["critic"]["model_score"] == 8
    assert final_state["quality_metrics"]["critic"]["effective_score"] == 5
    assert final_state["quality_metrics"]["citations"]["citation_coverage"] == .67
    assert "Previous unresolved writing issue." in result["unresolved_issues"]


def test_missing_prior_recovery_counters_remain_unmeasured(monkeypatch):
    query = "independent authoritative Redis caching implementation sources"
    search = _FakeSearch({query: _candidate()})
    _install_recovery_mocks(
        monkeypatch,
        search,
        extraction=json.dumps({"evidence": [], "claims": []}),
    )
    state = _recovery_state()
    state["research_rounds"] = 1
    state["recovery_metrics"] = {}
    state["targeted_queries_attempted"] = None
    state["feedback"] = (
        "Revision Required: YES\nResearch Gaps:\n- Multiple sources from one domain.\n"
        f"Targeted Queries:\n- {query}"
    )

    result = graph.research_recovery_node(state)

    assert result["recovery_metrics"]["new_candidate_urls"] is None
    assert result["recovery_metrics"]["duplicate_urls"] is None
    assert result["quality_metrics"]["recovery"]["new_candidate_urls"] is None
    assert result["quality_metrics"]["recovery"]["queries_attempted"] is None


def test_recovery_rejects_untraceable_excerpt(monkeypatch):
    query = "independent authoritative Redis caching implementation sources"
    search = _FakeSearch({query: _candidate()})
    _install_recovery_mocks(
        monkeypatch,
        search,
        extraction=json.dumps({
            "evidence": [{
                "source_url": "https://docs.python.org/guide",
                "excerpt": "This invented excerpt does not exist in the fetched page.",
                "supporting_text": "Unverified assertion.",
                "relevance_score": .9,
                "confidence": .9,
            }],
            "claims": [{
                "claim_text": "Invented claim.",
                "evidence_indices": [0],
                "confidence": .9,
            }],
        }),
    )
    state = _recovery_state()
    state["feedback"] = (
        "Revision Required: YES\nResearch Gaps:\n- Sources lack diversity.\n"
        f"Targeted Queries:\n- {query}"
    )

    result = graph.research_recovery_node(state)

    assert result["recovery_progress"] is False
    assert result["evidence"] == state["evidence"]
    assert result["claims"] == state["claims"]
    assert result["recovery_metrics"]["rejected_evidence"] == 1
    assert "Targeted research produced no new validated evidence." in result["unresolved_issues"]


def test_no_queries_and_duplicate_search_results_terminate_without_duplicate_fetch(monkeypatch):
    query = "independent authoritative Redis caching implementation sources"
    search = _FakeSearch({
        query: _candidate(),
        "independent Redis cache latency results": _candidate(
            "https://docs.python.org/guide/?utm_source=duplicate"
        ),
    })
    scrape_calls = _install_recovery_mocks(
        monkeypatch,
        search,
        extraction=json.dumps({"evidence": [], "claims": []}),
    )
    state = _recovery_state()
    state["feedback"] = (
        "Revision Required: YES\nResearch Gaps:\n- Multiple sources from one domain.\n"
        f"Targeted Queries:\n- {query}\n- independent Redis cache latency results"
    )
    result = graph.research_recovery_node(state)
    assert len(scrape_calls) == 1
    assert result["recovery_metrics"]["duplicate_urls"] == 1
    assert graph._route_after_recovery(result) == "export"

    no_query = _recovery_state()
    no_query["quality_metrics"]["sources"] = _good_quality()["sources"]
    no_query["feedback"] = "Revision Required: YES\nResearch Gaps:\n- None\nTargeted Queries:\n- None"
    monkeypatch.setattr(graph, "web_search", _FakeSearch({}))
    no_query_result = graph.research_recovery_node(no_query)
    assert no_query_result["targeted_queries_attempted"] == []
    assert graph._route_after_recovery(no_query_result) == "export"


def test_graph_uses_independent_bounded_research_and_revision_routes(monkeypatch):
    visited = []
    research_calls = []
    revision_calls = []
    monkeypatch.setattr(graph, "MAX_RESEARCH_ROUNDS", 1)
    monkeypatch.setattr(graph, "MAX_REVISION_ITERATIONS", 1)

    def search(state):
        visited.append("search")
        return {"ranked_sources": []}

    def reader(state):
        visited.append("reader")
        return {"reader_summary": "evidence"}

    def writer(state):
        visited.append("writer")
        return {"report": f"draft-{len(visited)}"}

    def critic(state):
        visited.append("critic")
        if state.get("research_rounds", 0) == 0:
            return {
                "quality_route": "research",
                "needs_revision": True,
                "feedback": "weak evidence",
            }
        if state.get("revision_count", 0) == 0:
            return {
                "quality_route": "revision",
                "needs_revision": True,
                "feedback": "writing only",
            }
        return {
            "quality_route": "revision",
            "needs_revision": True,
            "termination_reason": "revision_budget_exhausted",
            "unresolved_issues": ["Writing remains unclear."],
            "feedback": "writing only",
        }

    def recover(state):
        visited.append("research_recovery")
        research_calls.append(1)
        return {"research_rounds": state.get("research_rounds", 0) + 1, "recovery_progress": True}

    def revision(state):
        visited.append("revision")
        revision_calls.append(1)
        return {"revision_count": state.get("revision_count", 0) + 1, "report": "revised"}

    def export(state):
        visited.append("export")
        return {"output_paths": {}}

    for name, node in (
        ("search_node", search),
        ("reader_node", reader),
        ("writer_node", writer),
        ("critic_node", critic),
        ("research_recovery_node", recover),
        ("revision_node", revision),
        ("export_node", export),
    ):
        monkeypatch.setattr(graph, name, node)

    graph.build_research_graph().invoke({"topic": "bounded-loop"})

    assert research_calls == [1]
    assert revision_calls == [1]
    assert visited == [
        "search",
        "reader",
        "writer",
        "critic",
        "research_recovery",
        "writer",
        "critic",
        "revision",
        "critic",
        "export",
    ]


def test_critic_sets_truthful_termination_at_both_budget_caps(monkeypatch):
    monkeypatch.setattr(graph, "_cached_llm", lambda stage, key, invoke: (
        "Score: 10/10\nRevision Required: YES\n"
        "Research Gaps:\n- More independent evidence is needed.\n"
        "Writing Issues:\n- Clarify the conclusion.\nTargeted Queries:\n- None"
    ))
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    state = {
        "topic": "Redis",
        "report": "Draft with an unresolved gap.",
        "research_rounds": graph.MAX_RESEARCH_ROUNDS,
        "revision_count": graph.MAX_REVISION_ITERATIONS,
        "quality_metrics": _good_quality(),
    }
    state["quality_metrics"]["sources"]["unique_domains"] = 1
    state["quality_metrics"]["sources"]["total_sources"] = 2

    result = graph.critic_node(state)

    assert result["quality_approved"] is False
    assert result["quality_route"] == "research"
    assert result["termination_reason"] == "research_and_revision_budgets_exhausted"
    assert result["unresolved_issues"]
    assert graph._route_after_critic({**state, **result}) == "export"


def test_research_cap_uses_remaining_revision_budget_for_writing_or_validation(monkeypatch):
    monkeypatch.setattr(graph, "_cached_llm", lambda stage, key, invoke: (
        "Score: 8/10\nRevision Required: YES\n"
        "Research Gaps:\n- More independent evidence is needed.\n"
        "Writing Issues:\n- Clarify the conclusion.\n"
        "Targeted Queries:\n- None"
    ))
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    state = {
        "topic": "Redis",
        "report": "Draft with an unresolved gap.",
        "research_rounds": graph.MAX_RESEARCH_ROUNDS,
        "revision_count": 0,
        "quality_metrics": _good_quality(),
    }
    state["quality_metrics"]["sources"]["unique_domains"] = 1
    state["quality_metrics"]["sources"]["total_sources"] = 2

    result = graph.critic_node(state)

    assert result["quality_route"] == "revision"
    assert result["termination_reason"] == ""
    assert result["quality_approved"] is False
    assert graph._route_after_critic({**state, **result}) == "revision"
    assert result["unresolved_issues"]


def test_export_marks_failed_gates_not_approved_and_discloses_budget(monkeypatch):
    captured = {}
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    monkeypatch.setattr(
        graph,
        "save_report",
        lambda **kwargs: captured.update(kwargs) or {"markdown": "report.md", "pdf": "report.pdf"},
    )
    state = {
        "topic": "Redis",
        "report": "Latest report.",
        "quality_approved": False,
        "termination_reason": "research_budget_exhausted",
        "research_rounds": graph.MAX_RESEARCH_ROUNDS,
        "revision_count": graph.MAX_REVISION_ITERATIONS,
        "unresolved_issues": ["Insufficient independent evidence."],
        "quality_metrics": {
            "sources": {"unique_domains": 1, "unique_sources": 2, "total_sources": 2, "diversity_ratio": .5, "average_source_quality": 7},
            "evidence": {"evidence_coverage": 1, "supported_claims": 1, "unsupported_claims": 0, "total_claims": 1, "average_claim_confidence": .9},
            "citations": {"citation_coverage": 1},
            "critic": {"model_score": 9, "effective_score": 5},
            "recovery": {
                "queries_attempted": ["query"],
                "new_candidate_urls": 3,
                "newly_accepted_sources": 2,
                "duplicate_urls": 1,
                "successful_fetches": 2,
                "failed_fetches": 1,
                "accepted_evidence": 4,
                "rejected_evidence": 2,
            },
        },
    }

    result = graph.export_node(state)

    assert "NOT APPROVED" in captured["report"]
    assert "research_budget_exhausted" in captured["report"]
    assert "Insufficient independent evidence." in captured["report"]
    assert "Recovery queries / candidate URLs / new sources: 1 / 3 / 2" in captured["report"]
    assert "Duplicate recovery URLs / successful / failed fetches: 1 / 2 / 1" in captured["report"]
    assert "Recovery evidence accepted / rejected: 4 / 2" in captured["report"]
    assert result["quality_approved"] is False


def test_missing_citation_metrics_are_unknown_and_gate_fails_closed():
    quality = _good_quality()
    quality.pop("citations")

    assessment = graph._assessment({"quality_metrics": quality}, "Score: 9/10\nRevision Required: NO")

    assert "citation_coverage" not in quality.get("citations", {})
    assert assessment["decision"] == "revision"
    assert "Citation coverage was not measured." in assessment["mandatory_failures"]
    assert "Citation validation metrics are incomplete." in assessment["mandatory_failures"]


def test_unapproved_export_shows_actual_critic_and_unmeasured_citations(monkeypatch):
    captured = {}
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    monkeypatch.setattr(
        graph,
        "save_report",
        lambda **kwargs: captured.update(kwargs) or {"markdown": "report.md", "pdf": "report.pdf"},
    )
    state = {
        "topic": "Redis",
        "report": "Latest report.",
        "quality_approved": False,
        "termination_reason": "no_research_progress",
        "research_rounds": 2,
        "revision_count": 0,
        "critic_evaluations": [
            {"model_score": 8, "effective_score": 5, "approval": False}
        ],
        "quality_metrics": {
            "sources": {
                "unique_domains": 2,
                "unique_sources": 4,
                "total_sources": 4,
                "diversity_ratio": .5,
                "average_source_quality": 6.5,
            },
            "evidence": {
                "evidence_coverage": .88,
                "supported_claims": 15,
                "unsupported_claims": 2,
                "total_claims": 17,
                "average_claim_confidence": .87,
            },
            "recovery": {
                "rounds_used": 2,
                "queries_attempted": ["first", "second"],
                "new_candidate_urls": 3,
                "duplicate_urls": 2,
                "newly_accepted_sources": 2,
                "successful_fetches": 3,
                "failed_fetches": 1,
                "accepted_evidence": 4,
                "rejected_evidence": 1,
            },
        },
    }

    result = graph.export_node(state)
    report = captured["report"]
    assert "Quality status: NOT APPROVED" in report
    assert "Citation coverage: not measured" in report
    assert "Invalid citations: not measured" in report
    assert "Critic score: model 8.0/10; effective 5.0/10" in report
    assert "Recovery queries / candidate URLs / new sources: 2 / 3 / 2" in report
    assert "Duplicate recovery URLs / successful / failed fetches: 2 / 3 / 1" in report
    assert "Recovery evidence accepted / rejected: 4 / 1" in report
    assert result["quality_approved"] is False


def test_empty_research_export_marks_zero_denominator_metrics_unmeasured(monkeypatch):
    captured = {}
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    monkeypatch.setattr(
        graph,
        "save_report",
        lambda **kwargs: captured.update(kwargs) or {"markdown": "report.md", "pdf": "report.pdf"},
    )
    state = {
        "topic": "HTTP/3",
        "report": "Evidence is unavailable.",
        "quality_approved": False,
        "termination_reason": "no_research_progress",
        "research_rounds": 1,
        "revision_count": 0,
        "quality_metrics": {
            "sources": {
                "unique_domains": 0,
                "unique_sources": 0,
                "total_sources": 0,
                "diversity_ratio": None,
                "average_source_quality": None,
            },
            "evidence": {
                "evidence_coverage": None,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "total_claims": 0,
                "average_claim_confidence": None,
            },
            "citations": {
                "citation_coverage": None,
                "invalid_citations": None,
            },
        },
    }

    graph.export_node(state)
    report = captured["report"]

    assert "Source diversity: not measured" in report
    assert "Average source quality: not measured" in report
    assert "Evidence coverage: not measured" in report
    assert "Citation coverage: not measured" in report
    assert "Average claim confidence: not measured" in report


def test_export_quality_metrics_are_identical_in_markdown_and_pdf(tmp_path, monkeypatch):
    from pypdf import PdfReader
    import export

    monkeypatch.setattr(export, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(graph, "_save_stage", lambda *args: None)
    state = {
        "topic": "Redis",
        "report": "Latest report.",
        "quality_approved": False,
        "termination_reason": "no_research_progress",
        "research_rounds": 1,
        "revision_count": 0,
        "critic": {"overall_score": 4},
        "quality_metrics": {
            "sources": {
                "unique_domains": 1,
                "unique_sources": 2,
                "total_sources": 2,
                "diversity_ratio": .5,
                "average_source_quality": 6.4,
            },
            "evidence": {
                "evidence_coverage": .75,
                "supported_claims": 3,
                "unsupported_claims": 1,
                "total_claims": 4,
                "average_claim_confidence": .81,
            },
            "citations": {
                "citation_coverage": .5,
                "invalid_citations": 1,
            },
            "critic": {
                "model_score": 8,
                "effective_score": 5,
            },
            "recovery": {
                "rounds_used": 1,
                "queries_attempted": ["one"],
                "new_candidate_urls": 2,
                "duplicate_urls": 1,
                "newly_accepted_sources": 1,
                "successful_fetches": 1,
                "failed_fetches": 0,
                "accepted_evidence": 2,
                "rejected_evidence": 0,
            },
        },
    }

    result = graph.export_node(state)
    markdown_path = result["output_paths"]["markdown"]
    pdf_path = result["output_paths"]["pdf"]
    markdown_text = open(markdown_path, encoding="utf-8").read()
    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)

    expected = (
        "Quality status: NOT APPROVED",
        "Termination reason: no_research_progress",
        "Research recovery rounds: 1",
        "Recovery queries / candidate URLs / new sources: 1 / 2 / 1",
        "Duplicate recovery URLs / successful / failed fetches: 1 / 1 / 0",
        "Recovery evidence accepted / rejected: 2 / 0",
        "Evidence coverage: 75%",
        "Citation coverage: 50%",
        "Invalid citations: 1",
        "Critic score: model 8.0/10; effective 5.0/10",
    )
    for item in expected:
        assert item in markdown_text
        assert item in pdf_text


def test_resume_state_retains_counters_evidence_and_unresolved_state(monkeypatch):
    import cache
    from research_quality import RESEARCH_DATA_VERSION, RESEARCH_PROMPT_VERSION

    monkeypatch.setenv("OPENAI_MODEL", "gpt-current")
    checkpoint = {
        "topic": "Redis",
        "_research_version": RESEARCH_DATA_VERSION,
        "_research_prompt_version": RESEARCH_PROMPT_VERSION,
        "_llm_models": ["gpt-current"],
        "research_rounds": 1,
        "revision_count": 1,
        "processed_urls": ["https://redis.io/docs"],
        "evidence": [{"evidence_id": "E7"}],
        "claims": [{"claim_id": "C5"}],
        "termination_reason": "quality_pending",
        "unresolved_issues": ["Need an independent source."],
    }
    cache.set_checkpoint("Redis", checkpoint)

    resumed = cache.load_initial_state("Redis")

    assert resumed["research_rounds"] == 1
    assert resumed["revision_count"] == 1
    assert resumed["processed_urls"] == ["https://redis.io/docs"]
    assert resumed["evidence"] == [{"evidence_id": "E7"}]
    assert resumed["claims"] == [{"claim_id": "C5"}]
    assert resumed["termination_reason"] == "quality_pending"
    assert resumed["unresolved_issues"] == ["Need an independent source."]


def test_recovery_cache_hits_misses_are_counted_by_existing_tools(monkeypatch):
    from metrics import get_metrics, reset_metrics

    reset_metrics()
    monkeypatch.setattr(graph, "web_search", _FakeSearch({
        "independent authoritative Redis caching implementation sources": _candidate()
    }))
    _install_recovery_mocks(
        monkeypatch,
        graph.web_search,
        extraction=json.dumps({"evidence": [], "claims": []}),
    )
    state = _recovery_state()
    state["quality_metrics"]["sources"] = _good_quality()["sources"]
    result = graph.research_recovery_node(state)

    assert result["research_rounds"] == 1
    assert get_metrics().recovery_searches == 1
    reset_metrics()

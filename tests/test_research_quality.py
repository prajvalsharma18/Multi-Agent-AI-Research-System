from research_models import Claim
from research_quality import (calculate_domain_diversity, calculate_evidence_metrics, materialize_citations,
    parse_documents, parse_extraction, quality_gate, validate_extraction)
from source_scoring import canonicalize_url, domain_for_url, score_source


def test_url_normalization_and_registrable_domain():
    assert canonicalize_url("HTTPS://WWW.Example.com/a/?utm_source=x#part") == "https://example.com/a"
    assert domain_for_url("https://docs.redis.io/guide") == "redis.io"
    assert domain_for_url("https://www.gov.uk/page") == "gov.uk"
    assert canonicalize_url("javascript:bad") == ""


def test_source_scoring_breakdown_and_diversity_bonus():
    source = {"url": "https://redis.io/docs", "title": "Redis cache latency", "snippet": "Redis caching reduces latency"}
    total, parts = score_source(source, "Redis cache latency", True)
    second, repeated = score_source(source, "Redis cache latency", False)
    assert parts["authority"] >= 5 and parts["relevance"] > 0
    assert parts["diversity"] == 10 and total > second
    assert set(parts) == {"authority", "relevance", "recency", "evidence_quality", "diversity", "total"}


def test_document_evidence_traceability_and_claim_support():
    scraped = "Title: Redis guide\nURL: https://redis.io/docs\nContent:\nRead-through caching reduces database load."
    docs = parse_documents(scraped)
    raw = '{"evidence":[{"source_url":"https://redis.io/docs","excerpt":"Read-through   caching reduces database load.","supporting_text":"reduces backend requests","relevance_score":0.9,"confidence":0.8},{"source_url":"https://fake.invalid","excerpt":"made up","supporting_text":"x"},{"source_url":"https://redis.io/docs","excerpt":"not present","supporting_text":"x"}],"claims":[{"claim_text":"It reduces load.","evidence_indices":[0],"confidence":0.7},{"claim_text":"It is always faster.","evidence_indices":[1],"confidence":0.8},{"claim_text":"unsupported","evidence_indices":[]}]} '
    evidence, claims, notes, stats = validate_extraction("Redis caching", docs, parse_extraction(raw))
    assert len(evidence) == 1 and stats["rejected"] == 2
    assert claims[0].supported and claims[0].source_urls == ["https://redis.io/docs"]
    assert not claims[1].supported and claims[1].confidence == 0
    assert len(notes) == 3


def test_parse_extraction_diagnostics_distinguish_failure_boundaries():
    for raw, expected_status in (
        ("not json", "malformed_json"),
        ('{"evidence": "not a list"}', "schema_invalid"),
        ('{"evidence": [], "claims": []}', "empty_extraction"),
        ('{"evidence": [], "claims": [{"claim_text": "x"}]}', "parsed"),
    ):
        diagnostics = {}
        parse_extraction(raw, diagnostics)
        assert diagnostics["status"] == expected_status


def test_diversity_distinguishes_single_authority_and_low_authority_reuse():
    high = [{"url": "https://who.int/a", "score": 8.5, "authority_score": 10}]
    low = [{"url": "https://medium.com/a", "score": 5}, {"url": "https://medium.com/b", "score": 5}]
    assert calculate_domain_diversity(high)["classification"] == "single_authoritative_source"
    metrics = calculate_domain_diversity(low)
    assert metrics["classification"] == "single_domain_reuse" and metrics["diversity_ratio"] == .5


def test_quality_gate_uses_sources_that_support_claims_not_unused_candidates():
    sources = [
        {"url": "https://redis.io/docs/guide", "authority_score": 9, "score": 8},
        {"url": "https://docs.python.org/guide", "authority_score": 10, "score": 9},
        {"url": "https://who.int/report", "authority_score": 10, "score": 9},
    ]
    claims = [
        Claim(
            claim_id="C1",
            claim_text="Redis stores frequently requested values in memory.",
            evidence_ids=["E1"],
            source_urls=["https://redis.io/docs/guide"],
            confidence=.9,
        )
    ]

    metrics = calculate_domain_diversity(sources, claims)
    gate = quality_gate(
        metrics,
        {"total_claims": 1, "evidence_coverage": 1},
        {"citation_coverage": 1, "invalid_citations": 0},
    )

    assert metrics["unique_domains"] == 3
    assert metrics["evidence_sources"]["unique_domains"] == 1
    assert metrics["evidence_sources"]["urls"] == ["https://redis.io/docs/guide"]
    assert gate["research_gaps"] == []


def test_empty_research_denominators_remain_unmeasured():
    source_metrics = calculate_domain_diversity([])
    _, _, citation_metrics = materialize_citations("", [], [])
    evidence_metrics = calculate_evidence_metrics([], [], {})

    assert source_metrics["total_sources"] == 0
    assert source_metrics["diversity_ratio"] is None
    assert source_metrics["average_source_quality"] is None
    assert source_metrics["average_authority_score"] is None
    assert source_metrics["authoritative_source_ratio"] is None
    assert source_metrics["duplicate_source_ratio"] is None
    assert citation_metrics["total_claims"] == 0
    assert citation_metrics["citation_coverage"] is None
    assert evidence_metrics["total_claims"] == 0
    assert evidence_metrics["evidence_coverage"] is None
    assert evidence_metrics["average_claim_confidence"] is None


def test_citation_materialization_and_quality_gate():
    claim = Claim(claim_id="C1", claim_text="Redis reduces repeated reads", evidence_ids=["E1"], source_urls=["https://redis.io/docs"], confidence=.8)
    sources = [{"url": "https://redis.io/docs", "title": "Redis docs", "score": 9}]
    report, citation_map, metrics = materialize_citations(
        "Finding [C1].\n\n# Sources\n\n- [Forged label](https://redis.io/docs)", [claim], sources
    )
    assert "[Redis docs](https://redis.io/docs)" in report
    assert "— Quality Score:" in report
    assert "â€”" not in report
    assert "[Forged label]" not in report and "[unmapped citation removed]" not in report
    assert report.count("# Sources") == 1
    assert metrics["citation_coverage"] == 1 and citation_map["C1"]
    missing, _, missing_metrics = materialize_citations("Finding with no marker.", [claim], sources)
    assert missing.startswith("Finding with no marker.") and missing_metrics["citation_coverage"] == 0
    assert quality_gate({"classification": "single_low_authority_source", "unique_domains": 1, "total_sources": 1}, {"total_claims": 1, "evidence_coverage": .5}, missing_metrics)["revision_required"]
    ungrounded_report, _, ungrounded = materialize_citations(
        "Fact [Redis docs](https://redis.io/docs).", [claim], sources
    )
    assert "https://redis.io/docs" not in ungrounded_report
    assert "[unmapped citation removed]" not in ungrounded_report
    assert ungrounded["invalid_citations"] == 1 and ungrounded["citation_coverage"] == 0
    bare, _, bare_metrics = materialize_citations("Invented URL https://fake.example/path", [claim], sources)
    assert "https://fake.example/path" not in bare
    assert bare_metrics["invalid_citations"] == 1


def test_adjacent_claim_markers_deduplicate_shared_inline_source():
    claims = [
        Claim(claim_id=claim_id, claim_text="Supported claim", evidence_ids=["E1"],
              source_urls=["https://redis.io/docs"], confidence=.9)
        for claim_id in ("C1", "C2")
    ]
    sources = [{"url": "https://redis.io/docs", "title": "Redis docs", "score": 9}]

    report, citation_map, metrics = materialize_citations("Conclusion [C1] [C2].", claims, sources)
    inline_report, bibliography = report.split("# Sources", maxsplit=1)

    assert inline_report.count("[Redis docs](https://redis.io/docs)") == 1
    assert bibliography.count("[Redis docs](https://redis.io/docs)") == 1
    assert set(citation_map) == {"C1", "C2"}
    assert metrics["citation_coverage"] == 1


def test_three_adjacent_claim_markers_keep_distinct_validated_sources():
    shared_url = "https://redis.io/docs"
    other_url = "https://docs.python.org/guide"
    claims = [
        Claim(claim_id="C1", claim_text="First", evidence_ids=["E1"], source_urls=[shared_url], confidence=.9),
        Claim(claim_id="C2", claim_text="Second", evidence_ids=["E2"], source_urls=[shared_url], confidence=.9),
        Claim(claim_id="C3", claim_text="Third", evidence_ids=["E3"], source_urls=[other_url], confidence=.9),
    ]
    sources = [
        {"url": shared_url, "title": "Redis docs", "score": 9},
        {"url": other_url, "title": "Python docs", "score": 9},
    ]

    report, citation_map, metrics = materialize_citations("Conclusion [C1][C2][C3].", claims, sources)
    inline_report, bibliography = report.split("# Sources", maxsplit=1)

    assert inline_report.count("[Redis docs](https://redis.io/docs)") == 1
    assert inline_report.count("[Python docs](https://docs.python.org/guide)") == 1
    assert bibliography.count("[Redis docs](https://redis.io/docs)") == 1
    assert bibliography.count("[Python docs](https://docs.python.org/guide)") == 1
    assert set(citation_map) == {"C1", "C2", "C3"}
    assert metrics["citation_coverage"] == 1


def test_separated_claims_retain_citations_with_deduplicated_bibliography():
    url = "https://redis.io/docs"
    claims = [
        Claim(claim_id=claim_id, claim_text=claim_id, evidence_ids=["E1"], source_urls=[url], confidence=.9)
        for claim_id in ("C1", "C2")
    ]
    sources = [{"url": url, "title": "Redis docs", "score": 9}]

    report, citation_map, metrics = materialize_citations(
        "First point [C1].\n\nAn unrelated point between citations.\n\nSecond point [C2].",
        claims,
        sources,
    )
    inline_report, bibliography = report.split("# Sources", maxsplit=1)

    assert inline_report.count("[Redis docs](https://redis.io/docs)") == 2
    assert bibliography.count("[Redis docs](https://redis.io/docs)") == 1
    assert set(citation_map) == {"C1", "C2"}
    assert metrics["citation_coverage"] == 1


def test_invalid_source_url_is_never_materialized():
    claim = Claim(
        claim_id="C1",
        claim_text="Unsupported URL",
        evidence_ids=["E1"],
        source_urls=["javascript:alert(1)"],
        confidence=.9,
    )
    sources = [{"url": "javascript:alert(1)", "title": "Invalid source", "score": 9}]

    report, citation_map, metrics = materialize_citations("[C1]", [claim], sources)

    assert "javascript:" not in report
    assert "[Invalid source]" not in report
    assert citation_map == {}
    assert metrics["invalid_citations"] == 1


def test_empty_citation_mapping_removes_model_authored_urls():
    report, citation_map, metrics = materialize_citations(
        "Unmapped [Source](https://fake.example/path).",
        [],
        [],
    )

    assert "https://fake.example/path" not in report
    assert "[unmapped citation removed]" not in report
    assert citation_map == {}
    assert metrics["citation_coverage"] is None
    assert metrics["invalid_citations"] == 1
    assert "No validated sources were cited." in report

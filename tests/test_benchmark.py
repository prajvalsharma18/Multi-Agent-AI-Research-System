import json
from pathlib import Path

import pytest
import dotenv

import benchmark
import export
from metrics import PipelineMetrics


def test_benchmark_dataset_is_valid_and_covers_diverse_categories():
    dataset = benchmark.load_dataset()

    assert len(dataset["cases"]) == 10
    assert len({case["id"] for case in dataset["cases"]}) == 10
    assert any(case["fresh_information_required"] for case in dataset["cases"])
    assert any(case["minimum_unique_domains"] >= 2 for case in dataset["cases"])
    assert any(case["category"] == "insufficient-evidence" for case in dataset["cases"])


def test_dataset_validation_fails_clearly_for_missing_fields(tmp_path):
    dataset = benchmark.load_dataset()
    del dataset["cases"][0]["insufficient_evidence_behavior"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields: insufficient_evidence_behavior"):
        benchmark.load_dataset(path)


def test_case_selection_is_stable_and_rejects_unknown_ids():
    dataset = benchmark.load_dataset()
    selected = benchmark.select_cases(dataset, ["idempotency-subquestions", "redis-caching"])

    assert [case["id"] for case in selected] == ["redis-caching", "idempotency-subquestions"]
    with pytest.raises(ValueError, match="Unknown benchmark case id"):
        benchmark.select_cases(dataset, ["not-a-case"])


def test_run_ids_are_unique_and_include_the_evaluation_mode():
    first = benchmark.make_run_id("live", "2026-10-09T00:00:00+00:00")
    second = benchmark.make_run_id("live", "2026-10-09T00:00:00+00:00")

    assert first.startswith("live-20261009T000000Z-")
    assert first != second


def test_offline_runner_is_fixture_only_and_writes_both_aggregate_formats(tmp_path):
    report, (json_path, markdown_path) = benchmark.run_offline(
        benchmark.load_dataset(),
        ["redis-caching", "idempotency-subquestions"],
        tmp_path,
    )

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    summary = markdown_path.read_text(encoding="utf-8")
    assert saved["mode"] == "offline"
    assert saved["case_counts"]["fixture_verified"] == 2
    assert saved["case_counts"]["research_cases_executed"] == 0
    assert all(case["status"] == "fixture_verified" for case in report["per_case"])
    assert all(case["research_outcome"] == "not_evaluated_offline" for case in report["per_case"])
    assert "does not answer the benchmark research questions" in summary
    assert json_path.exists() and markdown_path.exists()


def test_offline_fixture_set_covers_requested_recovery_scenarios():
    fixture_file = benchmark._read_json(benchmark.OFFLINE_FIXTURES_PATH)
    scenarios = {item["scenario"] for item in fixture_file["scenarios"]}

    assert scenarios == {
        "initial_approval",
        "recovery_enables_approval",
        "candidates_without_new_evidence",
        "duplicate_recovery_sources",
        "recovery_fetch_failure",
        "evidence_improves_but_gate_rejects",
        "recovery_round_budget_exhausted",
        "remaining_budget_insufficient",
        "source_merge_and_id_remapping",
        "no_progress_preserves_last_critic",
    }
    assert all(benchmark._verify_offline_fixture(item)[0] for item in fixture_file["scenarios"])


def test_live_runner_requires_explicit_cache_setting_and_enforces_case_cap(tmp_path):
    dataset = benchmark.load_dataset()
    with pytest.raises(ValueError, match="exceeds --max-live-cases"):
        benchmark.run_live(
            dataset,
            ["redis-caching", "http3-performance"],
            max_live_cases=1,
            cache_setting="disabled",
            output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="explicit cache setting"):
        benchmark.run_live(
            dataset,
            ["redis-caching"],
            max_live_cases=1,
            cache_setting="",
            output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="between 1 and 3"):
        benchmark.run_live(dataset, None, 4, "disabled", tmp_path)


def test_live_cli_requires_explicit_opt_in_and_cache_flag(tmp_path):
    assert benchmark.main(["--live", "--case", "redis-caching", "--output-dir", str(tmp_path)]) == 2
    assert list(tmp_path.glob("*.json")) == []


def test_missing_credentials_block_live_run_without_serializing_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "load_dotenv", lambda: None)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("A_TEST_SECRET", "must-not-appear")

    report, paths = benchmark.run_live(
        benchmark.load_dataset(),
        ["redis-caching"],
        max_live_cases=1,
        cache_setting="disabled",
        output_dir=tmp_path,
    )

    serialized = paths[0].read_text(encoding="utf-8")
    assert report["per_case"][0]["status"] == "blocked"
    assert "OPENAI_API_KEY" in serialized and "TAVILY_API_KEY" in serialized
    assert "must-not-appear" not in serialized
    assert "api_key_value" not in serialized.lower()


def test_default_cli_mode_is_offline_and_never_calls_live_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "run_live",
        lambda *args, **kwargs: pytest.fail("offline CLI must not call live runner"),
    )

    assert benchmark.main(["--output-dir", str(tmp_path)]) == 0
    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8"))["mode"] == "offline"


def test_live_runner_records_only_selected_case_ids_without_mixing_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-secret")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(
        benchmark,
        "_live_case",
        lambda case, run_id, cache_setting, output_dir: _case_result(
            case["id"],
            cache=cache_setting,
        ),
    )

    report, paths = benchmark.run_live(
        benchmark.load_dataset(),
        ["redis-caching"],
        max_live_cases=1,
        cache_setting="disabled",
        output_dir=tmp_path,
    )

    assert report["mode"] == "live"
    assert report["selected_case_ids"] == ["redis-caching"]
    assert report["case_counts"]["selected"] == 1
    assert "test-openai-secret" not in paths[0].read_text(encoding="utf-8")
    assert "test-tavily-secret" not in paths[0].read_text(encoding="utf-8")


def test_live_case_failures_are_recorded_without_exception_text(tmp_path, monkeypatch):
    import graph

    def fail(state):
        raise RuntimeError("secret-token-value")

    monkeypatch.setattr(
        graph,
        "research_graph",
        type("BrokenGraph", (), {"invoke": staticmethod(fail)})(),
    )
    case = benchmark.load_dataset()["cases"][0]

    result = benchmark._live_case(case, "live-test-run", "disabled", tmp_path)

    assert result["status"] == "failed"
    assert result["failure"]["exception_type"] == "RuntimeError"
    assert "secret-token-value" not in json.dumps(result)
    assert result["operations"]["cache"]["configuration"] == "disabled"


def _case_result(
    case_id,
    *,
    status="passed",
    outcome="approved",
    evidence_coverage=None,
    citation_coverage=None,
    runtime=None,
    cache="disabled",
    attempted=False,
    productive=None,
):
    return {
        "case_id": case_id,
        "status": status,
        "research_outcome": outcome,
        "execution_kind": "live_langgraph",
        "quality_gate": {"approved": outcome == "approved"},
        "evidence": {"evidence_coverage": evidence_coverage},
        "citations": {"citation_coverage": citation_coverage},
        "recovery": {"attempted": attempted, "productive": productive},
        "operations": {
            "runtime_seconds": runtime,
            "openai_calls": None,
            "cache": {"configuration": cache},
        },
        "exports": {"checked": True, "passed": True},
    }


def test_aggregate_excludes_unknowns_and_uses_explicit_denominators():
    results = [
        _case_result("a", evidence_coverage=.5, citation_coverage=.8, runtime=10, attempted=True, productive=True),
        _case_result("b", status="rejected", outcome="rejected_by_quality_gate", evidence_coverage=None, citation_coverage=.4, runtime=None, attempted=True, productive=False),
        _case_result("c", status="failed", outcome="unknown", evidence_coverage=1, citation_coverage=1, runtime=20),
    ]
    results[0]["recovery"]["quality_gate_improved"] = True
    results[0]["recovery"]["resulted_in_approval"] = True
    results[1]["recovery"]["quality_gate_improved"] = False
    results[1]["recovery"]["resulted_in_approval"] = False
    report = benchmark.aggregate_results(results, {"mode": "live"})

    assert report["case_counts"]["quality_gate_approved"] == 1
    assert report["case_counts"]["quality_gate_approval_denominator_completed_runs"] == 2
    assert report["case_counts"]["passed"] == 2
    assert report["aggregates"]["evidence_coverage"] == {
        "sum": .5,
        "mean": .5,
        "median": None,
        "range": None,
        "measured_count": 1,
        "selected_count": 2,
    }
    assert report["aggregates"]["citation_coverage"]["mean"] == .6
    assert report["aggregates"]["citation_coverage"]["measured_count"] == 2
    assert report["aggregates"]["runtime_seconds"]["measured_count"] == 2
    assert report["aggregates"]["recovery_attempt_rate"] == 1
    assert report["aggregates"]["recovery_productivity_rate"] == .5
    assert report["aggregates"]["recovery_gate_improvement_rate"] == .5
    assert report["aggregates"]["recovery_final_approval_rate"] == .5


def test_aggregate_separates_execution_quality_and_export_outcomes():
    approved = _case_result("approved")
    approved.update(execution_status="completed", quality_gate_status="approved", export_status="passed")
    rejected = _case_result("rejected", status="rejected", outcome="rejected_by_quality_gate")
    rejected.update(execution_status="completed", quality_gate_status="rejected", export_status="passed")
    failed = _case_result("execution-failed", status="failed", outcome="unknown")
    failed.update(execution_status="failed", quality_gate_status="unknown", export_status="not_checked")
    incomplete = _case_result("blocked", status="blocked", outcome="unknown")
    incomplete.update(execution_status="blocked", quality_gate_status="unknown", export_status="not_checked")
    export_failed = _case_result("export-failed", status="failed", outcome="approved")
    export_failed.update(
        execution_status="completed",
        quality_gate_status="approved",
        export_status="failed",
        exports={"checked": True, "passed": False},
    )

    report = benchmark.aggregate_results(
        [approved, rejected, failed, incomplete, export_failed],
        {
            "mode": "live",
            "run_id": "live-test",
            "benchmark_version": "test",
            "started_at": "2026-01-01T00:00:00+00:00",
            "selected_case_ids": ["approved", "rejected", "execution-failed", "blocked", "export-failed"],
            "configuration": {"cache": "disabled"},
        },
    )
    approved["pipeline_diagnostics"] = {
        "availability": "measured",
        "initial_search": {
            "results_returned": 2,
            "parsed_candidates": 2,
            "accepted_sources": 1,
            "source_scoring_rejections": 1,
            "invalid_url_candidates": 0,
        },
        "initial_fetch": {
            "scrape_attempts": 1,
            "fetch_successes": 1,
            "fetch_failures": 0,
            "extraction_failures": 0,
            "unreported_outcomes": 0,
        },
        "extraction": {"status": "parsed"},
        "evidence_validation": {"accepted": 1, "rejected": 0},
        "recovery_rounds": [{
            "round": 1,
            "queries_attempted": ["targeted"],
            "candidate_urls_discovered": 1,
            "duplicate_urls": 1,
            "fetch": {"fetch_successes": 1, "fetch_failures": 0},
            "evidence_validation": {"accepted": 1, "rejected": 0},
            "new_validated_evidence": 1,
            "termination_reason": "progress",
        }],
    }
    summary = benchmark.render_summary(report)

    counts = report["case_counts"]
    assert counts["execution_failures"] == 1
    assert counts["quality_gate_rejections"] == 1
    assert counts["incomplete_runs"] == 1
    assert counts["export_failures"] == 1
    assert counts["approved_reports"] == 2
    assert "Execution failures: 1" in summary
    assert "Quality-gate rejections: 1" in summary
    assert "Export failures: 1" in summary
    assert "Initial search: 2 returned, 2 parsed, 1 accepted" in summary
    assert "Recovery round 1: 1 queries" in summary


def test_aggregate_rejects_mixed_cache_configurations_and_modes():
    first = _case_result("a", cache="enabled")
    second = _case_result("b", cache="disabled")

    with pytest.raises(ValueError, match="mix different cache configurations"):
        benchmark.aggregate_results([first, second], {"mode": "live"})
    with pytest.raises(ValueError, match="exactly offline or live"):
        benchmark.aggregate_results([], {"mode": "mocked"})
    with pytest.raises(ValueError, match="cannot contain offline"):
        benchmark.aggregate_results(
            [{"execution_kind": "deterministic_fixture"}],
            {"mode": "live"},
        )


def test_state_summary_preserves_unknown_metrics_and_reports_heuristic_labels(tmp_path):
    state = {
        "topic": "Redis",
        "research_rounds": 0,
        "quality_approved": False,
        "termination_reason": "quality_requirements_unresolved",
        "ranked_sources": [{
            "url": "https://redis.io/docs/guide",
            "domain": "redis.io",
            "authority_score": 9,
            "source_score_breakdown": {"relevance": 7.5, "authority": 9},
        }],
        "quality_metrics": {
            "sources": {"total_sources": 1, "diversity_ratio": 1.0},
            "evidence": {"valid_evidence": 1, "total_claims": 1},
            "citations": {},
        },
    }

    result = benchmark.summarize_state(
        {"id": "redis-caching", "question": "Redis", "category": "backend",
         "minimum_unique_urls": 2, "minimum_unique_domains": 2,
         "expected_source_characteristics": [], "fresh_information_required": False,
         "freshness_days": None},
        state,
        PipelineMetrics(),
        "disabled",
        "start",
        "finish",
    )

    assert result["citations"]["citation_coverage"] is None
    assert result["sources"]["authority"]["assessment_method"].startswith("existing domain-based")
    assert result["sources"]["relevance"]["average_score_0_to_10"] == 7.5
    assert result["sources"]["minimum_url_domain_requirements_met"] is False
    assert result["sources"]["independent_corroboration_verified"] is None
    assert result["operations"]["cache"] == {
        "configuration": "disabled",
        "status": "not_applicable",
        "hits": None,
        "misses": None,
    }


def test_state_summary_exposes_latest_score_diagnostics_and_separate_statuses(tmp_path, monkeypatch):
    state = _export_state(tmp_path, monkeypatch, approved=False)
    state.update({
        "topic": "Redis",
        "report": "Latest report.",
        "final_report": "Latest report.",
        "research_rounds": 2,
        "quality_approved": False,
        "termination_reason": "no_research_progress",
        "critic_evaluations": [{
            "model_score": 8,
            "raw_model_score": 8,
            "score_status": "valid",
            "effective_score": 5,
            "decision": "research",
        }],
    })
    state["quality_metrics"]["critic"].update({
        "model_score": 8,
        "raw_model_score": 8,
        "score_status": "valid",
        "effective_score": 5,
    })
    report = Path(state["output_paths"]["markdown"]).read_text(encoding="utf-8")
    report = report.replace("model 9.0/10; effective 9.0/10", "model 8.0/10; effective 5.0/10")
    state["output_paths"] = export.save_report("Redis", report)
    state["report"] = state["final_report"] = report
    state["quality_metrics"]["diagnostics"] = {
        "initial_search": {"results_returned": 5, "accepted_sources": 2},
        "initial_fetch": {"fetch_successes": 2, "fetch_failures": 1},
        "extraction": {"status": "parsed"},
        "evidence_validation": {"accepted": 3, "rejected": 1},
        "claim_support": {"total": 2, "supported": 1, "unsupported": 1},
    }
    state["quality_metrics"]["recovery"] = {
        "successful_fetches": 2,
        "failed_fetches": 1,
        "new_validated_evidence": 1,
        "diagnostics_by_round": [{"round": 1, "new_validated_evidence": 1}],
    }

    result = benchmark.summarize_state(
        {"id": "redis-caching", "question": "Redis", "category": "backend",
         "minimum_unique_urls": 1, "minimum_unique_domains": 1,
         "expected_source_characteristics": [], "fresh_information_required": False,
         "freshness_days": None},
        state,
        PipelineMetrics(),
        "disabled",
        "start",
        "finish",
    )

    assert result["execution_status"] == "completed"
    assert result["quality_gate_status"] == "rejected"
    assert result["export_status"] == "passed"
    assert result["quality_gate"]["critic_model_score"] == 8
    assert result["quality_gate"]["critic_raw_model_score"] == 8
    assert result["quality_gate"]["critic_effective_score"] == 5
    assert result["quality_gate"]["routing_decision"] == "research"
    assert result["pipeline_diagnostics"]["initial_search"]["results_returned"] == 5
    assert result["pipeline_diagnostics"]["recovery_rounds"] == [
        {"round": 1, "new_validated_evidence": 1}
    ]


def test_state_summary_keeps_zero_denominator_metrics_unknown_and_separates_gate_failures():
    state = {
        "research_rounds": 1,
        "quality_approved": False,
        "critic_evaluations": [{"gate_failures": ["initial issue"]}],
        "documents": [],
        "quality_metrics": {
            "sources": {"total_sources": 0, "diversity_ratio": 0.0, "average_source_quality": 0.0},
            "evidence": {"total_claims": 0, "evidence_coverage": 0.0},
            "citations": {
                "total_claims": 0,
                "citation_coverage": 0.0,
                "invalid_citations": 0,
            },
        },
    }
    metrics = PipelineMetrics(scrape_calls=3)

    result = benchmark.summarize_state(
        {"id": "redis-caching", "question": "Redis", "category": "backend",
         "minimum_unique_urls": 2, "minimum_unique_domains": 2,
         "expected_source_characteristics": [], "fresh_information_required": False,
         "freshness_days": None},
        state,
        metrics,
        "disabled",
        "start",
        "finish",
    )

    assert result["evidence"]["evidence_coverage"] is None
    assert result["citations"]["citation_coverage"] is None
    assert result["citations"]["validation_passed"] is None
    assert result["sources"]["source_diversity_ratio"] is None
    assert result["sources"]["composite_quality"]["average_score_0_to_10"] is None
    assert result["operations"]["fetch_attempts"] == 3
    assert result["operations"]["initial_fetch_successes"] is None
    assert result["operations"]["initial_fetch_failures"] is None
    assert result["recovery"]["quality_gate_improved"] is False


def test_state_summary_does_not_export_critic_gate_failures_as_fetch_failures():
    state = {
        "research_rounds": 1,
        "quality_approved": False,
        "critic_evaluations": [{
            "gate_failures": ["Evidence coverage below required threshold"]
        }],
        "documents": [{"extraction_status": "success"}],
        "quality_metrics": {
            "sources": {"total_sources": 1, "diversity_ratio": 1.0},
            "evidence": {"total_claims": 1, "evidence_coverage": 1.0},
            "citations": {"total_claims": 1, "citation_coverage": 1.0, "invalid_citations": 0},
        },
    }

    result = benchmark.summarize_state(
        {"id": "redis-caching", "question": "Redis", "category": "backend",
         "minimum_unique_urls": 1, "minimum_unique_domains": 1,
         "expected_source_characteristics": [], "fresh_information_required": False,
         "freshness_days": None},
        state,
        PipelineMetrics(),
        "disabled",
        "start",
        "finish",
    )

    assert result["operations"]["initial_fetch_failures"] is None
    assert result["operations"]["fetch_attempts"] == 0
    assert result["recovery"]["quality_gate_improved"] is False


def _export_state(tmp_path, monkeypatch, *, approved=True, url="https://redis.io/docs/caching"):
    monkeypatch.setattr(export, "OUTPUT_DIR", tmp_path)
    report = (
        "A validated finding [Source](https://redis.io/docs/caching).\n\n"
        "# Sources\n\n- [Source](https://redis.io/docs/caching)\n\n"
        "## Research Quality\n\n"
        f"- Quality status: {'APPROVED' if approved else 'NOT APPROVED'}\n"
        f"- Termination reason: {'approved' if approved else 'no_research_progress'}\n"
        "- Source diversity: 100%\n- Evidence coverage: 100%\n- Citation coverage: 100%\n"
        "- Critic score: model 9.0/10; effective 9.0/10\n"
    )
    paths = export.save_report("Redis", report)
    return {
        "output_paths": paths,
        "citation_map": {"C1": [url]},
        "quality_approved": approved,
        "termination_reason": "approved" if approved else "no_research_progress",
        "quality_metrics": {
            "sources": {"diversity_ratio": 1.0},
            "evidence": {"evidence_coverage": 1.0},
            "citations": {"citation_coverage": 1.0},
            "critic": {"model_score": 9, "effective_score": 9},
        },
    }


def test_export_integrity_checks_markdown_pdf_and_validated_link_parity(tmp_path, monkeypatch):
    state = _export_state(tmp_path, monkeypatch)
    result = benchmark.validate_exports(state)

    assert result["passed"] is True
    assert result["markdown_links"] == 2
    assert result["bibliography_urls"] == 1
    assert result["pdf_links"] == 2
    assert result["validated_url_count"] == 1


def test_export_integrity_rejects_unmapped_urls_and_duplicate_bibliography(tmp_path, monkeypatch):
    state = _export_state(tmp_path, monkeypatch)
    markdown_path = Path(state["output_paths"]["markdown"])
    text = markdown_path.read_text(encoding="utf-8")
    text = text.replace(
        "## Research Quality",
        "- [Duplicate](https://redis.io/docs/caching)\n\n## Research Quality",
    )
    text += "\n[Fabricated](https://example.com/fake)\n"
    markdown_path.write_text(text, encoding="utf-8")

    result = benchmark.validate_exports(state)

    assert result["passed"] is False
    assert "markdown_link_not_in_validated_citation_map" in result["issues"]
    assert "duplicate_bibliography_url" in result["issues"]


def test_export_integrity_requires_inline_and_pdf_links_in_bibliography(tmp_path, monkeypatch):
    state = _export_state(tmp_path, monkeypatch)
    markdown_path = Path(state["output_paths"]["markdown"])
    text = markdown_path.read_text(encoding="utf-8")
    text = text.replace("- [Source](https://redis.io/docs/caching)\n\n", "")
    markdown_path.write_text(text, encoding="utf-8")

    result = benchmark.validate_exports(state)

    assert result["passed"] is False
    assert "markdown_link_not_in_bibliography" in result["issues"]
    assert "pdf_link_not_in_bibliography" in result["issues"]


def test_isolated_run_configuration_does_not_touch_standard_cache_or_export_directory(tmp_path, monkeypatch):
    import cache

    original_cache = cache.CACHE_ROOT
    original_output = export.OUTPUT_DIR
    monkeypatch.setenv("PIPELINE_CACHE_ENABLED", "true")
    with benchmark._isolated_run_configuration("enabled", tmp_path / "cache", tmp_path / "artifacts"):
        assert cache.CACHE_ROOT == tmp_path / "cache"
        assert export.OUTPUT_DIR == tmp_path / "artifacts"
        assert cache.cache_enabled() is True
    assert cache.CACHE_ROOT == original_cache
    assert export.OUTPUT_DIR == original_output
    assert cache.cache_enabled() is True

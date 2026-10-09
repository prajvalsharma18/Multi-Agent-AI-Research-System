"""Offline conformance and explicitly opted-in live research benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from source_scoring import canonicalize_url, domain_for_url

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "benchmarks" / "benchmark_cases.json"
OFFLINE_FIXTURES_PATH = ROOT / "benchmarks" / "offline_scenarios.json"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "benchmarks"
MAX_ALLOWED_LIVE_CASES = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark data: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark data must be a JSON object: {path.name}")
    return value


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    dataset = _read_json(path)
    if not isinstance(dataset.get("benchmark_version"), str) or not dataset["benchmark_version"].strip():
        raise ValueError("Dataset requires a non-empty benchmark_version.")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not 8 <= len(cases) <= 12:
        raise ValueError("Dataset cases must be a list containing 8 to 12 cases.")
    seen: set[str] = set()
    required = {
        "id", "question", "category", "expected_source_characteristics",
        "minimum_unique_urls", "minimum_unique_domains", "fresh_information_required",
        "freshness_days", "insufficient_evidence_behavior", "evaluation_notes",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Dataset case {index} must be an object.")
        missing = required - case.keys()
        if missing:
            raise ValueError(f"Dataset case {index} missing fields: {', '.join(sorted(missing))}.")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"Dataset case {index} has an empty or duplicate id.")
        seen.add(case_id)
        for field in ("question", "category", "insufficient_evidence_behavior", "evaluation_notes"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"Dataset case {case_id} requires non-empty {field}.")
        if not isinstance(case["expected_source_characteristics"], list) or not case["expected_source_characteristics"]:
            raise ValueError(f"Dataset case {case_id} requires expected source characteristics.")
        for field in ("minimum_unique_urls", "minimum_unique_domains"):
            if not isinstance(case[field], int) or isinstance(case[field], bool) or case[field] < 1:
                raise ValueError(f"Dataset case {case_id} has invalid {field}.")
        if not isinstance(case["fresh_information_required"], bool):
            raise ValueError(f"Dataset case {case_id} has invalid fresh_information_required.")
        freshness = case["freshness_days"]
        if freshness is not None and (
            not isinstance(freshness, int) or isinstance(freshness, bool) or freshness < 1
        ):
            raise ValueError(f"Dataset case {case_id} has invalid freshness_days.")
        if case["fresh_information_required"] != (freshness is not None):
            raise ValueError(f"Dataset case {case_id} freshness flag and freshness_days disagree.")
    if not isinstance(dataset.get("evaluation_date"), str):
        raise ValueError("Dataset requires an evaluation_date.")
    try:
        date.fromisoformat(dataset["evaluation_date"])
    except ValueError as exc:
        raise ValueError("Dataset evaluation_date must be an ISO date.") from exc
    return dataset


def select_cases(dataset: dict[str, Any], case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    cases = dataset["cases"]
    if not case_ids:
        return list(cases)
    requested = set(case_ids)
    available = {case["id"] for case in cases}
    unknown = requested - available
    if unknown:
        raise ValueError(f"Unknown benchmark case id(s): {', '.join(sorted(unknown))}.")
    return [case for case in cases if case["id"] in requested]


def make_run_id(mode: str, started_at: str | None = None) -> str:
    stamp = (started_at or _utc_now()).replace("+00:00", "Z").replace("-", "").replace(":", "")
    return f"{mode}-{stamp}-{uuid.uuid4().hex[:8]}"


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _published_age_days(value: Any, reference_date: str | None) -> int | None:
    if not isinstance(value, str) or not value.strip() or not reference_date:
        return None
    try:
        reference = date.fromisoformat(reference_date)
        try:
            published = date.fromisoformat(value[:10])
        except ValueError:
            published = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        return max(0, (reference - published).days)
    except ValueError:
        return None


def _percent(value: Any) -> str:
    return f"{value:.0%}" if isinstance(value, (int, float)) else "not measured"


def _markdown_urls(text: str) -> list[str]:
    import re
    return re.findall(r"\]\((https?://[^)\s]+)\)", text)


def validate_exports(state: dict[str, Any]) -> dict[str, Any]:
    """Check emitted Markdown/PDF files against validated URLs and graph quality state."""
    import re
    from pypdf import PdfReader

    paths = state.get("output_paths") or {}
    markdown_path = Path(paths["markdown"]) if paths.get("markdown") else None
    pdf_path = Path(paths["pdf"]) if paths.get("pdf") else None
    issues: list[str] = []
    if not markdown_path or not markdown_path.is_file():
        issues.append("markdown_missing")
    if not pdf_path or not pdf_path.is_file():
        issues.append("pdf_missing")
    if issues:
        return {"checked": True, "passed": False, "issues": issues, "markdown_links": None, "pdf_links": None}

    try:
        markdown_text = markdown_path.read_text(encoding="utf-8")
        reader = PdfReader(str(pdf_path))
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        return {
            "checked": True,
            "passed": False,
            "issues": [f"artifact_unreadable:{type(exc).__name__}"],
            "markdown_links": None,
            "pdf_links": None,
        }

    citation_map = state.get("citation_map") or {}
    validated_urls = {
        normalized
        for urls in citation_map.values()
        if isinstance(urls, list)
        for url in urls
        if (normalized := canonicalize_url(str(url)))
    }
    markdown_links = _markdown_urls(markdown_text)
    if "[unmapped citation removed]" in markdown_text:
        issues.append("unmapped_citation_placeholder")
    if re.search(r"\]\(https?://[^)\s]*$", markdown_text, re.MULTILINE):
        issues.append("malformed_markdown_link")
    if any(canonicalize_url(url) not in validated_urls for url in markdown_links):
        issues.append("markdown_link_not_in_validated_citation_map")

    source_section = markdown_text.split("# Sources", 1)
    bibliography = source_section[1].split("## Research Quality", 1)[0] if len(source_section) == 2 else ""
    bibliography_links = _markdown_urls(bibliography)
    normalized_bibliography = [canonicalize_url(url) for url in bibliography_links]
    if len(normalized_bibliography) != len(set(normalized_bibliography)):
        issues.append("duplicate_bibliography_url")
    if any(url not in validated_urls for url in normalized_bibliography):
        issues.append("bibliography_url_not_validated")
    bibliography_set = set(normalized_bibliography)
    if any(canonicalize_url(url) not in bibliography_set for url in markdown_links):
        issues.append("markdown_link_not_in_bibliography")

    pdf_links: list[str] = []
    for page in reader.pages:
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A")
            if action and action.get("/URI"):
                pdf_links.append(str(action["/URI"]))
    if markdown_links and not pdf_links:
        issues.append("pdf_links_missing")
    if any(canonicalize_url(url) not in validated_urls for url in pdf_links):
        issues.append("pdf_link_not_validated")
    if any(canonicalize_url(url) not in bibliography_set for url in pdf_links):
        issues.append("pdf_link_not_in_bibliography")

    quality = state.get("quality_metrics") or {}
    approved = bool(state.get("quality_approved"))
    expected_values = [
        f"Quality status: {'APPROVED' if approved else 'NOT APPROVED'}",
        f"Termination reason: {state.get('termination_reason') or 'quality_requirements_unresolved'}",
    ]
    evidence = quality.get("evidence") or {}
    citations = quality.get("citations") or {}
    sources = quality.get("sources") or {}
    expected_values.extend([
        f"Source diversity: {_percent(sources.get('diversity_ratio'))}",
        f"Evidence coverage: {_percent(evidence.get('evidence_coverage'))}",
        f"Citation coverage: {_percent(citations.get('citation_coverage'))}",
    ])
    critic = quality.get("critic") or {}
    if critic.get("model_score") is not None and critic.get("effective_score") is not None:
        expected_values.append(
            f"Critic score: model {critic['model_score']:.1f}/10; "
            f"effective {critic['effective_score']:.1f}/10"
        )
    if any(value not in markdown_text for value in expected_values):
        issues.append("markdown_quality_metadata_mismatch")
    if any(value not in pdf_text for value in expected_values):
        issues.append("pdf_quality_metadata_mismatch")
    if not approved and "Quality status: APPROVED" in markdown_text:
        issues.append("rejected_report_mislabeled_approved")
    return {
        "checked": True,
        "passed": not issues,
        "issues": issues,
        "markdown_links": len(markdown_links),
        "bibliography_urls": len(normalized_bibliography),
        "pdf_links": len(pdf_links),
        "pdf_pages": len(reader.pages),
        "validated_url_count": len(validated_urls),
    }


def summarize_state(
    case: dict[str, Any],
    state: dict[str, Any],
    run_metrics: Any,
    cache_setting: str,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    quality = state.get("quality_metrics") or {}
    source_quality = quality.get("sources") or {}
    evidence_quality = quality.get("evidence") or {}
    citation_quality = quality.get("citations") or {}
    recovery = quality.get("recovery") or state.get("recovery_metrics") or {}
    diagnostics = quality.get("diagnostics") or {}
    source_records = state.get("ranked_sources") or []
    unique_sources: dict[str, dict[str, Any]] = {}
    for source in source_records:
        url = canonicalize_url(str(source.get("url", "")))
        if url:
            unique_sources.setdefault(url, source)
    urls = sorted(unique_sources)
    domains = sorted({domain_for_url(url) for url in urls if domain_for_url(url)})
    relevances = [
        value for source in unique_sources.values()
        if isinstance((value := (source.get("source_score_breakdown") or {}).get("relevance")), (int, float))
    ]
    authority_values = [
        float(source.get("authority_score", (source.get("source_score_breakdown") or {}).get("authority")))
        for source in unique_sources.values()
        if isinstance(source.get("authority_score", (source.get("source_score_breakdown") or {}).get("authority")), (int, float))
    ]
    accepted_source_scores = []
    for url, source in sorted(unique_sources.items()):
        breakdown = source.get("source_score_breakdown") or {}
        authority_score = source.get("authority_score", breakdown.get("authority"))
        relevance_score = breakdown.get("relevance")
        composite_score = source.get("score", source.get("source_score"))
        accepted_source_scores.append({
            "url": url,
            "domain": domain_for_url(url),
            "authority_score": _number_or_none(authority_score),
            "relevance_score": _number_or_none(relevance_score),
            "composite_score": _number_or_none(composite_score),
        })
    evidence_source_metrics = source_quality.get("evidence_sources") or {}
    published_ages = [
        age for source in unique_sources.values()
        if (age := _published_age_days(source.get("published_date"), case.get("_evaluation_date"))) is not None
    ]
    claim_domain_counts: list[int] = []
    evidence_by_id = {
        item.get("evidence_id"): item
        for item in state.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    for claim in state.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_urls = claim.get("source_urls") or [
            evidence_by_id[item_id].get("source_url")
            for item_id in claim.get("evidence_ids", [])
            if item_id in evidence_by_id
        ]
        claim_domains = {domain_for_url(url) for url in claim_urls if url and domain_for_url(url)}
        claim_domain_counts.append(len(claim_domains))
    recovery_successes = _int_or_none(recovery.get("successful_fetches"))
    recovery_failures = _int_or_none(recovery.get("failed_fetches"))
    initial_fetch = diagnostics.get("initial_fetch") or {}
    initial_search = diagnostics.get("initial_search") or {}
    extraction = diagnostics.get("extraction") or {}

    evaluations = state.get("critic_evaluations") or []
    initial_eval = evaluations[0] if evaluations else {}
    latest_eval = evaluations[-1] if evaluations else {}
    critic = quality.get("critic") or {}
    approval = state.get("quality_approved")
    recovery_attempted = _int_or_none(state.get("research_rounds")) not in (None, 0)
    validated_recovery_evidence = _int_or_none(recovery.get("accepted_evidence"))
    quality_improved = None
    initial_gate_failures = initial_eval.get("gate_failures")
    final_failures = latest_eval.get("gate_failures")
    if isinstance(initial_gate_failures, list) and isinstance(final_failures, list):
        quality_improved = len(final_failures) < len(initial_gate_failures)
    integrity = validate_exports(state)
    completed = bool(state.get("output_paths")) and bool(state.get("final_report") or state.get("report"))
    termination_reason = state.get("termination_reason")
    budget_exhausted = termination_reason in {
        "research_budget_exhausted",
        "revision_budget_exhausted",
        "research_and_revision_budgets_exhausted",
        "hard_validation_failure",
    }
    if not completed:
        status = "incomplete"
    elif not integrity["passed"]:
        status = "failed"
    else:
        status = "passed"
    source_requirements_met = (
        len(urls) >= case["minimum_unique_urls"]
        and len(domains) >= case["minimum_unique_domains"]
    )
    return {
        "case_id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "status": status,
        "execution_status": "completed" if completed else "incomplete",
        "research_outcome": (
            "approved" if approval is True else "rejected_by_quality_gate" if approval is False else "unknown"
        ),
        "quality_gate_status": (
            "approved" if approval is True else "rejected" if approval is False else "unknown"
        ),
        "started_at": started_at,
        "finished_at": finished_at,
        "termination_reason": state.get("termination_reason"),
        "quality_gate": {
            "approved": approval if isinstance(approval, bool) else None,
            "gate_threshold": "Existing deterministic quality_gate; no single numeric threshold",
            "critic_model_score": _number_or_none(
                critic.get("model_score")
                if critic.get("model_score") is not None else latest_eval.get("model_score")
            ),
            "critic_raw_model_score": _number_or_none(
                critic.get("raw_model_score")
                if critic.get("raw_model_score") is not None else latest_eval.get("raw_model_score")
            ),
            "critic_score_status": (
                critic.get("score_status")
                if critic.get("score_status") is not None else latest_eval.get("score_status")
            ),
            "critic_effective_score": _number_or_none(
                critic.get("effective_score")
                if critic.get("effective_score") is not None else latest_eval.get("effective_score")
            ),
            "routing_decision": state.get("quality_route", latest_eval.get("decision")),
            "revision_count": _int_or_none(state.get("revision_count")),
            "unresolved_issues": state.get("unresolved_issues"),
        },
        "pipeline_diagnostics": {
            "availability": "measured" if diagnostics else "unavailable_for_this_run",
            "initial_search": diagnostics.get("initial_search"),
            "initial_fetch": diagnostics.get("initial_fetch"),
            "extraction": diagnostics.get("extraction"),
            "evidence_validation": diagnostics.get("evidence_validation"),
            "claim_support": diagnostics.get("claim_support"),
            "recovery_rounds": recovery.get("diagnostics_by_round"),
        },
        "sources": {
            "accepted_unique_urls": len(urls),
            "unique_domains": len(domains),
            "urls": urls,
            "domains": domains,
            "accepted_source_scores": accepted_source_scores,
            "claim_supporting_sources": {
                "unique_urls": _int_or_none(evidence_source_metrics.get("total_sources")),
                "unique_domains": _int_or_none(evidence_source_metrics.get("unique_domains")),
                "classification": evidence_source_metrics.get("classification"),
                "average_authority_score_0_to_10": _number_or_none(
                    evidence_source_metrics.get("average_authority_score")
                ),
                "urls": evidence_source_metrics.get("urls"),
            },
            "duplicate_source_count": _int_or_none(source_quality.get("total_sources")) - len(urls)
            if _int_or_none(source_quality.get("total_sources")) is not None else None,
            "authority": {
                "assessment_method": "existing domain-based authority heuristic (source_scoring.score_url)",
                "average_score_0_to_10": round(sum(authority_values) / len(authority_values), 2) if authority_values else None,
                "measured_sources": len(authority_values),
            },
            "relevance": {
                "assessment_method": "existing keyword-overlap source_score_breakdown.relevance heuristic",
                "average_score_0_to_10": round(sum(relevances) / len(relevances), 2) if relevances else None,
                "measured_sources": len(relevances),
            },
            "composite_quality": {
                "assessment_method": "existing weighted source-scoring composite on a 0–10 scale",
                "average_score_0_to_10": (
                    _number_or_none(source_quality.get("average_source_quality")) if urls else None
                ),
            },
            "minimum_url_domain_requirements_met": source_requirements_met,
            "minimum_unique_urls_required": case["minimum_unique_urls"],
            "minimum_unique_domains_required": case["minimum_unique_domains"],
            "source_diversity_ratio": (
                _number_or_none(source_quality.get("diversity_ratio")) if urls else None
            ),
            "freshness": {
                "required": case["fresh_information_required"],
                "evaluation_date": case.get("_evaluation_date"),
                "freshness_days_required": case["freshness_days"],
                "fresh_sources_measured": sum(
                    age <= case["freshness_days"] for age in published_ages
                ) if case["fresh_information_required"] and case["freshness_days"] is not None and published_ages else None,
                "freshness_denominator_dated_sources": len(published_ages) if case["fresh_information_required"] else None,
                "measurement_method": "search-result publication-date metadata only; dates are not independently verified",
            },
            "multi_domain_claim_count": sum(count >= 2 for count in claim_domain_counts),
            "independent_corroboration_verified": None,
            "corroboration_note": "Multiple domains are a proxy only; independence/syndication is not verified automatically.",
        },
        "evidence": {
            "accepted_count": _int_or_none(evidence_quality.get("valid_evidence", evidence_quality.get("evidence_items"))),
            "rejected_count": _int_or_none(evidence_quality.get("rejected_evidence")),
            "total_claims_denominator": _int_or_none(evidence_quality.get("total_claims")),
            "supported_claims": _int_or_none(evidence_quality.get("supported_claims")),
            "unsupported_claims": _int_or_none(evidence_quality.get("unsupported_claims")),
            "evidence_coverage": (
                _number_or_none(evidence_quality.get("evidence_coverage"))
                if _int_or_none(evidence_quality.get("total_claims")) not in (None, 0)
                else None
            ),
            "mapping_integrity_failures": state.get("evidence_mapping_failures"),
        },
        "citations": {
            "citation_coverage": (
                _number_or_none(citation_quality.get("citation_coverage"))
                if _int_or_none(citation_quality.get("total_claims")) not in (None, 0)
                else None
            ),
            "coverage_denominator_claims": _int_or_none(citation_quality.get("total_claims")),
            "invalid_or_unmapped_count": _int_or_none(citation_quality.get("invalid_citations")),
            "duplicate_or_malformed_count": None,
            "validation_passed": (
                citation_quality.get("citation_coverage") == 1
                and citation_quality.get("invalid_citations") == 0
            ) if (
                _int_or_none(citation_quality.get("total_claims")) not in (None, 0)
                and "citation_coverage" in citation_quality
                and "invalid_citations" in citation_quality
            ) else None,
        },
        "recovery": {
            "attempted": recovery_attempted,
            "rounds": _int_or_none(state.get("research_rounds")),
            "budget_status": (
                "exhausted"
                if budget_exhausted
                else "not_required"
                if not recovery_attempted
                else "stopped_before_budget_or_budget_unknown"
            ),
            "candidate_urls": _int_or_none(recovery.get("new_candidate_urls")),
            "duplicate_urls": _int_or_none(recovery.get("duplicate_urls")),
            "newly_accepted_sources": _int_or_none(recovery.get("newly_accepted_sources")),
            "successful_fetches": _int_or_none(recovery.get("successful_fetches")),
            "failed_fetches": _int_or_none(recovery.get("failed_fetches")),
            "extraction_failures": _int_or_none(recovery.get("extraction_failures")),
            "scrape_failure_unknown": _int_or_none(recovery.get("scrape_failure_unknown")),
            "unreported_fetch_outcomes": _int_or_none(recovery.get("unreported_fetch_outcomes")),
            "new_validated_evidence": _int_or_none(recovery.get("new_validated_evidence", validated_recovery_evidence)),
            "productive": (
                _int_or_none(recovery.get("new_validated_evidence", validated_recovery_evidence)) > 0
                if _int_or_none(recovery.get("new_validated_evidence", validated_recovery_evidence)) is not None
                else None
            ),
            "quality_gate_improved": quality_improved,
            "resulted_in_approval": approval is True if recovery_attempted else None,
            "success_definition": "Productive means at least one new validated evidence item; gate improvement and final approval are separate.",
        },
        "operations": {
            "runtime_seconds": round(run_metrics.elapsed_seconds, 3),
            "initial_searches": _int_or_none(run_metrics.initial_searches),
            "recovery_searches": _int_or_none(run_metrics.recovery_searches),
            "search_results": _int_or_none(
                initial_search.get("results_returned", (quality.get("search") or {}).get("results"))
            ),
            "initial_search_candidates": _int_or_none(initial_search.get("parsed_candidates")),
            "initial_search_accepted_sources": _int_or_none(initial_search.get("accepted_sources")),
            "initial_search_scoring_rejections": _int_or_none(initial_search.get("source_scoring_rejections")),
            "fetch_attempts": _int_or_none(run_metrics.scrape_calls),
            "initial_fetch_attempts": _int_or_none(initial_fetch.get("scrape_attempts")),
            "initial_fetch_successes": _int_or_none(initial_fetch.get("fetch_successes")),
            "initial_fetch_failures": _int_or_none(initial_fetch.get("fetch_failures")),
            "initial_extraction_failures": _int_or_none(initial_fetch.get("extraction_failures")),
            "initial_fetch_unreported_outcomes": _int_or_none(initial_fetch.get("unreported_outcomes")),
            "extraction_status": extraction.get("status"),
            "evidence_validation_rejected": _int_or_none(
                (diagnostics.get("evidence_validation") or {}).get("rejected")
            ),
            "recovery_fetch_successes": recovery_successes,
            "recovery_fetch_failures": recovery_failures,
            "scrape_requests_instrumented": _int_or_none(run_metrics.scrape_calls),
            "openai_calls": _int_or_none(run_metrics.openai_calls),
            "token_usage": None,
            "monetary_cost": None,
            "cache": {
                "configuration": cache_setting,
                "status": "measured" if cache_setting == "enabled" else "not_applicable",
                "hits": _int_or_none(run_metrics.cache_hits) if cache_setting == "enabled" else None,
                "misses": _int_or_none(run_metrics.cache_misses) if cache_setting == "enabled" else None,
            },
            "exceptions": [],
        },
        "exports": {
            **integrity,
            "paths": state.get("output_paths") or {},
        },
        "export_status": "passed" if integrity["passed"] else "failed",
    }


def _numeric_aggregate(values: list[Any]) -> dict[str, Any]:
    known = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {
        "sum": sum(known) if known else None,
        "mean": round(statistics.mean(known), 3) if known else None,
        "median": round(statistics.median(known), 3) if len(known) >= 2 else None,
        "range": [min(known), max(known)] if len(known) >= 2 else None,
        "measured_count": len(known),
        "selected_count": len(values),
    }


def aggregate_results(results: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("mode") not in {"offline", "live"}:
        raise ValueError("Aggregate mode must be exactly offline or live.")
    execution_kinds = {item.get("execution_kind") for item in results if item.get("execution_kind")}
    if metadata["mode"] == "offline" and execution_kinds - {"deterministic_fixture"}:
        raise ValueError("Offline aggregates cannot contain live execution results.")
    if metadata["mode"] == "live" and "deterministic_fixture" in execution_kinds:
        raise ValueError("Live aggregates cannot contain offline fixture results.")
    cache_configurations = {
        item.get("operations", {}).get("cache", {}).get("configuration")
        for item in results
        if item.get("operations", {}).get("cache", {}).get("configuration") is not None
    }
    if len(cache_configurations) > 1:
        raise ValueError("A single aggregate cannot mix different cache configurations.")
    completed = [
        case for case in results
        if case.get("research_outcome") in {"approved", "rejected_by_quality_gate"}
    ]
    attempted = [case for case in completed if case.get("recovery", {}).get("attempted") is True]
    productive = [case for case in attempted if case.get("recovery", {}).get("productive") is True]
    gate_improvement_cases = [
        case for case in attempted
        if isinstance(case.get("recovery", {}).get("quality_gate_improved"), bool)
    ]
    recovery_approval_cases = [
        case for case in attempted
        if isinstance(case.get("recovery", {}).get("resulted_in_approval"), bool)
    ]
    cache_enabled = metadata.get("configuration", {}).get("cache") == "enabled"

    def case_source_metric(key: str) -> dict[str, Any]:
        values = [
            case.get("sources", {}).get(key, {}).get("average_score_0_to_10")
            for case in completed
            if isinstance(
                case.get("sources", {}).get(key, {}).get("average_score_0_to_10"),
                (int, float),
            )
        ]
        summary = _numeric_aggregate(values)
        summary["aggregation_method"] = "unweighted mean of measured per-case values"
        return summary

    approved = sum(case.get("quality_gate", {}).get("approved") is True for case in completed)
    statuses = {
        "passed": sum(
            case.get("status") in {"passed", "rejected"}
            and case.get("research_outcome") in {"approved", "rejected_by_quality_gate"}
            and case.get("exports", {}).get("checked") is True
            and case.get("exports", {}).get("passed") is True
            for case in results
        ),
        "failed": sum(case.get("status") == "failed" for case in results),
        "rejected_by_quality_gate": sum(
            case.get("research_outcome") == "rejected_by_quality_gate" for case in results
        ),
        "incomplete": sum(case.get("status") in {"incomplete", "blocked"} for case in results),
    }
    execution_failures = sum(
        case.get("execution_status") == "failed"
        or (
            case.get("execution_status") is None
            and case.get("status") == "failed"
            and case.get("exports", {}).get("checked") is not True
        )
        for case in results
    )
    incomplete_runs = sum(
        case.get("execution_status") in {"incomplete", "blocked"}
        or (
            case.get("execution_status") is None
            and case.get("status") in {"incomplete", "blocked"}
        )
        for case in results
    )
    export_failures = sum(
        case.get("exports", {}).get("checked") is True
        and case.get("exports", {}).get("passed") is False
        for case in results
    )
    return {
        **metadata,
        "case_counts": {
            "selected": len(results),
            "executed": len([case for case in results if case.get("status") not in {"blocked"}]),
            **statuses,
            "quality_gate_approved": approved,
            "quality_gate_approval_rate": approved / len(completed) if completed else None,
            "quality_gate_approval_denominator_completed_runs": len(completed),
            "execution_failures": execution_failures,
            "quality_gate_rejections": statuses["rejected_by_quality_gate"],
            "incomplete_runs": incomplete_runs,
            "export_failures": export_failures,
            "approved_reports": approved,
            "operational_pass_definition": "Graph returned a completed report and Markdown/PDF export integrity checks passed; not a correctness judgment.",
        },
        "aggregates": {
            "runtime_seconds": _numeric_aggregate([
                case.get("operations", {}).get("runtime_seconds") for case in results
            ]),
            "openai_calls": _numeric_aggregate([
                case.get("operations", {}).get("openai_calls") for case in results
            ]),
            "evidence_coverage": _numeric_aggregate([
                case.get("evidence", {}).get("evidence_coverage") for case in completed
            ]),
            "citation_coverage": _numeric_aggregate([
                case.get("citations", {}).get("citation_coverage") for case in completed
            ]),
            "source_authority_per_case_mean": case_source_metric("authority"),
            "source_relevance_per_case_mean": case_source_metric("relevance"),
            "unique_urls_per_case": _numeric_aggregate([
                case.get("sources", {}).get("accepted_unique_urls") for case in completed
            ]),
            "unique_domains_per_case": _numeric_aggregate([
                case.get("sources", {}).get("unique_domains") for case in completed
            ]),
            "source_minimum_requirements_met": {
                "count": sum(case.get("sources", {}).get("minimum_url_domain_requirements_met") is True for case in completed),
                "denominator": sum(isinstance(case.get("sources", {}).get("minimum_url_domain_requirements_met"), bool) for case in completed),
            },
            "recovery_attempt_rate": len(attempted) / len(completed) if completed else None,
            "recovery_attempt_denominator": len(completed),
            "recovery_productivity_rate": len(productive) / len(attempted) if attempted else None,
            "recovery_productivity_denominator": len(attempted),
            "recovery_gate_improvement_rate": (
                sum(case.get("recovery", {}).get("quality_gate_improved") is True for case in gate_improvement_cases)
                / len(gate_improvement_cases) if gate_improvement_cases else None
            ),
            "recovery_gate_improvement_denominator": len(gate_improvement_cases),
            "recovery_final_approval_rate": (
                sum(case.get("recovery", {}).get("resulted_in_approval") is True for case in recovery_approval_cases)
                / len(recovery_approval_cases) if recovery_approval_cases else None
            ),
            "recovery_final_approval_denominator": len(recovery_approval_cases),
            "initial_searches": _numeric_aggregate([
                case.get("operations", {}).get("initial_searches") for case in results
            ]),
            "recovery_searches": _numeric_aggregate([
                case.get("operations", {}).get("recovery_searches") for case in results
            ]),
            "fetch_attempts": _numeric_aggregate([
                case.get("operations", {}).get("fetch_attempts") for case in results
            ]),
            "cache_hits": _numeric_aggregate([
                case.get("operations", {}).get("cache", {}).get("hits") if cache_enabled else None
                for case in results
            ]),
            "cache_misses": _numeric_aggregate([
                case.get("operations", {}).get("cache", {}).get("misses") if cache_enabled else None
                for case in results
            ]),
            "export_integrity_pass_rate": (
                sum(case.get("exports", {}).get("passed") is True for case in results)
                / sum(case.get("exports", {}).get("checked") is True for case in results)
                if any(case.get("exports", {}).get("checked") is True for case in results) else None
            ),
        },
        "per_case": results,
        "limitations": [
            "Source relevance and authority are existing heuristic assessments, not human verification.",
            "Multiple domains do not establish independent corroboration; syndication is not detected.",
            "Token usage and monetary cost are unavailable from current instrumentation.",
            "Quality-gate approval is not a guarantee of factual correctness.",
        ],
    }


def render_summary(report: dict[str, Any]) -> str:
    counts = report["case_counts"]
    aggregates = report["aggregates"]
    selected = report.get("selected_case_ids", [])
    lines = [
        f"# Research Benchmark {report['run_id']}",
        "",
        f"- Benchmark version: {report['benchmark_version']}",
        f"- Mode: {report['mode']}",
        f"- Started: {report['started_at']}",
        f"- Cases selected: {counts['selected']}; executed: {counts['executed']}",
        f"- Selected case IDs: {', '.join(selected) or 'none'}",
        f"- Cache configuration: {report.get('configuration', {}).get('cache', 'not applicable')}",
        "",
        "## Outcomes",
        "",
        f"- Operationally passed: {counts['passed']}",
        f"- Failed: {counts['failed']}",
        f"- Rejected by quality gate: {counts['rejected_by_quality_gate']}",
        f"- Incomplete or blocked: {counts['incomplete']}",
        f"- Execution failures: {counts.get('execution_failures', 0)}",
        f"- Export failures: {counts.get('export_failures', 0)}",
        f"- Quality-gate rejections: {counts.get('quality_gate_rejections', 0)}",
        f"- Approved reports: {counts.get('approved_reports', 0)}",
        f"- Quality approvals: {counts['quality_gate_approved']} / "
        f"{counts['quality_gate_approval_denominator_completed_runs']} completed runs "
        f"({ _percent(counts['quality_gate_approval_rate']) })",
        "",
        "## Aggregated metrics",
        "",
        f"- Evidence coverage: {_percent(aggregates['evidence_coverage']['mean'])} "
        f"(measured {aggregates['evidence_coverage']['measured_count']} / "
        f"{aggregates['evidence_coverage']['selected_count']})",
        f"- Citation coverage: {_percent(aggregates['citation_coverage']['mean'])} "
        f"(measured {aggregates['citation_coverage']['measured_count']} / "
        f"{aggregates['citation_coverage']['selected_count']})",
        f"- Recovery attempts: {_percent(aggregates['recovery_attempt_rate'])} "
        f"({aggregates['recovery_attempt_denominator']} completed cases)",
        f"- Productive recovery: {_percent(aggregates['recovery_productivity_rate'])} "
        f"({aggregates['recovery_productivity_denominator']} attempted cases; new validated evidence)",
        f"- Export integrity pass rate: {_percent(aggregates['export_integrity_pass_rate'])}",
        "",
        "## Per-case results",
        "",
        "| Case | Status | Execution | Quality gate | Export | Domains / URLs | Evidence | Citation coverage | Recovery rounds |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for case in report["per_case"]:
        sources = case.get("sources", {})
        evidence = case.get("evidence", {})
        citation = case.get("citations", {})
        recovery = case.get("recovery", {})
        lines.append(
            f"| {case.get('case_id')} | {case.get('status')} | {case.get('execution_status', 'not measured')} | "
            f"{case.get('quality_gate_status', case.get('research_outcome', 'not evaluated'))} | "
            f"{case.get('export_status', 'not measured')} | "
            f"{sources.get('unique_domains', '—')} / {sources.get('accepted_unique_urls', '—')} | "
            f"{evidence.get('accepted_count', '—')} | {_percent(citation.get('citation_coverage'))} | "
            f"{recovery.get('rounds', '—')} |"
        )
    lines.extend(["", "## Pipeline diagnostics", ""])
    for case in report["per_case"]:
        pipeline = case.get("pipeline_diagnostics", {})
        lines.append(f"### {case.get('case_id')}")
        lines.append("")
        sources = case.get("sources") or {}
        supporting = sources.get("claim_supporting_sources") or {}
        scores = sources.get("accepted_source_scores")
        if isinstance(scores, list):
            if scores:
                score_lines = "; ".join(
                    f"{item.get('url')} "
                    f"(authority {item.get('authority_score') if item.get('authority_score') is not None else 'not measured'}/10; "
                    f"relevance {item.get('relevance_score') if item.get('relevance_score') is not None else 'not measured'}/10; "
                    f"composite {item.get('composite_score') if item.get('composite_score') is not None else 'not measured'}/10)"
                    for item in scores
                )
            else:
                score_lines = "no accepted sources"
            lines.append(f"- Accepted-source scores: {score_lines}.")
            lines.append(
                "- Sources mapped to supported claims: "
                f"{supporting.get('unique_urls', 'not measured')} URL(s), "
                f"{supporting.get('unique_domains', 'not measured')} domain(s), "
                f"average authority {supporting.get('average_authority_score_0_to_10', 'not measured')}."
            )
        if pipeline.get("availability") != "measured":
            lines.append("Pipeline diagnostics: not available for this run.")
            lines.append("")
            continue
        search = pipeline.get("initial_search") or {}
        fetch = pipeline.get("initial_fetch") or {}
        extraction = pipeline.get("extraction") or {}
        evidence_validation = pipeline.get("evidence_validation") or {}
        lines.append(
            "- Initial search: "
            f"{search.get('results_returned', 'not measured')} returned, "
            f"{search.get('parsed_candidates', 'not measured')} parsed, "
            f"{search.get('accepted_sources', 'not measured')} accepted, "
            f"{search.get('source_scoring_rejections', 'not measured')} rejected by source scoring, "
            f"{search.get('invalid_url_candidates', 'not measured')} invalid URLs."
        )
        lines.append(
            "- Initial fetch: "
            f"{fetch.get('scrape_attempts', 'not measured')} attempted, "
            f"{fetch.get('fetch_successes', 'not measured')} successful, "
            f"{fetch.get('fetch_failures', 'not measured')} fetch failures, "
            f"{fetch.get('extraction_failures', 'not measured')} extraction failures, "
            f"{fetch.get('unreported_outcomes', 'not measured')} unreported outcomes."
        )
        lines.append(
            "- Evidence extraction/validation: "
            f"{extraction.get('status', 'not measured')}; "
            f"{evidence_validation.get('accepted', 'not measured')} accepted and "
            f"{evidence_validation.get('rejected', 'not measured')} rejected."
        )
        rounds = pipeline.get("recovery_rounds")
        if isinstance(rounds, list) and rounds:
            for round_data in rounds:
                fetch_round = round_data.get("fetch") or {}
                validated = round_data.get("evidence_validation") or {}
                lines.append(
                    f"- Recovery round {round_data.get('round', 'unknown')}: "
                    f"{len(round_data.get('queries_attempted', []))} queries; "
                    f"{round_data.get('candidate_urls_discovered', 'not measured')} new candidates; "
                    f"{round_data.get('duplicate_urls', 'not measured')} duplicate URLs; "
                    f"{fetch_round.get('fetch_successes', 'not measured')} fetch successes / "
                    f"{fetch_round.get('fetch_failures', 'not measured')} fetch failures; "
                    f"{validated.get('accepted', 'not measured')} evidence accepted / "
                    f"{validated.get('rejected', 'not measured')} rejected; "
                    f"{round_data.get('new_validated_evidence', 'not measured')} new validated evidence; "
                    f"{round_data.get('termination_reason', 'unknown')}."
                )
        else:
            lines.append("- Recovery-round diagnostics: not measured.")
        lines.append("")
    lines.extend(["", "## Interpretation", ""])
    if report["mode"] == "offline":
        lines.append(
            "This is a deterministic conformance-fixture report. It does not answer the benchmark research questions "
            "or represent live model/search quality."
        )
        lines.append(
            f"{report['case_counts'].get('fixture_verified', 0)} fixture scenario(s) verified; "
            f"{report['case_counts'].get('research_cases_executed', 0)} research case(s) executed."
        )
    else:
        lines.append(
            "Live outcomes describe only the listed completed runs. Approval is the application's existing quality-gate "
            "decision, not a human fact-check or production-readiness claim."
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def _write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report['run_id']}.json"
    markdown_path = output_dir / f"{report['run_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_summary(report), encoding="utf-8")
    return json_path, markdown_path


def _load_offline_fixtures(dataset: dict[str, Any], case_ids: list[str] | None) -> list[dict[str, Any]]:
    fixture_file = _read_json(OFFLINE_FIXTURES_PATH)
    fixtures = fixture_file.get("scenarios")
    if not isinstance(fixtures, list) or len(fixtures) != 10:
        raise ValueError("Offline fixture file must contain exactly 10 scenarios.")
    selected = {case["id"] for case in select_cases(dataset, case_ids)}
    fixture_by_id = {item.get("case_id"): item for item in fixtures if isinstance(item, dict)}
    if set(fixture_by_id) != {case["id"] for case in dataset["cases"]}:
        raise ValueError("Offline fixture IDs must match dataset case IDs exactly.")
    return [fixture_by_id[case["id"]] for case in dataset["cases"] if case["id"] in selected]


def _verify_offline_fixture(fixture: dict[str, Any]) -> tuple[bool, list[str]]:
    data = fixture.get("scenario_data")
    if not isinstance(data, dict):
        return False, ["scenario_data_missing"]
    errors: list[str] = []
    integer_counters = [
        value for key, value in data.items()
        if key.endswith(("_urls", "_sources", "_fetches", "_added", "_rounds"))
    ]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in integer_counters):
        errors.append("negative_or_non_integer_counter")
    rounds = data.get("research_rounds")
    attempted = fixture.get("recovery_attempted")
    if isinstance(rounds, int) and (rounds > 0) != attempted:
        errors.append("recovery_attempt_and_round_count_disagree")
    evidence_added = data.get("recovery_evidence_added")
    if isinstance(evidence_added, int) and (evidence_added > 0) != fixture.get(
        "recovery_produced_new_validated_evidence"
    ):
        errors.append("recovery_productivity_definition_mismatch")
    if fixture.get("simulated_gate_approved") != (fixture.get("termination_reason") == "approved"):
        errors.append("approval_and_termination_disagree")
    scenario = fixture.get("scenario")
    if scenario == "initial_approval" and (attempted or data.get("revision_count") != 0):
        errors.append("initial_approval_should_skip_recovery_and_revision")
    if scenario == "recovery_enables_approval":
        if (
            not attempted
            or not fixture["recovery_produced_new_validated_evidence"]
            or data.get("gate_failures_after_recovery", 1) >= data.get("gate_failures_before_recovery", 0)
        ):
            errors.append("recovery_approval_transition_not_demonstrated")
    if scenario == "candidates_without_new_evidence" and (
        data.get("recovery_candidate_urls", 0) <= 0 or evidence_added != 0
    ):
        errors.append("candidates_must_not_be_counted_as_new_evidence")
    if scenario == "duplicate_recovery_sources" and data.get("recovery_duplicate_urls", 0) <= 0:
        errors.append("duplicate_source_count_missing")
    if scenario == "recovery_fetch_failure" and (
        data.get("recovery_failed_fetches", 0) <= 0 or evidence_added != 0
    ):
        errors.append("failed_fetch_created_evidence_or_was_not_counted")
    if scenario == "evidence_improves_but_gate_rejects" and (
        fixture["simulated_gate_approved"]
        or data.get("recovery_evidence_added", 0) <= 0
        or data.get("gate_failures_after_recovery", 0) >= data.get("gate_failures_before_recovery", 0)
    ):
        errors.append("evidence_improvement_must_remain_separate_from_approval")
    if scenario == "recovery_round_budget_exhausted" and (
        data.get("research_rounds") != data.get("max_research_rounds")
        or fixture["termination_reason"] != "research_budget_exhausted"
    ):
        errors.append("research_budget_not_exhausted_as_configured")
    if scenario == "remaining_budget_insufficient" and (
        attempted or data.get("remaining_budget_available") is not False
    ):
        errors.append("insufficient_budget_must_skip_recovery")
    if scenario == "source_merge_and_id_remapping":
        evidence_ids = data.get("evidence_ids_after", [])
        claim_ids = data.get("claim_evidence_ids_after", [])
        if (
            data.get("ids_unique_and_resolved") is not True
            or len(evidence_ids) != len(set(evidence_ids))
            or not set(claim_ids).issubset(set(evidence_ids))
        ):
            errors.append("recovery_ids_are_not_unique_and_resolved")
    if scenario == "no_progress_preserves_last_critic":
        if (
            data.get("latest_critic_score") is None
            or data.get("citation_coverage") is None
            or data.get("unresolved_issues_preserved") is not True
            or data.get("cumulative_counters_verified") is not True
            or data.get("quality_status") != "NOT APPROVED"
            or fixture["termination_reason"] != "no_research_progress"
        ):
            errors.append("no_progress_state_was_not_preserved")
    return not errors, errors


def run_offline(
    dataset: dict[str, Any],
    case_ids: list[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict[str, Any], tuple[Path, Path]]:
    started_at = _utc_now()
    run_id = make_run_id("offline", started_at)
    fixtures = _load_offline_fixtures(dataset, case_ids)
    results = []
    for fixture in fixtures:
        verified, errors = _verify_offline_fixture(fixture)
        results.append({
            "case_id": fixture["case_id"],
            "status": "fixture_verified" if verified else "fixture_failed",
            "research_outcome": "not_evaluated_offline",
            "execution_kind": "deterministic_fixture",
            "scenario": fixture["scenario"],
            "simulated_gate_approved": fixture["simulated_gate_approved"],
            "recovery": {
                "attempted": fixture["recovery_attempted"],
                "productive": fixture["recovery_produced_new_validated_evidence"],
                "quality_gate_improved": None,
                "resulted_in_approval": None,
                "success_definition": "Fixture flag only; not measured research behavior.",
            },
            "termination_reason": fixture["termination_reason"],
            "fixture_invariant": fixture["invariant"],
            "fixture_observations": fixture["scenario_data"],
            "fixture_errors": errors,
            "exports": {"checked": False, "passed": None, "reason": "No research graph or report export runs offline."},
        })
    metadata = {
        "run_id": run_id,
        "benchmark_version": dataset["benchmark_version"],
        "mode": "offline",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "evaluation_date": dataset["evaluation_date"],
        "selected_case_ids": [item["case_id"] for item in fixtures],
        "configuration": {"cache": "not_applicable", "search_provider": "mocked", "model": "mocked"},
        "offline_fixture_version": _read_json(OFFLINE_FIXTURES_PATH)["fixture_version"],
        "offline_semantics": "Fixture conformance only; no research question was executed.",
    }
    report = aggregate_results(results, metadata)
    report["case_counts"]["executed"] = 0
    report["case_counts"]["fixture_verified"] = sum(item["status"] == "fixture_verified" for item in results)
    report["case_counts"]["fixture_failed"] = sum(item["status"] == "fixture_failed" for item in results)
    report["case_counts"]["fixture_scenarios_executed"] = len(results)
    report["case_counts"]["research_cases_executed"] = 0
    report["aggregates"]["export_integrity_pass_rate"] = None
    report["limitations"] = [
        "Offline fixtures validate reporting contracts only; they are not executions of the production graph.",
        "Graph routing, evidence validation, recovery merging, and export behavior are also covered by deterministic unit tests.",
        "No offline fixture result is a live research-quality measurement.",
    ]
    paths = _write_report(report, output_dir)
    return report, paths


@contextmanager
def _isolated_run_configuration(
    cache_setting: str,
    cache_root: Path,
    report_dir: Path,
) -> Iterator[None]:
    import cache
    import export

    prior_env = os.environ.get("PIPELINE_CACHE_ENABLED")
    old_cache_dirs = {
        name: getattr(cache, name)
        for name in ("CACHE_ROOT", "SCRAPE_CACHE_DIR", "SEARCH_CACHE_DIR", "STAGE_CACHE_DIR", "PIPELINE_CACHE_DIR")
    }
    old_output_dir = export.OUTPUT_DIR
    cache.CACHE_ROOT = cache_root
    cache.SCRAPE_CACHE_DIR = cache_root / "scrapes"
    cache.SEARCH_CACHE_DIR = cache_root / "searches"
    cache.STAGE_CACHE_DIR = cache_root / "stages"
    cache.PIPELINE_CACHE_DIR = cache_root / "pipeline"
    export.OUTPUT_DIR = report_dir
    os.environ["PIPELINE_CACHE_ENABLED"] = "true" if cache_setting == "enabled" else "false"
    try:
        yield
    finally:
        for name, value in old_cache_dirs.items():
            setattr(cache, name, value)
        export.OUTPUT_DIR = old_output_dir
        if prior_env is None:
            os.environ.pop("PIPELINE_CACHE_ENABLED", None)
        else:
            os.environ["PIPELINE_CACHE_ENABLED"] = prior_env


def _live_case(
    case: dict[str, Any],
    run_id: str,
    cache_setting: str,
    output_dir: Path,
) -> dict[str, Any]:
    from agents import get_active_model
    from graph import research_graph
    from metrics import get_metrics, reset_metrics

    started_at = _utc_now()
    case_output = output_dir / "artifacts" / run_id / case["id"]
    case_cache = output_dir / "isolated-cache" / run_id / case["id"]
    metrics = reset_metrics()
    try:
        with _isolated_run_configuration(cache_setting, case_cache, case_output):
            state = dict(research_graph.invoke({"topic": case["question"]}))
        metrics = get_metrics()
        result = summarize_state(case, state, metrics, cache_setting, started_at, _utc_now())
        result["execution_kind"] = "live_langgraph"
        result["model"] = get_active_model()
        result["search_provider"] = "Tavily"
        result["source_expected_characteristics"] = case["expected_source_characteristics"]
        if result["status"] == "passed" and result["quality_gate"]["approved"] is False:
            result["status"] = "rejected"
    except Exception as exc:
        metrics = get_metrics()
        result = {
            "case_id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "status": "failed",
            "research_outcome": "unknown",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "failure": {"exception_type": type(exc).__name__, "message": "Details omitted to prevent accidental secret disclosure."},
            "execution_status": "failed",
            "quality_gate_status": "unknown",
            "export_status": "not_checked",
            "operations": {
                "runtime_seconds": round(metrics.elapsed_seconds, 3),
                "openai_calls": metrics.openai_calls,
                "cache": {"configuration": cache_setting, "status": "measured" if cache_setting == "enabled" else "not_applicable"},
                "exceptions": [type(exc).__name__],
            },
            "exports": {"checked": False, "passed": None, "issues": ["graph_execution_failed"]},
        }
    return result


def run_live(
    dataset: dict[str, Any],
    case_ids: list[str] | None,
    max_live_cases: int,
    cache_setting: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict[str, Any], tuple[Path, Path]]:
    load_dotenv()
    from agents import DEFAULT_OPENAI_MODEL, get_active_model

    selected = select_cases(dataset, case_ids)
    if max_live_cases < 1 or max_live_cases > MAX_ALLOWED_LIVE_CASES:
        raise ValueError(f"max_live_cases must be between 1 and {MAX_ALLOWED_LIVE_CASES}.")
    if len(selected) > max_live_cases:
        raise ValueError(
            f"{len(selected)} case(s) selected exceeds --max-live-cases {max_live_cases}; "
            "select fewer cases or explicitly raise the cap."
        )
    if cache_setting not in {"enabled", "disabled"}:
        raise ValueError("Live evaluation requires an explicit cache setting.")
    if get_active_model() != DEFAULT_OPENAI_MODEL:
        raise ValueError(
            f"Live evaluation requires the configured project model {DEFAULT_OPENAI_MODEL}; "
            "the current configured model differs."
        )
    missing = [key for key in ("OPENAI_API_KEY", "TAVILY_API_KEY") if not os.getenv(key)]
    started_at = _utc_now()
    run_id = make_run_id("live", started_at)
    if missing:
        results = [{
            "case_id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "status": "blocked",
            "research_outcome": "unknown",
            "execution_status": "blocked",
            "quality_gate_status": "unknown",
            "export_status": "not_checked",
            "failure": {
                "exception_type": "MissingConfiguration",
                "message": "Required credential variables are not configured: " + ", ".join(missing),
            },
            "exports": {"checked": False, "passed": None},
        } for case in selected]
    else:
        results = [
            _live_case(
                {**case, "_evaluation_date": dataset["evaluation_date"]},
                run_id,
                cache_setting,
                output_dir,
            )
            for case in selected
        ]
    metadata = {
        "run_id": run_id,
        "benchmark_version": dataset["benchmark_version"],
        "mode": "live",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "evaluation_date": dataset["evaluation_date"],
        "selected_case_ids": [case["id"] for case in selected],
        "configuration": {
            "cache": cache_setting,
            "cache_isolation": "run-and-case scoped; existing application cache/checkpoints are not read or overwritten",
            "search_provider": "Tavily",
            "model": get_active_model(),
            "credentials_configured": {"OPENAI_API_KEY": not bool(missing and "OPENAI_API_KEY" in missing),
                                       "TAVILY_API_KEY": not bool(missing and "TAVILY_API_KEY" in missing)},
            "reasoning_effort": "none",
        },
    }
    report = aggregate_results(results, metadata)
    if missing:
        report["limitations"] = [
            "Live execution was blocked because required credential variables were not configured: " + ", ".join(missing),
            *report["limitations"],
        ]
    paths = _write_report(report, output_dir)
    return report, paths


def _latest_report(output_dir: Path) -> int:
    reports = sorted(output_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not reports:
        print("No benchmark reports found.")
        return 1
    print(f"Latest benchmark report: {reports[0]}")
    json_path = reports[0].with_suffix(".json")
    if json_path.is_file():
        print(f"Machine-readable report: {json_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic offline and opt-in live research benchmark.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Run the real LangGraph; requires credentials and explicit --cache.")
    mode.add_argument("--offline", action="store_true", help="Run deterministic conformance fixtures without API calls.")
    mode.add_argument("--latest", action="store_true", help="Print the most recent aggregate report paths.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Select a benchmark case ID; repeatable.")
    parser.add_argument("--max-live-cases", type=int, default=2, help=f"Maximum live cases (1-{MAX_ALLOWED_LIVE_CASES}, default 2).")
    parser.add_argument("--cache", choices=("enabled", "disabled"), help="Required for live mode; uses an isolated per-run cache.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for aggregate JSON/Markdown and run artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        dataset = load_dataset()
        if args.latest:
            return _latest_report(args.output_dir)
        if args.live:
            if args.cache is None:
                raise ValueError("Live evaluation requires an explicit --cache enabled|disabled setting.")
            report, paths = run_live(dataset, args.case_ids, args.max_live_cases, args.cache, args.output_dir)
        else:
            report, paths = run_offline(dataset, args.case_ids, args.output_dir)
    except ValueError as exc:
        print(f"Benchmark configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"Benchmark mode: {report['mode']}; run ID: {report['run_id']}")
    print(f"JSON report: {paths[0]}")
    print(f"Markdown summary: {paths[1]}")
    counts = report["case_counts"]
    if report["mode"] == "offline":
        print(
            f"Deterministic fixture scenarios executed: {counts.get('fixture_scenarios_executed', 0)}; "
            f"verified: {counts.get('fixture_verified', 0)}; "
            f"fixture failures: {counts.get('fixture_failed', 0)}; research cases executed: 0"
        )
        return 2 if counts.get("fixture_failed", 0) else 0
    print(
        f"Cases: {counts['selected']} selected; {counts['passed']} passed; "
        f"{counts['failed']} failed; {counts['rejected_by_quality_gate']} quality-gate rejected; "
        f"{counts['incomplete']} incomplete/blocked"
    )
    return 2 if counts["failed"] or counts["incomplete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic validation and reporting for research evidence and citations."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from pydantic import ValidationError

from research_models import (
    Claim,
    Document,
    Evidence,
    ResearchExtraction,
    ResearchNote,
)
from source_scoring import canonicalize_url, domain_for_url

RESEARCH_DATA_VERSION = "evidence-v2"
RESEARCH_PROMPT_VERSION = "citations-v2"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def parse_documents(scraped_text: str, source_metadata: list[dict] | None = None) -> list[Document]:
    """Turn the current reader tool's text protocol into typed documents."""
    metadata_by_url = {
        canonicalize_url(item.get("url", "")): item
        for item in (source_metadata or [])
        if item.get("url")
    }
    documents: list[Document] = []
    for block in re.split(r"\n={10,}\n", scraped_text or ""):
        title_match = re.search(r"(?m)^Title:\s*(.*)$", block)
        url_match = re.search(r"(?m)^URL:\s*(https?://\S+)", block)
        content_match = re.search(r"(?ms)^Content:\s*\n?(.*)$", block)
        if not url_match:
            continue
        url = url_match.group(1).strip().rstrip(".,;")
        fallback = metadata_by_url.get(canonicalize_url(url), {})
        title = (title_match.group(1).strip() if title_match else "") or fallback.get("title", "N/A")
        content = content_match.group(1).strip() if content_match else ""
        failed = not content or any(
            marker in content.lower()
            for marker in ("could not fetch page", "could not extract article body", "could not scrape url")
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else ""
        documents.append(Document(
            url=url,
            title=title,
            domain=domain_for_url(url),
            text="" if failed else content,
            metadata={"source_score": fallback.get("score"), "published_date": fallback.get("published_date")},
            extraction_status="failed" if failed else "success",
            word_count=len(content.split()) if not failed else 0,
            content_hash=digest,
        ))
    return documents


def parse_extraction(raw: str) -> ResearchExtraction:
    """Parse JSON returned by the existing summarizer chain."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
        return ResearchExtraction.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return ResearchExtraction()


def validate_extraction(
    topic: str,
    documents: list[Document],
    extraction: ResearchExtraction,
) -> tuple[list[Evidence], list[Claim], list[ResearchNote], dict[str, int]]:
    """Keep only evidence tied to successful documents and traceable excerpts."""
    by_url = {canonicalize_url(doc.url): doc for doc in documents if doc.extraction_status == "success"}
    evidence: list[Evidence] = []
    evidence_index_map: dict[int, str] = {}
    rejected = 0
    for index, proposal in enumerate(extraction.evidence):
        document = by_url.get(canonicalize_url(proposal.source_url))
        excerpt = proposal.excerpt.strip()
        if not document or not excerpt or normalize_text(excerpt) not in normalize_text(document.text):
            rejected += 1
            continue
        evidence_id = f"E{len(evidence) + 1}"
        evidence_index_map[index] = evidence_id
        evidence.append(Evidence(
            evidence_id=evidence_id,
            source_url=document.url,
            source_title=document.title,
            excerpt=excerpt,
            supporting_text=proposal.supporting_text.strip(),
            relevance_score=proposal.relevance_score,
            confidence=proposal.confidence,
            location=proposal.location,
        ))

    claims: list[Claim] = []
    notes: list[ResearchNote] = []
    for proposal in extraction.claims:
        evidence_ids = list(dict.fromkeys(
            evidence_index_map[index]
            for index in proposal.evidence_indices
            if index in evidence_index_map
        ))
        supporting = [item for item in evidence if item.evidence_id in evidence_ids]
        source_urls = list(dict.fromkeys(item.source_url for item in supporting))
        confidence = proposal.confidence
        if len({domain_for_url(url) for url in source_urls}) > 1:
            confidence = min(1.0, confidence + 0.1)
        claim = Claim(
            claim_id=f"C{len(claims) + 1}",
            claim_text=proposal.claim_text.strip(),
            evidence_ids=evidence_ids,
            source_urls=source_urls,
            confidence=confidence if supporting else 0,
            importance=proposal.importance,
        )
        claims.append(claim)
        notes.append(ResearchNote(
            topic=topic,
            claim=claim.claim_text,
            supporting_evidence=evidence_ids,
            source_urls=source_urls,
            confidence=claim.confidence,
        ))

    stats = {"proposals": len(extraction.evidence), "valid": len(evidence), "rejected": rejected}
    return evidence, claims, notes, stats


def calculate_domain_diversity(
    sources: list[dict], claims: list[dict] | list[Claim] | None = None
) -> dict[str, Any]:
    unique: dict[str, dict] = {}
    for source in sources:
        url = source.get("url", "")
        key = canonicalize_url(url)
        if key:
            unique.setdefault(key, source)
    domains = [source.get("domain") or domain_for_url(source.get("url", "")) for source in unique.values()]
    counts = Counter(domain for domain in domains if domain)
    total = len(sources)
    unique_domains = len(counts)
    ratio = unique_domains / total if total else 0.0
    source_values = list(unique.values())
    if total == 1 and source_values:
        score = float(source_values[0].get("authority_score", source_values[0].get("source_score_breakdown", {}).get("authority", source_values[0].get("score", 0))) or 0)
        classification = "single_authoritative_source" if score >= 9 else "single_low_authority_source"
    elif unique_domains > 1:
        classification = "diverse"
    elif total:
        classification = "single_domain_reuse"
    else:
        classification = "no_sources"

    domain_claim_counts: Counter[str] = Counter()
    for claim in claims or []:
        urls = claim.source_urls if isinstance(claim, Claim) else claim.get("source_urls", [])
        for domain in {domain_for_url(url) for url in urls}:
            if domain:
                domain_claim_counts[domain] += 1
    return {
        "total_sources": total,
        "unique_sources": len(unique),
        "total_domains": len(domains),
        "unique_domains": unique_domains,
        "diversity_ratio": round(ratio, 3),
        "average_source_quality": round(sum(float(s.get("score", s.get("source_score", 0)) or 0) for s in unique.values()) / len(unique), 2) if unique else 0.0,
        "average_authority_score": round(sum(float(s.get("authority_score", s.get("source_score_breakdown", {}).get("authority", 0)) or 0) for s in unique.values()) / len(unique), 2) if unique else 0.0,
        "authoritative_source_ratio": round(sum(float(s.get("authority_score", s.get("source_score_breakdown", {}).get("authority", 0)) or 0) >= 9 for s in unique.values()) / len(unique), 3) if unique else 0.0,
        "duplicate_source_ratio": round((total - len(unique)) / total, 3) if total else 0.0,
        "classification": classification,
        "repeated_domain_claims": {domain: count for domain, count in domain_claim_counts.items() if count > 1},
    }


def build_citation_context(claims: list[Claim], evidence: list[Evidence], sources: list[dict]) -> str:
    source_by_url = {canonicalize_url(item.get("url", "")): item for item in sources}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    lines = ["Validated research claims and citation keys:"]
    for claim in claims:
        lines.append(f"{claim.claim_id} | supported={claim.supported} | {claim.claim_text}")
        for evidence_id in claim.evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if not item:
                continue
            source = source_by_url.get(canonicalize_url(item.source_url), {})
            score = source.get("score", source.get("source_score", 0))
            authority = source.get("authority_score", source.get("source_score_breakdown", {}).get("authority", 0))
            lines.append(
                f"  {evidence_id} | {item.source_title} | {item.source_url} | "
                f"composite={score}/10 | authority={authority}/10 | excerpt={item.excerpt}"
            )
    if not claims:
        lines.append("No claims passed evidence validation. State that evidence is insufficient.")
    return "\n".join(lines)


def materialize_citations(
    report: str, claims: list[Claim], sources: list[dict]
) -> tuple[str, dict[str, list[str]], dict[str, Any]]:
    """Replace [C1] markers with links from validated claim/source mappings."""
    claim_by_id = {claim.claim_id: claim for claim in claims}
    source_by_url = {
        normalized: item
        for item in sources
        if (normalized := canonicalize_url(item.get("url", "")))
    }
    cited_claims: set[str] = set()
    citation_map: dict[str, list[str]] = {}
    invalid = 0

    # Citations must be expressed as claim markers. Remove model-authored links
    # before inserting links only from the validated claim/source mapping.
    raw_report = report or ""
    # Replace the model's bibliography wholesale after claim markers are
    # resolved, so removed model links do not leave broken Sources entries.
    raw_report = re.sub(
        r"(?ims)^#{1,6}\s+Sources\s*$.*?(?=^#{1,6}\s+|\Z)",
        "",
        raw_report,
    ).strip()
    direct_links = re.findall(r"\[[^\]]+\]\(https?://[^)]+\)(?:\s*\([^)]*/10\))?", raw_report)
    invalid += len(direct_links)
    raw_report = re.sub(r"\[[^\]]+\]\(https?://[^)]+\)(?:\s*\([^)]*/10\))?", "", raw_report)
    bare_urls = re.findall(r"(?<!\()https?://[^\s<>\])]+", raw_report)
    invalid += len(bare_urls)
    raw_report = re.sub(r"(?<!\()https?://[^\s<>\])]+", "", raw_report)

    # Combine whitespace-adjacent claim markers before resolving them. This
    # lets the existing URL de-duplication avoid repeating shared sources.
    marker = r"\[(C\d+(?:\s*,\s*C\d+)*)\]"
    adjacent_markers = rf"{marker}[ \t]*{marker}"
    while re.search(adjacent_markers, raw_report):
        raw_report = re.sub(
            adjacent_markers,
            lambda match: "[" + ",".join(dict.fromkeys(
                re.findall(r"C\d+", match.group(1) + "," + match.group(2))
            )) + "]",
            raw_report,
        )

    def replace(match: re.Match) -> str:
        nonlocal invalid
        ids = re.findall(r"C\d+", match.group(1))
        links: list[str] = []
        for claim_id in ids:
            claim = claim_by_id.get(claim_id)
            if not claim or not claim.supported:
                invalid += 1
                continue
            cited_claims.add(claim_id)
            for url in claim.source_urls:
                normalized_url = canonicalize_url(url)
                source = source_by_url.get(normalized_url) if normalized_url else None
                if not source:
                    invalid += 1
                    continue
                validated_url = canonicalize_url(source.get("url", ""))
                if not validated_url:
                    invalid += 1
                    continue
                score = source.get("score", source.get("source_score", 0))
                links.append(f"[{source.get('title') or 'Source'}]({validated_url}) ({score}/10)")
                citation_map.setdefault(claim_id, [])
                if validated_url not in citation_map[claim_id]:
                    citation_map[claim_id].append(validated_url)
        return "; ".join(dict.fromkeys(links)) if links else "[citation unavailable]"

    output = re.sub(marker, replace, raw_report)
    mapped_urls = {canonicalize_url(url) for urls in citation_map.values() for url in urls}
    for raw_url in re.findall(r"\]\((https?://[^)]+)\)", output):
        if canonicalize_url(raw_url) not in mapped_urls:
            invalid += 1
    all_claim_ids = {claim.claim_id for claim in claims}
    total = len(claims)
    coverage = len(cited_claims) / total if total else 0.0
    domain_use: Counter[str] = Counter(
        domain_for_url(url)
        for urls in citation_map.values()
        for url in set(urls)
        if domain_for_url(url)
    )
    referenced_urls = {canonicalize_url(url) for urls in citation_map.values() for url in urls}
    available_urls = {
        normalized
        for item in sources
        if (normalized := canonicalize_url(item.get("url", "")))
    }
    source_lines = []
    for url in dict.fromkeys(url for urls in citation_map.values() for url in urls):
        source = source_by_url.get(canonicalize_url(url), {})
        source_lines.append(
            f"- [{source.get('title') or 'Source'}]({url}) — Quality Score: "
            f"{source.get('score', source.get('source_score', 0))}/10"
        )
    output = output.rstrip() + "\n\n# Sources\n\n" + ("\n".join(source_lines) if source_lines else "No validated sources were cited.") + "\n"
    metrics = {
        "total_claims": total,
        "supported_claims": sum(claim.supported for claim in claims),
        "unsupported_claims": sum(not claim.supported for claim in claims),
        "claims_with_citations": len(cited_claims & all_claim_ids),
        "citation_coverage": round(coverage, 3),
        "invalid_citations": invalid,
        "unused_sources": len(available_urls - referenced_urls),
        "overused_domains": {domain: count for domain, count in domain_use.items() if count > 1},
    }
    return output, citation_map, metrics


def quality_gate(
    source_metrics: dict[str, Any],
    evidence_metrics: dict[str, Any],
    citation_metrics: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    if "total_claims" in evidence_metrics and evidence_metrics.get("total_claims", 0) == 0:
        issues.append("No claims passed validated evidence extraction; report evidence is insufficient.")
    if evidence_metrics.get("total_claims", 0) and evidence_metrics.get("evidence_coverage", 0) < 1:
        issues.append("Some claims lack validated supporting evidence.")
    if citation_metrics.get("total_claims", 0) and citation_metrics.get("citation_coverage", 0) < 1:
        issues.append("Not every extracted claim is cited in the report.")
    if citation_metrics.get("invalid_citations", 0):
        issues.append("The report contains citations not mapped to validated sources.")
    if source_metrics.get("classification") == "single_low_authority_source":
        issues.append("The report depends on one lower-authority source; qualify its conclusions.")
    if source_metrics.get("unique_domains", 0) == 1 and source_metrics.get("total_sources", 0) > 1:
        authority = source_metrics.get("average_authority_score", source_metrics.get("average_source_quality", 0))
        if authority >= 9:
            issues.append("The report relies on several pages from one domain; these do not provide independent corroboration.")
        else:
            issues.append("Multiple sources from one lower-authority domain do not provide independent corroboration.")
    return {"revision_required": bool(issues), "issues": issues}


def calculate_evidence_metrics(evidence: list[Evidence], claims: list[Claim], stats: dict[str, int]) -> dict[str, Any]:
    total = len(claims)
    supported = sum(claim.supported for claim in claims)
    return {
        "evidence_items": len(evidence),
        "valid_evidence": len(evidence),
        "rejected_evidence": stats.get("rejected", 0),
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": total - supported,
        "evidence_coverage": round(supported / total, 3) if total else 0.0,
        "average_claim_confidence": round(sum(c.confidence for c in claims) / total, 3) if total else 0.0,
        "independently_supported_claims": sum(len({domain_for_url(u) for u in c.source_urls}) > 1 for c in claims),
    }

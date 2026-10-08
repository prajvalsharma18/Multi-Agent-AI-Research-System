"""Typed records for source-grounded research data."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    title: str
    url: str
    domain: str
    snippet: str = ""
    published_date: str | None = None
    score: float = Field(default=0, ge=0, le=10)
    source_score: float = Field(default=0, ge=0, le=10)
    source_score_breakdown: dict[str, float] = Field(default_factory=dict)
    query_used: str = ""


class Document(BaseModel):
    url: str
    title: str
    domain: str
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_status: Literal["success", "failed"]
    word_count: int = Field(default=0, ge=0)
    content_hash: str = ""


class Evidence(BaseModel):
    evidence_id: str
    source_url: str
    source_title: str
    excerpt: str
    supporting_text: str
    relevance_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    location: str | None = None


class Claim(BaseModel):
    claim_id: str
    claim_text: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)
    importance: float = Field(default=0.5, ge=0, le=1)
    citation_required: bool = True

    @property
    def supported(self) -> bool:
        return bool(self.evidence_ids)


class ResearchNote(BaseModel):
    topic: str
    claim: str
    supporting_evidence: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)
    contradictions: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    overall_score: float = Field(ge=0, le=10)
    factual_accuracy: float = Field(default=0, ge=0, le=10)
    citation_correctness: float = Field(default=0, ge=0, le=10)
    source_quality: float = Field(default=0, ge=0, le=10)
    source_diversity: float = Field(default=0, ge=0, le=10)
    evidence_coverage: float = Field(default=0, ge=0, le=10)
    completeness: float = Field(default=0, ge=0, le=10)
    contradictions: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    revision_required: bool = False
    actionable_feedback: list[str] = Field(default_factory=list)


class EvidenceProposal(BaseModel):
    source_url: str
    excerpt: str
    supporting_text: str
    relevance_score: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    location: str | None = None


class ClaimProposal(BaseModel):
    claim_text: str
    evidence_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    importance: float = Field(default=0.5, ge=0, le=1)


class ResearchExtraction(BaseModel):
    evidence: list[EvidenceProposal] = Field(default_factory=list)
    claims: list[ClaimProposal] = Field(default_factory=list)

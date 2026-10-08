"""Domain-based source quality scoring for research URLs."""

from datetime import datetime, timezone
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

MIN_SOURCE_SCORE = 7
SCORING_WEIGHTS = {"authority": .50, "relevance": .15, "recency": .10, "evidence_quality": .10, "diversity": .15}


def canonicalize_url(url: str) -> str:
    try:
        p = urlparse((url or "").strip())
        if p.scheme.lower() not in {"http", "https"} or not p.hostname:
            return ""
        host = p.hostname.lower().removeprefix("www.")
        port = p.port
        netloc = host if not port or (p.scheme.lower(), port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
        query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}))
        return urlunparse((p.scheme.lower(), netloc, p.path.rstrip("/"), "", query, ""))
    except (ValueError, TypeError):
        return ""


_MULTI_SUFFIX = {"co.uk", "org.uk", "ac.uk", "com.au", "net.au", "co.in", "ac.in", "com.br", "co.jp", "co.nz"}


def domain_for_url(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
        parts = host.split(".")
        if len(parts) < 2 or all(part.isdigit() for part in parts):
            return host
        suffix = ".".join(parts[-2:])
        return ".".join(parts[-3:]) if suffix in _MULTI_SUFFIX and len(parts) >= 3 else suffix
    except (ValueError, TypeError):
        return ""

# Higher score = more trustworthy for research
DOMAIN_SCORES: dict[str, int] = {
    # Government / official (10)
    "pib.gov.in": 10,
    "gov.in": 10,
    "nic.in": 10,
    "europa.eu": 10,
    "who.int": 10,
    "un.org": 10,
    "nasa.gov": 10,
    "data.gov": 10,
    "gov.uk": 10,
    "gov.au": 10,
    # Academic / encyclopedic (7–9)
    "wikipedia.org": 7,
    "britannica.com": 8,
    "scholar.google.com": 9,
    "arxiv.org": 9,
    "nature.com": 9,
    "redis.io": 9,
    "mongodb.com": 8,
    "developer.mozilla.org": 9,
    "docs.python.org": 9,
    "docs.aws.amazon.com": 9,
    "learn.microsoft.com": 9,
    "sciencedirect.com": 9,
    "ieee.org": 9,
    # Major news (8)
    "reuters.com": 9,
    "bbc.com": 9,
    "bbc.co.uk": 9,
    "thehindu.com": 8,
    "indiatoday.in": 8,
    "timesofindia.indiatimes.com": 8,
    "economictimes.indiatimes.com": 8,
    "ndtv.com": 8,
    "indianexpress.com": 8,
    # Low quality / social (2–4)
    "facebook.com": 2,
    "fb.com": 2,
    "twitter.com": 3,
    "x.com": 3,
    "instagram.com": 2,
    "tiktok.com": 2,
    "pinterest.com": 3,
    "quora.com": 4,
    "reddit.com": 4,
    "medium.com": 5,
    "blogspot.com": 4,
    "wordpress.com": 4,
}


def score_url(url: str) -> int:
    """Return a quality score (1–10) for a URL based on its domain."""
    try:
        domain = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return 3

    if domain in DOMAIN_SCORES:
        return DOMAIN_SCORES[domain]

    for pattern, score in DOMAIN_SCORES.items():
        if domain == pattern or domain.endswith("." + pattern):
            return score

    if domain.endswith(".gov") or domain.endswith(".gov.in"):
        return 10
    if domain.endswith(".edu") or domain.endswith(".ac.in"):
        return 9
    if domain.endswith(".org"):
        return 7

    return 5  # unknown domain — moderate default


def score_source(source: dict, query: str = "", unseen_domain: bool = True) -> tuple[float, dict[str, float]]:
    authority = float(score_url(source.get("url", "")))
    text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
    terms = {word for word in re.findall(r"[a-z0-9]{3,}", query.lower())}
    relevance = 10.0 if not terms else round(10 * len(terms & set(re.findall(r"[a-z0-9]{3,}", text))) / len(terms), 2)
    date = source.get("published_date")
    recency = 5.0
    try:
        age = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(str(date).replace("Z", "+00:00")).astimezone(timezone.utc)).days)
        recency = 10.0 if age <= 365 else 7.0 if age <= 3 * 365 else 4.0
    except (ValueError, TypeError):
        pass
    evidence_quality = min(10.0, 3 + min(len(source.get("snippet", "")), 500) / 100)
    diversity = 10.0 if unseen_domain else 0.0
    parts = {"authority": authority, "relevance": relevance, "recency": recency, "evidence_quality": round(evidence_quality, 2), "diversity": diversity}
    total = round(sum(parts[k] * SCORING_WEIGHTS[k] for k in parts), 2)
    return total, {**parts, "total": total}


def score_label(score: int) -> str:
    if score >= 9:
        return "Excellent"
    if score >= 7:
        return "Reliable"
    if score >= 5:
        return "Moderate"
    return "Low"

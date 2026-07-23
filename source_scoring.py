"""Domain-based source quality scoring for research URLs."""

from urllib.parse import urlparse

MIN_SOURCE_SCORE = 7

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
        if domain.endswith(pattern) or pattern in domain:
            return score

    if domain.endswith(".gov") or domain.endswith(".gov.in"):
        return 10
    if domain.endswith(".edu") or domain.endswith(".ac.in"):
        return 9
    if domain.endswith(".org"):
        return 7

    return 5  # unknown domain — moderate default


def score_label(score: int) -> str:
    if score >= 9:
        return "Excellent"
    if score >= 7:
        return "Reliable"
    if score >= 5:
        return "Moderate"
    return "Low"

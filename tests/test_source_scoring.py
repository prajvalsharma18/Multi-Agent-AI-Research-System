import pytest

from source_scoring import score_label, score_url


def test_authoritative_and_low_quality_domains_are_scored():
    assert score_url("https://www.who.int/news/item") == 10
    assert score_url("https://www.nature.com/articles/example") == 9
    assert score_url("https://www.facebook.com/post") == 2


def test_domain_matching_does_not_accept_domain_name_as_substring():
    assert score_url("https://notwho.int.example.org/article") == 7
    assert score_url("https://who.int.example.org/article") == 7
    assert score_url("https://example.com") == 5


@pytest.mark.parametrize("url", ["", "not a url", "https://", None])
def test_malformed_or_missing_urls_do_not_crash(url):
    assert isinstance(score_url(url), int)


def test_scoring_is_deterministic_and_labelled():
    url = "https://arxiv.org/abs/1234.5678"
    assert score_url(url) == score_url(url) == 9
    assert score_label(score_url(url)) == "Excellent"


def test_labels_cover_quality_bands():
    assert score_label(8) == "Reliable"
    assert score_label(5) == "Moderate"
    assert score_label(2) == "Low"

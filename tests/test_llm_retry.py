import pytest

import agents
import llm_retry
import metrics


class TransientError(Exception):
    status_code = 429


class MissingModelError(Exception):
    status_code = 404


@pytest.fixture(autouse=True)
def retry_setup(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    monkeypatch.setattr(llm_retry.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(agents, "get_model_candidates", lambda: ["primary"])
    monkeypatch.setattr(agents, "clear_agent_caches", lambda: None)
    monkeypatch.setattr(agents, "set_active_model", lambda model: None)
    metrics.reset_metrics()


def test_transient_error_retries_then_succeeds():
    attempts = 0
    def invoke():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise TransientError("rate limited")
        return "ok"
    assert llm_retry.invoke_with_llm_retry(invoke, step="writer") == "ok"
    assert attempts == 2
    assert metrics.get_metrics().retries == 1


@pytest.mark.parametrize("status_code", [408, 503])
def test_timeout_and_service_connection_statuses_are_retryable(status_code):
    error_type = type("TransientNetworkError", (Exception,), {"status_code": status_code})
    attempts = 0
    def invoke():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error_type("temporary network failure")
        return "ok"
    assert llm_retry.invoke_with_llm_retry(invoke) == "ok"
    assert attempts == 2
    assert metrics.get_metrics().retries == 1


def test_retry_exhaustion_raises_contextual_error():
    with pytest.raises(llm_retry.LLMRequestError, match="rate limited") as error:
        llm_retry.invoke_with_llm_retry(lambda: (_ for _ in ()).throw(TransientError("rate limited")), step="writer")
    assert error.value.step == "writer"
    assert metrics.get_metrics().failures == 1


def test_unavailable_primary_tries_configured_fallback(monkeypatch):
    monkeypatch.setattr(agents, "get_model_candidates", lambda: ["primary", "fallback"])
    seen = []
    def invoke():
        seen.append("call")
        if len(seen) == 1:
            raise MissingModelError("not found")
        return "fallback result"
    assert llm_retry.invoke_with_llm_retry(invoke, step="writer") == "fallback result"
    assert len(seen) == 2


def test_non_transient_failure_does_not_retry():
    attempts = 0
    def invoke():
        nonlocal attempts
        attempts += 1
        raise ValueError("bad request")
    with pytest.raises(llm_retry.LLMRequestError):
        llm_retry.invoke_with_llm_retry(invoke)
    assert attempts == 1

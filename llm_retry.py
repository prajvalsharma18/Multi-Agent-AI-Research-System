"""Retry handling for transient OpenAI API failures."""

from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class LLMRequestError(Exception):
    """OpenAI request failed after retries or configured model fallback."""

    def __init__(self, message: str, *, model: str | None = None, step: str | None = None):
        self.model = model
        self.step = step
        super().__init__(message)


def _is_retryable(exc: Exception) -> bool:
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)):
            return True
    except ImportError:
        pass
    return getattr(exc, "status_code", None) in (408, 409, 429, 500, 502, 503, 504)


def _is_model_unavailable(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404


def invoke_with_llm_retry(invoke_fn: Callable[[], T], *, step: str = "LLM") -> T:
    from agents import clear_agent_caches, get_model_candidates, set_active_model
    from metrics import get_metrics

    retries = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "2")))
    candidates = get_model_candidates()
    last_error: Exception | None = None

    for model_index, model in enumerate(candidates):
        set_active_model(model)
        clear_agent_caches()
        for attempt in range(retries + 1):
            get_metrics().log_openai_call(step, model)
            try:
                return invoke_fn()
            except Exception as exc:
                last_error = exc
                if _is_retryable(exc) and attempt < retries:
                    get_metrics().retries += 1
                    delay = min(2 ** attempt, 30)
                    print(f"[{step}] Temporary OpenAI error. Retrying in {delay}s…")
                    time.sleep(delay)
                    continue
                if _is_model_unavailable(exc) and model_index + 1 < len(candidates):
                    get_metrics().retries += 1
                    print(f"[{step}] Model '{model}' unavailable; trying configured fallback.")
                    break
                get_metrics().failures += 1
                raise LLMRequestError(
                    f"OpenAI request failed during '{step}' using model '{model}': {exc}",
                    model=model,
                    step=step,
                ) from exc
    if last_error:
        raise LLMRequestError(f"OpenAI request failed during '{step}': {last_error}", step=step) from last_error
    raise LLMRequestError(f"OpenAI request failed during '{step}'.", step=step)

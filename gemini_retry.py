"""Retry handling for Gemini API errors (quota-safe for 15 RPM / 500 RPD)."""

from __future__ import annotations

import os
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_last_gemini_request_at: float = 0.0


class GeminiQuotaError(Exception):
    """Raised when Gemini quota/rate-limit is exceeded."""

    def __init__(self, message: str, *, model: str | None = None, step: str | None = None) -> None:
        self.model = model
        self.step = step
        super().__init__(message)


def is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("429", "resource_exhausted", "quota exceeded", "rate limit")
    )


def is_model_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("404", "not_found", "no longer available", "not found")
    )


def is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("503", "500", "timeout", "temporarily unavailable"))


def should_try_fallback_model(exc: Exception) -> bool:
    return is_model_unavailable(exc)


def parse_retry_delay(exc: Exception, default: float = 5.0) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1, 30)
    return default


def pace_gemini_request() -> None:
    """Space requests to stay under gemini-3.5-flash-lite free tier (15 RPM)."""
    global _last_gemini_request_at

    delay = float(os.getenv("GEMINI_REQUEST_DELAY_SEC", "4"))
    if delay <= 0:
        return

    now = time.perf_counter()
    if _last_gemini_request_at > 0:
        elapsed = now - _last_gemini_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
    _last_gemini_request_at = time.perf_counter()


def invoke_with_gemini_retry(invoke_fn: Callable[[], T], *, step: str = "LLM") -> T:
    from agents import clear_agent_caches, get_active_model, get_model_candidates, set_active_model
    from cache import get_daily_gemini_count, get_daily_gemini_limit, record_gemini_request
    from metrics import get_metrics

    daily_limit = get_daily_gemini_limit()
    daily_count = get_daily_gemini_count()
    if daily_count >= daily_limit:
        raise GeminiQuotaError(
            f"Daily Gemini request budget reached ({daily_count}/{daily_limit}). "
            f"Cached results are preserved. Try again tomorrow or raise GEMINI_DAILY_LIMIT.",
            step=step,
        )

    pace_gemini_request()
    get_metrics().log_gemini_call(step)

    candidates = get_model_candidates()
    last_error: Exception | None = None
    active_before = get_active_model()

    for model_index, model in enumerate(candidates):
        if model != active_before and model_index > 0:
            set_active_model(model)
            clear_agent_caches()
            active_before = model
        elif model_index == 0:
            set_active_model(model)

        max_attempts = 2 if model_index == 0 else 1
        for attempt in range(max_attempts):
            try:
                result = invoke_fn()
                count = record_gemini_request()
                get_metrics().daily_gemini_count = count
                return result
            except Exception as exc:
                last_error = exc

                if is_quota_error(exc):
                    raise GeminiQuotaError(
                        f"Gemini quota/rate limit at stage '{step}' "
                        f"({get_daily_gemini_count()}/{daily_limit} requests today). "
                        f"Rerun the same topic to resume from cache. "
                        f"Recommended: gemini-3.5-flash-lite with GEMINI_REQUEST_DELAY_SEC=4.",
                        model=model,
                        step=step,
                    ) from exc

                if is_transient_error(exc) and attempt < max_attempts - 1:
                    delay = parse_retry_delay(exc)
                    print(f"[{step}] Transient error on '{model}'. Retrying in {delay:.0f}s…")
                    time.sleep(delay)
                    continue

                if should_try_fallback_model(exc) and model_index < len(candidates) - 1:
                    print(f"[{step}] Model '{model}' unavailable. Switching fallback…")
                    set_active_model(candidates[model_index + 1])
                    clear_agent_caches()
                    break

                raise

    if last_error:
        raise last_error
    raise RuntimeError(f"[{step}] invoke failed with no result")

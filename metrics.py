"""Lightweight pipeline metrics (no secrets logged)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:
    gemini_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    scrape_calls: int = 0
    daily_gemini_count: int = 0
    gemini_stages: list[str] = field(default_factory=list)
    cache_hit_stages: list[str] = field(default_factory=list)
    _start_time: float = field(default_factory=time.perf_counter)

    def log_gemini_call(self, stage: str) -> None:
        self.gemini_calls += 1
        self.gemini_stages.append(stage)

    def log_cache_hit(self, stage: str) -> None:
        self.cache_hits += 1
        self.cache_hit_stages.append(stage)

    def log_cache_miss(self, stage: str) -> None:
        self.cache_misses += 1

    def log_scrape(self, count: int = 1) -> None:
        self.scrape_calls += count

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    def summary(self) -> str:
        from cache import get_daily_gemini_count, get_daily_gemini_limit

        stages = ", ".join(self.gemini_stages) or "none"
        hits = ", ".join(self.cache_hit_stages) or "none"
        daily = self.daily_gemini_count or get_daily_gemini_count()
        limit = get_daily_gemini_limit()
        return (
            f"Gemini calls this run: {self.gemini_calls} [{stages}] | "
            f"Today: {daily}/{limit} RPD | "
            f"Cache hits: {self.cache_hits} [{hits}] | "
            f"Scrapes: {self.scrape_calls} | "
            f"Time: {self.elapsed_seconds:.1f}s"
        )


_metrics = PipelineMetrics()


def get_metrics() -> PipelineMetrics:
    return _metrics


def reset_metrics() -> PipelineMetrics:
    global _metrics
    _metrics = PipelineMetrics()
    return _metrics

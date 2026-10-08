"""Lightweight pipeline metrics (no secrets logged)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:
    openai_calls: int = 0
    retries: int = 0
    failures: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    scrape_calls: int = 0
    openai_stages: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    cache_hit_stages: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)

    def set_quality(self, **values) -> None:
        self.quality.update(values)
    _start_time: float = field(default_factory=time.perf_counter)

    def log_openai_call(self, stage: str, model: str) -> None:
        self.openai_calls += 1
        self.openai_stages.append(stage)
        self.models.append(model)

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
        stages = ", ".join(self.openai_stages) or "none"
        models = ", ".join(dict.fromkeys(self.models)) or "none"
        hits = ", ".join(self.cache_hit_stages) or "none"
        return (
            f"OpenAI calls this run: {self.openai_calls} [{stages}] | "
            f"Models: {models} | "
            f"Retries: {self.retries} | Failures: {self.failures} | "
            f"Cache hits: {self.cache_hits} [{hits}] | "
            f"Scrapes: {self.scrape_calls} | "
            f"Quality: {self.quality} | "
            f"Time: {self.elapsed_seconds:.1f}s"
        )


_metrics = PipelineMetrics()


def get_metrics() -> PipelineMetrics:
    return _metrics


def reset_metrics() -> PipelineMetrics:
    global _metrics
    _metrics = PipelineMetrics()
    return _metrics

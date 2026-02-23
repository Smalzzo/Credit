"""In-memory runtime metrics used by API and dashboards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RuntimeMetrics:
    request_count: int = 0
    error_count: int = 0
    latency_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)

    def record(self, latency: float, inference: float, is_error: bool) -> None:
        self.request_count += 1
        if is_error:
            self.error_count += 1
        self.latency_ms.append(latency)
        self.inference_ms.append(inference)

    def summary(self) -> dict:
        if not self.latency_ms:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
            }
        sorted_latency = sorted(self.latency_ms)
        p50_idx = int(0.50 * (len(sorted_latency) - 1))
        p95_idx = int(0.95 * (len(sorted_latency) - 1))
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "latency_p50": float(sorted_latency[p50_idx]),
            "latency_p95": float(sorted_latency[p95_idx]),
        }

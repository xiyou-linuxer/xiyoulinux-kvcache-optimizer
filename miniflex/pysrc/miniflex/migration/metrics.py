"""Observability metrics for the heat-driven migration engine.

Tracks, in a backend-agnostic way:
  - promotion / demotion counts per tier
  - prefetch decisions and their estimated cost
  - per-tier block distribution over time
  - simple latency / throughput estimates for the planning loop

All counters are lightweight and safe to call from a hot path; they are designed
to be wired into MiniFlex's existing ``common.metrics`` later but stay
self-contained for now so the module can be unit tested in isolation.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MigrationMetrics:
  """Thread-safe counters for migration / prefetch activity."""
  # promotions = moves towards GPU (H2D, DISK2H first hop)
  promotions: int = 0
  # demotions = moves away from GPU (D2H, H2DISK first hop)
  demotions: int = 0
  # prefetch decisions issued
  prefetch_decisions: int = 0
  # estimated prefetch cost (sum of tier_cost * blocks)
  prefetch_cost: float = 0.0
  # blocks that were already on their target tier (no-op) per round
  noops: int = 0
  # planning round latency samples (ms)
  latencies_ms: List[float] = field(default_factory=list)
  # per-tier distribution snapshots (most recent)
  tier_distribution: Dict[str, int] = field(default_factory=lambda: {"gpu": 0, "cpu": 0, "ssd": 0})

  _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

  def record_promotion(self, n: int = 1) -> None:
    with self._lock:
      self.promotions += n

  def record_demotion(self, n: int = 1) -> None:
    with self._lock:
      self.demotions += n

  def record_prefetch(self, blocks: int, cost: float) -> None:
    with self._lock:
      self.prefetch_decisions += 1
      self.prefetch_cost += cost

  def record_noop(self, n: int = 1) -> None:
    with self._lock:
      self.noops += n

  def record_latency(self, seconds: float) -> None:
    with self._lock:
      # keep bounded
      if len(self.latencies_ms) >= 1024:
        self.latencies_ms.pop(0)
      self.latencies_ms.append(seconds * 1000.0)

  def update_distribution(self, dist: Dict[str, int]) -> None:
    with self._lock:
      self.tier_distribution = dict(dist)

  def snapshot(self) -> Dict:
    with self._lock:
      lats = self.latencies_ms
      p50 = self._percentile(lats, 50) if lats else 0.0
      p95 = self._percentile(lats, 95) if lats else 0.0
      return {
        "promotions": self.promotions,
        "demotions": self.demotions,
        "prefetch_decisions": self.prefetch_decisions,
        "prefetch_cost": round(self.prefetch_cost, 3),
        "noops": self.noops,
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "tier_distribution": dict(self.tier_distribution),
      }

  @staticmethod
  def _percentile(values: List[float], pct: int) -> float:
    if not values:
      return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Stopwatch:
  """Context manager that records elapsed time into a callback."""

  def __init__(self, callback):
    self._callback = callback
    self._start = 0.0

  def __enter__(self):
    self._start = time.perf_counter()
    return self

  def __exit__(self, *exc):
    self._callback(time.perf_counter() - self._start)

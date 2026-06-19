"""Thread-safe lightweight metrics for miniflex.

Usage across miniflex modules::

    from miniflex.common.metrics import incr, observe, dump_json, reset

    # counter
    incr("get_hit_cpu_blocks", 4)

    # histogram
    observe("transfer_latency_sec", 0.003)

    # snapshot (e.g. for get_kv_connector_stats or debug)
    snapshot()  # → dict

    # persist
    dump_json("/tmp/miniflex_metrics.json")

Outside (e.g. vllm_baseline_runner.py)::

    # delete file before a run so the adapter resets counters
    # after the run, read the JSON and merge into summary.json
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional


class Counter:
    """Thread-safe monotonic counter."""

    def __init__(self) -> None:
        self._value: float = 0.0
        self._lock = threading.Lock()

    def incr(self, delta: float = 1.0) -> None:
        with self._lock:
            self._value += delta

    def get(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0


class Histogram:
    """Thread-safe histogram with reservoir sampling when full."""

    def __init__(self, max_samples: int = 100_000) -> None:
        self._values: list[float] = []
        self._max_samples = max_samples
        self._total: int = 0  # number of observations (not capped)
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._total += 1
            if len(self._values) < self._max_samples:
                self._values.append(value)
            else:
                # reservoir sampling: replace a random element
                import random
                idx = random.randint(0, self._total - 1)
                if idx < self._max_samples:
                    self._values[idx] = value

    def snapshot(self) -> dict:
        with self._lock:
            if not self._values:
                return {"count": 0}
            ordered = sorted(self._values)
            return {
                "count": self._total,
                "samples": len(ordered),
                "min": ordered[0],
                "max": ordered[-1],
                "p50": _percentile(ordered, 0.50),
                "p95": _percentile(ordered, 0.95),
                "p99": _percentile(ordered, 0.99),
                "sum": sum(ordered),
                "mean": sum(ordered) / len(ordered),
            }

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._total = 0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same logic as vllm_baseline_runner)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = rank - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


class MetricsRegistry:
    """Central registry of counters and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def incr(self, name: str, delta: float = 1.0) -> None:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter()
        self._counters[name].incr(delta)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram()
        self._histograms[name].observe(value)

    def snapshot(self) -> dict:
        with self._lock:
            result: dict = {}
            for name, c in sorted(self._counters.items()):
                result[name] = c.get()
            for name, h in sorted(self._histograms.items()):
                result[name] = h.snapshot()
            return result

    def reset(self) -> None:
        with self._lock:
            for c in self._counters.values():
                c.reset()
            for h in self._histograms.values():
                h.reset()

    def dump_json(self, path: str | Path) -> None:
        data = self.snapshot()
        data["_ts"] = time.time()
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Global singleton — the only instance across the whole scheduler process
# ---------------------------------------------------------------------------
_global = MetricsRegistry()


def incr(name: str, delta: float = 1.0) -> None:
    _global.incr(name, delta)


def observe(name: str, value: float) -> None:
    _global.observe(name, value)


def snapshot() -> dict:
    return _global.snapshot()


def reset() -> None:
    _global.reset()


def dump_json(path: str | Path) -> None:
    _global.dump_json(path)


def dump_json_if_missing(path: str | Path) -> None:
    """Dump JSON, resetting first if the file doesn't exist.

    Used by the adapter so a runner-side ``rm /tmp/miniflex_metrics.json``
    automatically causes a fresh reset before the next data is written.
    """
    if not Path(path).exists():
        reset()
    dump_json(path)

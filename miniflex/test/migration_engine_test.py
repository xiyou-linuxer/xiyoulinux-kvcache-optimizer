"""Unit tests for the unified MigrationEngine + metrics (system-side).

Run with:  PYTHONPATH=pysrc python -m pytest test/migration_engine_test.py
or simply: PYTHONPATH=pysrc python test/migration_engine_test.py
"""
import sys
import traceback

from miniflex.migration import (
  HeatTracker,
  MigrationEngine,
  MigrationEngineConfig,
  MigrationMetrics,
  Stopwatch,
  Tier,
)


class FakeClock:
  def __init__(self, start=1000.0):
    self.now = start

  def time(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


def _make_engine(clock=None, **kw):
  cfg = MigrationEngineConfig(
    decay=kw.get("decay", 0.9),
    hot_threshold=kw.get("hot_threshold", 3.0),
    cold_threshold=kw.get("cold_threshold", 0.5),
    max_promotions_per_round=kw.get("max_promotions", 8),
    max_demotions_per_round=kw.get("max_demotions", 8),
    max_inflight_blocks=kw.get("max_inflight", 32),
  )
  return MigrationEngine(config=cfg, time_func=(clock or FakeClock()).time)


def test_engine_basic_loop():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  # Make a block hot on SSD so it should be promoted.
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  assert not plan.is_empty
  assert plan.num_promotions >= 1
  assert "DISK2H" in plan.transfer_types()


def test_engine_inflight_dedup():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  p1 = eng.tick()
  assert len(p1.ops) >= 1
  # Second tick without marking complete must NOT reschedule the same block.
  p2 = eng.tick()
  assert p2.is_empty, "block should not be double-booked while in-flight"


def test_engine_mark_completed():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  eng.tick()
  # After completion, tracker reflects move to CPU (first hop of two-hop).
  eng.mark_completed("ssd", 1)
  assert eng.tracker.get("ssd", 1) is None
  assert eng.tracker.get("cpu", 1) is not None


def test_engine_demotion():
  clock = FakeClock()
  eng = _make_engine(clock=clock, hot_threshold=3.0, cold_threshold=0.5)
  # A block on GPU that goes cold (only touched once, long ago).
  eng.touch("gpu", 5)
  clock.advance(60)  # let it decay well below cold_threshold
  plan = eng.tick()
  assert plan.num_demotions >= 1
  assert "D2H" in plan.transfer_types()


def test_engine_bandwidth_cap():
  clock = FakeClock()
  eng = _make_engine(clock=clock, max_promotions=2, max_inflight=2)
  for i in range(10):
    for _ in range(10):
      eng.touch("ssd", i)
      clock.advance(0.05)
  plan = eng.tick()
  # Promotions capped by max_promotions_per_round.
  assert plan.num_promotions <= 2


def test_prefetch_inflight_budget():
  clock = FakeClock()
  eng = _make_engine(clock=clock, max_inflight=4)
  # Fill up in-flight slots.
  for i in range(4):
    for _ in range(10):
      eng.touch("ssd", i)
      clock.advance(0.05)
  eng.tick()  # schedules 4 promotions -> 4 in-flight
  # Now prefetch budget should be 0 (max_inflight - inflight).
  dec = eng.request_prefetch(
    request_id=99, matched={"cpu": 8, "ssd": 4}, gpu_blocks_available=10,
  )
  assert dec.total_blocks == 0, "prefetch must yield to in-flight bandwidth"


def test_metrics_counters():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  eng.tick()
  eng.request_prefetch(request_id=1, matched={"cpu": 4}, gpu_blocks_available=4)
  stats = eng.stats()
  assert stats["promotions"] >= 1
  assert stats["prefetch_decisions"] == 1
  assert stats["tracked_blocks"] == 1
  assert stats["inflight_blocks"] >= 1


def test_metrics_percentiles():
  m = MigrationMetrics()
  for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    m.record_latency(v / 1000.0)  # ms
  snap = m.snapshot()
  assert snap["latency_p50_ms"] > 0
  assert snap["latency_p95_ms"] >= snap["latency_p50_ms"]


def test_stopwatch():
  import time
  m = MigrationMetrics()
  with Stopwatch(m.record_latency):
    time.sleep(0.01)
  assert len(m.latencies_ms) == 1
  assert m.latencies_ms[0] >= 5  # at least ~5ms


def test_heat_tier_weight():
  """A cold-tier (SSD) hit should heat a block faster than a GPU hit."""
  clock = FakeClock()
  ht_gpu = HeatTracker(decay=0.9, time_func=clock.time, use_tier_weight=True)
  ht_ssd = HeatTracker(decay=0.9, time_func=clock.time, use_tier_weight=True)
  ht_gpu.touch("gpu", 1)
  ht_ssd.touch("ssd", 1)
  assert ht_ssd.get("ssd", 1).score > ht_gpu.get("gpu", 1).score


def test_heat_burst_detection():
  clock = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clock.time)
  # 3 hits within the burst window should flag bursting.
  for _ in range(3):
    ht.touch("cpu", 1)
    clock.advance(0.3)
  assert ht.get("cpu", 1).is_bursting is True
  # After a long gap, burst state should clear on next decay.
  clock.advance(10)
  ht.decay_all()
  assert ht.get("cpu", 1).is_bursting is False


def _main():
  funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
  failed = 0
  for fn in funcs:
    try:
      fn()
      print(f"  PASS {fn.__name__}")
    except Exception:
      failed += 1
      print(f"  FAIL {fn.__name__}")
      traceback.print_exc()
  print(f"\n{len(funcs) - failed} passed, {failed} failed")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(_main())

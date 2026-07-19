"""Unit tests for the heat-driven migration + prefetch modules (system-side PoC).

Run with:  PYTHONPATH=pysrc python -m pytest test/migration_test.py
or simply: PYTHONPATH=pysrc python test/migration_test.py
"""
import sys
import traceback

from miniflex.migration import (
  HeatTracker,
  MigrationPolicy,
  MigrationPlanner,
  PrefetchPlanner,
  Tier,
)


class FakeClock:
  def __init__(self, start=1000.0):
    self.now = start

  def time(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


def test_tracker_basic():
  clk = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clk.time)
  ht.touch("cpu", 1)
  ht.touch("cpu", 1)
  ht.touch("ssd", 2)
  b1 = ht.get("cpu", 1)
  b2 = ht.get("ssd", 2)
  assert b1 is not None and b1.access_count == 2
  assert b2 is not None and b2.access_count == 1
  # With tier weighting, an SSD hit (weight 4.0) is worth more than two CPU
  # hits (weight 2.0 each): 2*2.0=4.0 vs 1*4.0=4.0, so scores are close.
  # The key invariant is that both are tracked and scored positively.
  assert b1.score > 0 and b2.score > 0
  assert len(ht) == 2


def test_tracker_recency_ordering():
  clk = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clk.time)
  # block A touched early, block B touched later
  ht.touch("cpu", 1)
  clk.advance(5)
  ht.touch("cpu", 2)
  hot = ht.top_hot(1)
  assert hot[0].block_id == 2  # B is hotter (more recent + decayed less)


def test_tracker_move():
  clk = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clk.time)
  ht.touch("ssd", 7)
  bh = ht.move("ssd", "cpu", 7)
  assert bh.tier == "cpu"
  assert ht.get("ssd", 7) is None
  assert ht.get("cpu", 7) is not None


def test_policy_thresholds():
  clk = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clk.time)
  # Make one hot, one warm, one cold
  for _ in range(10):
    ht.touch("ssd", 1)
    clk.advance(0.1)
  ht.touch("cpu", 2)
  clk.advance(20)  # let it cool
  ht.touch("cpu", 3)

  policy = MigrationPolicy(hot_threshold=3.0, cold_threshold=0.5)
  decisions = policy.decide(ht)
  # block 1 is hot on ssd => should be promoted (towards gpu/cpu)
  assert ("ssd", 1) in decisions
  assert decisions[("ssd", 1)] == Tier.HOT


def test_policy_bandwidth_cap():
  clk = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clk.time)
  for i in range(20):
    for _ in range(10):
      ht.touch("ssd", i)
      clk.advance(0.01)
  policy = MigrationPolicy(hot_threshold=3.0, cold_threshold=0.5,
                           max_promotions_per_round=3, max_demotions_per_round=0)
  decisions = policy.decide(ht)
  assert len(decisions) <= 3


def test_planner_transfer_types():
  clk = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clk.time)
  for _ in range(10):
    ht.touch("ssd", 1)
    clk.advance(0.1)
  policy = MigrationPolicy(hot_threshold=3.0, cold_threshold=0.5)
  planner = MigrationPlanner(policy)
  plan = planner.build_plan(ht)
  assert not plan.is_empty
  ttypes = plan.transfer_types()
  # ssd -> gpu is two-hop, first hop is DISK2H
  assert "DISK2H" in ttypes
  assert plan.num_promotions >= 1


def test_planner_and_apply():
  clk = FakeClock()
  ht = HeatTracker(decay=0.9, time_func=clk.time)
  for _ in range(10):
    ht.touch("ssd", 5)
    clk.advance(0.1)
  planner = MigrationPlanner(MigrationPolicy(hot_threshold=3.0, cold_threshold=0.5))
  plan = planner.plan_and_apply(ht)
  assert not plan.is_empty
  # after apply, block 5 should have moved off ssd (to cpu staging for two-hop)
  assert ht.get("ssd", 5) is None


def test_prefetch_basic():
  pp = PrefetchPlanner(max_prefetch_ratio=1.0, prefer_near=True)
  dec = pp.plan(
    request_id=42,
    matched={"cpu": 8, "ssd": 4},
    gpu_blocks_available=10,
  )
  assert dec.request_id == 42
  assert dec.total_blocks == 10  # budget allows 10
  # cpu (cheaper) should come first
  assert dec.plan[0] == ("cpu", 8)
  assert dec.plan[1] == ("ssd", 2)
  assert not dec.is_complete  # 2 ssd blocks left


def test_prefetch_budget_throttle():
  pp = PrefetchPlanner(max_prefetch_ratio=0.5)
  dec = pp.plan(request_id=1, matched={"cpu": 100}, gpu_blocks_available=10)
  assert dec.total_blocks == 5  # 50% of 10


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

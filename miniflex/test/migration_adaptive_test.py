"""Unit tests for adaptive threshold tuning (Commit 2).

Run with:  PYTHONPATH=pysrc python -m pytest test/migration_adaptive_test.py
or simply: PYTHONPATH=pysrc python test/migration_adaptive_test.py
"""
import sys
import traceback

from miniflex.migration import (
  MigrationEngine,
  MigrationEngineConfig,
  AdaptiveConfig,
  AdaptiveTuner,
)


class FakeClock:
  def __init__(self, start=1000.0):
    self.now = start

  def time(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


def test_adaptive_tuner_expands_hot_tier():
  """High cold-tier access should lower hot threshold (expand hot tier)."""
  cfg = AdaptiveConfig(cold_access_trigger=10, cooldown_rounds=1)
  tuner = AdaptiveTuner(cfg)
  hot, cold = tuner.tune(
    current_hot=3.0, current_cold=0.5,
    promotions=0, demotions=0, cold_accesses=15,
  )
  assert hot < 3.0, "hot threshold should decrease when cold access is high"
  assert cold < 0.5, "cold threshold should decrease when cold access is high"
  assert hot > cold, "hot must stay above cold"


def test_adaptive_tuner_respects_bounds():
  cfg = AdaptiveConfig(min_hot_threshold=0.5, max_hot_threshold=5.0)
  tuner = AdaptiveTuner(cfg)
  hot, cold = tuner.tune(
    current_hot=0.5, current_cold=0.1,
    promotions=0, demotions=0, cold_accesses=100,
  )
  assert hot >= 0.5, "hot should not go below min"
  assert hot <= 5.0, "hot should not exceed max"


def test_adaptive_tuner_cooldown():
  cfg = AdaptiveConfig(cold_access_trigger=10, cooldown_rounds=3)
  tuner = AdaptiveTuner(cfg)
  # First call triggers adjustment (cold_accesses >= trigger), entering cooldown.
  h1, c1 = tuner.tune(3.0, 0.5, promotions=0, demotions=0, cold_accesses=100)
  assert h1 < 3.0, "should adjust when cold access is high"
  # During cooldown, no adjustment even with high cold access.
  h2, c2 = tuner.tune(h1, c1, promotions=0, demotions=0, cold_accesses=100)
  assert h2 == h1 and c2 == c1, "should not adjust during cooldown"
  # After cooldown expires (3 more calls), it can adjust again.
  for _ in range(2):
    tuner.tune(h1, c1, promotions=1, demotions=0, cold_accesses=0)
  h3, c3 = tuner.tune(h1, c1, promotions=0, demotions=0, cold_accesses=100)
  assert h3 < h1, "should adjust again after cooldown expires"


def test_engine_adaptive_disabled_by_default():
  clock = FakeClock()
  cfg = MigrationEngineConfig(decay=0.9, hot_threshold=3.0, cold_threshold=0.5)
  eng = MigrationEngine(config=cfg, time_func=clock.time)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  # Without adaptive config, thresholds must not change.
  assert eng.policy.hot_threshold == 3.0
  assert eng.policy.cold_threshold == 0.5


def test_engine_adaptive_enabled():
  clock = FakeClock()
  adaptive = AdaptiveConfig(cold_access_trigger=5, cooldown_rounds=1)
  cfg = MigrationEngineConfig(
    decay=0.9, hot_threshold=3.0, cold_threshold=0.5, adaptive=adaptive,
  )
  eng = MigrationEngine(config=cfg, time_func=clock.time)
  # Create enough cold-tier accesses to trigger tuning.
  for i in range(10):
    for _ in range(5):
      eng.touch("ssd", i)
      clock.advance(0.05)
  eng.tick()
  # After tick, thresholds should have been adjusted (hot lowered).
  assert eng.policy.hot_threshold < 3.0, "adaptive should lower hot threshold"
  assert eng.policy.hot_threshold > eng.policy.cold_threshold


def test_engine_adaptive_keeps_thresholds_valid():
  clock = FakeClock()
  adaptive = AdaptiveConfig(min_hot_threshold=0.5, max_hot_threshold=10.0)
  cfg = MigrationEngineConfig(
    decay=0.9, hot_threshold=3.0, cold_threshold=0.5, adaptive=adaptive,
  )
  eng = MigrationEngine(config=cfg, time_func=clock.time)
  for i in range(5):
    for _ in range(5):
      eng.touch("ssd", i)
      clock.advance(0.05)
  eng.tick()
  # Thresholds must stay within bounds and hot > cold.
  assert 0.5 <= eng.policy.hot_threshold <= 10.0
  assert eng.policy.hot_threshold > eng.policy.cold_threshold




def test_adaptive_tuner_idle_restore_requires_sustained_idleness():
  """A single quiet round must not undo a cold-pressure expansion."""
  cfg = AdaptiveConfig(cold_access_trigger=10, cooldown_rounds=0, idle_restore_rounds=3)
  tuner = AdaptiveTuner(cfg)
  # Expand once under cold pressure.
  h1, c1 = tuner.tune(3.0, 0.5, promotions=0, demotions=0, cold_accesses=100)
  assert h1 < 3.0
  # Two idle rounds: below the restore threshold, thresholds must hold.
  h2, c2 = tuner.tune(h1, c1, promotions=0, demotions=0, cold_accesses=0)
  assert (h2, c2) == (h1, c1), "first idle round must not restore"
  h3, c3 = tuner.tune(h2, c2, promotions=0, demotions=0, cold_accesses=0)
  assert (h3, c3) == (h1, c1), "second idle round must not restore"
  # Third consecutive idle round: restore kicks in.
  h4, c4 = tuner.tune(h3, c3, promotions=0, demotions=0, cold_accesses=0)
  assert h4 > h3 and c4 > c3, "sustained idleness should restore thresholds"


def test_adaptive_tuner_cold_pressure_resets_idle_streak():
  """Cold pressure between idle rounds resets the restore countdown."""
  cfg = AdaptiveConfig(cold_access_trigger=10, cooldown_rounds=0, idle_restore_rounds=3)
  tuner = AdaptiveTuner(cfg)
  tuner.tune(3.0, 0.5, promotions=0, demotions=0, cold_accesses=100)  # expand
  tuner.tune(2.5, 0.4, promotions=0, demotions=0, cold_accesses=0)    # idle 1
  tuner.tune(2.5, 0.4, promotions=0, demotions=0, cold_accesses=0)    # idle 2
  # Cold pressure returns: streak resets.
  tuner.tune(2.5, 0.4, promotions=0, demotions=0, cold_accesses=100)
  h, c = tuner.tune(2.0, 0.3, promotions=0, demotions=0, cold_accesses=0)  # idle 1 again
  assert (h, c) == (2.0, 0.3), "idle streak must restart after cold pressure"


def test_adaptive_tuner_mixed_signals_reset_idle_streak():
  """Migration activity below the cold trigger counts as not-idle."""
  cfg = AdaptiveConfig(cold_access_trigger=10, cooldown_rounds=0, idle_restore_rounds=2)
  tuner = AdaptiveTuner(cfg)
  tuner.tune(3.0, 0.5, promotions=0, demotions=0, cold_accesses=100)  # expand
  tuner.tune(2.5, 0.4, promotions=0, demotions=0, cold_accesses=0)    # idle 1
  # Active migration round (not idle): resets streak.
  h, c = tuner.tune(2.5, 0.4, promotions=3, demotions=0, cold_accesses=0)
  assert (h, c) == (2.5, 0.3 + 0.1) or (h, c) == (2.5, 0.4), "mixed round must not restore"
  h2, c2 = tuner.tune(h, c, promotions=0, demotions=0, cold_accesses=0)  # idle 1 again
  assert (h2, c2) == (h, c), "streak must have been reset by the active round"


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

"""Unit tests for AccessBridge (data-plane access reporting).

Run with:  PYTHONPATH=pysrc python -m pytest test/migration_access_bridge_test.py
"""
import sys
import traceback

from miniflex.migration import (
  AccessBridge,
  AccessReport,
  MigrationEngine,
  MigrationEngineConfig,
)


class FakeClock:
  def __init__(self, start=1000.0):
    self.now = start

  def time(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


def _make_engine(clock=None):
  cfg = MigrationEngineConfig(decay=0.9, hot_threshold=3.0, cold_threshold=0.5)
  return MigrationEngine(config=cfg, time_func=(clock or FakeClock()).time)


def test_report_touches_all_tiers():
  eng = _make_engine()
  bridge = AccessBridge(eng)
  n = bridge.report(AccessReport(
    request_id="r1",
    gpu_blocks=[1, 2],
    cpu_blocks=[3],
    ssd_blocks=[4, 5, 6],
  ))
  assert n == 6
  assert eng.tracker.get("gpu", 1) is not None
  assert eng.tracker.get("gpu", 2) is not None
  assert eng.tracker.get("cpu", 3) is not None
  assert eng.tracker.get("ssd", 4) is not None
  assert eng.tracker.get("ssd", 6) is not None


def test_report_forwards_phase_and_decode_step():
  eng = _make_engine()
  bridge = AccessBridge(eng)
  bridge.report(AccessReport(
    request_id="r2",
    phase="decode",
    decode_step=7,
    gpu_blocks=[10],
  ))
  bh = eng.tracker.get("gpu", 10)
  assert bh is not None
  assert bh.phase == "decode"
  assert bh.decode_step == 7


def test_empty_report_is_noop():
  eng = _make_engine()
  bridge = AccessBridge(eng)
  n = bridge.report(AccessReport(request_id="r3"))
  assert n == 0
  assert len(eng.tracker) == 0


def test_report_feeds_engine_policy():
  """Repeated SSD hits via the bridge should drive a promotion decision."""
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  bridge = AccessBridge(eng)
  for _ in range(10):
    bridge.report(AccessReport(request_id="r4", ssd_blocks=[42]))
    clock.advance(0.1)
  plan = eng.tick()
  # The hot SSD block should be scheduled for promotion (DISK2H first hop).
  assert "DISK2H" in plan.transfer_types()


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

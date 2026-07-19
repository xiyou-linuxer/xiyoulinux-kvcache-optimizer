"""Unit tests for MigrationExecutor (plan -> TransferOpGraph bridge).

Run with:  PYTHONPATH=pysrc python -m pytest test/migration_executor_test.py
or simply: PYTHONPATH=pysrc python test/migration_executor_test.py
"""
import sys
import traceback
import numpy as np

from miniflex.common.transfer import TransferType
from miniflex.migration import (
  MigrationEngine,
  MigrationEngineConfig,
  MigrationExecutor,
)


class FakeClock:
  def __init__(self, start=1000.0):
    self.now = start

  def time(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


class FakeTransferEngine:
  """Captures submitted TransferOpGraphs for inspection."""

  def __init__(self):
    self.graphs = []

  def submit(self, graph):
    self.graphs.append(graph)


def _resolve_blocks(tier: str, block_id: int) -> np.ndarray:
  # Simple deterministic mapping: block_id -> array of 1 block_id.
  # GPU gets block_id + 1000 to distinguish tiers in assertions.
  if tier == "gpu":
    return np.array([block_id + 1000], dtype=np.int64)
  if tier == "cpu":
    return np.array([block_id], dtype=np.int64)
  if tier == "ssd":
    return np.array([block_id + 2000], dtype=np.int64)
  return np.array([], dtype=np.int64)


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


def test_plan_to_graph_empty():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  te = FakeTransferEngine()
  ex = MigrationExecutor(eng, te.submit, _resolve_blocks)
  plan = eng.planner.build_plan(eng.tracker)  # empty plan
  assert plan.is_empty
  assert ex.plan_to_graph(plan) is None


def test_plan_to_graph_single_promotion():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  assert not plan.is_empty
  te = FakeTransferEngine()
  ex = MigrationExecutor(eng, te.submit, _resolve_blocks)
  graph = ex.plan_to_graph(plan)
  assert graph is not None
  # First hop of SSD->GPU is DISK2H (ssd->cpu).
  types = [op.transfer_type for op in graph._op_map.values()]
  assert TransferType.DISK2H in types
  # Physical block IDs resolved correctly.
  disk2h = [op for op in graph._op_map.values() if op.transfer_type == TransferType.DISK2H][0]
  assert disk2h.src_block_ids[0] == 2001  # ssd block 1 -> 2001
  assert disk2h.dst_block_ids[0] == 1     # cpu block 1


def test_submit_plan_and_completion():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  te = FakeTransferEngine()
  ex = MigrationExecutor(eng, te.submit, _resolve_blocks)
  gid = ex.submit_plan(plan)
  assert gid is not None
  assert len(te.graphs) == 1
  # Completion feedback should update the tracker and free in-flight slot.
  ex.handle_completion(gid)
  assert eng.tracker.get("ssd", 1) is None
  assert eng.tracker.get("cpu", 1) is not None  # first hop lands on CPU


def test_completion_chains_on_completion_hook():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  te = FakeTransferEngine()
  chained = []
  ex = MigrationExecutor(
    eng, te.submit, _resolve_blocks,
    on_completion=lambda gid: chained.append(gid),
  )
  gid = ex.submit_plan(plan)
  ex.handle_completion(gid)
  assert chained == [gid]


def test_resolution_failure_skips_op():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  te = FakeTransferEngine()

  def bad_resolve(tier, block_id):
    raise ValueError("no such block")

  ex = MigrationExecutor(eng, te.submit, bad_resolve)
  graph = ex.plan_to_graph(plan)
  # Resolution failure -> None (treated as no-op, not a crash).
  assert graph is None


def test_two_hop_marks_first_hop_correctly():
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  te = FakeTransferEngine()
  ex = MigrationExecutor(eng, te.submit, _resolve_blocks)
  gid = ex.submit_plan(plan)
  ex.handle_completion(gid)
  # After first hop, block should be on CPU (staging), not GPU.
  assert eng.tracker.get("ssd", 1) is None
  assert eng.tracker.get("cpu", 1) is not None




def test_two_hop_second_leg_submitted_automatically():
  """SSD->GPU: first hop DISK2H completes -> H2D second hop auto-submitted."""
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 1)
    clock.advance(0.1)
  plan = eng.tick()
  te = FakeTransferEngine()
  ex = MigrationExecutor(eng, te.submit, _resolve_blocks)
  gid1 = ex.submit_plan(plan)
  assert len(te.graphs) == 1
  ex.handle_completion(gid1)
  # Second hop should have been submitted automatically.
  assert len(te.graphs) == 2
  second = te.graphs[1]
  types = [op.transfer_type for op in second._op_map.values()]
  assert TransferType.H2D in types
  h2d = [op for op in second._op_map.values() if op.transfer_type == TransferType.H2D][0]
  assert h2d.src_block_ids[0] == 1      # cpu staging block
  assert h2d.dst_block_ids[0] == 1001   # gpu block


def test_two_hop_second_leg_demotion_submitted_automatically():
  """GPU->SSD: first hop D2H completes -> H2DISK second hop auto-submitted."""
  clock = FakeClock()
  # Force GPU block to be cold so it demotes to SSD.
  eng = _make_engine(clock=clock, hot_threshold=100.0, cold_threshold=99.0)
  eng.touch("gpu", 7)  # score ~1.0 * weights, well below cold_threshold
  clock.advance(0.1)
  plan = eng.tick()
  te = FakeTransferEngine()
  ex = MigrationExecutor(eng, te.submit, _resolve_blocks)
  gid1 = ex.submit_plan(plan)
  assert gid1 is not None
  first_types = [op.transfer_type for op in te.graphs[0]._op_map.values()]
  assert TransferType.D2H in first_types
  ex.handle_completion(gid1)
  assert len(te.graphs) == 2
  second_types = [op.transfer_type for op in te.graphs[1]._op_map.values()]
  assert TransferType.H2DISK in second_types
  h2disk = [op for op in te.graphs[1]._op_map.values()
            if op.transfer_type == TransferType.H2DISK][0]
  assert h2disk.src_block_ids[0] == 7      # cpu staging block
  assert h2disk.dst_block_ids[0] == 2007   # ssd block


def test_two_hop_completion_of_second_leg_frees_engine_state():
  """Completing both hops leaves the tracker at the final tier."""
  clock = FakeClock()
  eng = _make_engine(clock=clock)
  for _ in range(10):
    eng.touch("ssd", 3)
    clock.advance(0.1)
  plan = eng.tick()
  te = FakeTransferEngine()
  ex = MigrationExecutor(eng, te.submit, _resolve_blocks)
  gid1 = ex.submit_plan(plan)
  ex.handle_completion(gid1)
  gid2 = te.graphs[1].graph_id
  ex.handle_completion(gid2)
  # Engine tracker should now show the block on GPU (final tier).
  assert eng.tracker.get("ssd", 3) is None
  assert eng.tracker.get("cpu", 3) is None
  assert eng.tracker.get("gpu", 3) is not None


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

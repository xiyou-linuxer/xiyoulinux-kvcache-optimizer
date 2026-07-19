"""Tests for MigrationRuntimeBridge."""
import sys
import traceback
import numpy as np

from miniflex.common.transfer import CompletedOp
from miniflex.migration import (
  MigrationEngine,
  MigrationEngineConfig,
  MigrationExecutor,
  MigrationRuntimeBridge,
)


class FakeClock:
  def __init__(self, start=1000.0):
    self.now = start

  def time(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


class FakeTransferEngine:
  def __init__(self):
    self.graphs = []
    self.completed = []

  def submit_transfer_graph(self, graph):
    self.graphs.append(graph)

  def get_completed_graphs_and_ops(self, timeout=None):
    out = list(self.completed)
    self.completed.clear()
    return out


def _resolve_blocks(tier: str, block_id: int) -> np.ndarray:
  if tier == 'gpu':
    return np.array([block_id + 1000], dtype=np.int64)
  if tier == 'cpu':
    return np.array([block_id], dtype=np.int64)
  if tier == 'ssd':
    return np.array([block_id + 2000], dtype=np.int64)
  return np.array([], dtype=np.int64)


def _make_engine(clock):
  cfg = MigrationEngineConfig(decay=0.9, hot_threshold=3.0, cold_threshold=0.5)
  return MigrationEngine(config=cfg, time_func=clock.time)


def test_drive_once_submits_graph():
  clk = FakeClock()
  eng = _make_engine(clk)
  for _ in range(10):
    eng.touch('ssd', 1)
    clk.advance(0.1)
  host = FakeTransferEngine()
  ex = MigrationExecutor(eng, host.submit_transfer_graph, _resolve_blocks)
  bridge = MigrationRuntimeBridge(eng, ex, host.get_completed_graphs_and_ops)
  gid = bridge.drive_once()
  assert gid is not None
  assert len(host.graphs) == 1
  assert bridge.stats.submitted_graphs == 1


def test_poll_completions_handles_graph_feedback():
  clk = FakeClock()
  eng = _make_engine(clk)
  for _ in range(10):
    eng.touch('ssd', 1)
    clk.advance(0.1)
  host = FakeTransferEngine()
  ex = MigrationExecutor(eng, host.submit_transfer_graph, _resolve_blocks)
  bridge = MigrationRuntimeBridge(eng, ex, host.get_completed_graphs_and_ops)
  gid = bridge.drive_once()
  host.completed.append(CompletedOp.completed_graph(gid))
  handled = bridge.poll_completions()
  assert handled == 1
  assert bridge.stats.completed_graphs == 1
  assert gid in bridge.stats.handled_graph_ids
  assert eng.tracker.get('cpu', 1) is not None


def test_poll_completions_ignores_non_graph_ops():
  clk = FakeClock()
  eng = _make_engine(clk)
  host = FakeTransferEngine()
  ex = MigrationExecutor(eng, host.submit_transfer_graph, _resolve_blocks)
  bridge = MigrationRuntimeBridge(eng, ex, host.get_completed_graphs_and_ops)
  host.completed.append(CompletedOp(graph_id=1, op_id=123))
  handled = bridge.poll_completions()
  assert handled == 0
  assert bridge.stats.ignored_ops == 1


def _main():
  funcs = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
  failed = 0
  for fn in funcs:
    try:
      fn()
      print(f'  PASS {fn.__name__}')
    except Exception:
      failed += 1
      print(f'  FAIL {fn.__name__}')
      traceback.print_exc()
  print(f'\n{len(funcs) - failed} passed, {failed} failed')
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(_main())

"""Tests for TransferEngineHostAdapter and build_runtime_bridge."""
import numpy as np

from miniflex.common.transfer import CompletedOp
from miniflex.migration import (
  MigrationEngine,
  MigrationEngineConfig,
  TransferEngineHostAdapter,
  build_runtime_bridge,
)


class FakeClock:
  def __init__(self, start=1000.0):
    self.now = start

  def time(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


class FakeHost:
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


def test_build_runtime_bridge():
  clk = FakeClock()
  eng = MigrationEngine(MigrationEngineConfig(decay=0.9, hot_threshold=3.0, cold_threshold=0.5), time_func=clk.time)
  host = FakeHost()
  adapter = TransferEngineHostAdapter(host)
  bridge = build_runtime_bridge(eng, adapter, _resolve_blocks)

  for _ in range(10):
    eng.touch('ssd', 1)
    clk.advance(0.1)

  gid = bridge.drive_once()
  assert gid is not None
  assert len(host.graphs) == 1

  host.completed.append(CompletedOp.completed_graph(gid))
  handled = bridge.poll_completions()
  assert handled == 1
  assert eng.tracker.get('cpu', 1) is not None

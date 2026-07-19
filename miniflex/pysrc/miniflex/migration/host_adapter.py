"""Host adapter for wiring MigrationRuntimeBridge to a real TransferEngine.

This is a tiny, typed wrapper around the existing TransferEngine interface used
by MiniFlex.  It exists for two reasons:

1. make the runtime bridge easier to plug into the current codebase without
   threading raw callables everywhere;
2. keep the integration testable by depending only on the small subset of the
   host API we actually need.

Expected host methods:
- submit_transfer_graph(graph)
- get_completed_graphs_and_ops(timeout=None)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from miniflex.common.transfer import CompletedOp, TransferOpGraph
from miniflex.migration.engine import MigrationEngine
from miniflex.migration.executor import MigrationExecutor
from miniflex.migration.runtime_bridge import MigrationRuntimeBridge


@dataclass
class TransferEngineHostAdapter:
  """Minimal adapter exposing the subset of TransferEngine used by migration."""
  host: object

  def submit_graph(self, graph: TransferOpGraph) -> None:
    self.host.submit_transfer_graph(graph)

  def poll_completed(self, timeout: Optional[float] = None) -> Iterable[CompletedOp]:
    return self.host.get_completed_graphs_and_ops(timeout)


def build_runtime_bridge(
  migration_engine: MigrationEngine,
  host_adapter: TransferEngineHostAdapter,
  resolve_blocks,
) -> MigrationRuntimeBridge:
  """Construct a ready-to-use runtime bridge on top of a TransferEngine host."""
  executor = MigrationExecutor(
    migration_engine,
    host_adapter.submit_graph,
    resolve_blocks,
  )
  return MigrationRuntimeBridge(
    migration_engine,
    executor,
    host_adapter.poll_completed,
  )

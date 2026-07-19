"""Runtime bridge wiring MigrationEngine/Executor to a TransferEngine loop.

This module provides a tiny integration layer that turns the existing control
plane pieces into a runnable execution loop:

  MigrationEngine.tick() -> MigrationPlan
      -> MigrationExecutor.submit_plan(plan) -> graph_id
      -> host polls TransferEngine.get_completed_graphs_and_ops()
      -> completed graph_ids are routed back into
         MigrationExecutor.handle_completion(graph_id)

The bridge intentionally stays pure-Python and only depends on a narrow host
interface:
- ``submit_transfer_graph(graph)``
- ``get_completed_graphs_and_ops(timeout=None)``

This keeps it unit-testable without the real C++/CUDA transfer engine while
making the integration path explicit and reusable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Set

from miniflex.common.transfer import CompletedOp
from miniflex.migration.engine import MigrationEngine
from miniflex.migration.executor import MigrationExecutor


@dataclass
class RuntimeBridgeStats:
  submitted_graphs: int = 0
  completed_graphs: int = 0
  ignored_ops: int = 0
  last_submitted_graph_id: int = -1
  handled_graph_ids: Set[int] = field(default_factory=set)


class MigrationRuntimeBridge:
  """Lightweight driver that closes the execution loop.

  Usage pattern:
    1. call ``drive_once()`` periodically; it runs one migration round and, if a
       plan exists, submits it through the executor.
    2. call ``poll_completions()`` to consume completed ops/graphs from the
       transfer engine and feed completed graph ids back into the executor.

  The bridge only completes graph-level feedback. Non-graph CompletedOp items
  (individual op completions) are ignored here because the executor/engine use
  graph completion to free in-flight state.
  """

  def __init__(
    self,
    migration_engine: MigrationEngine,
    migration_executor: MigrationExecutor,
    get_completed_graphs_and_ops: Callable[[Optional[float]], Iterable[CompletedOp]],
  ):
    self.migration_engine = migration_engine
    self.migration_executor = migration_executor
    self.get_completed_graphs_and_ops = get_completed_graphs_and_ops
    self.stats = RuntimeBridgeStats()

  def drive_once(self) -> Optional[int]:
    """Run one migration round and submit a graph if there is useful work."""
    plan = self.migration_engine.tick()
    graph_id = self.migration_executor.submit_plan(plan)
    if graph_id is not None:
      self.stats.submitted_graphs += 1
      self.stats.last_submitted_graph_id = graph_id
    return graph_id

  def poll_completions(self, timeout: Optional[float] = None) -> int:
    """Consume completed graphs from the host transfer engine.

    Returns the number of completed graphs handled in this poll.
    """
    handled = 0
    completed_items = self.get_completed_graphs_and_ops(timeout)
    for item in completed_items:
      if not isinstance(item, CompletedOp):
        continue
      if not item.is_graph_completed():
        self.stats.ignored_ops += 1
        continue
      self.migration_executor.handle_completion(item.graph_id)
      self.stats.completed_graphs += 1
      self.stats.handled_graph_ids.add(item.graph_id)
      handled += 1
    return handled

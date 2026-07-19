"""Bridge between MigrationEngine and the real TransferEngine.

This module translates ``MigrationPlan`` objects (which operate on logical
block IDs and tier names) into concrete ``TransferOpGraph`` objects that the
existing ``TransferEngine`` can schedule and execute.

Design notes:
  - We deliberately do NOT import ``TransferEngine`` directly; the host passes a
    callable ``submit_graph`` and a callable ``resolve_blocks`` so this module
    stays pure-Python and unit-testable without the C++/CUDA extensions.
  - ``resolve_blocks`` maps a logical ``(tier, block_id)`` to the physical
    ``np.ndarray`` of block IDs the transfer engine expects.  This indirection
    keeps the migration engine decoupled from the storage allocator.
  - Two-hop SSD<->GPU moves are chained: we emit the first hop, and when it
    completes the host schedules the second hop via ``on_first_hop_done``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from miniflex.common.transfer import TransferOp, TransferOpGraph, TransferType
from miniflex.migration.engine import MigrationEngine
from miniflex.migration.planner import MigrationOp, MigrationPlan



@dataclass
class ResolvedMigrationOp:
  """A MigrationOp with its physical block IDs resolved."""
  migration_op: MigrationOp
  src_block_ids: np.ndarray
  dst_block_ids: np.ndarray


class MigrationExecutor:
  """Submit MigrationPlans to a TransferEngine and feed completions back.

  Args:
    engine: the :class:`MigrationEngine` producing plans.
    submit_graph: callable that accepts a ``TransferOpGraph`` and hands it to
      the real ``TransferEngine`` (usually ``transfer_engine.submit_transfer_graph``).
    resolve_blocks: callable ``(tier: str, block_id: int) -> np.ndarray`` that
      returns the physical block ID array for a logical block.  The host owns
      the mapping from logical to physical.
    on_completion: optional callable ``(graph_id: int) -> None`` invoked when a
      migration graph finishes (used for chaining two-hop moves).
  """

  def __init__(
    self,
    engine: MigrationEngine,
    submit_graph: Callable[[TransferOpGraph], None],
    resolve_blocks: Callable[[str, int], np.ndarray],
    on_completion: Optional[Callable[[int], None]] = None,
  ):
    self.engine = engine
    self.submit_graph = submit_graph
    self.resolve_blocks = resolve_blocks
    self.on_completion = on_completion
    # graph_id -> list of (src_tier, block_id) for completion feedback.
    self._graph_ops: Dict[int, List[Tuple[str, int]]] = {}
    self._lock = threading.Lock()

  # -- plan -> graph ---------------------------------------------------------
  def plan_to_graph(self, plan: MigrationPlan) -> Optional[TransferOpGraph]:
    """Convert a MigrationPlan into a TransferOpGraph.

    Returns ``None`` if the plan is empty or if any op fails to resolve its
    physical block IDs (treated as a no-op, not an error).
    """
    if plan.is_empty:
      return None
    graph = TransferOpGraph()
    resolved: List[ResolvedMigrationOp] = []
    for mop in plan.ops:
      # For two-hop moves (SSD<->GPU via CPU), the first hop's dst_tier is the
      # CPU staging tier, not the final GPU/SSD tier.  Resolve dst blocks
      # against the staging tier so the DISK2H / D2H op gets CPU block IDs.
      effective_dst_tier = "cpu" if mop.needs_second_hop else mop.dst_tier
      try:
        src_ids = self.resolve_blocks(mop.src_tier, mop.block_id)
        dst_ids = self.resolve_blocks(effective_dst_tier, mop.block_id)
      except Exception:
        # Resolution failure -> skip this op (defensive: never break the loop).
        continue
      if src_ids.size == 0 or dst_ids.size == 0:
        continue
      resolved.append(ResolvedMigrationOp(mop, src_ids, dst_ids))
    if not resolved:
      return None

    # First pass: add all first-hop ops.
    first_hop_ops: List[TransferOp] = []
    for rmop in resolved:
      ttype = TransferType[rmop.migration_op.transfer_type]
      op = TransferOp(
        transfer_type=ttype,
        graph_id=graph.graph_id,
        src_block_ids=rmop.src_block_ids,
        dst_block_ids=rmop.dst_block_ids,
      )
      graph.add_transfer_op(op)
      first_hop_ops.append(op)
    return graph

  # -- submission ------------------------------------------------------------
  def submit_plan(self, plan: MigrationPlan) -> Optional[int]:
    """Convert + submit a plan; returns the graph_id or None."""
    graph = self.plan_to_graph(plan)
    if graph is None:
      return None
    # Remember which (tier, block) pairs this graph carries for completion.
    with self._lock:
      self._graph_ops[graph.graph_id] = [
        (op.src_tier, op.block_id) for op in plan.ops
      ]
    self.submit_graph(graph)
    return graph.graph_id

  # -- completion feedback ---------------------------------------------------
  def handle_completion(self, graph_id: int) -> None:
    """Notify the engine that a migration graph finished.

    Calls ``MigrationEngine.mark_completed`` for each op in the graph so the
    heat tracker is updated and the in-flight slot is freed.  Then invokes the
    optional ``on_completion`` hook (used to chain two-hop SSD<->GPU moves).
    """
    with self._lock:
      ops = self._graph_ops.pop(graph_id, [])
    for src_tier, block_id in ops:
      self.engine.mark_completed(src_tier, block_id)
    if self.on_completion is not None:
      self.on_completion(graph_id)

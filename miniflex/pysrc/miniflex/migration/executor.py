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
  - Two-hop SSD<->GPU moves are chained *automatically*: when the first-hop
    graph completes, ``handle_completion`` resolves the staging-tier block and
    submits the second hop (SSD->CPU->GPU via H2D, GPU->CPU->SSD via H2DISK)
    through the same ``submit_graph`` callable.  Hosts that need a hook after
    each hop can still pass ``on_completion``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from miniflex.common.transfer import TransferOp, TransferOpGraph, TransferType
from miniflex.migration.engine import MigrationEngine, _InflightOp
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
    # graph_id -> list of (original_src_tier, block_id, final_dst_tier) for
    # ops that were scheduled as two-hop moves and whose second hop has not
    # been submitted yet.  Keyed by the *first-hop* graph id.
    self._pending_second_hops: Dict[int, List[Tuple[str, int, str]]] = {}
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
    # Two-hop ops additionally register a pending second hop so completion can
    # chain the CPU<->GPU/SSD leg without host involvement.
    with self._lock:
      self._graph_ops[graph.graph_id] = [
        (op.src_tier, op.block_id) for op in plan.ops
      ]
      pending = [
        (op.src_tier, op.block_id, op.dst_tier)
        for op in plan.ops
        if op.needs_second_hop
      ]
      if pending:
        self._pending_second_hops[graph.graph_id] = pending
    self.submit_graph(graph)
    return graph.graph_id

  # -- second-hop construction -------------------------------------------------
  # After the first hop of a two-hop move lands on CPU staging, the second hop
  # is a plain CPU<->GPU/SSD transfer.  These are fixed per original direction.
  _SECOND_HOP_TRANSFER_TYPE: Dict[Tuple[str, str], str] = {
    ("ssd", "gpu"): "H2D",       # ssd -> cpu (done) -> gpu
    ("gpu", "ssd"): "H2DISK",    # gpu -> cpu (done) -> ssd
  }

  def _submit_second_hop(
    self,
    original_src_tier: str,
    block_id: int,
    final_dst_tier: str,
  ) -> Optional[int]:
    """Resolve + submit the second hop for a two-hop move.

    Returns the new graph_id, or ``None`` if the staging block cannot be
    resolved (the engine has already been told the first hop completed, so the
    block will sit on CPU until the policy re-decides).
    """
    ttype_name = self._SECOND_HOP_TRANSFER_TYPE[(original_src_tier, final_dst_tier)]
    try:
      src_ids = self.resolve_blocks("cpu", block_id)
      dst_ids = self.resolve_blocks(final_dst_tier, block_id)
    except Exception:
      return None
    if src_ids.size == 0 or dst_ids.size == 0:
      return None

    graph = TransferOpGraph()
    op = TransferOp(
      transfer_type=TransferType[ttype_name],
      graph_id=graph.graph_id,
      src_block_ids=src_ids,
      dst_block_ids=dst_ids,
    )
    graph.add_transfer_op(op)
    with self._lock:
      self._graph_ops[graph.graph_id] = [("cpu", block_id)]
    # Re-register the in-flight slot on the engine so the tracker's tier
    # follows the block across the second hop (cpu -> final_dst_tier) and the
    # slot is freed when the second-hop graph completes.
    self.engine._inflight[("cpu", block_id)] = _InflightOp(
      src_tier="cpu",
      dst_tier=final_dst_tier,
      block_id=block_id,
      scheduled_at=self.engine.time_func(),
    )
    self.submit_graph(graph)
    return graph.graph_id

  # -- completion feedback ---------------------------------------------------
  def handle_completion(self, graph_id: int) -> None:
    """Notify the engine that a migration graph finished.

    Calls ``MigrationEngine.mark_completed`` for each op in the graph so the
    heat tracker is updated and the in-flight slot is freed.  If any op in the
    graph was the first hop of a two-hop SSD<->GPU move, automatically submits
    the second hop.  Then invokes the optional ``on_completion`` hook.
    """
    with self._lock:
      ops = self._graph_ops.pop(graph_id, [])
      pending = self._pending_second_hops.pop(graph_id, [])
    for src_tier, block_id in ops:
      self.engine.mark_completed(src_tier, block_id)
    # Chain second hops for any two-hop ops that just finished their first leg.
    for original_src_tier, block_id, final_dst_tier in pending:
      self._submit_second_hop(original_src_tier, block_id, final_dst_tier)
    if self.on_completion is not None:
      self.on_completion(graph_id)

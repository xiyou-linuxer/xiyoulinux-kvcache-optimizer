"""Unified heat-aware tiered migration engine.

This ties the heat tracker, tiering policy, migration planner and prefetch
planner into a single closed loop that can run periodically (or on demand) to
keep KV cache blocks balanced across GPU / CPU / SSD tiers.

Design goals:
  - One entry point: ``MigrationEngine.tick()`` runs one full round.
  - Bandwidth-safe: promotions (online-critical) are scheduled before demotions
    (background), and both respect the policy's per-round caps.
  - Deduplication: a block already scheduled for migration in an unfinished round
    is not scheduled again until it completes (prevents double-booking).
  - Prefetch integration: ``request_prefetch()`` lets the scheduler ask for
    lookahead prefetch on an incoming request before it actually needs the KV.
  - Observability: every round updates a ``MigrationMetrics`` instance.

The engine is intentionally decoupled from the real transfer engine: it emits
``MigrationPlan`` objects whose ops align with MiniFlex's existing
``TransferType`` enum.  The host engine is responsible for submitting those ops
to the real ``TransferOpGraph`` and calling ``mark_completed()`` when done.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from miniflex.migration.heat import HeatTracker
from miniflex.migration.metrics import MigrationMetrics, Stopwatch
from miniflex.migration.planner import MigrationPlan, MigrationPlanner
from miniflex.migration.policy import MigrationPolicy
from miniflex.migration.prefetch import PrefetchDecision, PrefetchPlanner


@dataclass
class MigrationEngineConfig:
  decay: float = 0.95
  hot_threshold: float = 2.0
  cold_threshold: float = 0.5
  max_promotions_per_round: int = 64
  max_demotions_per_round: int = 64
  max_prefetch_ratio: float = 1.0
  # Global in-flight bandwidth cap (max blocks scheduled but not completed).
  max_inflight_blocks: int = 256


@dataclass
class _InflightOp:
  src_tier: str
  dst_tier: str
  block_id: int
  scheduled_at: float


class MigrationEngine:
  """Closed-loop heat-aware migration + prefetch coordinator."""

  def __init__(
    self,
    config: Optional[MigrationEngineConfig] = None,
    time_func=None,
  ):
    self.config = config or MigrationEngineConfig()
    self.time_func = time_func or time.time
    self.tracker = HeatTracker(decay=self.config.decay, time_func=self.time_func)
    self.policy = MigrationPolicy(
      hot_threshold=self.config.hot_threshold,
      cold_threshold=self.config.cold_threshold,
      max_promotions_per_round=self.config.max_promotions_per_round,
      max_demotions_per_round=self.config.max_demotions_per_round,
    )
    self.planner = MigrationPlanner(self.policy)
    self.prefetch_planner = PrefetchPlanner(max_prefetch_ratio=self.config.max_prefetch_ratio)
    self.metrics = MigrationMetrics()
    # Blocks currently scheduled but not yet completed (src_tier, block_id).
    self._inflight: Dict[Tuple[str, int], _InflightOp] = {}
    self._last_plan: Optional[MigrationPlan] = None

  # -- access passthrough --------------------------------------------------
  def touch(self, tier: str, block_id: int):
    """Record an access (called by the host engine on cache hit/miss)."""
    return self.tracker.touch(tier, block_id)

  def block_heat(self, tier: str, block_id: int) -> Optional[float]:
    bh = self.tracker.get(tier, block_id)
    return bh.score if bh else None

  @property
  def last_plan(self) -> Optional[MigrationPlan]:
    return self._last_plan

  # -- core loop -----------------------------------------------------------
  def tick(self) -> MigrationPlan:
    """Run one migration round: decay -> decide -> plan -> metrics."""
    with Stopwatch(self.metrics.record_latency):
      self.tracker.decay_all()
      self._prune_completed()
      plan = self.planner.build_plan(self.tracker)
      plan = self._filter_inflight(plan)
      self._register_inflight(plan)
      self.metrics.update_distribution(self.tracker.tier_distribution())
      noops = max(0, len(self.tracker) - len(plan.ops))
      self.metrics.record_noop(noops)
      self.metrics.record_promotion(plan.num_promotions)
      self.metrics.record_demotion(plan.num_demotions)
    self._last_plan = plan
    return plan

  def request_prefetch(
    self,
    request_id: int,
    matched: Dict[str, int],
    gpu_blocks_available: int,
    bandwidth_budget: Optional[int] = None,
  ) -> PrefetchDecision:
    """Ask for lookahead prefetch for an incoming request.

    Prefetch competes with background migration for bandwidth; we deduct
    in-flight blocks from the budget so prefetch never overloads the pipe.
    """
    if bandwidth_budget is None:
      bandwidth_budget = max(0, self.config.max_inflight_blocks - len(self._inflight))
    decision = self.prefetch_planner.plan(
      request_id=request_id,
      matched=matched,
      gpu_blocks_available=gpu_blocks_available,
      bandwidth_budget=bandwidth_budget,
    )
    self.metrics.record_prefetch(decision.total_blocks, decision.estimated_cost)
    return decision

  # -- completion tracking -------------------------------------------------
  def mark_completed(self, src_tier: str, block_id: int) -> None:
    """Notify the engine that a scheduled migration op has finished.

    The host engine should call this after the corresponding TransferOpGraph
    completes; it updates the heat tracker to reflect the new tier and frees the
    in-flight slot.
    """
    key = (src_tier, block_id)
    op = self._inflight.pop(key, None)
    if op is None:
      return
    self.tracker.move(op.src_tier, op.dst_tier, block_id)

  # -- internals -----------------------------------------------------------
  def _prune_completed(self) -> None:
    # Safety net: drop ops that have been in-flight too long (> 30s) to avoid
    # permanent bandwidth leaks if the host forgets to call mark_completed.
    now = self.time_func()
    stale = [k for k, op in self._inflight.items() if now - op.scheduled_at > 30.0]
    for k in stale:
      self._inflight.pop(k, None)

  def _filter_inflight(self, plan: MigrationPlan) -> MigrationPlan:
    """Drop ops for blocks already in flight to avoid double-booking."""
    if not plan.ops:
      return plan
    kept = [
      op for op in plan.ops
      if (op.src_tier, op.block_id) not in self._inflight
    ]
    return MigrationPlan(ops=kept)

  def _register_inflight(self, plan: MigrationPlan) -> None:
    now = self.time_func()
    for op in plan.ops:
      if len(self._inflight) >= self.config.max_inflight_blocks:
        break
      key = (op.src_tier, op.block_id)
      if key in self._inflight:
        continue
      # For two-hop moves (SSD<->GPU via CPU), the first hop lands on the CPU
      # staging tier, so the in-flight destination is CPU, not the final
      # GPU/SSD tier.  The engine chains the second hop after completion.
      dst = "cpu" if op.needs_second_hop else op.dst_tier
      self._inflight[key] = _InflightOp(
        src_tier=op.src_tier,
        dst_tier=dst,
        block_id=op.block_id,
        scheduled_at=now,
      )

  # -- reporting -----------------------------------------------------------
  def stats(self) -> Dict:
    snap = self.metrics.snapshot()
    snap["inflight_blocks"] = len(self._inflight)
    snap["tracked_blocks"] = len(self.tracker)
    return snap

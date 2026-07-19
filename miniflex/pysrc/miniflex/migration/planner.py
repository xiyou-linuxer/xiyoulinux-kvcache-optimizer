"""Turn heat-driven tiering decisions into concrete migration operations.

The planner produces :class:`MigrationPlan` objects whose ops align with
MiniFlex's existing ``TransferType`` enum, so the plans can be handed straight to
a ``TransferOpGraph`` once block IDs are mapped to physical slots.  This keeps the
migration logic decoupled from the actual transfer engine while staying wire-
compatible.

Each :class:`MigrationOp` carries:
  - ``src_tier`` / ``dst_tier``: where the block lives now vs. where it should go
  - ``block_id``: the logical block identifier
  - ``transfer_type``: the matching MiniFlex ``TransferType`` string

We deliberately operate on *logical* block IDs here; mapping them to physical
slot IDs is the responsibility of the engine that consumes the plan (it has the
mempool / radix-tree state).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from miniflex.migration.heat import HeatTracker
from miniflex.migration.policy import MigrationPolicy, Tier


# Map a (src_tier, dst_tier) pair to the corresponding MiniFlex TransferType.
# Note: GPU<->SSD always stages through CPU in the current implementation, so a
# GPU<->SSD move is expressed as the leg that the planner can actually schedule.
_TRANSFER_TYPE_MAP: Dict[Tuple[str, str], str] = {
  ("cpu", "gpu"): "H2D",
  ("gpu", "cpu"): "D2H",
  ("ssd", "cpu"): "DISK2H",
  ("cpu", "ssd"): "H2DISK",
  # GPU <-> SSD is two-hop; surface the first hop plus a deferred second hop.
  ("ssd", "gpu"): "DISK2H",   # first bring to CPU; H2D is scheduled after
  ("gpu", "ssd"): "D2H",      # first bring to CPU; H2DISK is scheduled after
}


@dataclass
class MigrationOp:
  src_tier: str
  dst_tier: str
  block_id: int
  transfer_type: str
  # True when this move requires a follow-up second hop (e.g. SSD->GPU via CPU).
  needs_second_hop: bool = False

  def __post_init__(self):
    if (self.src_tier, self.dst_tier) not in _TRANSFER_TYPE_MAP:
      raise ValueError(f"unsupported migration leg: {self.src_tier}->{self.dst_tier}")


@dataclass
class MigrationPlan:
  ops: List[MigrationOp] = field(default_factory=list)

  @property
  def num_promotions(self) -> int:
    return sum(1 for op in self.ops if op.transfer_type in ("H2D", "DISK2H"))

  @property
  def num_demotions(self) -> int:
    return sum(1 for op in self.ops if op.transfer_type in ("D2H", "H2DISK"))

  @property
  def is_empty(self) -> bool:
    return len(self.ops) == 0

  def transfer_types(self) -> List[str]:
    return [op.transfer_type for op in self.ops]


class MigrationPlanner:
  """Compose a :class:`HeatTracker` and :class:`MigrationPolicy` into plans."""

  def __init__(self, policy: Optional[MigrationPolicy] = None):
    self.policy = policy or MigrationPolicy()

  def build_plan(self, tracker: HeatTracker) -> MigrationPlan:
    decisions = self.policy.decide(tracker)
    ops: List[MigrationOp] = []
    for (src_tier, block_id), target in decisions.items():
      dst_tier = target.value
      two_hop = {("ssd", "gpu"), ("gpu", "ssd")}
      # For two-hop moves we only emit the first hop here; the engine chains the
      # second hop once the staging tier acknowledges completion.
      ops.append(MigrationOp(
        src_tier=src_tier,
        dst_tier=dst_tier,
        block_id=block_id,
        transfer_type=_TRANSFER_TYPE_MAP[(src_tier, dst_tier)],
        needs_second_hop=(src_tier, dst_tier) in two_hop,
      ))
    return MigrationPlan(ops=ops)

  def plan_and_apply(self, tracker: HeatTracker) -> MigrationPlan:
    """Build a plan and update the tracker to reflect intended moves.

    This is a convenience for simulation/testing: in production the engine would
    update the tracker only after each transfer actually completes.
    """
    plan = self.build_plan(tracker)
    for op in plan.ops:
      # For two-hop, record the intermediate staging tier for now.
      if op.needs_second_hop:
        staging = "cpu"
        tracker.move(op.src_tier, staging, op.block_id)
      else:
        tracker.move(op.src_tier, op.dst_tier, op.block_id)
    return plan

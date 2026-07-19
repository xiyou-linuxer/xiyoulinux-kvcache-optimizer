"""Heat-driven tiering policy.

Decides, for each block, whether it should live on GPU (hot), CPU (warm) or SSD
(cold).  The policy is intentionally simple and interpretable: thresholds on the
heat score produced by :class:`HeatTracker`.

Tuning knobs:
  - ``hot_threshold``: score at or above which a block belongs on GPU.
  - ``cold_threshold``: score at or below which a block belongs on SSD.
  - Scores between the two belong on CPU (warm tier).

A block that is already on its correct tier is a no-op; only blocks that should
move produce a migration decision.  We also cap the number of *promotions* and
*demotions* per planning round to avoid bandwidth spikes (QoS).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from miniflex.migration.heat import BlockHeat, HeatTracker, TIER_ORDER


class Tier(Enum):
  HOT = "gpu"
  WARM = "cpu"
  COLD = "ssd"


TIER_FROM_STR = {"gpu": Tier.HOT, "cpu": Tier.WARM, "ssd": Tier.COLD}


@dataclass
class MigrationPolicy:
  hot_threshold: float = 2.0
  cold_threshold: float = 0.5
  max_promotions_per_round: int = 64   # bandwidth throttle: promote at most N
  max_demotions_per_round: int = 64    # bandwidth throttle: demote at most N

  def __post_init__(self):
    if self.cold_threshold > self.hot_threshold:
      raise ValueError("cold_threshold must be <= hot_threshold")
    if self.max_promotions_per_round < 0 or self.max_demotions_per_round < 0:
      raise ValueError("bandwidth caps must be non-negative")

  def target_tier(self, bh: BlockHeat) -> Tier:
    if bh.score >= self.hot_threshold:
      return Tier.HOT
    if bh.score <= self.cold_threshold:
      return Tier.COLD
    return Tier.WARM

  def should_move(self, bh: BlockHeat) -> Optional[Tier]:
    """Return the target tier if the block should move, else None."""
    current = TIER_FROM_STR[bh.tier]
    target = self.target_tier(bh)
    if target == current:
      return None
    return target

  def decide(self, tracker: HeatTracker) -> Dict[Tuple[str, int], Tier]:
    """Compute per-block migration decisions for one planning round.

    Applies bandwidth throttling: promotions (moving hotter, i.e. towards GPU)
    and demotions (moving colder, i.e. towards SSD) are each capped.
    Returns ``{(tier, block_id): target_tier}`` for blocks that should move.
    """
    promotions: List[Tuple[BlockHeat, Tier]] = []
    demotions: List[Tuple[BlockHeat, Tier]] = []

    for bh in tracker.all_blocks():
      target = self.should_move(bh)
      if target is None:
        continue
      # A move towards GPU (lower order index) is a promotion.
      if TIER_ORDER[target.value] < TIER_ORDER[bh.tier]:
        promotions.append((bh, target))
      else:
        demotions.append((bh, target))

    # Hottest promotions first; coldest demotions first.
    promotions.sort(key=lambda x: x[0].score, reverse=True)
    demotions.sort(key=lambda x: (x[0].score, x[0].last_access))

    decisions: Dict[Tuple[str, int], Tier] = {}
    for bh, target in promotions[: self.max_promotions_per_round]:
      decisions[(bh.tier, bh.block_id)] = target
    for bh, target in demotions[: self.max_demotions_per_round]:
      decisions[(bh.tier, bh.block_id)] = target
    return decisions

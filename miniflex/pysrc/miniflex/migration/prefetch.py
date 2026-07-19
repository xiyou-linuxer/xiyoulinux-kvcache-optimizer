"""Lookahead prefetch decision logic.

Architecture/system-side optimization: instead of waiting until a request
actually needs a KV block to start refilling it (which makes the GPU stall on
H2D / DISK2H), prefetch decides *ahead of time* which blocks should be pulled
closer so the refill latency is hidden behind the previous request's compute.

Inputs are intentionally abstract so the planner is unit-testable without the
real engines:
  - ``request_id``
  - ``matched``: dict mapping tier -> number of blocks matched on that tier
  - ``gpu_blocks_available``: how many GPU slots the request can use
  - ``bandwidth_budget``: max bytes we are willing to prefetch for this request

The planner returns a :class:`PrefetchDecision` describing which tiers to pull
from and how many blocks, prioritizing the cheapest (closest) matches first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Relative cost (higher = farther from GPU). Used to prioritize prefetch source.
TIER_COST = {"cpu": 1, "ssd": 10}


@dataclass
class PrefetchDecision:
  request_id: int
  # Ordered list of (tier, num_blocks) to prefetch, cheapest first.
  plan: List = field(default_factory=list)
  # Total estimated cost (sum of tier_cost * num_blocks).
  estimated_cost: float = 0.0
  # Whether the plan fully satisfies the request's matched blocks.
  is_complete: bool = True

  @property
  def total_blocks(self) -> int:
    return sum(n for _, n in self.plan)

  @property
  def is_empty(self) -> bool:
    return len(self.plan) == 0


class PrefetchPlanner:
  """Decide which matched blocks to prefetch for an incoming request.

  Args:
    max_prefetch_ratio: max fraction of ``gpu_blocks_available`` to prefetch.
    prefer_near: when True, pull cheaper (CPU) tiers before SSD.
  """

  def __init__(self, max_prefetch_ratio: float = 1.0, prefer_near: bool = True):
    if not 0.0 < max_prefetch_ratio <= 1.0:
      raise ValueError(f"max_prefetch_ratio must be in (0, 1], got {max_prefetch_ratio}")
    self.max_prefetch_ratio = max_prefetch_ratio
    self.prefer_near = prefer_near

  def plan(
    self,
    request_id: int,
    matched: Dict[str, int],
    gpu_blocks_available: int,
    bandwidth_budget: Optional[int] = None,
  ) -> PrefetchDecision:
    """Compute a prefetch plan.

    Args:
      request_id: logical request identifier.
      matched: ``{"cpu": n_cpu, "ssd": n_ssd}`` blocks matched on each tier.
      gpu_blocks_available: number of free GPU slots this request can fill.
      bandwidth_budget: optional max total blocks to prefetch (QoS throttle).
    """
    if gpu_blocks_available < 0:
      raise ValueError("gpu_blocks_available must be non-negative")
    budget = int(gpu_blocks_available * self.max_prefetch_ratio)
    if bandwidth_budget is not None:
      budget = min(budget, int(bandwidth_budget))

    # Order tiers by cost so we pull the cheapest first.
    tiers = [t for t in ("cpu", "ssd") if matched.get(t, 0) > 0]
    if self.prefer_near:
      tiers.sort(key=lambda t: TIER_COST[t])
    else:
      tiers.sort(key=lambda t: -TIER_COST[t])

    plan = []
    remaining = budget
    estimated_cost = 0.0
    total_matched = sum(matched.get(t, 0) for t in ("cpu", "ssd"))

    for tier in tiers:
      if remaining <= 0:
        break
      n = min(matched[tier], remaining)
      if n > 0:
        plan.append((tier, n))
        estimated_cost += TIER_COST[tier] * n
        remaining -= n

    total_planned = sum(n for _, n in plan)
    return PrefetchDecision(
      request_id=request_id,
      plan=plan,
      estimated_cost=estimated_cost,
      is_complete=(total_planned >= total_matched),
    )

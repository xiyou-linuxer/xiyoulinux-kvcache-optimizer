"""Per-block KV access tracking and heat scoring (enhanced heat model).

We model heat as a combination of:
  - access frequency (how often the block is touched)
  - recency (how recently it was touched)
  - tier weight (a hit on a cold tier is more valuable to promote than a hit on
    a hot tier, because it currently costs more to serve)
  - burst detection (sudden access spikes should heat a block up faster)

The tracker is intentionally decoupled from the storage backend: it only knows
about ``(tier, block_id)`` keys and the current time, so it can be unit tested
in isolation and wired into the real engines later.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple


# Canonical tier ordering, from hottest (closest to compute) to coldest.
TIER_ORDER = {"gpu": 0, "cpu": 1, "ssd": 2}

# A hit on a colder tier is worth more heat: it currently costs more to serve,
# so promoting it yields a bigger win.  GPU hit -> 1.0x, CPU -> 2.0x, SSD -> 4.0x.
TIER_WEIGHT = {"gpu": 1.0, "cpu": 2.0, "ssd": 4.0}

# Phase-aware weighting: decode-time hits are more valuable than prefill-time
# hits because decode sits on the critical path and blocking there hurts
# throughput/latency more directly.
PHASE_WEIGHT = {"prefill": 1.0, "decode": 1.5}

# Burst detection: a "burst" is >= N accesses within a short window.
BURST_WINDOW_SECONDS = 2.0
BURST_MIN_ACCESSES = 3
BURST_BONUS = 1.5  # multiplier applied to a touch during a burst


@dataclass
class BlockHeat:
  """Heat state for a single KV block."""
  tier: str
  block_id: int
  access_count: int = 0
  last_access: float = 0.0
  # Exponentially-decayed heat score (higher = hotter).
  score: float = 0.0
  # Recent access timestamps for burst detection (bounded in size).
  recent_accesses: Deque[float] = field(default_factory=lambda: deque(maxlen=16))
  # Set when the block is currently in a burst (>=N hits in the window).
  is_bursting: bool = False
  layer_id: Optional[int] = None
  phase: Optional[str] = None

  def touch(self, now: float, decay: float, tier_weight: float = 1.0) -> None:
    self.access_count += 1
    self.last_access = now
    self.recent_accesses.append(now)
    # Decay the existing score then add a unit of heat, so old heat fades.
    # The tier weight makes cold-tier hits count for more.
    self.score = self.score * decay + tier_weight
    # Burst detection: are there >= BURST_MIN_ACCESSES within the window?
    window_start = now - BURST_WINDOW_SECONDS
    in_window = sum(1 for t in self.recent_accesses if t >= window_start)
    self.is_bursting = in_window >= BURST_MIN_ACCESSES
    if self.is_bursting:
      self.score *= BURST_BONUS

  def age(self, now: float, decay: float) -> None:
    """Apply time decay without a fresh access."""
    if self.last_access <= 0.0:
      return
    seconds = max(0.0, now - self.last_access)
    # Exponential decay: with decay=0.95, ~13.9s half-life (retains ~70.7%
    # after 10s).  decay=0.9 gives a true 10s half-life.
    effective = math.exp(-seconds * (1.0 - decay) * 0.693)
    self.score *= effective
    # Burst state decays too: if no recent hits, it's no longer bursting.
    window_start = now - BURST_WINDOW_SECONDS
    in_window = sum(1 for t in self.recent_accesses if t >= window_start)
    self.is_bursting = in_window >= BURST_MIN_ACCESSES


class HeatTracker:
  """Tracks per-block heat and reports hot/cold candidates.

  Args:
    decay: per-touch exponential decay factor in [0, 1). Lower = shorter memory.
    time_func: injectable clock for deterministic testing.
    use_tier_weight: whether cold-tier hits should heat a block up faster.
  """

  def __init__(
    self,
    decay: float = 0.95,
    time_func=None,
    use_tier_weight: bool = True,
    layer_weights: Optional[Dict[int, float]] = None,
    phase_weights: Optional[Dict[str, float]] = None,
  ):
    if not 0.0 <= decay < 1.0:
      raise ValueError(f"decay must be in [0, 1), got {decay}")
    self.decay = decay
    self.time_func = time_func or time.time
    self.use_tier_weight = use_tier_weight
    self.layer_weights = layer_weights or {}
    self.phase_weights = {**PHASE_WEIGHT, **(phase_weights or {})}
    self._blocks: Dict[Tuple[str, int], BlockHeat] = {}

  # -- basic API -----------------------------------------------------------
  def _key(self, tier: str, block_id: int) -> Tuple[str, int]:
    if tier not in TIER_ORDER:
      raise ValueError(f"unknown tier: {tier}")
    return (tier, block_id)

  def register(
    self,
    tier: str,
    block_id: int,
    layer_id: Optional[int] = None,
    phase: Optional[str] = None,
  ) -> BlockHeat:
    key = self._key(tier, block_id)
    if key not in self._blocks:
      self._blocks[key] = BlockHeat(
        tier=tier,
        block_id=block_id,
        layer_id=layer_id,
        phase=phase,
      )
    elif layer_id is not None:
      # Preserve the most specific layer information we have seen.
      self._blocks[key].layer_id = layer_id
    if phase is not None:
      self._blocks[key].phase = phase
    return self._blocks[key]

  def touch(
    self,
    tier: str,
    block_id: int,
    layer_id: Optional[int] = None,
    phase: Optional[str] = None,
  ) -> BlockHeat:
    bh = self.register(tier, block_id, layer_id, phase)
    weight = TIER_WEIGHT[tier] if self.use_tier_weight else 1.0
    if layer_id is not None:
      weight *= self.layer_weights.get(layer_id, 1.0)
    if phase is not None:
      weight *= self.phase_weights.get(phase, 1.0)
    bh.touch(self.time_func(), self.decay, tier_weight=weight)
    return bh

  def remove(self, tier: str, block_id: int) -> None:
    self._blocks.pop(self._key(tier, block_id), None)

  def move(self, src_tier: str, dst_tier: str, block_id: int) -> BlockHeat:
    """Record that a block migrated between tiers (preserving its heat)."""
    src_key = self._key(src_tier, block_id)
    bh = self._blocks.pop(src_key, None)
    if bh is None:
      bh = BlockHeat(tier=dst_tier, block_id=block_id)
    bh.tier = dst_tier
    dst_key = self._key(dst_tier, block_id)
    self._blocks[dst_key] = bh
    return bh

  def get(self, tier: str, block_id: int) -> Optional[BlockHeat]:
    return self._blocks.get(self._key(tier, block_id))

  def __len__(self) -> int:
    return len(self._blocks)

  # -- analytics -----------------------------------------------------------
  def all_blocks(self) -> Iterable[BlockHeat]:
    return self._blocks.values()

  def top_hot(self, n: int, tier: Optional[str] = None) -> List[BlockHeat]:
    """Return the ``n`` hottest blocks, optionally filtered by current tier.

    Sorted by score desc, then recency desc (most recent first) as tiebreaker,
    then burst status (bursting blocks prioritized).
    """
    candidates = [b for b in self._blocks.values() if tier is None or b.tier == tier]
    candidates.sort(key=lambda b: (b.score, b.last_access, float(b.is_bursting)), reverse=True)
    return candidates[:n]

  def top_cold(self, n: int, tier: Optional[str] = None) -> List[BlockHeat]:
    """Return the ``n`` coldest blocks (lowest score, oldest access)."""
    candidates = [b for b in self._blocks.values() if tier is None or b.tier == tier]
    candidates.sort(key=lambda b: (b.score, b.last_access))
    return candidates[:n]

  def tier_distribution(self) -> Dict[str, int]:
    """Count tracked blocks per tier."""
    dist: Dict[str, int] = {"gpu": 0, "cpu": 0, "ssd": 0}
    for bh in self._blocks.values():
      dist[bh.tier] = dist.get(bh.tier, 0) + 1
    return dist

  def decay_all(self) -> None:
    """Apply time decay to every tracked block (call periodically)."""
    now = self.time_func()
    for bh in self._blocks.values():
      bh.age(now, self.decay)

  def snapshot(self) -> Dict[Tuple[str, int], float]:
    return {k: bh.score for k, bh in self._blocks.items()}

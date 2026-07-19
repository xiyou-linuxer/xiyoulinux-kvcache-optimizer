"""Per-block KV access tracking and heat scoring.

We model heat as a combination of:
  - access frequency (how often the block is touched)
  - recency (how recently it was touched)
  - current tier (blocks already on a cold tier get a small penalty so we prefer
    to promote hot blocks that are far from GPU)

The tracker is intentionally decoupled from the storage backend: it only knows
about ``(tier, block_id)`` keys and the current time, so it can be unit tested
in isolation and wired into the real engines later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# Canonical tier ordering, from hottest (closest to compute) to coldest.
TIER_ORDER = {"gpu": 0, "cpu": 1, "ssd": 2}


@dataclass
class BlockHeat:
  """Heat state for a single KV block."""
  tier: str
  block_id: int
  access_count: int = 0
  last_access: float = 0.0
  # Exponentially-decayed heat score (higher = hotter).
  score: float = 0.0

  def touch(self, now: float, decay: float) -> None:
    self.access_count += 1
    self.last_access = now
    # Decay the existing score then add a unit of heat, so old heat fades.
    self.score = self.score * decay + 1.0

  def age(self, now: float, decay: float) -> None:
    """Apply time decay without a fresh access."""
    if self.last_access <= 0.0:
      return
    # Approximate continuous decay based on seconds since last access.
    seconds = max(0.0, now - self.last_access)
    # Normalize so that ~10s of silence halves the heat.
    effective = 1.0 - (1.0 - decay) * seconds
    effective = max(0.0, min(1.0, effective))
    self.score *= effective


class HeatTracker:
  """Tracks per-block heat and reports hot/cold candidates.

  Args:
    decay: per-touch exponential decay factor in [0, 1). Lower = shorter memory.
    time_func: injectable clock for deterministic testing.
  """

  def __init__(self, decay: float = 0.95, time_func=None):
    if not 0.0 <= decay < 1.0:
      raise ValueError(f"decay must be in [0, 1), got {decay}")
    self.decay = decay
    self.time_func = time_func or time.time
    self._blocks: Dict[Tuple[str, int], BlockHeat] = {}

  # -- basic API -----------------------------------------------------------
  def _key(self, tier: str, block_id: int) -> Tuple[str, int]:
    if tier not in TIER_ORDER:
      raise ValueError(f"unknown tier: {tier}")
    return (tier, block_id)

  def register(self, tier: str, block_id: int) -> BlockHeat:
    key = self._key(tier, block_id)
    if key not in self._blocks:
      self._blocks[key] = BlockHeat(tier=tier, block_id=block_id)
    return self._blocks[key]

  def touch(self, tier: str, block_id: int) -> BlockHeat:
    bh = self.register(tier, block_id)
    bh.touch(self.time_func(), self.decay)
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
    """Return the ``n`` hottest blocks, optionally filtered by current tier."""
    candidates = [b for b in self._blocks.values() if tier is None or b.tier == tier]
    # Sort by score desc, break ties by recency (most recent first).
    candidates.sort(key=lambda b: (b.score, b.last_access), reverse=True)
    return candidates[:n]

  def top_cold(self, n: int, tier: Optional[str] = None) -> List[BlockHeat]:
    """Return the ``n`` coldest blocks (lowest score, oldest access)."""
    candidates = [b for b in self._blocks.values() if tier is None or b.tier == tier]
    # Coldest = lowest score, then oldest access (ascending on both).
    candidates.sort(key=lambda b: (b.score, b.last_access))
    return candidates[:n]

  def decay_all(self) -> None:
    """Apply time decay to every tracked block (call periodically)."""
    now = self.time_func()
    for bh in self._blocks.values():
      bh.age(now, self.decay)

  def snapshot(self) -> Dict[Tuple[str, int], float]:
    return {k: bh.score for k, bh in self._blocks.items()}

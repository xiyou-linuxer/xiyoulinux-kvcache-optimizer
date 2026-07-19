"""Adaptive threshold tuning for the heat-aware migration engine.

Instead of using fixed hot/cold thresholds, this module adjusts them based on
observed hit behavior.  The intuition:

  - If too many accesses are landing on cold tiers (SSD), the hot/warm tiers are
    too small.  We should **lower** the hot threshold (making more blocks
    eligible for GPU) and **lower** the cold threshold (making fewer blocks
    eligible for SSD) to expand the warm/hot tiers.
  - If the hot tier is under-utilized (few promotions), the thresholds may be
    too aggressive; we can relax them to avoid thrashing.

The tuner uses the ``MigrationMetrics`` counters as signals and clamps the
thresholds to safe bounds so they never drift into nonsense values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AdaptiveConfig:
  # Bounds for thresholds.
  min_hot_threshold: float = 0.5
  max_hot_threshold: float = 10.0
  min_cold_threshold: float = 0.05
  max_cold_threshold: float = 2.0
  # Step sizes for tuning.
  hot_step: float = 0.5
  cold_step: float = 0.1
  # How many cold-tier accesses trigger a threshold relaxation.
  cold_access_trigger: int = 20
  # How many rounds to wait between adjustments (avoid oscillation).
  cooldown_rounds: int = 3


class AdaptiveTuner:
  """Adjusts MigrationPolicy thresholds based on metrics feedback."""

  def __init__(self, config: Optional[AdaptiveConfig] = None):
    self.config = config or AdaptiveConfig()
    self._rounds_since_adjust = 0

  def tune(
    self,
    current_hot: float,
    current_cold: float,
    promotions: int,
    demotions: int,
    cold_accesses: int,
  ) -> tuple[float, float]:
    """Return (new_hot, new_cold) thresholds.

    Cooldown semantics: after an adjustment, we enter a cooldown of
    ``cooldown_rounds`` calls before we allow another adjustment.  This
    prevents oscillation when the system is reacting to its own changes.
    """
    # If we're still in cooldown, don't adjust.
    if self._rounds_since_adjust > 0:
      self._rounds_since_adjust -= 1
      return current_hot, current_cold

    new_hot = current_hot
    new_cold = current_cold
    adjusted = False

    # If cold-tier accesses are high, the hot/warm tiers are too small:
    # expand them by lowering both thresholds.
    if cold_accesses >= self.config.cold_access_trigger:
      new_hot = max(self.config.min_hot_threshold, current_hot - self.config.hot_step)
      new_cold = max(self.config.min_cold_threshold, current_cold - self.config.cold_step)
      adjusted = True
    # Only relax (shrink hot tier) when the system is idle: no cold pressure
    # and no migration activity at all.
    elif promotions == 0 and demotions == 0 and cold_accesses == 0:
      new_hot = min(self.config.max_hot_threshold, current_hot + self.config.hot_step)
      new_cold = min(self.config.max_cold_threshold, current_cold + self.config.cold_step)
      adjusted = True

    # Clamp to bounds.
    new_hot = max(self.config.min_hot_threshold, min(self.config.max_hot_threshold, new_hot))
    new_cold = max(self.config.min_cold_threshold, min(self.config.max_cold_threshold, new_cold))
    # Ensure hot > cold.
    if new_hot <= new_cold:
      new_hot = new_cold + 0.5

    # Enter cooldown only if we actually adjusted.
    if adjusted:
      self._rounds_since_adjust = self.config.cooldown_rounds

    return new_hot, new_cold

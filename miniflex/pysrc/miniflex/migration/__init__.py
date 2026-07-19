"""MiniFlex heat-driven migration + lookahead prefetch modules.

System-side optimization: when KV cache gets large, store it smartly.

- ``heat``: per-block access tracking and heat scoring (frequency + recency).
- ``policy``: threshold-based Hot/Warm/Cold tiering decisions.
- ``planner``: turn tiering decisions into concrete GPU<->CPU<->SSD migration
  operations that align with MiniFlex's existing ``TransferType`` enum, so the
  plans can be fed straight into ``TransferOpGraph``.
- ``prefetch``: lookahead prefetch decision logic that decides which blocks to
  pull closer before they are actually needed, to hide refill latency.

All pure-Python and self-contained for unit testing without C++/CUDA extensions.
"""
from miniflex.migration.heat import HeatTracker, BlockHeat
from miniflex.migration.policy import MigrationPolicy, Tier
from miniflex.migration.planner import MigrationPlanner, MigrationPlan, MigrationOp
from miniflex.migration.prefetch import PrefetchPlanner, PrefetchDecision

__all__ = [
  "HeatTracker",
  "BlockHeat",
  "MigrationPolicy",
  "Tier",
  "MigrationPlanner",
  "MigrationPlan",
  "MigrationOp",
  "PrefetchPlanner",
  "PrefetchDecision",
]

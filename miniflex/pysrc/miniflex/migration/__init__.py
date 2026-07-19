"""MiniFlex Heat-Aware Tiered Migration Engine.

System-side optimization: when KV cache gets large, store it smartly and move
it proactively.  The big picture is a closed loop:

  access -> heat scoring -> tiering decision -> migration plan -> execution
              ^                                          |
              +------------ completion feedback ----------+

Modules:
- ``heat``: per-block access tracking and heat scoring (frequency + recency +
  tier weight + burst detection).
- ``policy``: threshold-based Hot/Warm/Cold tiering decisions with bandwidth
  throttling.
- ``planner``: turn tiering decisions into concrete GPU<->CPU<->SSD migration
  operations that align with MiniFlex's existing ``TransferType`` enum, so the
  plans can be fed straight into ``TransferOpGraph`` (incl. two-hop SSD<->GPU).
- ``prefetch``: lookahead prefetch decision logic that decides which blocks to
  pull closer before they are actually needed, to hide refill latency.
- ``metrics``: thread-safe counters + latency samples for observability.
- ``engine``: the unified ``MigrationEngine`` that ties heat -> policy -> plan
  -> prefetch together, with in-flight deduplication, bandwidth caps and
  completion feedback.
- ``executor``: bridges MigrationPlan -> TransferOpGraph for the real
  TransferEngine, with completion feedback and two-hop chaining.
- ``adaptive``: adaptive threshold tuning based on metrics feedback, so the
  policy can expand/contract the hot tier automatically.

All pure-Python and self-contained for unit testing without C++/CUDA extensions.
"""
from miniflex.migration.heat import HeatTracker, BlockHeat
from miniflex.migration.policy import MigrationPolicy, Tier
from miniflex.migration.planner import MigrationPlanner, MigrationPlan, MigrationOp
from miniflex.migration.prefetch import PrefetchPlanner, PrefetchDecision
from miniflex.migration.metrics import MigrationMetrics, Stopwatch
from miniflex.migration.engine import MigrationEngine, MigrationEngineConfig
from miniflex.migration.executor import MigrationExecutor, ResolvedMigrationOp
from miniflex.migration.adaptive import AdaptiveTuner, AdaptiveConfig
from miniflex.migration.runtime_bridge import MigrationRuntimeBridge, RuntimeBridgeStats
from miniflex.migration.host_adapter import TransferEngineHostAdapter, build_runtime_bridge

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
  "MigrationMetrics",
  "Stopwatch",
  "MigrationEngine",
  "MigrationEngineConfig",
  "MigrationExecutor",
  "ResolvedMigrationOp",
  "AdaptiveTuner",
  "AdaptiveConfig",
  "MigrationRuntimeBridge",
  "RuntimeBridgeStats",
  "TransferEngineHostAdapter",
  "build_runtime_bridge",
]

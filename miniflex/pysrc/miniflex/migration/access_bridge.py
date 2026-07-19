"""Optional data-plane access reporting bridge for the migration engine.

The migration control plane (``MigrationEngine``) only becomes useful once it
*sees* real cache accesses.  Previously the only way to feed it was for the
host (e.g. the vLLM connector) to call ``MigrationEngine.touch()`` directly
with the right ``(tier, block_id)`` pairs, which left the data-plane wiring as
an exercise for the integrator.

This module closes that gap with a small, explicit, *opt-in* helper:

- ``AccessReport``: a plain data object describing "this request touched these
  blocks on these tiers" (GPU-resident hits, lower-tier hits, misses).
- ``AccessBridge``: translates one ``AccessReport`` into the corresponding
  ``MigrationEngine.touch()`` calls, applying the engine's weighting model
  (tier weight, phase weight, decode-step weight) exactly once per block.

The bridge is intentionally passive: it does not hook into the vLLM connector
by itself and is a no-op unless the host constructs it and calls
``report()``.  This keeps the default runtime behaviour unchanged while giving
integrators a single, documented wiring point.

Typical wiring (scheduler side of the vLLM connector):

  bridge = AccessBridge(migration_engine, block_size=config.cache_config.tokens_per_block)
  # inside get_num_new_matched_tokens, after the match result is known:
  bridge.report(AccessReport(
      request_id=request.request_id,
      phase="prefill",
      gpu_blocks=gpu_hit_block_ids,
      cpu_blocks=cpu_hit_block_ids,
      ssd_blocks=ssd_hit_block_ids,
  ))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from miniflex.migration.engine import MigrationEngine


@dataclass
class AccessReport:
  """One request's worth of cache-access evidence for the migration engine.

  Attributes:
    request_id: opaque request identifier (used only for metrics/debugging).
    phase: "prefill" or "decode"; forwarded to the heat tracker's phase
      weighting.  Defaults to "prefill".
    decode_step: current decode step index when ``phase == "decode"``.
    gpu_blocks / cpu_blocks / ssd_blocks: logical block IDs that this request
      touched on each tier.  A block should appear on at most one tier per
      report (the tier it was actually served from).
  """
  request_id: str
  phase: str = "prefill"
  decode_step: Optional[int] = None
  gpu_blocks: List[int] = field(default_factory=list)
  cpu_blocks: List[int] = field(default_factory=list)
  ssd_blocks: List[int] = field(default_factory=list)

  def counts_by_tier(self) -> Dict[str, int]:
    return {
      "gpu": len(self.gpu_blocks),
      "cpu": len(self.cpu_blocks),
      "ssd": len(self.ssd_blocks),
    }


class AccessBridge:
  """Translate :class:`AccessReport` objects into engine ``touch()`` calls.

  The bridge owns no state beyond the engine reference, so it is safe to
  construct one per connector instance and call it from the scheduler's hot
  path.  All it does is iterate the three tier lists and forward each block
  to ``MigrationEngine.touch()`` with the report's phase / decode-step
  metadata.
  """

  def __init__(self, engine: MigrationEngine):
    self.engine = engine

  def report(self, access: AccessReport) -> int:
    """Report one request's accesses; returns the number of blocks touched."""
    touched = 0
    for tier, blocks in (
      ("gpu", access.gpu_blocks),
      ("cpu", access.cpu_blocks),
      ("ssd", access.ssd_blocks),
    ):
      for block_id in blocks:
        self.engine.touch(
          tier,
          block_id,
          phase=access.phase,
          decode_step=access.decode_step,
        )
        touched += 1
    return touched

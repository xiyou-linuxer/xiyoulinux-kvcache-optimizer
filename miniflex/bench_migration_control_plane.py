"""Lightweight benchmark for the migration control plane.

This script does NOT require the real C++/CUDA TransferEngine.  It validates the
behaviour of the system-side control plane in a deterministic, fast way by
simulating accesses and showing:

- per-round tier distribution
- promotion / demotion counts
- prefetch decisions
- hot-threshold / cold-threshold adaptation
- top-hot block ranking (showing that layer / phase / decode-step weighting
  changes the ordering as intended)

Usage:
  cd miniflex
  PYTHONPATH=pysrc python bench_migration_control_plane.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from miniflex.migration import (
  AdaptiveConfig,
  MigrationEngine,
  MigrationEngineConfig,
)


class FakeClock:
  def __init__(self, start: float = 1000.0):
    self.now = start

  def time(self) -> float:
    return self.now

  def advance(self, seconds: float) -> None:
    self.now += seconds


@dataclass
class RoundSnapshot:
  round_idx: int
  tier_distribution: Dict[str, int]
  promotions: int
  demotions: int
  prefetch_blocks: int
  hot_threshold: float
  cold_threshold: float
  top_hot: List[tuple]


def seed_workload(engine: MigrationEngine, clock: FakeClock) -> None:
  """Create a representative synthetic KV workload.

  We intentionally mix:
  - SSD-heavy hot blocks (should be promoted)
  - CPU warm blocks
  - GPU hot blocks
  - different layers
  - prefill vs decode
  - different decode steps
  """
  # A few SSD blocks that are accessed often during decode (very valuable)
  for _ in range(8):
    engine.touch("ssd", 1, layer_id=20, phase="decode", decode_step=8)
    clock.advance(0.05)
  for _ in range(6):
    engine.touch("ssd", 2, layer_id=12, phase="decode", decode_step=3)
    clock.advance(0.05)

  # CPU warm blocks
  for _ in range(4):
    engine.touch("cpu", 10, layer_id=6, phase="decode", decode_step=2)
    clock.advance(0.05)
  for _ in range(3):
    engine.touch("cpu", 11, layer_id=2, phase="prefill")
    clock.advance(0.05)

  # GPU blocks that are already hot
  for _ in range(5):
    engine.touch("gpu", 100, layer_id=24, phase="decode", decode_step=10)
    clock.advance(0.05)


def run_benchmark(rounds: int = 5) -> List[RoundSnapshot]:
  clock = FakeClock()
  config = MigrationEngineConfig(
    decay=0.9,
    hot_threshold=3.0,
    cold_threshold=0.5,
    max_promotions_per_round=4,
    max_demotions_per_round=4,
    max_prefetch_ratio=0.5,
    max_inflight_blocks=16,
    adaptive=AdaptiveConfig(cold_access_trigger=2, cooldown_rounds=1),
  )
  engine = MigrationEngine(config=config, time_func=clock.time)
  seed_workload(engine, clock)

  snapshots: List[RoundSnapshot] = []
  for round_idx in range(1, rounds + 1):
    plan = engine.tick()
    # Simulate an incoming request that matched some blocks on CPU/SSD.
    dec = engine.request_prefetch(
      request_id=round_idx,
      matched={"cpu": 3, "ssd": 2},
      gpu_blocks_available=4,
    )
    stats = engine.stats()
    top_hot = [
      (b.tier, b.block_id, round(b.score, 3), b.layer_id, b.phase, b.decode_step)
      for b in engine.tracker.top_hot(3)
    ]
    snapshots.append(
      RoundSnapshot(
        round_idx=round_idx,
        tier_distribution=stats["tier_distribution"],
        promotions=plan.num_promotions,
        demotions=plan.num_demotions,
        prefetch_blocks=dec.total_blocks,
        hot_threshold=engine.policy.hot_threshold,
        cold_threshold=engine.policy.cold_threshold,
        top_hot=top_hot,
      )
    )
    # Mark all currently planned ops as completed so the next round can make
    # further progress in this synthetic benchmark.
    for op in plan.ops:
      engine.mark_completed(op.src_tier, op.block_id)
    clock.advance(0.2)
  return snapshots


def main() -> None:
  snaps = run_benchmark()
  print("=== Migration Control Plane Benchmark ===")
  for s in snaps:
    print(f"\n[round {s.round_idx}]")
    print(f"tier_distribution={s.tier_distribution}")
    print(f"promotions={s.promotions}, demotions={s.demotions}, prefetch_blocks={s.prefetch_blocks}")
    print(f"thresholds: hot={s.hot_threshold:.2f}, cold={s.cold_threshold:.2f}")
    print("top_hot=")
    for item in s.top_hot:
      tier, block_id, score, layer_id, phase, decode_step = item
      print(
        f"  - tier={tier} block={block_id} score={score} layer={layer_id} phase={phase} step={decode_step}"
      )


if __name__ == "__main__":
  main()

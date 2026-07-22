"""Configuration and type-registration tests for GDS wiring.

They intentionally cover only public configuration and transfer-op validation.
Planner routing and CPU two-hop fallback are outside this PR; direct GDS I/O is
covered separately on a CUDA machine with cuFile.

Run: PYTHONPATH=pysrc:test python -m pytest test/gds_worker_test.py -q
"""
import numpy as np

from miniflex.common.config import CacheConfig
from miniflex.common.transfer import TransferType, TransferOp, TransferOpGraph


def test_gds_transfer_types_exist():
  assert TransferType.D2DISK.value == "D2DISK"
  assert TransferType.DISK2D.value == "DISK2D"


def test_gds_ops_are_valid_transfer_ops():
  # These direct types must pass TransferOp validation. This does not imply
  # that the planner emits them yet.
  graph = TransferOpGraph()
  op = TransferOp(
    transfer_type=TransferType.D2DISK,
    graph_id=graph.graph_id,
    src_block_ids=np.array([0, 1], dtype=np.int64),
    dst_block_ids=np.array([2, 3], dtype=np.int64),
  )
  graph.add_transfer_op(op)
  assert graph.num_ops == 1


def test_enable_gds_requires_ssd():
  try:
    CacheConfig(tokens_per_block=16, enable_ssd=False, enable_gds=True)
  except ValueError as e:
    assert "enable_gds requires enable_ssd" in str(e)
  else:
    raise AssertionError("expected ValueError when enable_gds without enable_ssd")


def test_enable_gds_with_ssd_ok(tmp_path):
  cfg = CacheConfig(
    tokens_per_block=16,
    enable_ssd=True,
    ssd_cache_dir=str(tmp_path),
    enable_gds=True,
  )
  assert cfg.enable_gds is True


def test_default_gds_disabled():
  cfg = CacheConfig(tokens_per_block=16)
  assert cfg.enable_gds is False


def test_gds_env_override(monkeypatch):
  import miniflex.integration.config as icfg
  monkeypatch.setenv("MINIFLEX_ENABLE_GDS", "1")
  overrides = icfg._env_overrides()
  assert overrides.get("enable_gds") == "1"
  assert icfg._parse_bool(overrides["enable_gds"]) is True


def test_gds_json_config_override():
  import miniflex.integration.config as icfg
  from miniflex.common.config import CacheConfig
  cfg = CacheConfig(tokens_per_block=16, enable_ssd=True, ssd_cache_dir="/tmp/x")
  icfg._apply_cache_overrides(cfg, {"enable_gds": True})
  assert cfg.enable_gds is True

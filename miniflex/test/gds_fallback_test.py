"""Unavailable-GDS behavior tests for the direct SSD<->GPU worker.

Validates that the worker never reports a direct transfer as successful when
cuFile is unavailable. CPU two-hop fallback is a planner/engine decision and
is not performed inside this worker.

Run: PYTHONPATH=pysrc:test python -m pytest test/gds_fallback_test.py -q
"""
import pytest
import torch

from miniflex.common.storage import (
  KVCacheLayout,
  KVCacheLayoutType,
  StorageHandle,
  StorageHandlerType,
)
from miniflex.common.transfer import TransferType

# The GDS worker needs miniflex._C (built with the C++ extension). On machines
# without that extension, keep the pure-Python suite independent of native GDS.
try:
  import miniflex._C  # noqa: F401
except ImportError:  # pragma: no cover - environment without the extension
  GDSTransferWorker = None
  _HAS_C = False
else:
  from miniflex.transfer.worker import GDSTransferWorker
  _HAS_C = True

pytestmark = pytest.mark.skipif(
  not _HAS_C or not torch.cuda.is_available(),
  reason="GDS native-worker tests require miniflex._C and CUDA",
)


def _gpu_handle(tokens_per_block=16, num_blocks=8, num_layers=2, num_heads=2, head_size=4):
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
    use_mla=False,
  )
  data = torch.zeros(
    layout.get_total_elements(),
    dtype=torch.float32,
    device="cuda",
  )
  return StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    kv_layout=layout,
    dtype=torch.float32,
    data=data,
    gpu_device_id=torch.cuda.current_device(),
  )


def _ssd_handle(tmp_path, tokens_per_block=16, num_blocks=8, num_layers=2, num_heads=2, head_size=4):
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.BLOCKFIRST,
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
    use_mla=False,
  )
  f = tmp_path / "gds_test.bin"
  f.write_bytes(b"\x00" * layout.get_total_elements() * 4)
  return StorageHandle(
    handle_type=StorageHandlerType.FILE,
    kv_layout=layout,
    dtype=torch.float32,
    data=[str(f)],
    num_blocks_per_file=num_blocks,
  )


class _DummyPipe:
  def poll(self, timeout=0):
    return False

  def recv(self):
    raise EOFError


class _DummyQueue:
  def __init__(self):
    self.items = []

  def put(self, x):
    self.items.append(x)


def _make_worker(tmp_path):
  gpu = _gpu_handle()
  ssd = _ssd_handle(tmp_path)
  op_buf = torch.zeros((4, 16), dtype=torch.int64)
  return GDSTransferWorker(
    worker_id=0,
    transfer_conn=_DummyPipe(),
    finished_ops_queue=_DummyQueue(),
    op_buffer_tensor=op_buf,
    gpu_storage_handle=gpu,
    ssd_storage_handle=ssd,
  )


def test_gds_ctx_exposes_availability(tmp_path):
  worker = _make_worker(tmp_path)
  assert isinstance(worker.is_available(), bool)


def test_worker_raises_when_unavailable(tmp_path):
  worker = _make_worker(tmp_path)
  if worker.is_available():
    pytest.skip("the local cuFile backend initialized")
  src = torch.tensor([0, 1], dtype=torch.int64)
  dst = torch.tensor([0, 1], dtype=torch.int64)
  for transfer_type in (TransferType.DISK2D, TransferType.D2DISK):
    with pytest.raises(RuntimeError, match="GDSIOCTX.transfer_blocks failed"):
      worker._transfer_impl(src, dst, transfer_type)


def test_worker_rejects_invalid_transfer_type(tmp_path):
  worker = _make_worker(tmp_path)
  src = torch.tensor([0], dtype=torch.int64)
  dst = torch.tensor([0], dtype=torch.int64)
  for bad in (TransferType.H2D, TransferType.D2H, TransferType.H2DISK, TransferType.DISK2H):
    with pytest.raises(ValueError):
      worker._transfer_impl(src, dst, bad)


def test_worker_bounds_checking(tmp_path):
  worker = _make_worker(tmp_path)
  src = torch.tensor([0, 999], dtype=torch.int64)
  dst = torch.tensor([0, 1], dtype=torch.int64)
  with pytest.raises(ValueError, match="out of range"):
    worker._transfer_impl(src, dst, TransferType.D2DISK)

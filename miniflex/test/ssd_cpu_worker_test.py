import tempfile
import sys
import types
from pathlib import Path

import torch

from miniflex.common.storage import (
  KVCacheLayout,
  KVCacheLayoutType,
  StorageHandle,
  StorageHandlerType,
)
from miniflex.common.transfer import TransferType


class FakeSSDIOCTX:
  instances: list["FakeSSDIOCTX"] = []

  def __init__(
      self,
      queue_depth: int,
      blocks_per_file: int,
      cpu_tensor: torch.Tensor,
      layer_num: int,
      kv_dim: int,
      cpu_num_blocks: int,
      slice_bytes: int,
      file_paths: list[str],
      use_direct_io: bool = True,
  ):
    self.queue_depth = queue_depth
    self.blocks_per_file = blocks_per_file
    self.cpu_tensor = cpu_tensor
    self.layer_num = layer_num
    self.kv_dim = kv_dim
    self.cpu_num_blocks = cpu_num_blocks
    self.slice_bytes = slice_bytes
    self.file_paths = file_paths
    self.use_direct_io = use_direct_io
    self.calls = []
    FakeSSDIOCTX.instances.append(self)

  def transfer_blocks(
      self,
      src_block_ids: torch.Tensor,
      dst_block_ids: torch.Tensor,
      is_read: bool,
  ) -> bool:
    self.calls.append((
      src_block_ids.clone(),
      dst_block_ids.clone(),
      is_read,
    ))
    return True


class FakeCExtension:
  SSDIOCTX = FakeSSDIOCTX


fake_c_extension = types.ModuleType("miniflex._C")
fake_c_extension.SSDIOCTX = FakeSSDIOCTX
sys.modules["miniflex._C"] = fake_c_extension

from miniflex.transfer import worker as worker_module
from miniflex.transfer.worker import SSDCPUTransferWorker


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def make_layout(
    layout_type: KVCacheLayoutType,
    num_blocks: int,
    num_layers: int = 2,
    tokens_per_block: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
    use_mla: bool = False,
) -> KVCacheLayout:
  return KVCacheLayout(
    layout_type=layout_type,
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
    use_mla=use_mla,
  )


def make_handles(
    tmp_path: Path,
    dtype: torch.dtype = torch.float16,
    cpu_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERFIRST,
    ssd_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKFIRST,
    cpu_num_blocks: int = 4,
    ssd_num_blocks: int = 6,
    num_blocks_per_file: int = 3,
):
  cpu_layout = make_layout(cpu_layout_type, cpu_num_blocks)
  ssd_layout = make_layout(ssd_layout_type, ssd_num_blocks)
  cpu_tensor = torch.empty(cpu_layout.get_total_elements(), dtype=dtype)
  file_paths = []
  for file_idx in range((ssd_num_blocks + num_blocks_per_file - 1) // num_blocks_per_file):
    file_path = tmp_path / f"ssd_cache_{file_idx}.bin"
    file_path.write_bytes(b"\0" * 4096)
    file_paths.append(str(file_path))
  cpu_handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=cpu_tensor,
    kv_layout=cpu_layout,
    dtype=dtype,
  )
  ssd_handle = StorageHandle(
    handle_type=StorageHandlerType.FILE,
    data=file_paths,
    kv_layout=ssd_layout,
    dtype=dtype,
    num_blocks_per_file=num_blocks_per_file,
  )
  return cpu_handle, ssd_handle


def make_worker(cpu_handle, ssd_handle, use_direct_io: bool = False):
  return SSDCPUTransferWorker(
    worker_id=0,
    transfer_conn=None,
    finished_ops_queue=None,
    op_buffer_tensor=torch.empty((2, 4), dtype=torch.int64),
    cpu_storage_handle=cpu_handle,
    ssd_storage_handle=ssd_handle,
    use_direct_io=use_direct_io,
  )


def patch_fake_c_extension():
  FakeSSDIOCTX.instances.clear()
  old_c_extension = worker_module._C
  worker_module._C = FakeCExtension
  return old_c_extension


def restore_c_extension(old_c_extension):
  worker_module._C = old_c_extension


def test_ssdcpu_init_builds_ssd_io_ctx_with_expected_shape_and_direct_io():
  old_c_extension = patch_fake_c_extension()
  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      cpu_handle, ssd_handle = make_handles(Path(tmp_dir))
      worker = make_worker(cpu_handle, ssd_handle, use_direct_io=True)
      ctx = FakeSSDIOCTX.instances[-1]

      assert worker.num_layers == 2
      assert worker.kv_dim == 2
      assert worker.cpu_num_blocks == 4
      assert worker.ssd_num_blocks == 6
      assert worker.slice_bytes == 4 * 2 * 8 * torch.float16.itemsize
      assert ctx.queue_depth == 128
      assert ctx.blocks_per_file == 3
      assert ctx.cpu_tensor is cpu_handle.data
      assert ctx.layer_num == 2
      assert ctx.kv_dim == 2
      assert ctx.cpu_num_blocks == 4
      assert ctx.slice_bytes == worker.slice_bytes
      assert ctx.file_paths == ssd_handle.get_file_list()
      assert ctx.use_direct_io is True
  finally:
    restore_c_extension(old_c_extension)


def test_ssdcpu_init_validation_rejects_bad_layout_and_dtype():
  old_c_extension = patch_fake_c_extension()
  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = Path(tmp_dir)
      cpu_handle, ssd_handle = make_handles(tmp_path)

      bad_cpu, good_ssd = make_handles(tmp_path, cpu_layout_type=KVCacheLayoutType.BLOCKFIRST)
      assert_raises(ValueError, lambda: make_worker(bad_cpu, good_ssd))

      good_cpu, bad_ssd = make_handles(tmp_path, ssd_layout_type=KVCacheLayoutType.LAYERFIRST)
      assert_raises(ValueError, lambda: make_worker(good_cpu, bad_ssd))

      dtype_cpu, dtype_ssd = make_handles(tmp_path)
      dtype_ssd.dtype = torch.float32
      assert_raises(ValueError, lambda: make_worker(dtype_cpu, dtype_ssd))

      bad_file_handle = StorageHandle(
        handle_type=StorageHandlerType.FILE,
        data=ssd_handle.get_file_list(),
        kv_layout=ssd_handle.kv_layout,
        dtype=ssd_handle.dtype,
        num_blocks_per_file=ssd_handle.num_blocks_per_file,
      )
      assert_raises(ValueError, lambda: make_worker(bad_file_handle, ssd_handle))
  finally:
    restore_c_extension(old_c_extension)


def test_ssdcpu_transfer_impl_maps_h2disk_and_disk2h_to_cpp_direction():
  old_c_extension = patch_fake_c_extension()
  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      cpu_handle, ssd_handle = make_handles(Path(tmp_dir))
      worker = make_worker(cpu_handle, ssd_handle)
      ctx = FakeSSDIOCTX.instances[-1]

      cpu_blocks = torch.tensor([3, 1], dtype=torch.int64)
      ssd_blocks = torch.tensor([5, 0], dtype=torch.int64)
      assert worker._transfer_impl(cpu_blocks, ssd_blocks, TransferType.H2DISK) is True
      src, dst, is_read = ctx.calls[-1]
      assert torch.equal(src, cpu_blocks)
      assert torch.equal(dst, ssd_blocks)
      assert is_read is False

      assert worker._transfer_impl(ssd_blocks, cpu_blocks, TransferType.DISK2H) is True
      src, dst, is_read = ctx.calls[-1]
      assert torch.equal(src, ssd_blocks)
      assert torch.equal(dst, cpu_blocks)
      assert is_read is True
  finally:
    restore_c_extension(old_c_extension)


def test_ssdcpu_transfer_impl_validation():
  old_c_extension = patch_fake_c_extension()
  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      cpu_handle, ssd_handle = make_handles(Path(tmp_dir))
      worker = make_worker(cpu_handle, ssd_handle)

      assert_raises(
        ValueError,
        lambda: worker._transfer_impl(
          torch.tensor([0], dtype=torch.int64),
          torch.tensor([0], dtype=torch.int64),
          TransferType.H2D,
        ),
      )
      assert_raises(
        ValueError,
        lambda: worker._transfer_impl(
          torch.tensor([0.0], dtype=torch.float32),
          torch.tensor([0], dtype=torch.int64),
          TransferType.H2DISK,
        ),
      )
      assert_raises(
        ValueError,
        lambda: worker._transfer_impl(
          torch.tensor([[0]], dtype=torch.int64),
          torch.tensor([[0]], dtype=torch.int64),
          TransferType.H2DISK,
        ),
      )
      assert_raises(
        ValueError,
        lambda: worker._transfer_impl(
          torch.tensor([4], dtype=torch.int64),
          torch.tensor([0], dtype=torch.int64),
          TransferType.H2DISK,
        ),
      )
      assert_raises(
        ValueError,
        lambda: worker._transfer_impl(
          torch.tensor([6], dtype=torch.int64),
          torch.tensor([0], dtype=torch.int64),
          TransferType.DISK2H,
        ),
      )

      assert worker._transfer_impl(
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
        TransferType.H2DISK,
      ) is True
  finally:
    restore_c_extension(old_c_extension)


TEST_CASES = [
  ("SSDCPU 初始化构造 C++ ctx 并传递 O_DIRECT 选项", test_ssdcpu_init_builds_ssd_io_ctx_with_expected_shape_and_direct_io),
  ("SSDCPU 初始化轻量校验", test_ssdcpu_init_validation_rejects_bad_layout_and_dtype),
  ("SSDCPU transfer_impl 方向映射", test_ssdcpu_transfer_impl_maps_h2disk_and_disk2h_to_cpp_direction),
  ("SSDCPU transfer_impl 参数校验", test_ssdcpu_transfer_impl_validation),
]


def run_all_tests():
  print("开始运行 SSDCPUTransferWorker 校验测试")
  total = len(TEST_CASES)
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{total}] 开始：{name}")
    test_fn()
    print(f"[{index}/{total}] 通过：{name}")
  print(f"SSDCPUTransferWorker 校验测试完成：通过 {total}/{total}")


if __name__ == "__main__":
  run_all_tests()

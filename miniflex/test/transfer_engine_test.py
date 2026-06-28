# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：TransferEngine 的 submit（需 start、单图/图列表）、完成查询超时、handle 不匹配校验、op buffer 复用，以及 H2DISK/DISK2H 真实文件往返。

import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.storage import (
  KVCacheLayout,
  KVCacheLayoutType,
  StorageHandle,
  StorageHandlerType,
)
from miniflex.common.transfer import TransferOp, TransferOpGraph, TransferType
from miniflex.transfer import transfer_engine as transfer_engine_module
from miniflex.transfer.transfer_engine import TransferEngine


def blocks(ids: list[int]) -> np.ndarray:
  return np.array(ids, dtype=np.int64)


def make_layerfirst_layout(
    num_layers: int = 2,
    num_blocks: int = 4,
    tokens_per_block: int = 2,
    num_heads: int = 1,
    head_size: int = 4,
) -> KVCacheLayout:
  return KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
    use_mla=False,
  )


def make_blockfirst_layout(
    num_layers: int = 2,
    num_blocks: int = 6,
    tokens_per_block: int = 2,
    num_heads: int = 1,
    head_size: int = 4,
) -> KVCacheLayout:
  return KVCacheLayout(
    layout_type=KVCacheLayoutType.BLOCKFIRST,
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
    use_mla=False,
  )


def make_model(dtype: torch.dtype = torch.float32) -> ModelConfig:
  return ModelConfig(
    num_layers=2,
    num_kv_heads=1,
    head_size=4,
    dtype=dtype,
  )


def make_cpu_tensor_and_handle(dtype: torch.dtype = torch.float32):
  layout = make_layerfirst_layout()
  cpu_tensor = torch.zeros(layout.kv_shape, dtype=dtype).contiguous().share_memory_()
  handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=cpu_tensor,
    kv_layout=layout,
    dtype=dtype,
  )
  return cpu_tensor, handle


def make_dummy_gpu_handle(dtype: torch.dtype = torch.float32) -> StorageHandle:
  layout = make_layerfirst_layout()
  tensor = torch.empty(layout.get_total_elements(), dtype=dtype)
  return StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=tensor,
    kv_layout=layout,
    dtype=dtype,
    gpu_device_id=0,
  )


def make_ssd_handle(tmp_path: Path, dtype: torch.dtype = torch.float32) -> StorageHandle:
  ssd_layout = make_blockfirst_layout()
  block_bytes = ssd_layout.get_elements_per_block() * dtype.itemsize
  num_blocks_per_file = 3
  file_paths = []
  for file_idx in range(2):
    file_path = tmp_path / f"transfer_engine_ssd_{file_idx}.bin"
    with open(file_path, "wb") as file:
      file.truncate(num_blocks_per_file * block_bytes)
    file_paths.append(str(file_path))
  return StorageHandle(
    handle_type=StorageHandlerType.FILE,
    data=file_paths,
    kv_layout=ssd_layout,
    dtype=dtype,
    num_blocks_per_file=num_blocks_per_file,
  )


def make_cache_config(tmp_dir: str, enable_ssd: bool = True) -> CacheConfig:
  return CacheConfig(
    tokens_per_block=2,
    enable_ssd=enable_ssd,
    num_cpu_blocks=4,
    num_ssd_blocks=6,
    ssd_cache_dir=tmp_dir if enable_ssd else None,
    use_direct_io=False,
  )


def make_transfer_engine(tmp_dir: str, enable_ssd: bool = True):
  dtype = torch.float32
  model = make_model(dtype)
  cache = make_cache_config(tmp_dir, enable_ssd=enable_ssd)
  cpu_tensor, cpu_handle = make_cpu_tensor_and_handle(dtype)
  gpu_handle = make_dummy_gpu_handle(dtype)
  ssd_handle = make_ssd_handle(Path(tmp_dir), dtype) if enable_ssd else None
  engine = TransferEngine(model, cache, gpu_handle, cpu_handle, ssd_handle)
  return engine, cpu_tensor


def close_unstarted_engine(engine: TransferEngine) -> None:
  for fd in (engine.shutdown_read_fd, engine.shutdown_write_fd):
    try:
      os.close(fd)
    except OSError:
      pass


def make_single_op_graph(
    transfer_type: TransferType,
    src_ids: list[int],
    dst_ids: list[int],
) -> tuple[TransferOpGraph, TransferOp]:
  graph = TransferOpGraph()
  op = TransferOp(
    transfer_type=transfer_type,
    graph_id=graph.graph_id,
    src_block_ids=blocks(src_ids),
    dst_block_ids=blocks(dst_ids),
  )
  graph.add_transfer_op(op)
  return graph, op


def wait_for_graph_completed(engine: TransferEngine, graph_id: int, timeout: float = 10.0):
  completed = wait_for_completed_graph_ids(engine, {graph_id}, timeout=timeout)
  return completed


def wait_for_completed_graph_ids(
    engine: TransferEngine,
    graph_ids: set[int],
    timeout: float = 10.0,
):
  deadline = time.time() + timeout
  completed = []
  completed_graph_ids = set()
  while time.time() < deadline:
    completed.extend(engine.get_completed_graphs_and_ops(timeout=0.1))
    completed_graph_ids.update(
      op.graph_id for op in completed if op.is_graph_completed()
    )
    if graph_ids.issubset(completed_graph_ids):
      return completed
    time.sleep(0.01)
  raise AssertionError(
    f"timeout waiting for completed graphs {sorted(graph_ids)}; got={completed}"
  )


def fill_cpu_tensor_blocks(cpu_tensor: torch.Tensor, base: int) -> None:
  for layer_id in range(cpu_tensor.shape[0]):
    for kv_id in range(cpu_tensor.shape[1]):
      for block_id in range(cpu_tensor.shape[2]):
        value = float(base + layer_id * 1000 + kv_id * 100 + block_id)
        cpu_tensor[layer_id, kv_id, block_id].fill_(value)


def assert_cpu_blocks_equal(
    cpu_tensor: torch.Tensor,
    expected_tensor: torch.Tensor,
    dst_to_src: dict[int, int],
    zero_blocks: list[int],
) -> None:
  for dst_block_id, src_block_id in dst_to_src.items():
    actual = cpu_tensor[:, :, dst_block_id]
    expected = expected_tensor[:, :, src_block_id]
    assert torch.equal(actual, expected), (
      f"CPU block {dst_block_id} should equal original block {src_block_id}"
    )
  for block_id in zero_blocks:
    block = cpu_tensor[:, :, block_id]
    assert torch.equal(block, torch.zeros_like(block)), (
      f"CPU block {block_id} should remain zero"
    )


class FakeReadyEvent:
  def wait(self, timeout=None):
    del timeout
    return True


class FakeWorkerHandle:
  _next_worker_id = 0

  def __init__(self):
    self.worker_id = FakeWorkerHandle._next_worker_id
    FakeWorkerHandle._next_worker_id += 1
    self.ready_event = FakeReadyEvent()
    self.submitted_ops = []
    self.shutdown_called = False

  def submit_transfer(self, transfer_op: TransferOp) -> None:
    self.submitted_ops.append(transfer_op)

  def shutdown(self) -> None:
    self.shutdown_called = True


class FakeGPUCPUTransferWorker:
  @classmethod
  def create_worker(cls, *args, **kwargs):
    del args, kwargs
    return FakeWorkerHandle()


def patch_fake_gpucpu_worker():
  old_worker_cls = transfer_engine_module.GPUCPUTransferWorker
  transfer_engine_module.GPUCPUTransferWorker = FakeGPUCPUTransferWorker
  return old_worker_cls


def restore_gpucpu_worker(old_worker_cls):
  transfer_engine_module.GPUCPUTransferWorker = old_worker_cls


def test_submit_transfer_graph_requires_start_and_accepts_single_graph():
  old_worker_cls = patch_fake_gpucpu_worker()
  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      engine, _ = make_transfer_engine(tmp_dir, enable_ssd=False)
      try:
        graph, _ = make_single_op_graph(TransferType.VIRTUAL, [], [])
        engine.submit_transfer_graph(graph)
        assert not engine._running
        assert engine.get_completed_graphs_and_ops(timeout=0.01) == []

        engine.start()
        completed = wait_for_graph_completed(engine, graph.graph_id)
        assert any(op.graph_id == graph.graph_id and op.is_graph_completed() for op in completed)
      finally:
        engine.shutdown()
  finally:
    restore_gpucpu_worker(old_worker_cls)


def test_submit_transfer_graph_accepts_graph_list_and_rejects_bad_input():
  old_worker_cls = patch_fake_gpucpu_worker()
  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      engine, _ = make_transfer_engine(tmp_dir, enable_ssd=False)
      try:
        first_graph, _ = make_single_op_graph(TransferType.VIRTUAL, [], [])
        second_graph, _ = make_single_op_graph(TransferType.VIRTUAL, [], [])
        engine.start()
        engine.submit_transfer_graph([first_graph, second_graph])
        completed = wait_for_completed_graph_ids(
          engine,
          {first_graph.graph_id, second_graph.graph_id},
        )
        completed_graph_ids = {op.graph_id for op in completed if op.is_graph_completed()}
        assert {first_graph.graph_id, second_graph.graph_id}.issubset(completed_graph_ids)

        try:
          engine.submit_transfer_graph((first_graph, second_graph))
        except ValueError:
          pass
        else:
          raise AssertionError("tuple input should be rejected")

        try:
          engine.submit_transfer_graph([first_graph, object()])
        except ValueError:
          pass
        else:
          raise AssertionError("mixed list input should be rejected")
      finally:
        engine.shutdown()
  finally:
    restore_gpucpu_worker(old_worker_cls)


def test_get_completed_graphs_and_ops_empty_respects_timeout():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, _ = make_transfer_engine(tmp_dir, enable_ssd=False)
    try:
      started = time.time()
      assert engine.get_completed_graphs_and_ops(timeout=0.01) == []
      assert time.time() - started < 0.5
    finally:
      close_unstarted_engine(engine)


def test_init_rejects_obvious_handle_mismatches():
  with tempfile.TemporaryDirectory() as tmp_dir:
    dtype = torch.float32
    model = make_model(dtype)
    cache = make_cache_config(tmp_dir, enable_ssd=True)
    _, cpu_handle = make_cpu_tensor_and_handle(dtype)
    gpu_handle = make_dummy_gpu_handle(dtype)
    ssd_handle = make_ssd_handle(Path(tmp_dir), dtype)

    bad_gpu_dtype = make_dummy_gpu_handle(torch.float16)
    try:
      TransferEngine(model, cache, bad_gpu_dtype, cpu_handle, ssd_handle)
    except ValueError:
      pass
    else:
      raise AssertionError("gpu/cpu dtype mismatch should be rejected")

    bad_gpu_layout = make_dummy_gpu_handle(dtype)
    bad_gpu_layout.kv_layout = make_layerfirst_layout(num_layers=3)
    try:
      TransferEngine(model, cache, bad_gpu_layout, cpu_handle, ssd_handle)
    except ValueError:
      pass
    else:
      raise AssertionError("gpu/cpu layout mismatch should be rejected")

    bad_ssd_dtype = make_ssd_handle(Path(tmp_dir), torch.float16)
    try:
      TransferEngine(model, cache, gpu_handle, cpu_handle, bad_ssd_dtype)
    except ValueError:
      pass
    else:
      raise AssertionError("cpu/ssd dtype mismatch should be rejected")

    bad_ssd_coverage = make_ssd_handle(Path(tmp_dir), dtype)
    bad_ssd_coverage.num_blocks_per_file = 1
    try:
      TransferEngine(model, cache, gpu_handle, cpu_handle, bad_ssd_coverage)
    except ValueError:
      pass
    else:
      raise AssertionError("SSD files that do not cover layout blocks should be rejected")

    try:
      TransferEngine(model, cache, gpu_handle, cpu_handle, None)
    except ValueError:
      pass
    else:
      raise AssertionError("missing SSD handle should be rejected when SSD is enabled")


def test_register_op_buffer_uses_device_prefix_for_same_block_ids():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, _ = make_transfer_engine(tmp_dir, enable_ssd=False)
    try:
      op = TransferOp(
        transfer_type=TransferType.H2DISK,
        graph_id=0,
        src_block_ids=blocks([1, 2]),
        dst_block_ids=blocks([1, 2]),
      )
      engine._register_op_buffer(op)
      assert op.src_slot_id != -1
      assert op.dst_slot_id != -1
      assert op.src_slot_id != op.dst_slot_id

      engine._free_op_buffer(op)
      assert op.src_slot_id == -1
      assert op.dst_slot_id == -1
    finally:
      close_unstarted_engine(engine)


def test_transfer_engine_h2disk_disk2h_roundtrip_real_files():
  old_worker_cls = patch_fake_gpucpu_worker()
  try:
    with tempfile.TemporaryDirectory() as tmp_dir:
      engine, cpu_tensor = make_transfer_engine(tmp_dir, enable_ssd=True)
      fill_cpu_tensor_blocks(cpu_tensor, base=900)
      original = cpu_tensor.clone()

      try:
        engine.start()

        write_graph, write_op = make_single_op_graph(
          TransferType.H2DISK,
          src_ids=[3, 1],
          dst_ids=[4, 0],
        )
        engine.submit_transfer_graph(write_graph)
        write_completed = wait_for_graph_completed(engine, write_graph.graph_id)
        assert (write_graph.graph_id, write_op.op_id) in [op.to_tuple() for op in write_completed]

        cpu_tensor.zero_()

        read_graph, read_op = make_single_op_graph(
          TransferType.DISK2H,
          src_ids=[4, 0],
          dst_ids=[0, 2],
        )
        engine.submit_transfer_graph(read_graph)
        read_completed = wait_for_graph_completed(engine, read_graph.graph_id)
        assert (read_graph.graph_id, read_op.op_id) in [op.to_tuple() for op in read_completed]

        assert_cpu_blocks_equal(
          cpu_tensor,
          original,
          dst_to_src={0: 3, 2: 1},
          zero_blocks=[1, 3],
        )
      finally:
        engine.shutdown()
  finally:
    restore_gpucpu_worker(old_worker_cls)


TEST_CASES = [
  ("submit_transfer_graph 需要显式 start 且接受单 graph", test_submit_transfer_graph_requires_start_and_accepts_single_graph),
  ("submit_transfer_graph 接受 graph list 并拒绝坏输入", test_submit_transfer_graph_accepts_graph_list_and_rejects_bad_input),
  ("get_completed_graphs_and_ops 空队列按 timeout 返回", test_get_completed_graphs_and_ops_empty_respects_timeout),
  ("TransferEngine 初始化拒绝明显 handle 错误", test_init_rejects_obvious_handle_mismatches),
  ("register_op_buffer 使用设备前缀隔离相同 block id", test_register_op_buffer_uses_device_prefix_for_same_block_ids),
  ("TransferEngine H2DISK/DISK2H 真实文件往返校验", test_transfer_engine_h2disk_disk2h_roundtrip_real_files),
]


def run_all_tests():
  print("开始运行 TransferEngine 测试")
  total = len(TEST_CASES)
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{total}] 开始：{name}")
    try:
      test_fn()
    except Exception as exc:
      print(f"[{index}/{total}] 失败：{name}，错误：{type(exc).__name__}: {exc}")
      raise
    print(f"[{index}/{total}] 通过：{name}")
  print(f"TransferEngine 测试完成：通过 {total}/{total}")


if __name__ == "__main__":
  run_all_tests()

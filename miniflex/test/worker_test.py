# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：TransferWorkerBase worker 进程的 D2H/H2D/H2DISK/DISK2H 提交与张量更新、共享 op slot、shutdown 排空任务，以及 GPU↔CPU、SSD↔CPU 端到端吞吐基准。

import queue
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from miniflex.common.ring_buffer import SharedOpPool
from miniflex.common.storage import (
  KVCacheLayout,
  KVCacheLayoutType,
  StorageHandle,
  StorageHandlerType,
)
from miniflex.common.transfer import TransferOp, TransferType
from miniflex.transfer.worker import GPUCPUTransferWorker, SSDCPUTransferWorker, TransferWorkerBase


class SkipTest(Exception):
  pass


def require_cuda() -> None:
  if not torch.cuda.is_available():
    raise SkipTest("CUDA 不可用，跳过真实 GPUCPU 跨进程 worker 测试")
  torch.cuda.set_device(0)


def blocks(ids: list[int]) -> np.ndarray:
  return np.array(ids, dtype=np.int64)


def make_op(
    transfer_type: TransferType,
    src_ids: list[int],
    dst_ids: list[int],
    graph_id: int,
) -> TransferOp:
  return TransferOp(
    transfer_type=transfer_type,
    graph_id=graph_id,
    src_block_ids=blocks(src_ids),
    dst_block_ids=blocks(dst_ids),
  )


def wait_queue_item(result_queue, timeout: float = 10):
  deadline = time.time() + timeout
  while time.time() < deadline:
    try:
      return result_queue.get(timeout=0.1)
    except queue.Empty:
      pass
  raise AssertionError("timeout waiting for worker finished op")


def wait_completed_ids(result_queue, count: int) -> list[int]:
  return [wait_queue_item(result_queue) for _ in range(count)]


def transfer_payload_bytes(layout: KVCacheLayout, dtype: torch.dtype, num_blocks: int) -> int:
  return layout.get_elements_per_block() * dtype.itemsize * num_blocks


def format_bytes(num_bytes: int) -> str:
  if num_bytes < 1024:
    return f"{num_bytes} B"
  if num_bytes < 1024 * 1024:
    return f"{num_bytes / 1024:.3f} KiB"
  if num_bytes < 1024 * 1024 * 1024:
    return f"{num_bytes / 1024 / 1024:.3f} MiB"
  return f"{num_bytes / 1024 / 1024 / 1024:.3f} GiB"


def format_rate(num_bytes: int, elapsed_s: float) -> str:
  if elapsed_s <= 0:
    return "inf B/s"
  bytes_per_second = num_bytes / elapsed_s
  if bytes_per_second < 1024:
    return f"{bytes_per_second:.3f} B/s"
  if bytes_per_second < 1024 * 1024:
    return f"{bytes_per_second / 1024:.3f} KiB/s"
  if bytes_per_second < 1024 * 1024 * 1024:
    return f"{bytes_per_second / 1024 / 1024:.3f} MiB/s"
  return f"{bytes_per_second / 1024 / 1024 / 1024:.3f} GiB/s"


def print_transfer_timing(label: str, payload_bytes: int, elapsed_s: float, num_blocks: int) -> None:
  elapsed_ms = elapsed_s * 1000
  print(
    f"    单次端到端：{label} blocks={num_blocks} "
    f"payload={format_bytes(payload_bytes)} "
    f"elapsed={elapsed_ms:.3f} ms "
    f"rate={format_rate(payload_bytes, elapsed_s)}"
  )


def print_transfer_benchmark(
    label: str,
    payload_bytes_per_op: int,
    repeats: int,
    elapsed_s: float,
    num_blocks: int,
) -> None:
  total_payload_bytes = payload_bytes_per_op * repeats
  print(
    f"    端到端吞吐：{label} blocks/op={num_blocks} repeats={repeats} "
    f"payload/op={format_bytes(payload_bytes_per_op)} "
    f"total={format_bytes(total_payload_bytes)} "
    f"elapsed={elapsed_s * 1000:.3f} ms "
    f"rate={format_rate(total_payload_bytes, elapsed_s)}"
  )


def submit_and_wait_with_timing(
    worker,
    finished_ops_queue,
    op: TransferOp,
    layout: KVCacheLayout,
    dtype: torch.dtype,
    label: str,
) -> int:
  payload_bytes = transfer_payload_bytes(layout, dtype, op.valid_block_num)
  start = time.perf_counter()
  worker.submit_transfer(op)
  completed_op_id = wait_queue_item(finished_ops_queue)
  elapsed_s = time.perf_counter() - start
  print_transfer_timing(label, payload_bytes, elapsed_s, op.valid_block_num)
  return completed_op_id


def benchmark_worker_transfer(
    worker,
    finished_ops_queue,
    transfer_type: TransferType,
    src_ids: list[int],
    dst_ids: list[int],
    layout: KVCacheLayout,
    dtype: torch.dtype,
    label: str,
    warmup: int = 2,
    repeats: int = 8,
) -> None:
  num_blocks = len(src_ids)
  payload_bytes_per_op = transfer_payload_bytes(layout, dtype, num_blocks)
  for index in range(warmup):
    op = make_op(transfer_type, src_ids, dst_ids, graph_id=9000 + index)
    worker.submit_transfer(op)
    assert wait_queue_item(finished_ops_queue) == op.op_id

  start = time.perf_counter()
  for index in range(repeats):
    op = make_op(transfer_type, src_ids, dst_ids, graph_id=10000 + index)
    worker.submit_transfer(op)
    assert wait_queue_item(finished_ops_queue) == op.op_id
  elapsed_s = time.perf_counter() - start
  print_transfer_benchmark(label, payload_bytes_per_op, repeats, elapsed_s, num_blocks)


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


def make_gpu_cpu_storage_handles(
    num_layers: int = 2,
    num_blocks: int = 4,
    tokens_per_block: int = 2,
    num_heads: int = 1,
    head_size: int = 4,
    dtype: torch.dtype = torch.float32,
):
  layout = make_layerfirst_layout(
    num_layers=num_layers,
    num_blocks=num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
  )
  layer_shape = layout.kv_shape[1:]
  gpu_layers = [
    torch.zeros(layer_shape, dtype=dtype, device="cuda")
    for _ in range(num_layers)
  ]
  # CE/C++ 路径要求 CPU 端是单块连续 LAYERFIRST 缓冲;按层切视图供校验。
  cpu_buf = torch.zeros(layout.kv_shape, dtype=dtype, device="cpu").contiguous().share_memory_()
  cpu_layers = [cpu_buf[layer_id] for layer_id in range(num_layers)]
  gpu_handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=gpu_layers,
    kv_layout=layout,
    dtype=dtype,
    gpu_device_id=0,
  )
  cpu_handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=cpu_buf.view(-1),
    kv_layout=layout,
    dtype=dtype,
  )
  return gpu_handle, cpu_handle, gpu_layers, cpu_layers


def fill_blocks(layers: list[torch.Tensor], base: int) -> None:
  for layer_id, layer in enumerate(layers):
    for block_id in range(layer.shape[1]):
      layer[:, block_id].fill_(float(base + layer_id * 100 + block_id))


def assert_transferred_blocks(
    layers: list[torch.Tensor],
    dst_to_src: dict[int, int],
    src_base: int,
    untouched_blocks: list[int],
) -> None:
  for layer_id, layer in enumerate(layers):
    for dst_block_id, src_block_id in dst_to_src.items():
      expected = float(src_base + layer_id * 100 + src_block_id)
      assert layer[:, dst_block_id].eq(expected).all(), (
        f"layer={layer_id} block={dst_block_id} expected {expected}"
      )
    for block_id in untouched_blocks:
      assert layer[:, block_id].eq(0).all(), (
        f"layer={layer_id} block={block_id} should remain zero"
      )


def create_gpucpu_worker_via_base(op_buffer_tensor, gpu_handle, cpu_handle):
  ctx = mp.get_context("spawn")
  finished_ops_queue = ctx.Queue()
  worker = TransferWorkerBase.create_worker.__func__(
    GPUCPUTransferWorker,
    ctx,
    finished_ops_queue,
    op_buffer_tensor,
    gpu_handle,
    cpu_handle,
  )
  assert worker.ready_event.wait(timeout=10), "worker did not become ready"
  assert worker.process.is_alive(), "worker process should be alive after ready"
  return worker, finished_ops_queue


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


def make_ssd_cpu_storage_handles(
    tmp_path: Path,
    dtype: torch.dtype = torch.float32,
    num_layers: int = 2,
    cpu_num_blocks: int = 4,
    ssd_num_blocks: int = 6,
    tokens_per_block: int = 2,
    num_heads: int = 1,
    head_size: int = 4,
    num_blocks_per_file: int = 3,
):
  cpu_layout = make_layerfirst_layout(
    num_layers=num_layers,
    num_blocks=cpu_num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
  )
  ssd_layout = make_blockfirst_layout(
    num_layers=num_layers,
    num_blocks=ssd_num_blocks,
    tokens_per_block=tokens_per_block,
    num_heads=num_heads,
    head_size=head_size,
  )
  cpu_tensor = torch.zeros(cpu_layout.kv_shape, dtype=dtype).contiguous().share_memory_()
  block_bytes = ssd_layout.get_elements_per_block() * dtype.itemsize
  file_paths = []
  for file_idx in range(2):
    file_path = tmp_path / f"ssd_worker_cache_{file_idx}.bin"
    with open(file_path, "wb") as file:
      file.truncate(num_blocks_per_file * block_bytes)
    file_paths.append(str(file_path))
  cpu_handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=cpu_tensor.view(-1),
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
  return cpu_handle, ssd_handle, cpu_tensor


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


def create_ssdcpu_worker_via_base(op_buffer_tensor, cpu_handle, ssd_handle):
  ctx = mp.get_context("spawn")
  finished_ops_queue = ctx.Queue()
  worker = TransferWorkerBase.create_worker.__func__(
    SSDCPUTransferWorker,
    ctx,
    finished_ops_queue,
    op_buffer_tensor,
    cpu_handle,
    ssd_handle,
    False,
  )
  assert worker.ready_event.wait(timeout=10), "SSDCPU worker did not become ready"
  assert worker.process.is_alive(), "SSDCPU worker process should be alive after ready"
  return worker, finished_ops_queue


def shutdown_worker(worker) -> None:
  worker.shutdown()
  assert not worker.process.is_alive(), "worker process should exit after shutdown"


def test_base_create_worker_submit_d2h_updates_cpu_tensor():
  require_cuda()
  op_buffer_tensor = torch.empty((4, 8), dtype=torch.int64).share_memory_()
  gpu_handle, cpu_handle, gpu_layers, cpu_layers = make_gpu_cpu_storage_handles()
  fill_blocks(gpu_layers, base=10)

  worker, finished_ops_queue = create_gpucpu_worker_via_base(
    op_buffer_tensor,
    gpu_handle,
    cpu_handle,
  )
  try:
    op = make_op(TransferType.D2H, src_ids=[0, 2], dst_ids=[3, 1], graph_id=1)

    assert submit_and_wait_with_timing(
      worker,
      finished_ops_queue,
      op,
      gpu_handle.kv_layout,
      gpu_handle.dtype,
      "GPUCPU D2H 跨进程端到端",
    ) == op.op_id
    assert_transferred_blocks(
      cpu_layers,
      dst_to_src={3: 0, 1: 2},
      src_base=10,
      untouched_blocks=[0, 2],
    )
  finally:
    shutdown_worker(worker)
    torch.cuda.empty_cache()


def test_base_create_worker_submit_h2d_updates_gpu_tensor():
  require_cuda()
  op_buffer_tensor = torch.empty((4, 8), dtype=torch.int64).share_memory_()
  gpu_handle, cpu_handle, gpu_layers, cpu_layers = make_gpu_cpu_storage_handles()
  fill_blocks(cpu_layers, base=100)

  worker, finished_ops_queue = create_gpucpu_worker_via_base(
    op_buffer_tensor,
    gpu_handle,
    cpu_handle,
  )
  try:
    op = make_op(TransferType.H2D, src_ids=[0, 2], dst_ids=[3, 1], graph_id=2)

    assert submit_and_wait_with_timing(
      worker,
      finished_ops_queue,
      op,
      cpu_handle.kv_layout,
      cpu_handle.dtype,
      "GPUCPU H2D 跨进程端到端",
    ) == op.op_id
    torch.cuda.synchronize()
    assert_transferred_blocks(
      gpu_layers,
      dst_to_src={3: 0, 1: 2},
      src_base=100,
      untouched_blocks=[0, 2],
    )
  finally:
    shutdown_worker(worker)
    torch.cuda.empty_cache()


def test_base_get_transfer_block_ids_uses_shared_op_slots_for_gpucpu():
  require_cuda()
  pool = SharedOpPool(max_op_num=4, max_block_num=8)
  src_slot = pool.allocate_slot(blocks([1, 3]))
  dst_slot = pool.allocate_slot(blocks([0, 2]))
  gpu_handle, cpu_handle, gpu_layers, cpu_layers = make_gpu_cpu_storage_handles()
  fill_blocks(cpu_layers, base=200)

  worker, finished_ops_queue = create_gpucpu_worker_via_base(
    pool.get_buffer(),
    gpu_handle,
    cpu_handle,
  )
  try:
    op = make_op(TransferType.H2D, src_ids=[0, 0], dst_ids=[0, 0], graph_id=3)
    op.src_slot_id = src_slot
    op.dst_slot_id = dst_slot
    op.valid_block_num = 2
    assert submit_and_wait_with_timing(
      worker,
      finished_ops_queue,
      op,
      cpu_handle.kv_layout,
      cpu_handle.dtype,
      "GPUCPU H2D SharedOpPool 端到端",
    ) == op.op_id
    torch.cuda.synchronize()
    assert_transferred_blocks(
      gpu_layers,
      dst_to_src={0: 1, 2: 3},
      src_base=200,
      untouched_blocks=[1, 3],
    )
  finally:
    shutdown_worker(worker)
    torch.cuda.empty_cache()


def test_base_shutdown_after_submitted_task_finishes_transfer():
  require_cuda()
  op_buffer_tensor = torch.empty((4, 8), dtype=torch.int64).share_memory_()
  gpu_handle, cpu_handle, gpu_layers, cpu_layers = make_gpu_cpu_storage_handles()
  fill_blocks(cpu_layers, base=300)

  worker, finished_ops_queue = create_gpucpu_worker_via_base(
    op_buffer_tensor,
    gpu_handle,
    cpu_handle,
  )
  op = make_op(TransferType.H2D, src_ids=[2], dst_ids=[1], graph_id=4)
  worker.submit_transfer(op)
  shutdown_worker(worker)

  assert wait_queue_item(finished_ops_queue) == op.op_id
  torch.cuda.synchronize()
  assert_transferred_blocks(
    gpu_layers,
    dst_to_src={1: 2},
    src_base=300,
    untouched_blocks=[0, 2, 3],
  )
  torch.cuda.empty_cache()


def test_base_shutdown_processes_multiple_queued_tasks_before_exit():
  require_cuda()
  op_buffer_tensor = torch.empty((4, 8), dtype=torch.int64).share_memory_()
  gpu_handle, cpu_handle, gpu_layers, cpu_layers = make_gpu_cpu_storage_handles()
  fill_blocks(cpu_layers, base=400)

  worker, finished_ops_queue = create_gpucpu_worker_via_base(
    op_buffer_tensor,
    gpu_handle,
    cpu_handle,
  )
  first_op = make_op(TransferType.H2D, src_ids=[0], dst_ids=[3], graph_id=5)
  second_op = make_op(TransferType.H2D, src_ids=[2], dst_ids=[1], graph_id=5)
  worker.submit_transfer(first_op)
  worker.submit_transfer(second_op)
  shutdown_worker(worker)

  assert wait_completed_ids(finished_ops_queue, 2) == [first_op.op_id, second_op.op_id]
  torch.cuda.synchronize()
  assert_transferred_blocks(
    gpu_layers,
    dst_to_src={3: 0, 1: 2},
    src_base=400,
    untouched_blocks=[0, 2],
  )
  torch.cuda.empty_cache()


def test_base_create_worker_submit_h2disk_and_disk2h_roundtrip_real_file():
  # 覆盖 SSDCPU worker 的真实跨进程 H2DISK -> DISK2H 文件读写与数据校验。
  with tempfile.TemporaryDirectory() as tmp_dir:
    op_buffer_tensor = torch.empty((4, 8), dtype=torch.int64).share_memory_()
    cpu_handle, ssd_handle, cpu_tensor = make_ssd_cpu_storage_handles(Path(tmp_dir))
    fill_cpu_tensor_blocks(cpu_tensor, base=500)
    original = cpu_tensor.clone()

    worker, finished_ops_queue = create_ssdcpu_worker_via_base(
      op_buffer_tensor,
      cpu_handle,
      ssd_handle,
    )
    try:
      write_op = make_op(TransferType.H2DISK, src_ids=[3, 1], dst_ids=[4, 0], graph_id=6)
      assert submit_and_wait_with_timing(
        worker,
        finished_ops_queue,
        write_op,
        cpu_handle.kv_layout,
        cpu_handle.dtype,
        "SSDCPU H2DISK 跨进程端到端",
      ) == write_op.op_id

      cpu_tensor.zero_()

      read_op = make_op(TransferType.DISK2H, src_ids=[4, 0], dst_ids=[0, 2], graph_id=6)
      assert submit_and_wait_with_timing(
        worker,
        finished_ops_queue,
        read_op,
        cpu_handle.kv_layout,
        cpu_handle.dtype,
        "SSDCPU DISK2H 跨进程端到端",
      ) == read_op.op_id

      assert_cpu_blocks_equal(
        cpu_tensor,
        original,
        dst_to_src={0: 3, 2: 1},
        zero_blocks=[1, 3],
      )
    finally:
      shutdown_worker(worker)


def test_base_get_transfer_block_ids_uses_shared_op_slots_for_ssdcpu_real_file():
  # 覆盖 SSDCPU worker 通过 SharedOpPool slot 读取 block id 后的真实文件传输。
  with tempfile.TemporaryDirectory() as tmp_dir:
    pool = SharedOpPool(max_op_num=4, max_block_num=8)
    cpu_handle, ssd_handle, cpu_tensor = make_ssd_cpu_storage_handles(Path(tmp_dir))
    fill_cpu_tensor_blocks(cpu_tensor, base=700)
    original = cpu_tensor.clone()

    worker, finished_ops_queue = create_ssdcpu_worker_via_base(
      pool.get_buffer(),
      cpu_handle,
      ssd_handle,
    )
    try:
      src_slot = pool.allocate_slot(blocks([2]))
      dst_slot = pool.allocate_slot(blocks([5]))
      write_op = make_op(TransferType.H2DISK, src_ids=[0], dst_ids=[0], graph_id=7)
      write_op.src_slot_id = src_slot
      write_op.dst_slot_id = dst_slot
      write_op.valid_block_num = 1
      assert submit_and_wait_with_timing(
        worker,
        finished_ops_queue,
        write_op,
        cpu_handle.kv_layout,
        cpu_handle.dtype,
        "SSDCPU H2DISK SharedOpPool 端到端",
      ) == write_op.op_id

      cpu_tensor.zero_()

      src_slot = pool.allocate_slot(blocks([5]))
      dst_slot = pool.allocate_slot(blocks([1]))
      read_op = make_op(TransferType.DISK2H, src_ids=[0], dst_ids=[0], graph_id=7)
      read_op.src_slot_id = src_slot
      read_op.dst_slot_id = dst_slot
      read_op.valid_block_num = 1
      assert submit_and_wait_with_timing(
        worker,
        finished_ops_queue,
        read_op,
        cpu_handle.kv_layout,
        cpu_handle.dtype,
        "SSDCPU DISK2H SharedOpPool 端到端",
      ) == read_op.op_id

      assert_cpu_blocks_equal(
        cpu_tensor,
        original,
        dst_to_src={1: 2},
        zero_blocks=[0, 2, 3],
      )
    finally:
      shutdown_worker(worker)


def test_base_gpucpu_end_to_end_benchmark_reports_throughput():
  # 使用较大的 payload 和多次重复，输出跨进程 GPU<->CPU 端到端吞吐。
  require_cuda()
  num_blocks = 512
  transfer_blocks = 64
  dtype = torch.float16
  op_buffer_tensor = torch.empty((4, transfer_blocks), dtype=torch.int64).share_memory_()
  gpu_handle, cpu_handle, gpu_layers, cpu_layers = make_gpu_cpu_storage_handles(
    num_layers=4,
    num_blocks=num_blocks,
    tokens_per_block=16,
    num_heads=8,
    head_size=64,
    dtype=dtype,
  )
  fill_blocks(cpu_layers, base=1000)
  fill_blocks(gpu_layers, base=2000)

  worker, finished_ops_queue = create_gpucpu_worker_via_base(
    op_buffer_tensor,
    gpu_handle,
    cpu_handle,
  )
  try:
    src_ids = list(range(transfer_blocks))
    dst_ids = list(range(num_blocks - transfer_blocks, num_blocks))
    print("    benchmark 说明：吞吐按逻辑 KV payload 计算，包含 Pipe/worker 调度开销")
    benchmark_worker_transfer(
      worker,
      finished_ops_queue,
      TransferType.H2D,
      src_ids,
      dst_ids,
      cpu_handle.kv_layout,
      cpu_handle.dtype,
      "GPUCPU H2D",
      warmup=2,
      repeats=8,
    )
    benchmark_worker_transfer(
      worker,
      finished_ops_queue,
      TransferType.D2H,
      dst_ids,
      src_ids,
      gpu_handle.kv_layout,
      gpu_handle.dtype,
      "GPUCPU D2H",
      warmup=2,
      repeats=8,
    )
  finally:
    shutdown_worker(worker)
    torch.cuda.empty_cache()


def test_base_ssdcpu_end_to_end_benchmark_reports_throughput():
  # 使用较大的 payload 和多次重复，输出跨进程 CPU<->SSD 端到端吞吐。
  with tempfile.TemporaryDirectory() as tmp_dir:
    transfer_blocks = 32
    dtype = torch.float16
    op_buffer_tensor = torch.empty((4, transfer_blocks), dtype=torch.int64).share_memory_()
    cpu_handle, ssd_handle, cpu_tensor = make_ssd_cpu_storage_handles(
      Path(tmp_dir),
      dtype=dtype,
      num_layers=4,
      cpu_num_blocks=128,
      ssd_num_blocks=128,
      tokens_per_block=16,
      num_heads=8,
      head_size=64,
      num_blocks_per_file=128,
    )
    fill_cpu_tensor_blocks(cpu_tensor, base=3000)

    worker, finished_ops_queue = create_ssdcpu_worker_via_base(
      op_buffer_tensor,
      cpu_handle,
      ssd_handle,
    )
    try:
      cpu_ids = list(range(transfer_blocks))
      ssd_ids = list(range(64, 64 + transfer_blocks))
      print("    benchmark 说明：吞吐按逻辑 KV payload 计算，包含 Pipe/worker/文件 IO 开销")
      benchmark_worker_transfer(
        worker,
        finished_ops_queue,
        TransferType.H2DISK,
        cpu_ids,
        ssd_ids,
        cpu_handle.kv_layout,
        cpu_handle.dtype,
        "SSDCPU H2DISK",
        warmup=1,
        repeats=4,
      )
      cpu_tensor.zero_()
      benchmark_worker_transfer(
        worker,
        finished_ops_queue,
        TransferType.DISK2H,
        ssd_ids,
        cpu_ids,
        cpu_handle.kv_layout,
        cpu_handle.dtype,
        "SSDCPU DISK2H",
        warmup=1,
        repeats=4,
      )
    finally:
      shutdown_worker(worker)


def test_base_shutdown_without_task_exits_worker_process():
  require_cuda()
  op_buffer_tensor = torch.empty((4, 8), dtype=torch.int64).share_memory_()
  gpu_handle, cpu_handle, _, _ = make_gpu_cpu_storage_handles()

  worker, _ = create_gpucpu_worker_via_base(
    op_buffer_tensor,
    gpu_handle,
    cpu_handle,
  )
  shutdown_worker(worker)
  torch.cuda.empty_cache()


TEST_CASES = [
  ("Base.create_worker + submit D2H：GPU block 写到 CPU block", test_base_create_worker_submit_d2h_updates_cpu_tensor),
  ("Base.create_worker + submit H2D：CPU block 写到 GPU block", test_base_create_worker_submit_h2d_updates_gpu_tensor),
  ("Base.get_transfer_block_ids：SharedOpPool slot 驱动真实 GPUCPU 传输", test_base_get_transfer_block_ids_uses_shared_op_slots_for_gpucpu),
  ("Base.shutdown：已提交任务完成传输后 worker 退出", test_base_shutdown_after_submitted_task_finishes_transfer),
  ("Base.shutdown：多个已提交任务完成传输后 worker 退出", test_base_shutdown_processes_multiple_queued_tasks_before_exit),
  ("Base.create_worker + submit H2DISK/DISK2H：SSDCPU 真实文件往返传输", test_base_create_worker_submit_h2disk_and_disk2h_roundtrip_real_file),
  ("Base.get_transfer_block_ids：SharedOpPool slot 驱动 SSDCPU 真实文件传输", test_base_get_transfer_block_ids_uses_shared_op_slots_for_ssdcpu_real_file),
  ("Base.benchmark：GPUCPU 跨进程端到端吞吐", test_base_gpucpu_end_to_end_benchmark_reports_throughput),
  ("Base.benchmark：SSDCPU 跨进程端到端吞吐", test_base_ssdcpu_end_to_end_benchmark_reports_throughput),
  ("Base.shutdown：无任务时 worker 直接退出", test_base_shutdown_without_task_exits_worker_process),
]


def run_all_tests():
  print("开始运行 Base + Worker 跨进程测试（小用例和 benchmark 都打印端到端速率）")
  passed = 0
  skipped = 0
  total = len(TEST_CASES)
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{total}] 开始：{name}")
    try:
      test_fn()
    except SkipTest as exc:
      skipped += 1
      print(f"[{index}/{total}] 跳过：{name}，原因：{exc}")
      continue
    except Exception as exc:
      print(f"[{index}/{total}] 失败：{name}，错误：{type(exc).__name__}: {exc}")
      raise
    passed += 1
    print(f"[{index}/{total}] 通过：{name}")
  print(f"Base + Worker 跨进程测试完成：通过 {passed}/{total}，跳过 {skipped}/{total}")


if __name__ == "__main__":
  run_all_tests()

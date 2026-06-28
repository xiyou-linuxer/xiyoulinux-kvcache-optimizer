# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：独立进程 TransferManager 的真实 GPU↔CPU↔SSD 往返集成测试。

import os
import tempfile
import time
import traceback

import numpy as np
import torch

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.storage import KVCacheLayout, KVCacheLayoutType
from miniflex.common.transfer import CompletedOp, TransferOp, TransferOpGraph, TransferType
from miniflex.server.client import MiniFlexGPURegisterClient
from miniflex.server.utils import normalize_zmq_endpoint
from miniflex.transfer_manager import TransferManagerHandle


def blocks(ids: list[int]) -> np.ndarray:
  """把 Python block id 列表转换成 TransferOp 需要的 int64 numpy 数组。"""
  return np.array(ids, dtype=np.int64)


def make_model(dtype: torch.dtype = torch.float32) -> ModelConfig:
  """构造测试用的小模型配置，保证真实传输数据量足够小。"""
  return ModelConfig(
    num_layers=2,
    num_kv_heads=1,
    head_size=2,
    dtype=dtype,
  )


def make_cache_config(cache_dir: str) -> CacheConfig:
  """构造启用 SSD 的测试缓存配置，覆盖 CPU/SSD 两层真实 worker。"""
  return CacheConfig(
    tokens_per_block=2,
    enable_ssd=True,
    num_cpu_blocks=4,
    num_ssd_blocks=4,
    ssd_cache_dir=cache_dir,
    ssd_file_prefix="transfer_manager_test",
    use_direct_io=False,
  )


def make_gpu_layout(model: ModelConfig, cache: CacheConfig, num_blocks: int = 4) -> KVCacheLayout:
  """根据测试模型和 cache 配置构造 vLLM 侧 GPU KV cache layout。"""
  return KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=model.num_layers,
    num_blocks=num_blocks,
    tokens_per_block=cache.tokens_per_block,
    num_heads=model.num_kv_heads,
    head_size=model.head_size,
    use_mla=model.use_mla,
  )


def make_gpu_layer_tensors(layout: KVCacheLayout, dtype: torch.dtype) -> list[torch.Tensor]:
  """创建分层 GPU KV tensor，并给每个 block 写入可校验的唯一值。"""
  layer_shape = tuple(layout.kv_shape[1:])
  tensors = [
    torch.empty(layer_shape, device="cuda:0", dtype=dtype)
    for _ in range(layout.num_layers)
  ]
  for layer_id, tensor in enumerate(tensors):
    for block_id in range(layout.num_blocks):
      value = float(1000 + layer_id * 100 + block_id)
      tensor[:, block_id].fill_(value)
  return tensors


def register_gpu_blocks(gpu_register_port: str, gpu_blocks: list[torch.Tensor], layout: KVCacheLayout) -> None:
  """通过 MiniFlexGPURegisterClient 向 TransferManager 注册 GPU KV cache。"""
  client = MiniFlexGPURegisterClient(gpu_register_port, device_id=0, dp_rank=0, tp_rank=0)
  try:
    client.register_to_server(gpu_blocks, layout)
    time.sleep(0.1)
  finally:
    client.close()


def wait_until_ready(handle: TransferManagerHandle, timeout: float = 20.0) -> None:
  """等待独立进程 TransferManager 完成 GPU 注册和 TransferEngine/worker 初始化。"""
  deadline = time.time() + timeout
  while time.time() < deadline:
    if handle.is_ready():
      return
    time.sleep(0.05)
  raise AssertionError("TransferManagerHandle 未在超时时间内 ready")


def wait_for_graph_completed(
    handle: TransferManagerHandle,
    graph_id: int,
    timeout: float = 30.0,
) -> list[CompletedOp]:
  """轮询 TransferManagerHandle.wait，直到指定 graph 上报完成或超时。"""
  completed: list[CompletedOp] = []
  deadline = time.time() + timeout
  while time.time() < deadline:
    completed.extend(handle.wait(timeout=0.1))
    if any(op.graph_id == graph_id and op.is_graph_completed() for op in completed):
      return completed
    time.sleep(0.01)
  raise AssertionError(f"等待 graph {graph_id} 完成超时，completed={completed}")


def make_gpu_cpu_ssd_roundtrip_graph() -> tuple[TransferOpGraph, list[TransferOp]]:
  """构造 D2H -> H2DISK -> DISK2H -> H2D 的真实跨层往返传输图。"""
  graph = TransferOpGraph()
  d2h = TransferOp(
    transfer_type=TransferType.D2H,
    graph_id=graph.graph_id,
    src_block_ids=blocks([1, 3]),
    dst_block_ids=blocks([0, 1]),
  )
  h2disk = TransferOp(
    transfer_type=TransferType.H2DISK,
    graph_id=graph.graph_id,
    src_block_ids=blocks([0, 1]),
    dst_block_ids=blocks([2, 0]),
  )
  disk2h = TransferOp(
    transfer_type=TransferType.DISK2H,
    graph_id=graph.graph_id,
    src_block_ids=blocks([2, 0]),
    dst_block_ids=blocks([2, 3]),
  )
  h2d = TransferOp(
    transfer_type=TransferType.H2D,
    graph_id=graph.graph_id,
    src_block_ids=blocks([2, 3]),
    dst_block_ids=blocks([0, 2]),
  )

  for op in (d2h, h2disk, disk2h, h2d):
    graph.add_transfer_op(op)
  graph.add_dependency(h2disk.op_id, d2h.op_id)
  graph.add_dependency(disk2h.op_id, h2disk.op_id)
  graph.add_dependency(h2d.op_id, disk2h.op_id)
  return graph, [d2h, h2disk, disk2h, h2d]


def assert_gpu_blocks_equal(
    tensors: list[torch.Tensor],
    original_tensors: list[torch.Tensor],
    dst_to_src: dict[int, int],
) -> None:
  """校验目标 GPU block 内容等于传输前指定源 block 的内容。"""
  torch.cuda.synchronize()
  for layer_id, tensor in enumerate(tensors):
    for dst_block_id, src_block_id in dst_to_src.items():
      actual = tensor[:, dst_block_id].detach().cpu()
      expected = original_tensors[layer_id][:, src_block_id].detach().cpu()
      assert torch.equal(actual, expected), (
        f"layer={layer_id} 的 dst_block={dst_block_id} 应该等于 src_block={src_block_id}"
      )


def cleanup_ipc_endpoint(gpu_register_port: str) -> None:
  """清理测试生成的 IPC socket 路径，避免后续测试复用旧文件。"""
  endpoint = normalize_zmq_endpoint(gpu_register_port)
  if endpoint.startswith("ipc://"):
    try:
      os.unlink(endpoint[len("ipc://"):])
    except FileNotFoundError:
      pass


def test_inter_process_manager_real_gpu_cpu_ssd_roundtrip():
  """覆盖独立进程 TransferManager、ZMQ GPU 注册和真实 GPU/CPU/SSD worker 往返传输。"""
  if not torch.cuda.is_available():
    print("跳过：CUDA 不可用，真实 GPU 传输测试需要 CUDA")
    return

  torch.cuda.set_device(0)
  dtype = torch.float32
  with tempfile.TemporaryDirectory() as cache_dir:
    model = make_model(dtype)
    cache = make_cache_config(cache_dir)
    gpu_layout = make_gpu_layout(model, cache)
    gpu_tensors = make_gpu_layer_tensors(gpu_layout, dtype)
    original_tensors = [tensor.clone() for tensor in gpu_tensors]

    for tensor in gpu_tensors:
      tensor[:, 0].zero_()
      tensor[:, 2].zero_()

    manager = TransferManagerHandle(model, cache, mode="process")
    try:
      manager.start()
      register_gpu_blocks(
        manager.gpu_register_port,
        gpu_tensors,
        gpu_layout,
      )
      wait_until_ready(manager)

      graph, ops = make_gpu_cpu_ssd_roundtrip_graph()
      manager.submit(graph)
      completed = wait_for_graph_completed(manager, graph.graph_id)
      completed_tuples = {op.to_tuple() for op in completed}
      assert (graph.graph_id, ops[-1].op_id) in completed_tuples
      assert (graph.graph_id, -1) in completed_tuples

      assert_gpu_blocks_equal(
        gpu_tensors,
        original_tensors,
        dst_to_src={0: 1, 2: 3},
      )
    finally:
      manager.shutdown()
      cleanup_ipc_endpoint(manager.gpu_register_port)


TEST_CASES = [
  ("独立进程 TransferManager 真实 GPU/CPU/SSD 往返传输", test_inter_process_manager_real_gpu_cpu_ssd_roundtrip),
]


def run_all_tests():
  print("开始运行 TransferManager 测试")
  total = len(TEST_CASES)
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{total}] 开始：{name}")
    try:
      test_fn()
    except Exception as exc:
      print(f"[{index}/{total}] 失败：{name}，错误：{type(exc).__name__}: {exc}")
      traceback.print_exc()
      raise
    print(f"[{index}/{total}] 通过：{name}")
  print(f"TransferManager 测试完成：通过 {total}/{total}")


if __name__ == "__main__":
  run_all_tests()

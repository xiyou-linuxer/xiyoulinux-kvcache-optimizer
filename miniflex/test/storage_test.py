# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：KVCacheLayout 三种布局（LAYERFIRST/BLOCKFIRST/MLA）的 shape 与 stride、StorageHandle（tensor/handle/file）的 getter 与校验、SSD allocator 建文件，以及 TensorSharedHandle 的校验与跨进程 CUDA tensor 共享。

import traceback
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp
import zmq

from miniflex.storage.allocator import SSDStorageAllocator
from miniflex.common.memory_handle import TensorSharedHandle
from miniflex.common.storage import (
  KVCacheLayout,
  KVCacheLayoutType,
  StorageHandle,
  StorageHandlerType,
)


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def make_layout(layout_type=KVCacheLayoutType.LAYERFIRST, dtype=torch.float16):
  del dtype
  return KVCacheLayout(
    layout_type=layout_type,
    num_layers=2,
    num_blocks=3,
    tokens_per_block=4,
    num_heads=2,
    head_size=8,
    use_mla=False,
  )


def make_fake_tensor_handle(tensor: torch.Tensor) -> TensorSharedHandle:
  handle = object.__new__(TensorSharedHandle)
  handle.rebuild_func = None
  handle.rebuild_args = None
  handle.device = tensor.device
  handle.use_direct_ipc = False
  handle.ipc_handle = None
  handle.tensor_shape = tuple(tensor.shape)
  handle.tensor_dtype = tensor.dtype
  handle.tensor_numel = tensor.numel()
  handle.offset = 0
  handle._cached_tensor = tensor
  handle.get_tensor = lambda: tensor
  return handle


def cuda_available() -> bool:
  return torch.cuda.is_available() and torch.cuda.device_count() > 0


def test_layerfirst_layout_shape_and_strides():
  layout = make_layout(KVCacheLayoutType.LAYERFIRST)

  assert tuple(layout.kv_shape) == (2, 2, 3, 4, 2, 8)
  assert layout.kv_dim == 2
  assert layout.get_chunk_size() == 4 * 2 * 8
  assert layout.get_total_elements() == 2 * 2 * 3 * 4 * 2 * 8
  assert layout.get_elements_per_block() == 2 * 2 * 4 * 2 * 8
  assert layout.get_layer_stride() == 2 * 3 * 4 * 2 * 8
  assert layout.get_block_stride() == 4 * 2 * 8
  assert layout.get_kv_stride() == 3 * 4 * 2 * 8


def test_blockfirst_layout_shape_and_strides():
  layout = make_layout(KVCacheLayoutType.BLOCKFIRST)

  assert tuple(layout.kv_shape) == (3, 2, 2, 4, 2, 8)
  assert layout.get_layer_stride() == 2 * 4 * 2 * 8
  assert layout.get_block_stride() == 2 * 2 * 4 * 2 * 8
  assert layout.get_kv_stride() == 4 * 2 * 8


def test_mla_layout_and_derivatives():
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=4,
    num_blocks=6,
    tokens_per_block=8,
    num_heads=1,
    head_size=16,
    use_mla=True,
  )

  assert tuple(layout.kv_shape) == (4, 1, 6, 8, 1, 16)
  assert layout.kv_dim == 1
  assert layout.with_num_blocks(2).num_blocks == 2
  assert layout.div_blocks(3).num_blocks == 2
  assert layout.div_blocks(4, padding=True).num_blocks == 2
  assert layout.div_layers(2).num_layers == 2
  assert layout.div_heads(1).num_heads == 1

  assert_raises(ValueError, lambda: layout.with_num_blocks(-1))
  assert_raises(ValueError, lambda: layout.div_blocks(4))
  assert_raises(ValueError, lambda: layout.div_layers(3))
  assert_raises(ValueError, lambda: layout.div_heads(2))


def test_layout_validation():
  valid_kwargs = dict(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=1,
    num_blocks=1,
    tokens_per_block=2,
    num_heads=1,
    head_size=1,
    use_mla=False,
  )

  assert_raises(ValueError, lambda: KVCacheLayout(**{**valid_kwargs, "layout_type": "bad"}))
  assert_raises(ValueError, lambda: KVCacheLayout(**{**valid_kwargs, "num_layers": 0}))
  assert_raises(ValueError, lambda: KVCacheLayout(**{**valid_kwargs, "num_blocks": -1}))
  assert_raises(ValueError, lambda: KVCacheLayout(**{**valid_kwargs, "tokens_per_block": 0}))
  assert_raises(ValueError, lambda: KVCacheLayout(**{**valid_kwargs, "num_heads": 0}))
  assert_raises(ValueError, lambda: KVCacheLayout(**{**valid_kwargs, "head_size": 0}))
  assert_raises(ValueError, lambda: KVCacheLayout(**{**valid_kwargs, "num_heads": 2, "use_mla": True}))

  empty_layout = KVCacheLayout(**{**valid_kwargs, "num_blocks": 0})
  assert tuple(empty_layout.kv_shape) == (1, 2, 0, 2, 1, 1)
  assert empty_layout.get_total_elements() == 0
  assert empty_layout.get_elements_per_block() == 4


def test_tensor_storage_single_tensor():
  layout = make_layout()
  tensor = torch.empty(layout.get_total_elements(), dtype=torch.bfloat16)
  handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=tensor,
    kv_layout=layout,
    dtype=torch.bfloat16,
  )

  assert handle.get_tensor() is tensor
  assert_raises(ValueError, handle.get_tensor_list)
  assert_raises(ValueError, handle.get_tensor_handle_list)
  assert_raises(ValueError, handle.get_file_list)


def test_tensor_storage_tensor_list():
  layout = make_layout()
  tensors = [
    torch.empty(tuple(layout.kv_shape[1:]), dtype=torch.float16),
    torch.empty(tuple(layout.kv_shape[1:]), dtype=torch.float16),
  ]
  handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=tensors,
    kv_layout=layout,
    dtype=torch.float16,
  )

  assert handle.get_tensor_list() is tensors
  assert_raises(ValueError, handle.get_tensor)
  assert_raises(ValueError, handle.get_file_list)
  assert_raises(ValueError, handle.get_tensor_handle_list)


def test_tensor_handle_storage_flexkv_like_getters():
  layout = make_layout()
  tensor = torch.empty(tuple(layout.kv_shape[1:]), dtype=torch.float16)
  tensor_handle = make_fake_tensor_handle(tensor)
  handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR_HANDLE,
    data=[tensor_handle],
    kv_layout=layout,
    dtype=torch.float16,
    gpu_device_id=0,
  )

  assert handle.get_tensor_handle_list() == [tensor_handle]
  assert handle.get_tensor_list() == [tensor]
  assert_raises(ValueError, handle.get_tensor)
  assert_raises(ValueError, handle.get_file_list)


def test_file_storage_handle():
  layout = make_layout(KVCacheLayoutType.BLOCKFIRST)
  files = ["/tmp/miniflex_storage_0.bin", "/tmp/miniflex_storage_1.bin"]
  handle = StorageHandle(
    handle_type=StorageHandlerType.FILE,
    data=files,
    kv_layout=layout,
    dtype=torch.float16,
    num_blocks_per_file=2,
  )

  assert handle.get_file_list() is files
  assert_raises(ValueError, handle.get_tensor)
  assert_raises(ValueError, handle.get_tensor_list)
  assert_raises(ValueError, handle.get_tensor_handle_list)


def test_storage_handle_validation():
  layout = make_layout()

  assert_raises(
    ValueError,
    lambda: StorageHandle(
      handle_type="tensor",
      data=torch.empty(1, dtype=torch.float16),
      kv_layout=layout,
      dtype=torch.float16,
    ),
  )
  assert_raises(
    ValueError,
    lambda: StorageHandle(
      handle_type=StorageHandlerType.TENSOR,
      data=torch.empty(1, dtype=torch.float32),
      kv_layout=layout,
      dtype=torch.float16,
    ),
  )
  assert_raises(
    ValueError,
    lambda: StorageHandle(
      handle_type=StorageHandlerType.TENSOR,
      data=[],
      kv_layout=layout,
      dtype=torch.float16,
    ),
  )
  assert_raises(
    ValueError,
    lambda: StorageHandle(
      handle_type=StorageHandlerType.TENSOR_HANDLE,
      data=[make_fake_tensor_handle(torch.empty(1, dtype=torch.float32))],
      kv_layout=layout,
      dtype=torch.float16,
    ),
  )
  assert_raises(
    ValueError,
    lambda: StorageHandle(
      handle_type=StorageHandlerType.FILE,
      data=[],
      kv_layout=layout,
      dtype=torch.float16,
    ),
  )
  assert_raises(
    ValueError,
    lambda: StorageHandle(
      handle_type=StorageHandlerType.FILE,
      data=["/tmp/cache.bin"],
      kv_layout=layout,
      dtype=torch.float16,
      num_blocks_per_file=0,
    ),
  )
  assert_raises(
    ValueError,
    lambda: StorageHandle(
      handle_type=StorageHandlerType.TENSOR,
      data=torch.empty(1, dtype=torch.float16),
      kv_layout=layout,
      dtype=torch.float16,
      gpu_device_id=-1,
    ),
  )


def test_file_storage_allocator_creates_cache_files():
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.BLOCKFIRST,
    num_layers=1,
    num_blocks=4,
    tokens_per_block=2,
    num_heads=1,
    head_size=2,
    use_mla=False,
  )
  block_size = layout.get_elements_per_block() * torch.float16.itemsize

  with tempfile.TemporaryDirectory() as cache_dir:
    handle = SSDStorageAllocator.allocate(
      layout,
      torch.float16,
      cache_dir=cache_dir,
      file_prefix="miniflex_test",
      max_file_size_gb=-1,
    )

    files = handle.get_file_list()
    assert handle.handle_type == StorageHandlerType.FILE
    assert handle.kv_layout == layout
    assert handle.dtype == torch.float16
    assert handle.num_blocks_per_file == layout.num_blocks
    assert len(files) == 1
    assert Path(files[0]).name == "miniflex_test_0_0.bin"
    assert Path(files[0]).is_file()
    assert Path(files[0]).stat().st_size == block_size * layout.num_blocks


def test_file_storage_allocator_multiple_dirs_and_raw_data():
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.BLOCKFIRST,
    num_layers=1,
    num_blocks=4,
    tokens_per_block=1,
    num_heads=1,
    head_size=1,
    use_mla=False,
  )
  block_size = layout.get_elements_per_block() * torch.float32.itemsize

  with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
    handle = SSDStorageAllocator.allocate(
      layout,
      torch.float32,
      cache_dir=[first_dir, second_dir],
      file_prefix="split",
      max_file_size_gb=-1,
    )

    files = handle.get_file_list()
    assert len(files) == 2
    assert handle.num_blocks_per_file == 2
    assert Path(files[0]).parent == Path(first_dir)
    assert Path(files[1]).parent == Path(second_dir)
    assert Path(files[0]).stat().st_size == block_size * 2
    assert Path(files[1]).stat().st_size == block_size * 2

    raw_handle = SSDStorageAllocator.from_raw_data(
      files,
      layout,
      torch.float32,
      num_blocks_per_file=2,
    )
    assert raw_handle.get_file_list() == files
    assert raw_handle.num_blocks_per_file == 2


def test_file_storage_allocator_validation():
  layout = make_layout(KVCacheLayoutType.BLOCKFIRST)

  assert_raises(ValueError, lambda: SSDStorageAllocator.allocate(layout, torch.float16))
  assert_raises(ValueError, lambda: SSDStorageAllocator.allocate(layout, torch.float16, cache_dir=[]))
  assert_raises(ValueError, lambda: SSDStorageAllocator.allocate(layout, torch.float16, cache_dir="/tmp", file_prefix=123))
  assert_raises(ValueError, lambda: SSDStorageAllocator.allocate(layout, torch.float16, cache_dir=["/tmp", "/tmp"]))
  assert_raises(
    ValueError,
    lambda: SSDStorageAllocator.from_raw_data(
      ["/tmp/miniflex_storage_0.bin"],
      layout,
      torch.float16,
      num_blocks_per_file=1,
    ),
  )
  assert_raises(
    ValueError,
    lambda: SSDStorageAllocator.from_raw_data(
      ["/tmp/miniflex_storage_0.bin"],
      layout,
      torch.float16,
    ),
  )


def test_tensor_shared_handle_cpu_safe_validation():
  assert_raises(ValueError, lambda: TensorSharedHandle(torch.empty(1)))
  assert_raises(NotImplementedError, lambda: TensorSharedHandle(bytes(64), device_id=0))
  assert_raises(NotImplementedError, lambda: TensorSharedHandle(torch.empty(1), force_direct_ipc=True))


def _cuda_storage_worker(request_endpoint, response_endpoint, device_id):
  context = None
  recv_socket = None
  send_socket = None
  try:
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.setsockopt(zmq.LINGER, 0)
    recv_socket.connect(request_endpoint)
    send_socket = context.socket(zmq.PUSH)
    send_socket.setsockopt(zmq.LINGER, 0)
    send_socket.connect(response_endpoint)

    msg = recv_socket.recv_pyobj()
    storage_handle = msg["storage_handle"]
    expected_before = msg["expected_before"]
    delta = msg["delta"]

    torch.cuda.set_device(device_id)
    tensors = storage_handle.get_tensor_list()
    if len(tensors) != 1:
      raise AssertionError(f"expected one tensor, got {len(tensors)}")
    tensor = tensors[0]
    before = tensor.detach().cpu().tolist()
    if before != expected_before:
      raise AssertionError(f"unexpected child tensor before mutation: {before}")
    tensor.add_(delta)
    torch.cuda.synchronize(device_id)
    after = tensor.detach().cpu().tolist()
    del tensor
    del tensors
    del storage_handle
    torch.cuda.ipc_collect()
    send_socket.send_pyobj(("ok", before, after))
  except Exception:
    if send_socket is not None:
      send_socket.send_pyobj(("error", traceback.format_exc()))
    else:
      raise
  finally:
    if recv_socket is not None:
      recv_socket.close(0)
    if send_socket is not None:
      send_socket.close(0)
    if context is not None:
      context.term()


def test_cuda_tensor_handle_metadata_and_conversion():
  if not cuda_available():
    print("跳过：当前环境没有 CUDA")
    return

  device_id = 0
  device = torch.device(f"cuda:{device_id}")
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=1,
    num_blocks=1,
    tokens_per_block=1,
    num_heads=1,
    head_size=4,
    use_mla=False,
  )
  tensor = torch.arange(4, dtype=torch.float32, device=device)
  handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR,
    data=[tensor],
    kv_layout=layout,
    dtype=torch.float32,
    gpu_device_id=device_id,
  )

  tensor_handles = handle.get_tensor_handle_list()
  assert len(tensor_handles) == 1
  assert isinstance(tensor_handles[0], TensorSharedHandle)
  assert tensor_handles[0].device == device
  assert tensor_handles[0].tensor_shape == (4,)
  assert tensor_handles[0].tensor_dtype == torch.float32
  assert tensor_handles[0].tensor_numel == 4
  assert tensor_handles[0].use_direct_ipc is False


def test_multiprocess_tensor_handle_storage_shares_cuda_tensor():
  if not cuda_available():
    print("跳过：当前环境没有 CUDA")
    return

  device_id = 0
  device = torch.device(f"cuda:{device_id}")
  layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=1,
    num_blocks=1,
    tokens_per_block=1,
    num_heads=1,
    head_size=4,
    use_mla=False,
  )
  tensor = torch.arange(4, dtype=torch.float32, device=device)
  expected_before = [0.0, 1.0, 2.0, 3.0]
  delta = 10.0

  tensor_handle = TensorSharedHandle(tensor, device_id=device_id)
  storage_handle = StorageHandle(
    handle_type=StorageHandlerType.TENSOR_HANDLE,
    data=[tensor_handle],
    kv_layout=layout,
    dtype=torch.float32,
    gpu_device_id=device_id,
  )

  context = zmq.Context()
  parent_send_socket = context.socket(zmq.PUSH)
  parent_send_socket.setsockopt(zmq.LINGER, 0)
  request_port = parent_send_socket.bind_to_random_port("tcp://127.0.0.1")
  request_endpoint = f"tcp://127.0.0.1:{request_port}"
  parent_recv_socket = context.socket(zmq.PULL)
  parent_recv_socket.setsockopt(zmq.LINGER, 0)
  response_port = parent_recv_socket.bind_to_random_port("tcp://127.0.0.1")
  response_endpoint = f"tcp://127.0.0.1:{response_port}"

  ctx = mp.get_context("spawn")
  process = ctx.Process(
    target=_cuda_storage_worker,
    args=(request_endpoint, response_endpoint, device_id),
  )
  process.start()
  parent_send_socket.send_pyobj({
    "storage_handle": storage_handle,
    "expected_before": expected_before,
    "delta": delta,
  })
  try:
    if not parent_recv_socket.poll(60000):
      raise TimeoutError("child process timed out while validating shared CUDA tensor")
    status, *payload = parent_recv_socket.recv_pyobj()
  except TimeoutError as exc:
    process.terminate()
    process.join(timeout=10)
    raise AssertionError(str(exc)) from exc
  process.join(timeout=30)
  if process.is_alive():
    process.terminate()
    process.join(timeout=10)
    raise AssertionError("child process did not exit")
  if process.exitcode != 0:
    raise AssertionError(f"child process exited with code {process.exitcode}")
  if status != "ok":
    raise AssertionError(payload[0])

  before, after = payload
  assert before == expected_before
  assert after == [value + delta for value in expected_before]
  torch.cuda.synchronize(device_id)
  assert torch.allclose(
    tensor.detach().cpu(),
    torch.tensor(after, dtype=torch.float32),
  )
  del storage_handle
  del tensor_handle
  torch.cuda.synchronize(device_id)
  torch.cuda.ipc_collect()
  parent_send_socket.close(0)
  parent_recv_socket.close(0)
  context.term()


TEST_CASES = [
  ("LAYERFIRST layout shape 和 stride", test_layerfirst_layout_shape_and_strides),
  ("BLOCKFIRST layout shape 和 stride", test_blockfirst_layout_shape_and_strides),
  ("MLA layout 和派生 layout", test_mla_layout_and_derivatives),
  ("layout 参数校验", test_layout_validation),
  ("Tensor 单对象 storage handle", test_tensor_storage_single_tensor),
  ("Tensor list storage handle", test_tensor_storage_tensor_list),
  ("TensorHandle storage handle 的 FlexKV getter 行为", test_tensor_handle_storage_flexkv_like_getters),
  ("File storage handle", test_file_storage_handle),
  ("StorageHandle 参数校验", test_storage_handle_validation),
  ("SSDStorageAllocator 创建 SSD cache 文件", test_file_storage_allocator_creates_cache_files),
  ("SSDStorageAllocator 多目录和 raw data 注册", test_file_storage_allocator_multiple_dirs_and_raw_data),
  ("SSDStorageAllocator 参数校验", test_file_storage_allocator_validation),
  ("TensorSharedHandle CPU-safe 参数校验", test_tensor_shared_handle_cpu_safe_validation),
  ("CUDA Tensor 转 TensorSharedHandle 元数据", test_cuda_tensor_handle_metadata_and_conversion),
  ("多进程 CUDA TensorSharedHandle 共享与修改验证", test_multiprocess_tensor_handle_storage_shares_cuda_tensor),
]


def run_all_tests():
  print("开始运行 Storage 测试")
  passed = 0
  total = len(TEST_CASES)
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{total}] 开始：{name}")
    try:
      test_fn()
    except Exception as exc:
      print(f"[{index}/{total}] 失败：{name}，错误：{type(exc).__name__}: {exc}")
      raise
    passed += 1
    print(f"[{index}/{total}] 通过：{name}")
  print(f"Storage 测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

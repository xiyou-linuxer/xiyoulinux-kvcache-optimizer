import tempfile
import traceback
from pathlib import Path

import torch

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.memory_handle import TensorSharedHandle
from miniflex.common.storage import KVCacheLayout, KVCacheLayoutType, StorageHandlerType
from miniflex.common.transfer import DeviceType
from miniflex.storage.storage_engine import StorageEngine


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def make_model(dtype=torch.float16, use_mla=False):
  return ModelConfig(
    num_layers=2,
    num_kv_heads=1 if use_mla else 2,
    head_size=8,
    use_mla=use_mla,
    dtype=dtype,
  )


def make_gpu_layout(layout_type=KVCacheLayoutType.LAYERFIRST, num_blocks=3, dtype=torch.float16):
  del dtype
  return KVCacheLayout(
    layout_type=layout_type,
    num_layers=2,
    num_blocks=num_blocks,
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


def test_cpu_storage_initialized():
  model = make_model()
  cache = CacheConfig(tokens_per_block=4, num_cpu_blocks=3)
  engine = StorageEngine(cache, model)

  assert engine.has_storage_handle(DeviceType.CPU)
  handle = engine.get_storage_handle(DeviceType.CPU)
  assert handle.handle_type == StorageHandlerType.TENSOR
  assert handle.dtype == model.dtype
  assert handle.kv_layout.layout_type == KVCacheLayoutType.LAYERFIRST
  assert handle.kv_layout.num_blocks == cache.num_cpu_blocks
  assert handle.kv_layout.tokens_per_block == cache.tokens_per_block
  assert handle.get_tensor().numel() == handle.kv_layout.get_total_elements()
  assert not engine.has_storage_handle(DeviceType.SSD)
  assert_raises(ValueError, lambda: engine.get_storage_handle(DeviceType.SSD))


def test_ssd_storage_initialized_and_prefix_used():
  model = make_model()
  with tempfile.TemporaryDirectory() as cache_dir:
    cache = CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
      ssd_cache_dir=cache_dir,
      ssd_file_prefix="engine_test",
    )
    engine = StorageEngine(cache, model)

    cpu_handle = engine.get_storage_handle(DeviceType.CPU)
    ssd_handle = engine.get_storage_handle(DeviceType.SSD)
    assert cpu_handle.kv_layout.layout_type == KVCacheLayoutType.LAYERFIRST
    assert ssd_handle.kv_layout.layout_type == KVCacheLayoutType.BLOCKFIRST
    assert ssd_handle.handle_type == StorageHandlerType.FILE
    assert ssd_handle.kv_layout.num_blocks == cache.num_ssd_blocks
    files = ssd_handle.get_file_list()
    assert len(files) == 1
    assert Path(files[0]).name.startswith("engine_test_")
    assert Path(files[0]).exists()
    assert ssd_handle.num_blocks_per_file == cache.num_ssd_blocks


def test_ssd_storage_initialized_with_cache_dir_list():
  model = make_model()
  with tempfile.TemporaryDirectory() as cache_dir_0, tempfile.TemporaryDirectory() as cache_dir_1:
    cache = CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
      ssd_cache_dir=[cache_dir_0, cache_dir_1],
      ssd_file_prefix="engine_list_test",
    )
    engine = StorageEngine(cache, model)

    ssd_handle = engine.get_storage_handle(DeviceType.SSD)
    files = ssd_handle.get_file_list()
    assert len(files) == 2
    assert Path(files[0]).parent == Path(cache_dir_0)
    assert Path(files[1]).parent == Path(cache_dir_1)
    assert all(Path(file).name.startswith("engine_list_test_") for file in files)
    assert ssd_handle.num_blocks_per_file == cache.num_ssd_blocks // 2


def test_ssd_config_validation():
  assert_raises(
    ValueError,
    lambda: CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
    ),
  )
  assert_raises(
    ValueError,
    lambda: CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
      ssd_cache_dir=[],
    ),
  )
  assert_raises(
    ValueError,
    lambda: CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
      ssd_cache_dir=["/tmp", ""],
    ),
  )
  assert_raises(
    ValueError,
    lambda: CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
      ssd_cache_dir="/tmp",
      ssd_file_prefix="",
    ),
  )
  assert_raises(
    ValueError,
    lambda: CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
      ssd_cache_dir="/tmp",
      ssd_max_file_size_gb=0,
    ),
  )


def test_ssd_layout_can_differ_from_cpu_layout():
  with tempfile.TemporaryDirectory() as cache_dir:
    cache = CacheConfig(
      tokens_per_block=4,
      enable_ssd=True,
      num_cpu_blocks=3,
      num_ssd_blocks=4,
      ssd_cache_dir=cache_dir,
      cpu_layout_type=KVCacheLayoutType.LAYERFIRST,
      ssd_layout_type=KVCacheLayoutType.BLOCKFIRST,
    )
    engine = StorageEngine(cache, make_model())
    cpu_handle = engine.get_storage_handle(DeviceType.CPU)
    ssd_handle = engine.get_storage_handle(DeviceType.SSD)
    assert cpu_handle.kv_layout.layout_type == KVCacheLayoutType.LAYERFIRST
    assert ssd_handle.kv_layout.layout_type == KVCacheLayoutType.BLOCKFIRST

  cache = CacheConfig(
    tokens_per_block=4,
    enable_ssd=False,
    num_cpu_blocks=3,
    num_ssd_blocks=4,
    cpu_layout_type=KVCacheLayoutType.LAYERFIRST,
    ssd_layout_type=KVCacheLayoutType.BLOCKFIRST,
  )
  cache.enable_ssd = True
  with tempfile.TemporaryDirectory() as cache_dir:
    cache.ssd_cache_dir = cache_dir
    engine = StorageEngine(cache, make_model())
    assert engine.get_storage_handle(DeviceType.CPU).kv_layout.layout_type == KVCacheLayoutType.LAYERFIRST
    assert engine.get_storage_handle(DeviceType.SSD).kv_layout.layout_type == KVCacheLayoutType.BLOCKFIRST


def test_register_gpu_blocks_with_tensor_list():
  model = make_model()
  engine = StorageEngine(CacheConfig(tokens_per_block=4, num_cpu_blocks=3), model)
  layout = make_gpu_layout()
  tensors = [
    torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype),
    torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype),
  ]

  engine.register_gpu_blocks(tensors, layout, device_id=0)
  handle = engine.get_storage_handle(DeviceType.GPU, 0)
  assert handle.handle_type == StorageHandlerType.TENSOR
  assert handle.kv_layout == layout
  assert handle.gpu_device_id == 0
  assert handle.get_tensor_list() == tensors


def test_register_gpu_blocks_with_tensor_handle_list():
  model = make_model()
  engine = StorageEngine(CacheConfig(tokens_per_block=4, num_cpu_blocks=3), model)
  layout = make_gpu_layout()
  tensor = torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype)
  tensor_handle = make_fake_tensor_handle(tensor)

  engine.register_gpu_blocks([tensor_handle, tensor_handle], layout, device_id=0)
  handle = engine.get_storage_handle(DeviceType.GPU, 0)
  assert handle.handle_type == StorageHandlerType.TENSOR_HANDLE
  assert handle.kv_layout == layout
  assert handle.gpu_device_id == 0
  assert handle.get_tensor_handle_list() == [tensor_handle, tensor_handle]


def test_register_gpu_blocks_validation_and_duplicate_registration():
  model = make_model()
  engine = StorageEngine(CacheConfig(tokens_per_block=4, num_cpu_blocks=3), model)
  layout = make_gpu_layout()
  tensors = [
    torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype),
    torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype),
  ]

  assert_raises(ValueError, lambda: engine.register_gpu_blocks([], layout, device_id=0))
  assert_raises(ValueError, lambda: engine.register_gpu_blocks(tensors, layout, device_id=-1))
  assert_raises(ValueError, lambda: engine.register_gpu_blocks(tensors, layout, device_id=1))

  engine.register_gpu_blocks(tensors, layout, device_id=0)
  assert engine.get_storage_handle(DeviceType.GPU, 0).get_tensor_list() == tensors
  assert_raises(ValueError, lambda: engine.register_gpu_blocks(tensors, layout, device_id=0))


def test_register_gpu_blocks_layout_and_tensor_validation():
  model = make_model()
  cache = CacheConfig(tokens_per_block=4, num_cpu_blocks=3)
  layout = make_gpu_layout()
  tensors = [
    torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype),
    torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype),
  ]

  bad_tokens_layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=2,
    num_blocks=3,
    tokens_per_block=8,
    num_heads=2,
    head_size=8,
    use_mla=False,
  )
  assert_raises(
    ValueError,
    lambda: StorageEngine(cache, model).register_gpu_blocks(tensors, bad_tokens_layout, device_id=0),
  )

  bad_num_layers_layout = KVCacheLayout(
    layout_type=KVCacheLayoutType.LAYERFIRST,
    num_layers=1,
    num_blocks=3,
    tokens_per_block=4,
    num_heads=2,
    head_size=8,
    use_mla=False,
  )
  assert_raises(
    ValueError,
    lambda: StorageEngine(cache, model).register_gpu_blocks([tensors[0]], bad_num_layers_layout, device_id=0),
  )

  wrong_dtype_tensors = [
    torch.empty(tuple(layout.kv_shape[1:]), dtype=torch.float32),
    torch.empty(tuple(layout.kv_shape[1:]), dtype=torch.float32),
  ]
  assert_raises(
    ValueError,
    lambda: StorageEngine(cache, model).register_gpu_blocks(wrong_dtype_tensors, layout, device_id=0),
  )


TEST_CASES = [
  ("CPU storage 初始化", test_cpu_storage_initialized),
  ("SSD storage 初始化并使用 prefix", test_ssd_storage_initialized_and_prefix_used),
  ("SSD storage 支持 cache_dir list", test_ssd_storage_initialized_with_cache_dir_list),
  ("SSD 配置校验", test_ssd_config_validation),
  ("SSD layout 可以和 CPU layout 不同", test_ssd_layout_can_differ_from_cpu_layout),
  ("注册 GPU tensor list", test_register_gpu_blocks_with_tensor_list),
  ("注册 GPU tensor handle list", test_register_gpu_blocks_with_tensor_handle_list),
  ("注册 GPU 参数校验和禁止重复注册", test_register_gpu_blocks_validation_and_duplicate_registration),
  ("注册 GPU layout 和 tensor 校验", test_register_gpu_blocks_layout_and_tensor_validation),
]


def run_all_tests():
  print("开始运行 StorageEngine 测试")
  passed = 0
  total = len(TEST_CASES)
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{total}] 开始：{name}")
    try:
      test_fn()
    except Exception as exc:
      print(f"[{index}/{total}] 失败：{name}，错误：{type(exc).__name__}: {exc}")
      traceback.print_exc()
      raise
    passed += 1
    print(f"[{index}/{total}] 通过：{name}")
  print(f"StorageEngine 测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

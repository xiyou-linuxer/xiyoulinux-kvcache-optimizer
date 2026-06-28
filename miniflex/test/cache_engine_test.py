# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：单层 CacheEngine 的初始化校验、block 申请/回收、前缀匹配与共享前缀、pending/ready 状态、淘汰（含 protected 保护、pin/unpin 控制）与 reset。

import numpy as np

from miniflex.cache.cache_engine import CacheEngine
from miniflex.common.block import SequenceMeta
from miniflex.common.transfer import DeviceType


def tokens(ids):
  """把普通列表转换成 SequenceMeta 需要的一维 np.int64 token 数组。"""
  return np.array(ids, dtype=np.int64)


def blocks(ids):
  """把普通列表转换成 CacheEngine/Mempool 使用的一维 np.int64 block id 数组。"""
  return np.array(ids, dtype=np.int64)


def seq(ids, tokens_per_block=2, namespace=None):
  """创建测试用 SequenceMeta，默认每 2 个 token 一个 block。"""
  return SequenceMeta(tokens(ids), tokens_per_block=tokens_per_block, namespace=namespace)


def assert_raises(exc_type, fn):
  """确认某段代码会抛出指定异常，方便测试错误路径。"""
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def new_engine(num_total_blocks=8, tokens_per_block=2, **kwargs):
  """创建一个 CPU CacheEngine；单元测试只验证逻辑缓存层，不涉及真实设备数据。"""
  return CacheEngine(
    device_type=DeviceType.CPU,
    num_total_blocks=num_total_blocks,
    tokens_per_block=tokens_per_block,
    **kwargs,
  )


def test_init_validation_and_basic_fields():
  engine = new_engine(num_total_blocks=4, tokens_per_block=2)
  assert engine.device_type == DeviceType.CPU
  assert engine.num_total_blocks == 4
  assert engine.tokens_per_block == 2
  assert engine.mempool.num_free_blocks == 4
  assert engine.radix_tree.is_empty()

  # MiniFlex 的逻辑 CacheEngine 只用于 CPU/SSD，GPU block 由后续 storage/transfer 层处理。
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.GPU, 4, 2))
  assert_raises(ValueError, lambda: CacheEngine("CPU", 4, 2))

  # 初始化参数错误应尽早失败，避免把错误推迟到 radix tree 或 mempool 内部。
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 0, 2))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 0))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 3))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 2, eviction_policy="bad"))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 2, evict_ratio=-0.1))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 2, evict_ratio=1.0))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 2, evict_start_threshold=0))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 2, evict_start_threshold=1.1))
  assert_raises(ValueError, lambda: CacheEngine(DeviceType.CPU, 4, 2, protected_threshold=0))

  # evict_ratio=0 和 evict_start_threshold=1.0 是有效配置：只在需要时按需求驱逐。
  CacheEngine(DeviceType.CPU, 4, 2, evict_ratio=0)
  CacheEngine(DeviceType.CPU, 4, 2, evict_start_threshold=1.0)


def test_take_zero_negative_and_recycle():
  engine = new_engine(num_total_blocks=4)

  # take(0) 是合法空申请，便于上层在无缺失 block 时保持统一路径。
  empty = engine.take(0)
  assert empty.dtype == np.int64
  assert empty.tolist() == []
  assert engine.mempool.num_free_blocks == 4

  assert_raises(ValueError, lambda: engine.take(-1))

  first = engine.take(2)
  assert first.dtype == np.int64
  assert first.tolist() == [0, 1]
  assert engine.mempool.num_free_blocks == 2
  assert engine.mempool.num_used_blocks == 2

  engine.recycle(first)
  assert engine.mempool.num_free_blocks == 4
  assert engine.mempool.num_used_blocks == 0

  # recycle 之后 mempool 使用 lazy refresh，下一次 allocate 应能拿回释放的 block。
  second = engine.take(2)
  assert second.tolist() == [0, 1]


def test_match_insert_and_shared_prefix():
  engine = new_engine(num_total_blocks=8)
  first = seq([1, 2, 3, 4, 5, 6])
  second = seq([1, 2, 3, 4, 7, 8])

  first_blocks = engine.take(first.num_blocks)
  first_node = engine.insert(first, first_blocks, is_ready=True)
  assert first_node is not None
  assert engine.radix_tree.total_cached_blocks() == 3

  first_match = engine.match(first)
  assert first_match.num_matched_blocks == 3
  assert first_match.num_ready_matched_blocks == 3
  assert first_match.physical_block_ids.tolist() == [0, 1, 2]

  # 第二条序列共享前两个 block，CacheEngine.insert 应只插入未命中后缀。
  second_match = engine.match(second)
  assert second_match.num_matched_blocks == 2
  assert second_match.physical_block_ids.tolist() == [0, 1]

  suffix_blocks = engine.take(second.num_blocks - second_match.num_matched_blocks)
  second_node = engine.insert(second, suffix_blocks, is_ready=True, match_result=second_match)
  assert second_node is not None

  assert engine.radix_tree.total_cached_blocks() == 4
  assert engine.radix_tree.total_node_size() == 3
  assert engine.match(first).physical_block_ids.tolist() == [0, 1, 2]
  assert engine.match(second).physical_block_ids.tolist() == [0, 1, 3]

  # 完整命中时再次 insert 不应创建新节点，也不应改变树规模。
  duplicate = engine.insert(second, blocks([]), is_ready=True, match_result=engine.match(second))
  assert duplicate is None
  assert engine.radix_tree.total_cached_blocks() == 4


def test_pending_ready_state_and_set_ready():
  engine = new_engine(num_total_blocks=4)
  request = seq([1, 2, 3, 4])

  node = engine.insert(request, engine.take(request.num_blocks), is_ready=False)
  assert node is not None
  assert not node.is_ready()

  # pending 节点参与“逻辑已占用”的匹配，但不计入 ready matched，也不可驱逐。
  match = engine.match(request)
  assert match.num_matched_blocks == 2
  assert match.num_ready_matched_blocks == 0
  assert match.physical_block_ids.tolist() == [0, 1]

  engine.set_ready(node, True)
  assert node.is_ready()
  ready_match = engine.match(request)
  assert ready_match.num_matched_blocks == 2
  assert ready_match.num_ready_matched_blocks == 2

  # 当前 radix tree 不支持 ready 回滚。
  assert_raises(ValueError, lambda: engine.set_ready(node, False))


def test_evict_recycle_and_strict_modes():
  engine = new_engine(num_total_blocks=2)
  request = seq([1, 2, 3, 4])

  node = engine.insert(request, engine.take(request.num_blocks), is_ready=True)
  assert node is not None
  assert engine.mempool.num_free_blocks == 0
  assert engine.radix_tree.total_cached_blocks() == 2

  # strict=True 时，如果节点可驱逐，take 会先 evict，再把 block 回收到 mempool 里分配出来。
  replacement = engine.take(2, strict=True)
  assert replacement.tolist() == [0, 1]
  assert engine.mempool.num_free_blocks == 0
  assert engine.radix_tree.total_cached_blocks() == 0
  assert engine.match(request).num_matched_blocks == 0

  # pending 节点不可驱逐：strict=False 返回空，strict=True 抛 RuntimeError。
  pending_request = seq([9, 10])
  pending_node = engine.insert(pending_request, replacement[:1], is_ready=False)
  assert pending_node is not None
  non_strict = engine.take(1, strict=False)
  assert non_strict.tolist() == []
  assert engine.mempool.num_free_blocks == 0
  assert engine.radix_tree.total_cached_blocks() == 1
  assert_raises(RuntimeError, lambda: engine.take(1, strict=True))


def test_protected_node_blocks_eviction_temporarily():
  engine = new_engine(num_total_blocks=1)
  request = seq([1, 2])

  node = engine.insert(request, engine.take(1), is_ready=True)
  assert node is not None
  assert engine.mempool.num_free_blocks == 0

  # protected_node 会在 take 期间临时 pin，保护该节点不被 eviction。
  non_strict = engine.take(1, protected_node=node, strict=False)
  assert non_strict.tolist() == []
  assert node._pin_count == 0
  assert engine.radix_tree.total_cached_blocks() == 1
  assert engine.match(request).num_matched_blocks == 1

  assert_raises(RuntimeError, lambda: engine.take(1, protected_node=node, strict=True))
  assert node._pin_count == 0
  assert engine.radix_tree.total_cached_blocks() == 1

  # 不再保护后，同一个 ready leaf 可以被驱逐。
  replacement = engine.take(1, strict=True)
  assert replacement.tolist() == [0]
  assert engine.radix_tree.total_cached_blocks() == 0


def test_pin_unpin_control_eviction():
  engine = new_engine(num_total_blocks=1)
  request = seq([1, 2])
  node = engine.insert(request, engine.take(1), is_ready=True)

  engine.pin(node)
  assert node._pin_count == 1
  assert engine.take(1, strict=False).tolist() == []
  assert engine.radix_tree.total_cached_blocks() == 1

  engine.unpin(node)
  assert node._pin_count == 0
  assert_raises(ValueError, lambda: engine.unpin(node))

  replacement = engine.take(1, strict=True)
  assert replacement.tolist() == [0]
  assert engine.radix_tree.total_cached_blocks() == 0


def test_reset_clears_index_and_mempool():
  engine = new_engine(num_total_blocks=4)
  request = seq([1, 2, 3, 4])

  engine.insert(request, engine.take(request.num_blocks), is_ready=True)
  assert engine.radix_tree.total_cached_blocks() == 2
  assert engine.mempool.num_used_blocks == 2

  engine.reset()
  assert engine.radix_tree.is_empty()
  assert engine.radix_tree.total_cached_blocks() == 0
  assert engine.mempool.num_free_blocks == 4
  assert engine.mempool.num_used_blocks == 0
  assert engine.match(request).num_matched_blocks == 0


TEST_CASES = [
  ("初始化参数校验和基础字段", test_init_validation_and_basic_fields),
  ("take 空申请、负数申请和 recycle", test_take_zero_negative_and_recycle),
  ("match/insert 以及共享前缀插入", test_match_insert_and_shared_prefix),
  ("pending 节点 ready 状态和 set_ready", test_pending_ready_state_and_set_ready),
  ("驱逐、回收和 strict/non-strict 申请", test_evict_recycle_and_strict_modes),
  ("protected_node 临时保护驱逐", test_protected_node_blocks_eviction_temporarily),
  ("pin/unpin 控制驱逐", test_pin_unpin_control_eviction),
  ("reset 清空 index 和 mempool", test_reset_clears_index_and_mempool),
]


def run_all_tests():
  print("开始运行缓存引擎测试")
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
  print(f"缓存引擎测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

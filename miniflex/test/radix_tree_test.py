import numpy as np

from miniflex.cache.radix_tree import RadixTree, RadixTreeNode
from miniflex.common.block import SequenceMeta


def seq(tokens, tokens_per_block=2, namespace=None):
  return SequenceMeta(np.array(tokens, dtype=np.int64), tokens_per_block, namespace)


def phys(ids):
  return np.array(ids, dtype=np.int64)


def assert_match(tree, sequence, matched, ready, physical):
  result = tree.match_prefix(sequence)
  assert result.num_matched_blocks == matched
  assert result.num_ready_matched_blocks == ready
  assert result.physical_block_ids.dtype == np.int64
  assert result.physical_block_ids.tolist() == physical
  return result


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def test_node_validation_and_leaf_semantics():
  assert_raises(
    ValueError,
    lambda: RadixTreeNode(
      np.array([[1]], dtype=np.uint64),
      np.array([1], dtype=np.int64),
    ),
  )
  assert_raises(
    ValueError,
    lambda: RadixTreeNode(
      np.array([1, 2], dtype=np.uint64),
      np.array([1], dtype=np.int64),
    ),
  )

  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  assert tree.root.is_root()
  assert not tree.root.is_leaf()
  assert tree.root.is_ready()
  assert tree.is_empty()
  assert tree.total_cached_blocks() == 0
  assert tree.total_node_size() == 0


def test_match_empty_insert_and_prefix_queries():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  request = seq([1, 2, 3, 4, 5, 6])

  empty_match = assert_match(tree, request, 0, 0, [])
  assert empty_match.is_empty()
  node = tree.insert(request, phys([10, 11, 12]), empty_match)

  assert node is not None
  assert node.is_leaf()
  assert node.is_ready()
  assert tree.leaf_nodes[node.head_hash()] is node
  assert not tree.is_empty()
  assert tree.total_cached_blocks() == 3
  assert tree.total_node_size() == 1

  assert_match(tree, request, 3, 3, [10, 11, 12])
  assert tree.num_matched(request) == 3
  assert_match(tree, seq([1, 2, 3, 4]), 2, 2, [10, 11])
  assert_match(tree, seq([9, 9, 9, 9]), 0, 0, [])

  duplicate = tree.insert(request, phys([]))
  assert duplicate is None
  assert tree.total_cached_blocks() == 3
  assert tree.total_node_size() == 1


def test_insert_splits_existing_leaf_and_matches_both_branches():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  first = seq([1, 2, 3, 4, 5, 6])
  second = seq([1, 2, 3, 4, 7, 8])

  first_node = tree.insert(first, phys([10, 11, 12]))
  second_match = assert_match(tree, second, 2, 2, [10, 11])
  second_node = tree.insert(second, phys([20]), second_match)

  assert first_node is not None
  assert second_node is not None
  assert first_node.parent is second_node.parent
  prefix_node = first_node.parent
  assert prefix_node is not None
  assert prefix_node.parent is tree.root
  assert prefix_node.size() == 2
  assert not prefix_node.is_leaf()
  assert first_node.size() == 1
  assert second_node.size() == 1
  assert first_node.head_hash() in prefix_node.children
  assert second_node.head_hash() in prefix_node.children
  assert first_node.head_hash() in tree.leaf_nodes
  assert second_node.head_hash() in tree.leaf_nodes
  assert prefix_node.head_hash() not in tree.leaf_nodes

  assert_match(tree, first, 3, 3, [10, 11, 12])
  assert_match(tree, second, 3, 3, [10, 11, 20])
  assert tree.total_cached_blocks() == 4
  assert tree.total_node_size() == 3


def test_insert_validation_does_not_mutate_tree():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  original = seq([1, 2, 3, 4])
  tree.insert(original, phys([10, 11]))

  assert_raises(ValueError, lambda: tree.insert(seq([5, 6]), np.array([20], dtype=np.int32)))
  assert_raises(ValueError, lambda: tree.insert(seq([7, 8, 9, 10]), np.array([[20, 21]], dtype=np.int64)))

  before_blocks = tree.total_cached_blocks()
  before_nodes = tree.total_node_size()
  before_leaf_heads = set(tree.leaf_nodes.keys())
  shared_prefix = seq([1, 2, 5, 6])
  bad_match = tree.match_prefix(shared_prefix)
  assert bad_match.num_matched_blocks == 1

  assert_raises(ValueError, lambda: tree.insert(shared_prefix, phys([20, 21]), bad_match))
  assert tree.total_cached_blocks() == before_blocks
  assert tree.total_node_size() == before_nodes
  assert set(tree.leaf_nodes.keys()) == before_leaf_heads
  assert_match(tree, original, 2, 2, [10, 11])


def test_pending_ready_state_affects_ready_match_and_eviction():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  request = seq([1, 2, 3, 4])
  node = tree.insert(request, phys([10, 11]), is_ready=False)

  assert node is not None
  assert not node.is_ready()
  assert node.is_in_use()
  assert not node.is_evictable()
  assert_match(tree, request, 2, 0, [10, 11])
  assert tree.evict(8).tolist() == []
  assert tree.total_cached_blocks() == 2

  tree.set_ready(node, True)
  assert node.is_ready()
  assert not node.is_in_use()
  assert node.is_evictable()
  assert_match(tree, request, 2, 2, [10, 11])
  assert_raises(ValueError, lambda: tree.set_ready(node, False))


def test_ready_length_marks_split_pending_prefix():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  first = seq([1, 2, 3, 4])
  second = seq([1, 2, 5, 6])

  first_node = tree.insert(first, phys([10, 11]), is_ready=False)
  second_match = tree.match_prefix(second)
  second_node = tree.insert(second, phys([12]), match_result=second_match, is_ready=False)

  assert first_node is not None
  assert second_node is not None
  assert_match(tree, first, 2, 0, [10, 11])
  assert_match(tree, second, 2, 0, [10, 12])

  tree.set_ready(second_node, True, ready_length=second.num_blocks)
  assert_match(tree, second, 2, 2, [10, 12])
  assert_match(tree, first, 2, 1, [10, 11])

  tree.set_ready(first_node, True, ready_length=first.num_blocks)
  assert_match(tree, first, 2, 2, [10, 11])


def test_pin_and_unpin_control_eviction():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  request = seq([1, 2, 3, 4])
  node = tree.insert(request, phys([10, 11]))

  tree.pin(node)
  assert node.is_in_use()
  assert not node.is_evictable()
  assert tree.evict(2).tolist() == []
  assert_match(tree, request, 2, 2, [10, 11])

  tree.unpin(node)
  assert not node.is_in_use()
  assert node.is_evictable()
  assert_raises(ValueError, lambda: tree.unpin(node))

  evicted = tree.evict(1)
  assert evicted.dtype == np.int64
  assert evicted.tolist() == [11]
  assert_match(tree, request, 1, 1, [10])
  assert tree.total_cached_blocks() == 1


def test_evict_zero_negative_partial_and_over_evict():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  request = seq([1, 2, 3, 4, 5, 6])
  tree.insert(request, phys([10, 11, 12]))

  evicted_zero = tree.evict(0)
  assert evicted_zero.dtype == np.int64
  assert evicted_zero.tolist() == []
  assert_raises(ValueError, lambda: tree.evict(-1))

  evicted_one = tree.evict(1)
  assert evicted_one.tolist() == [12]
  assert tree.total_cached_blocks() == 2
  assert tree.total_node_size() == 1
  assert_match(tree, request, 2, 2, [10, 11])

  evicted_rest = tree.evict(99)
  assert evicted_rest.tolist() == [10, 11]
  assert tree.is_empty()
  assert tree.total_cached_blocks() == 0
  assert tree.total_node_size() == 0
  assert_match(tree, request, 0, 0, [])


def test_evict_split_tree_promotes_parent_to_leaf():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  first = seq([1, 2, 3, 4, 5, 6])
  second = seq([1, 2, 3, 4, 7, 8])
  first_node = tree.insert(first, phys([10, 11, 12]))
  second_node = tree.insert(second, phys([20]), tree.match_prefix(second))
  prefix_node = first_node.parent

  first_node.grace_time = 1
  second_node.grace_time = 2
  prefix_node.grace_time = 3

  assert tree.evict(1).tolist() == [12]
  assert first_node.parent is None
  assert second_node.parent is prefix_node
  assert prefix_node.head_hash() not in tree.leaf_nodes
  assert second_node.head_hash() in tree.leaf_nodes
  assert tree.total_cached_blocks() == 3
  assert tree.total_node_size() == 2
  assert_match(tree, first, 2, 2, [10, 11])
  assert_match(tree, second, 3, 3, [10, 11, 20])

  assert tree.evict(1).tolist() == [20]
  assert second_node.parent is None
  assert prefix_node.is_leaf()
  assert prefix_node.head_hash() in tree.leaf_nodes
  assert tree.total_cached_blocks() == 2
  assert tree.total_node_size() == 1
  assert_match(tree, first, 2, 2, [10, 11])
  assert_match(tree, second, 2, 2, [10, 11])

  assert tree.evict(99).tolist() == [10, 11]
  assert tree.is_empty()
  assert tree.total_cached_blocks() == 0
  assert tree.total_node_size() == 0


def test_reset_clears_tree():
  tree = RadixTree(tokens_per_block=2, max_num_blocks=16)
  request = seq([1, 2, 3, 4])
  tree.insert(request, phys([10, 11]))
  assert not tree.is_empty()

  tree.reset()
  assert tree.root.is_root()
  assert not tree.root.is_leaf()
  assert tree.root.is_ready()
  assert tree.is_empty()
  assert tree.total_cached_blocks() == 0
  assert tree.total_node_size() == 0
  assert_match(tree, request, 0, 0, [])


def test_eviction_policy_priorities():
  node = RadixTreeNode(
    np.array([1], dtype=np.uint64),
    np.array([10], dtype=np.int64),
    _is_ready=True,
    hit_count=3,
    create_time=7,
    last_access_time=11,
    grace_time=13,
  )

  assert RadixTree(2, 16, eviction_policy="lru")._get_evictable_priority(node) == 13
  assert RadixTree(2, 16, eviction_policy="lfu")._get_evictable_priority(node) == (3, 11)
  assert RadixTree(2, 16, eviction_policy="slru", protected_threshold=4)._get_evictable_priority(node) == (0, 13)
  assert RadixTree(2, 16, eviction_policy="slru", protected_threshold=3)._get_evictable_priority(node) == (1, 13)
  assert RadixTree(2, 16, eviction_policy="fifo")._get_evictable_priority(node) == 7
  assert_raises(ValueError, lambda: RadixTree(2, 16, eviction_policy="bad")._get_evictable_priority(node))


TEST_CASES = [
  ("节点校验与 root/leaf 语义", test_node_validation_and_leaf_semantics),
  ("空树匹配、插入、前缀查询", test_match_empty_insert_and_prefix_queries),
  ("共享前缀插入后的 split 和双分支匹配", test_insert_splits_existing_leaf_and_matches_both_branches),
  ("插入参数校验失败时不污染树", test_insert_validation_does_not_mutate_tree),
  ("pending/ready 状态影响 ready match 和驱逐", test_pending_ready_state_affects_ready_match_and_eviction),
  ("ready_length 标记 split 后的 pending 前缀", test_ready_length_marks_split_pending_prefix),
  ("pin/unpin 控制驱逐", test_pin_and_unpin_control_eviction),
  ("驱逐 0、负数、部分 shrink 和超量驱逐", test_evict_zero_negative_partial_and_over_evict),
  ("split 树驱逐后父节点变 leaf", test_evict_split_tree_promotes_parent_to_leaf),
  ("reset 清空前缀树", test_reset_clears_tree),
  ("驱逐策略优先级", test_eviction_policy_priorities),
]


def run_all_tests():
  print("开始运行前缀树测试")
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
  print(f"前缀树测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

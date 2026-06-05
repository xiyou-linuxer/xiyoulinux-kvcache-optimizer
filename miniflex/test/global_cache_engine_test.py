import numpy as np

from miniflex.cache.global_cache_engine import GlobalCacheEngine
from miniflex.common.block import SequenceMeta
from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.transfer import TransferType


def tokens(ids):
  return np.array(ids, dtype=np.int64)


def mask(values):
  return np.array(values, dtype=np.bool_)


def slots(values):
  return np.array(values, dtype=np.int64)


def seq(ids, tokens_per_block=2, namespace=None):
  return SequenceMeta(tokens(ids), tokens_per_block, namespace)


def new_engine(enable_ssd=True, num_cpu_blocks=8, num_ssd_blocks=8):
  return GlobalCacheEngine(
    CacheConfig(
      tokens_per_block=2,
      enable_ssd=enable_ssd,
      num_cpu_blocks=num_cpu_blocks,
      num_ssd_blocks=num_ssd_blocks,
      ssd_cache_dir="/tmp/miniflex_global_cache_engine_test" if enable_ssd else None,
    ),
    ModelConfig(),
  )


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def sorted_ops(graph):
  return sorted(graph._op_map.values(), key=lambda op: op.op_id)


def test_input_helpers_and_validation():
  engine = new_engine()
  assert engine.slot_mapping_to_block_ids(slots([0, 1, 2, 3]), 2).tolist() == [0, 1]
  assert engine._get_block_range(mask([False, False, True, True])) == (1, 2)
  assert engine._get_block_range(mask([False, False, False, False])) == (0, 0)
  engine._check_block_aligned_mask(mask([False, False, True, True]))
  assert_raises(ValueError, lambda: engine._check_block_aligned_mask(mask([True, False, True, True])))
  assert_raises(ValueError, lambda: engine._check_block_aligned_mask(mask([False, True, True, True])))

  valid_tokens = tokens([1, 2, 3, 4])
  valid_mask = mask([True, True, True, True])
  valid_slots = slots([0, 1, 2, 3])
  engine._check_input(0, valid_tokens, valid_mask, valid_slots)

  assert_raises(ValueError, lambda: engine._check_input(-1, valid_tokens, valid_mask, valid_slots))
  assert_raises(ValueError, lambda: engine._check_input(0, valid_tokens.astype(np.int32), valid_mask, valid_slots))
  assert_raises(ValueError, lambda: engine._check_input(0, valid_tokens, valid_mask.astype(np.int64), valid_slots))
  assert_raises(ValueError, lambda: engine._check_input(0, valid_tokens, valid_mask, slots([0, 1])))


def test_get_miss_returns_empty_graph():
  engine = new_engine()
  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    1,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )

  assert graph.num_ops == 0
  assert return_mask.tolist() == [False, False, False, False]
  assert op_callbacks == {}
  assert task_end == -1
  callback()


def test_get_cpu_hit_builds_h2d():
  engine = new_engine()
  request = seq([1, 2, 3, 4])
  node = engine.cpu_cache_engine.insert(request, engine.cpu_cache_engine.take(request.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    2,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  ops = sorted_ops(graph)

  assert graph.num_ops == 1
  assert ops[0].transfer_type == TransferType.H2D
  assert ops[0].src_block_ids.tolist() == [0, 1]
  assert ops[0].dst_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True]
  assert op_callbacks == {}
  assert task_end == ops[0].op_id
  assert node._pin_count == 1
  assert engine.cpu_cache_engine.radix_tree.evict(2).tolist() == []
  callback()
  assert node._pin_count == 0


def test_get_ssd_hit_builds_disk2h_then_h2d_and_fills_cpu_cache():
  engine = new_engine()
  request = seq([1, 2, 3, 4])
  engine.ssd_cache_engine.insert(request, engine.ssd_cache_engine.take(request.num_blocks), is_ready=True)
  free_before = engine.cpu_cache_engine.mempool.num_free_blocks

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    3,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  disk2h, h2d = sorted_ops(graph)

  assert graph.num_ops == 2
  assert disk2h.transfer_type == TransferType.DISK2H
  assert h2d.transfer_type == TransferType.H2D
  assert h2d.depends_on == {disk2h.op_id}
  assert disk2h.src_block_ids.tolist() == [0, 1]
  assert disk2h.dst_block_ids.tolist() == h2d.src_block_ids.tolist()
  assert h2d.dst_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True]
  assert set(op_callbacks.keys()) == {disk2h.op_id}
  assert task_end == h2d.op_id
  assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - 2
  cpu_match = engine.cpu_cache_engine.match(request)
  assert cpu_match.num_matched_blocks == 2
  assert cpu_match.num_ready_matched_blocks == 0
  cpu_node = cpu_match.last_node
  ssd_node = engine.ssd_cache_engine.match(request).last_ready_node
  assert cpu_node._pin_count == 1
  assert ssd_node._pin_count == 1
  op_callbacks[disk2h.op_id]()
  cpu_match = engine.cpu_cache_engine.match(request)
  assert cpu_match.num_matched_blocks == 2
  assert cpu_match.num_ready_matched_blocks == 2
  assert cpu_node._pin_count == 1
  callback()
  assert cpu_node._pin_count == 0
  assert ssd_node._pin_count == 0
  assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - 2


def test_get_cpu_prefix_and_ssd_extra_builds_split_read():
  engine = new_engine()
  cpu_prefix = seq([1, 2])
  full_request = seq([1, 2, 3, 4])
  engine.cpu_cache_engine.insert(cpu_prefix, engine.cpu_cache_engine.take(cpu_prefix.num_blocks), is_ready=True)
  engine.ssd_cache_engine.insert(full_request, engine.ssd_cache_engine.take(full_request.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    4,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  disk2h, h2d = sorted_ops(graph)

  assert graph.num_ops == 2
  assert disk2h.transfer_type == TransferType.DISK2H
  assert h2d.transfer_type == TransferType.H2D
  assert disk2h.src_block_ids.tolist() == [1]
  assert h2d.src_block_ids.tolist() == [0, 1]
  assert h2d.dst_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True]
  assert set(op_callbacks.keys()) == {disk2h.op_id}
  assert task_end == h2d.op_id
  op_callbacks[disk2h.op_id]()
  cpu_match = engine.cpu_cache_engine.match(full_request)
  assert cpu_match.num_matched_blocks == 2
  assert cpu_match.num_ready_matched_blocks == 2
  callback()


def test_get_ignores_trailing_incomplete_block():
  engine = new_engine()
  cached_prefix = seq([1, 2, 3, 4])
  engine.cpu_cache_engine.insert(cached_prefix, engine.cpu_cache_engine.take(cached_prefix.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    5,
    tokens([1, 2, 3, 4, 5]),
    mask([True, True, True, True, True]),
    slots([0, 1, 2, 3, 4]),
  )
  ops = sorted_ops(graph)

  assert graph.num_ops == 1
  assert ops[0].transfer_type == TransferType.H2D
  assert ops[0].src_block_ids.tolist() == [0, 1]
  assert ops[0].dst_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True, False]
  assert op_callbacks == {}
  assert task_end == ops[0].op_id
  callback()


def test_get_nonzero_block_range_cpu_hit():
  engine = new_engine()
  request = seq([1, 2, 3, 4, 5, 6])
  engine.cpu_cache_engine.insert(request, engine.cpu_cache_engine.take(request.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    6,
    tokens([1, 2, 3, 4, 5, 6]),
    mask([False, False, True, True, True, True]),
    slots([2, 3, 4, 5]),
  )
  ops = sorted_ops(graph)

  assert graph.num_ops == 1
  assert ops[0].transfer_type == TransferType.H2D
  assert ops[0].src_block_ids.tolist() == [1, 2]
  assert ops[0].dst_block_ids.tolist() == [1, 2]
  assert return_mask.tolist() == [False, False, True, True, True, True]
  assert op_callbacks == {}
  assert task_end == ops[0].op_id
  callback()


def test_get_nonzero_block_range_ssd_fill():
  engine = new_engine()
  cpu_prefix = seq([1, 2])
  full_request = seq([1, 2, 3, 4, 5, 6])
  engine.cpu_cache_engine.insert(cpu_prefix, engine.cpu_cache_engine.take(cpu_prefix.num_blocks), is_ready=True)
  engine.ssd_cache_engine.insert(full_request, engine.ssd_cache_engine.take(full_request.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    7,
    tokens([1, 2, 3, 4, 5, 6]),
    mask([False, False, True, True, True, True]),
    slots([2, 3, 4, 5]),
  )
  disk2h, h2d = sorted_ops(graph)

  assert graph.num_ops == 2
  assert disk2h.transfer_type == TransferType.DISK2H
  assert h2d.transfer_type == TransferType.H2D
  assert disk2h.src_block_ids.tolist() == [1, 2]
  assert disk2h.dst_block_ids.tolist() == h2d.src_block_ids.tolist()
  assert h2d.dst_block_ids.tolist() == [1, 2]
  assert return_mask.tolist() == [False, False, True, True, True, True]
  assert set(op_callbacks.keys()) == {disk2h.op_id}
  assert task_end == h2d.op_id
  op_callbacks[disk2h.op_id]()
  cpu_match = engine.cpu_cache_engine.match(full_request)
  assert cpu_match.num_matched_blocks == 3
  assert cpu_match.num_ready_matched_blocks == 3
  callback()


def test_get_cpu_pending_falls_back_to_temporary_buffer():
  engine = new_engine()
  request = seq([1, 2, 3, 4])
  engine.cpu_cache_engine.insert(request, engine.cpu_cache_engine.take(request.num_blocks), is_ready=False)
  engine.ssd_cache_engine.insert(request, engine.ssd_cache_engine.take(request.num_blocks), is_ready=True)
  free_before = engine.cpu_cache_engine.mempool.num_free_blocks

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    8,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  disk2h, h2d = sorted_ops(graph)

  assert graph.num_ops == 2
  assert disk2h.transfer_type == TransferType.DISK2H
  assert h2d.transfer_type == TransferType.H2D
  assert op_callbacks == {}
  assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - 2
  callback()
  assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before
  cpu_match = engine.cpu_cache_engine.match(request)
  assert cpu_match.num_matched_blocks == 2
  assert cpu_match.num_ready_matched_blocks == 0


def test_get_cpu_space_shortage_returns_empty_graph():
  engine = new_engine(num_cpu_blocks=1)
  request = seq([1, 2, 3, 4])
  engine.ssd_cache_engine.insert(request, engine.ssd_cache_engine.take(request.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.get(
    9,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )

  assert graph.num_ops == 0
  assert return_mask.tolist() == [False, False, False, False]
  assert op_callbacks == {}
  assert task_end == -1
  assert engine.cpu_cache_engine.mempool.num_free_blocks == 1
  callback()


def test_get_rejects_unaligned_or_noncontiguous_mask():
  engine = new_engine()
  request = seq([1, 2, 3, 4])
  engine.cpu_cache_engine.insert(request, engine.cpu_cache_engine.take(request.num_blocks), is_ready=True)

  assert_raises(
    ValueError,
    lambda: engine.get(10, tokens([1, 2, 3, 4]), mask([True, False, True, True]), slots([0, 2, 3])),
  )
  assert_raises(
    ValueError,
    lambda: engine.get(11, tokens([1, 2, 3, 4]), mask([False, True, True, True]), slots([1, 2, 3])),
  )


def test_put_cpu_only_builds_d2h_and_fills_cpu_cache():
  engine = new_engine(enable_ssd=False)
  request = seq([1, 2, 3, 4])
  free_before = engine.cpu_cache_engine.mempool.num_free_blocks

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    12,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  ops = sorted_ops(graph)

  assert graph.num_ops == 1
  assert ops[0].transfer_type == TransferType.D2H
  assert ops[0].src_block_ids.tolist() == [0, 1]
  assert ops[0].dst_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True]
  assert set(op_callbacks.keys()) == {ops[0].op_id}
  assert task_end == ops[0].op_id
  assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - 2
  cpu_match = engine.cpu_cache_engine.match(request)
  assert cpu_match.num_matched_blocks == 2
  assert cpu_match.num_ready_matched_blocks == 0
  cpu_node = cpu_match.last_node
  assert cpu_node._pin_count == 1
  op_callbacks[ops[0].op_id]()
  cpu_match = engine.cpu_cache_engine.match(request)
  assert cpu_match.num_ready_matched_blocks == 2
  callback()
  assert cpu_node._pin_count == 0


def test_put_cpu_and_ssd_builds_d2h_then_h2disk():
  engine = new_engine()
  request = seq([1, 2, 3, 4])
  cpu_free_before = engine.cpu_cache_engine.mempool.num_free_blocks
  ssd_free_before = engine.ssd_cache_engine.mempool.num_free_blocks

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    13,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  d2h, h2disk = sorted_ops(graph)

  assert graph.num_ops == 2
  assert d2h.transfer_type == TransferType.D2H
  assert h2disk.transfer_type == TransferType.H2DISK
  assert h2disk.depends_on == {d2h.op_id}
  assert d2h.src_block_ids.tolist() == [0, 1]
  assert d2h.dst_block_ids.tolist() == h2disk.src_block_ids.tolist()
  assert h2disk.dst_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True]
  assert set(op_callbacks.keys()) == {d2h.op_id, h2disk.op_id}
  assert task_end == d2h.op_id
  assert engine.cpu_cache_engine.mempool.num_free_blocks == cpu_free_before - 2
  assert engine.ssd_cache_engine.mempool.num_free_blocks == ssd_free_before - 2
  cpu_node = engine.cpu_cache_engine.match(request).last_node
  ssd_node = engine.ssd_cache_engine.match(request).last_node
  assert cpu_node._pin_count == 1
  assert ssd_node._pin_count == 1
  op_callbacks[d2h.op_id]()
  assert engine.cpu_cache_engine.match(request).num_ready_matched_blocks == 2
  assert engine.ssd_cache_engine.match(request).num_ready_matched_blocks == 0
  op_callbacks[h2disk.op_id]()
  assert engine.ssd_cache_engine.match(request).num_ready_matched_blocks == 2
  callback()
  assert cpu_node._pin_count == 0
  assert ssd_node._pin_count == 0


def test_put_skips_existing_cpu_prefix_and_writes_suffix():
  engine = new_engine()
  cpu_prefix = seq([1, 2])
  full_request = seq([1, 2, 3, 4, 5, 6])
  engine.cpu_cache_engine.insert(cpu_prefix, engine.cpu_cache_engine.take(cpu_prefix.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    14,
    tokens([1, 2, 3, 4, 5, 6]),
    mask([True, True, True, True, True, True]),
    slots([0, 1, 2, 3, 4, 5]),
  )
  d2h, h2disk = sorted_ops(graph)

  assert graph.num_ops == 2
  assert d2h.transfer_type == TransferType.D2H
  assert h2disk.transfer_type == TransferType.H2DISK
  assert d2h.src_block_ids.tolist() == [1, 2]
  assert h2disk.src_block_ids.tolist() == [0, 1, 2]
  assert h2disk.dst_block_ids.tolist() == [0, 1, 2]
  assert return_mask.tolist() == [False, False, True, True, True, True]
  assert set(op_callbacks.keys()) == {d2h.op_id, h2disk.op_id}
  assert task_end == d2h.op_id
  op_callbacks[d2h.op_id]()
  op_callbacks[h2disk.op_id]()
  assert engine.cpu_cache_engine.match(full_request).num_ready_matched_blocks == 3
  assert engine.ssd_cache_engine.match(full_request).num_ready_matched_blocks == 3
  callback()


def test_put_uses_existing_ssd_prefix_for_h2disk_suffix():
  engine = new_engine()
  cpu_prefix = seq([1, 2])
  ssd_prefix = seq([1, 2, 3, 4])
  full_request = seq([1, 2, 3, 4, 5, 6])
  engine.cpu_cache_engine.insert(cpu_prefix, engine.cpu_cache_engine.take(cpu_prefix.num_blocks), is_ready=True)
  engine.ssd_cache_engine.insert(ssd_prefix, engine.ssd_cache_engine.take(ssd_prefix.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    15,
    tokens([1, 2, 3, 4, 5, 6]),
    mask([True, True, True, True, True, True]),
    slots([0, 1, 2, 3, 4, 5]),
  )
  d2h, h2disk = sorted_ops(graph)

  assert graph.num_ops == 2
  assert d2h.src_block_ids.tolist() == [1, 2]
  assert h2disk.src_block_ids.tolist() == [2]
  assert h2disk.dst_block_ids.tolist() == [2]
  assert return_mask.tolist() == [False, False, True, True, True, True]
  assert set(op_callbacks.keys()) == {d2h.op_id, h2disk.op_id}
  assert task_end == d2h.op_id
  op_callbacks[d2h.op_id]()
  op_callbacks[h2disk.op_id]()
  assert engine.cpu_cache_engine.match(full_request).num_ready_matched_blocks == 3
  assert engine.ssd_cache_engine.match(full_request).num_ready_matched_blocks == 3
  callback()


def test_put_full_cpu_hit_returns_empty_graph():
  engine = new_engine()
  request = seq([1, 2, 3, 4])
  engine.cpu_cache_engine.insert(request, engine.cpu_cache_engine.take(request.num_blocks), is_ready=True)

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    16,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )

  assert graph.num_ops == 0
  assert return_mask.tolist() == [False, False, False, False]
  assert op_callbacks == {}
  assert task_end == -1
  callback()


def test_put_space_shortage_returns_empty_graph_and_recycles():
  engine = new_engine(num_cpu_blocks=1)
  cpu_free_before = engine.cpu_cache_engine.mempool.num_free_blocks
  ssd_free_before = engine.ssd_cache_engine.mempool.num_free_blocks

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    17,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )

  assert graph.num_ops == 0
  assert return_mask.tolist() == [False, False, False, False]
  assert op_callbacks == {}
  assert task_end == -1
  assert engine.cpu_cache_engine.mempool.num_free_blocks == cpu_free_before
  assert engine.ssd_cache_engine.mempool.num_free_blocks == ssd_free_before
  callback()


def test_put_cpu_insert_none_returns_empty_graph_and_recycles():
  engine = new_engine()
  cpu_free_before = engine.cpu_cache_engine.mempool.num_free_blocks
  ssd_free_before = engine.ssd_cache_engine.mempool.num_free_blocks
  original_insert = engine.cpu_cache_engine.insert
  insert_calls = []

  def fake_cpu_insert(*args, **kwargs):
    insert_calls.append((args, kwargs))
    return None

  engine.cpu_cache_engine.insert = fake_cpu_insert
  try:
    graph, return_mask, callback, op_callbacks, task_end = engine.put(
      18,
      tokens([1, 2, 3, 4]),
      mask([True, True, True, True]),
      slots([0, 1, 2, 3]),
    )
  finally:
    engine.cpu_cache_engine.insert = original_insert

  assert len(insert_calls) == 1
  assert graph.num_ops == 0
  assert return_mask.tolist() == [False, False, False, False]
  assert op_callbacks == {}
  assert task_end == -1
  assert engine.cpu_cache_engine.mempool.num_free_blocks == cpu_free_before
  assert engine.ssd_cache_engine.mempool.num_free_blocks == ssd_free_before
  callback()


def test_put_ssd_insert_none_recycles_ssd_buffer_on_callback():
  engine = new_engine()
  request = seq([1, 2, 3, 4])
  cpu_free_before = engine.cpu_cache_engine.mempool.num_free_blocks
  ssd_free_before = engine.ssd_cache_engine.mempool.num_free_blocks
  original_insert = engine.ssd_cache_engine.insert
  insert_calls = []

  def fake_ssd_insert(*args, **kwargs):
    insert_calls.append((args, kwargs))
    return None

  engine.ssd_cache_engine.insert = fake_ssd_insert
  try:
    graph, return_mask, callback, op_callbacks, task_end = engine.put(
      19,
      tokens([1, 2, 3, 4]),
      mask([True, True, True, True]),
      slots([0, 1, 2, 3]),
    )
  finally:
    engine.ssd_cache_engine.insert = original_insert

  ops = sorted_ops(graph)
  assert len(insert_calls) == 1
  assert graph.num_ops == 2
  assert ops[0].transfer_type == TransferType.D2H
  assert ops[1].transfer_type == TransferType.H2DISK
  assert return_mask.tolist() == [True, True, True, True]
  assert set(op_callbacks.keys()) == {ops[0].op_id}
  assert task_end == ops[0].op_id
  assert engine.cpu_cache_engine.mempool.num_free_blocks == cpu_free_before - 2
  assert engine.ssd_cache_engine.mempool.num_free_blocks == ssd_free_before - 2

  op_callbacks[ops[0].op_id]()
  assert engine.cpu_cache_engine.match(request).num_ready_matched_blocks == 2
  assert engine.ssd_cache_engine.match(request).num_matched_blocks == 0
  callback()
  assert engine.cpu_cache_engine.mempool.num_free_blocks == cpu_free_before - 2
  assert engine.ssd_cache_engine.mempool.num_free_blocks == ssd_free_before


def test_overlapping_put_split_keeps_ready_prefix_contiguous():
  engine = new_engine(enable_ssd=False)
  first_request = seq([1, 2, 3, 4])
  second_request = seq([1, 2, 5, 6])

  first_graph, _, first_callback, first_op_callbacks, _ = engine.put(
    24,
    tokens([1, 2, 3, 4]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  first_d2h = sorted_ops(first_graph)[0]

  second_graph, _, second_callback, second_op_callbacks, _ = engine.put(
    25,
    tokens([1, 2, 5, 6]),
    mask([True, True, True, True]),
    slots([4, 5, 6, 7]),
  )
  second_d2h = sorted_ops(second_graph)[0]

  second_op_callbacks[second_d2h.op_id]()
  second_callback()

  first_match = engine.cpu_cache_engine.match(first_request)
  second_match = engine.cpu_cache_engine.match(second_request)
  assert first_match.num_matched_blocks == 2
  assert first_match.num_ready_matched_blocks == 0
  assert second_match.num_matched_blocks == 2
  assert second_match.num_ready_matched_blocks == 0

  get_graph, get_return_mask, get_callback, get_op_callbacks, get_task_end = engine.get(
    26,
    tokens([1, 2, 5, 6]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  assert get_graph.num_ops == 0
  assert get_return_mask.tolist() == [False, False, False, False]
  assert get_op_callbacks == {}
  assert get_task_end == -1
  get_callback()

  first_op_callbacks[first_d2h.op_id]()
  first_callback()
  assert engine.cpu_cache_engine.match(first_request).num_ready_matched_blocks == 2
  assert engine.cpu_cache_engine.match(second_request).num_ready_matched_blocks == 2

  get_graph, get_return_mask, get_callback, get_op_callbacks, get_task_end = engine.get(
    27,
    tokens([1, 2, 5, 6]),
    mask([True, True, True, True]),
    slots([0, 1, 2, 3]),
  )
  get_ops = sorted_ops(get_graph)
  assert get_return_mask.tolist() == [True, True, True, True]
  assert len(get_ops) == 1
  assert get_ops[0].transfer_type == TransferType.H2D
  assert get_ops[0].src_block_ids.tolist() == second_match.physical_block_ids.tolist()
  assert get_op_callbacks == {}
  assert get_task_end == get_ops[0].op_id
  get_callback()


def test_put_uses_mask_covered_complete_prefix_for_sequence():
  engine = new_engine(enable_ssd=False)
  prefix_request = seq([1, 2, 3, 4])
  full_request = seq([1, 2, 3, 4, 5, 6])
  free_before = engine.cpu_cache_engine.mempool.num_free_blocks

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    20,
    tokens([1, 2, 3, 4, 5, 6]),
    mask([True, True, True, True, False, False]),
    slots([0, 1, 2, 3]),
  )
  ops = sorted_ops(graph)

  assert graph.num_ops == 1
  assert ops[0].transfer_type == TransferType.D2H
  assert ops[0].src_block_ids.tolist() == [0, 1]
  assert ops[0].dst_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True, False, False]
  assert set(op_callbacks.keys()) == {ops[0].op_id}
  assert task_end == ops[0].op_id
  assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - 2

  cpu_match = engine.cpu_cache_engine.match(prefix_request)
  assert cpu_match.num_matched_blocks == 2
  assert cpu_match.num_ready_matched_blocks == 0
  cpu_node = cpu_match.last_node
  assert cpu_node._pin_count == 1

  op_callbacks[ops[0].op_id]()
  assert engine.cpu_cache_engine.match(prefix_request).num_ready_matched_blocks == 2
  full_match = engine.cpu_cache_engine.match(full_request)
  assert full_match.num_matched_blocks == 2
  assert full_match.num_ready_matched_blocks == 2
  callback()
  assert cpu_node._pin_count == 0
  assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - 2


def test_put_ignores_trailing_incomplete_block():
  engine = new_engine(enable_ssd=False)
  request = seq([1, 2, 3, 4])

  graph, return_mask, callback, op_callbacks, task_end = engine.put(
    21,
    tokens([1, 2, 3, 4, 5]),
    mask([True, True, True, True, True]),
    slots([0, 1, 2, 3, 4]),
  )
  ops = sorted_ops(graph)

  assert graph.num_ops == 1
  assert ops[0].transfer_type == TransferType.D2H
  assert ops[0].src_block_ids.tolist() == [0, 1]
  assert return_mask.tolist() == [True, True, True, True, False]
  assert set(op_callbacks.keys()) == {ops[0].op_id}
  assert task_end == ops[0].op_id
  op_callbacks[ops[0].op_id]()
  assert engine.cpu_cache_engine.match(request).num_ready_matched_blocks == 2
  callback()


def test_put_rejects_nonzero_or_unaligned_mask():
  engine = new_engine()
  assert_raises(
    ValueError,
    lambda: engine.put(22, tokens([1, 2, 3, 4]), mask([False, False, True, True]), slots([2, 3])),
  )
  assert_raises(
    ValueError,
    lambda: engine.put(23, tokens([1, 2, 3, 4]), mask([True, False, True, True]), slots([0, 2, 3])),
  )


TEST_CASES = [
  ("输入 helper 和校验", test_input_helpers_and_validation),
  ("GET 全 miss 返回空图", test_get_miss_returns_empty_graph),
  ("GET CPU 命中生成 H2D", test_get_cpu_hit_builds_h2d),
  ("GET SSD 命中生成 DISK2H -> H2D 并回填 CPU cache", test_get_ssd_hit_builds_disk2h_then_h2d_and_fills_cpu_cache),
  ("GET CPU 前缀命中和 SSD 后缀命中", test_get_cpu_prefix_and_ssd_extra_builds_split_read),
  ("GET 忽略尾部不完整 block", test_get_ignores_trailing_incomplete_block),
  ("GET 非 0 起始 block 的 CPU 命中", test_get_nonzero_block_range_cpu_hit),
  ("GET 非 0 起始 block 的 SSD 回填", test_get_nonzero_block_range_ssd_fill),
  ("GET CPU pending 时退化为临时 buffer", test_get_cpu_pending_falls_back_to_temporary_buffer),
  ("GET CPU 空间不足返回空图", test_get_cpu_space_shortage_returns_empty_graph),
  ("GET 拒绝非连续或非 block 对齐 mask", test_get_rejects_unaligned_or_noncontiguous_mask),
  ("PUT CPU-only 生成 D2H 并写入 CPU cache", test_put_cpu_only_builds_d2h_and_fills_cpu_cache),
  ("PUT CPU+SSD 生成 D2H -> H2DISK", test_put_cpu_and_ssd_builds_d2h_then_h2disk),
  ("PUT 跳过已有 CPU 前缀并写入后缀", test_put_skips_existing_cpu_prefix_and_writes_suffix),
  ("PUT 使用已有 SSD 前缀只写 SSD 后缀", test_put_uses_existing_ssd_prefix_for_h2disk_suffix),
  ("PUT 全 CPU 命中返回空图", test_put_full_cpu_hit_returns_empty_graph),
  ("PUT 空间不足返回空图并回收", test_put_space_shortage_returns_empty_graph_and_recycles),
  ("PUT CPU insert 返回 None 时返回空图并回收", test_put_cpu_insert_none_returns_empty_graph_and_recycles),
  ("PUT SSD insert 返回 None 时 callback 回收 SSD buffer", test_put_ssd_insert_none_recycles_ssd_buffer_on_callback),
  ("重叠 PUT split 后保持连续 ready 前缀", test_overlapping_put_split_keeps_ready_prefix_contiguous),
  ("PUT 只用 mask 覆盖的完整 block 前缀构造序列", test_put_uses_mask_covered_complete_prefix_for_sequence),
  ("PUT 忽略尾部不完整 block", test_put_ignores_trailing_incomplete_block),
  ("PUT 拒绝非 0 起点或非对齐 mask", test_put_rejects_nonzero_or_unaligned_mask),
]


def run_all_tests():
  print("开始运行 GlobalCacheEngine 测试")
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
  print(f"GlobalCacheEngine 测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

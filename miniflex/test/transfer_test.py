import numpy as np

from miniflex.common.transfer import (
  CompletedOp,
  TransferOp,
  TransferOpGraph,
  TransferOpStatus,
  TransferType,
)


def blocks(ids):
  """把普通列表转换成 transfer.py 要求的一维 np.int64 block id 数组。"""
  return np.array(ids, dtype=np.int64)


def assert_raises(exc_type, fn):
  """确认某段代码会抛出指定异常，方便测试错误路径。"""
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def ready_ids(graph):
  """取出当前可运行 op，并返回 op_id 集合；take_ready_ops 会把 op 标成 RUNNING。"""
  return {op.op_id for op in graph.take_ready_ops()}


def make_op(graph, transfer_type, src_ids, dst_ids):
  """创建一个属于 graph 的 TransferOp，减少测试里的重复样板。"""
  return TransferOp(
    transfer_type=transfer_type,
    graph_id=graph.graph_id,
    src_block_ids=blocks(src_ids),
    dst_block_ids=blocks(dst_ids),
  )


def test_transfer_op_validation_and_completed_op():
  graph = TransferOpGraph()

  # 普通传输 op 要求 src/dst block 数量一致，并自动记录有效 block 数。
  op = make_op(graph, TransferType.H2D, [1, 2], [10, 11])
  assert op.transfer_type == TransferType.H2D
  assert op.status == TransferOpStatus.PENDING
  assert op.valid_block_num == 2
  assert isinstance(op.op_id, int)

  # 非虚拟 op 的 block id 必须是一维 np.int64，且 src/dst 数量相同。
  assert_raises(
    ValueError,
    lambda: TransferOp(
      transfer_type=TransferType.H2D,
      graph_id=graph.graph_id,
      src_block_ids=np.array([[1]], dtype=np.int64),
      dst_block_ids=blocks([10]),
    ),
  )
  assert_raises(
    ValueError,
    lambda: TransferOp(
      transfer_type=TransferType.H2D,
      graph_id=graph.graph_id,
      src_block_ids=np.array([1], dtype=np.int32),
      dst_block_ids=blocks([10]),
    ),
  )
  assert_raises(
    ValueError,
    lambda: make_op(graph, TransferType.H2D, [1, 2], [10]),
  )
  assert_raises(
    ValueError,
    lambda: TransferOp(
      transfer_type=TransferType.H2D,
      graph_id=-1,
      src_block_ids=blocks([1]),
      dst_block_ids=blocks([10]),
    ),
  )

  # 虚拟 op 不做真实搬运，可以使用空 block 数组作为完成屏障。
  virtual_op = make_op(graph, TransferType.VIRTUAL, [], [])
  assert virtual_op.valid_block_num == 0

  # CompletedOp 用 op_id=-1 表示整张 graph 完成。
  completed_op = CompletedOp(graph_id=7, op_id=3)
  assert not completed_op.is_graph_completed()
  assert completed_op.to_tuple() == (7, 3)
  assert CompletedOp.from_tuple((7, 3)) == completed_op

  completed_graph = CompletedOp.completed_graph(graph_id=7)
  assert completed_graph.is_graph_completed()
  assert completed_graph.to_tuple() == (7, -1)


def test_single_transfer_op_lifecycle():
  graph = TransferOpGraph()
  op = make_op(graph, TransferType.H2D, [1, 2], [10, 11])
  graph.add_transfer_op(op)

  # 新加入且无依赖的 op 应立即 ready，取出后进入 RUNNING。
  assert graph.num_ops == 1
  assert not graph.all_transfer_ops_completed()
  assert ready_ids(graph) == {op.op_id}
  assert op.status == TransferOpStatus.RUNNING

  # RUNNING op 不会被重复取出；完成后整张 graph 才算完成。
  assert ready_ids(graph) == set()
  graph.mark_completed(op.op_id)
  assert op.status == TransferOpStatus.COMPLETED
  assert graph.all_transfer_ops_completed()
  assert_raises(ValueError, lambda: graph.mark_completed(op.op_id))


def test_dependency_disk2h_then_h2d():
  graph = TransferOpGraph()
  disk2h = make_op(graph, TransferType.DISK2H, [100, 101], [10, 11])
  h2d = make_op(graph, TransferType.H2D, [10, 11], [0, 1])

  graph.add_transfer_op(disk2h)
  graph.add_transfer_op(h2d)
  graph.add_dependency(h2d.op_id, disk2h.op_id)

  # SSD 命中读回时，H2D 必须等待 DISK2H 完成。
  assert h2d.depends_on == {disk2h.op_id}
  assert disk2h.dependents == {h2d.op_id}
  assert ready_ids(graph) == {disk2h.op_id}
  assert disk2h.status == TransferOpStatus.RUNNING
  assert h2d.status == TransferOpStatus.PENDING

  graph.mark_completed(disk2h.op_id)
  assert h2d.depends_on == set()
  assert ready_ids(graph) == {h2d.op_id}
  assert h2d.status == TransferOpStatus.RUNNING

  graph.mark_completed(h2d.op_id)
  assert graph.all_transfer_ops_completed()


def test_dependency_d2h_then_h2disk():
  graph = TransferOpGraph()
  d2h = make_op(graph, TransferType.D2H, [0, 1], [10, 11])
  h2disk = make_op(graph, TransferType.H2DISK, [10, 11], [100, 101])

  graph.add_transfer_op(d2h)
  graph.add_transfer_op(h2disk)
  graph.add_dependency(h2disk.op_id, d2h.op_id)

  # PUT 路径里，SSD 写入必须等待 GPU 到 CPU 的 D2H 完成。
  assert ready_ids(graph) == {d2h.op_id}
  graph.mark_completed(d2h.op_id)
  assert ready_ids(graph) == {h2disk.op_id}
  graph.mark_completed(h2disk.op_id)
  assert graph.all_transfer_ops_completed()


def test_virtual_op_waits_for_multiple_predecessors():
  graph = TransferOpGraph()
  disk2h = make_op(graph, TransferType.DISK2H, [100], [10])
  d2h = make_op(graph, TransferType.D2H, [0], [11])
  virtual = make_op(graph, TransferType.VIRTUAL, [], [])

  graph.add_transfer_op(disk2h)
  graph.add_transfer_op(d2h)
  graph.add_virtual_op(virtual)
  graph.add_dependency(virtual.op_id, disk2h.op_id)
  graph.add_dependency(virtual.op_id, d2h.op_id)

  # 虚拟 op 用来表达“多个真实 op 都完成后再通知上层”。
  assert virtual.depends_on == {disk2h.op_id, d2h.op_id}
  first_ready = ready_ids(graph)
  assert first_ready == {disk2h.op_id, d2h.op_id}
  assert virtual.status == TransferOpStatus.PENDING

  # 只完成一个前置 op 时，虚拟 op 仍不能运行。
  graph.mark_completed(disk2h.op_id)
  assert ready_ids(graph) == set()
  assert virtual.depends_on == {d2h.op_id}

  # 两个前置 op 都完成后，虚拟 op 才会 ready。
  graph.mark_completed(d2h.op_id)
  assert ready_ids(graph) == {virtual.op_id}
  graph.mark_completed(virtual.op_id)
  assert graph.all_transfer_ops_completed()


def test_add_dependency_validation():
  graph = TransferOpGraph()
  first = make_op(graph, TransferType.DISK2H, [100], [10])
  second = make_op(graph, TransferType.H2D, [10], [0])

  graph.add_transfer_op(first)
  graph.add_transfer_op(second)

  # 依赖关系只能建立在当前 graph 中已经登记过的 op 之间。
  assert_raises(ValueError, lambda: graph.add_dependency(second.op_id, first.op_id + 99999))

  # 当前 MiniFlex 实现要求被添加依赖的 op 仍是 PENDING，避免运行后再改依赖。
  assert ready_ids(graph) == {first.op_id, second.op_id}
  assert_raises(ValueError, lambda: graph.add_dependency(second.op_id, first.op_id))


def test_set_gpu_blocks_binds_h2d_and_d2h():
  graph = TransferOpGraph()
  h2d = make_op(graph, TransferType.H2D, [10, 11], [-1, -1])
  d2h = make_op(graph, TransferType.D2H, [-1, -1], [20, 21])
  h2disk = make_op(graph, TransferType.H2DISK, [30], [100])

  graph.add_transfer_op(h2d)
  graph.add_transfer_op(d2h)
  graph.add_transfer_op(h2disk)

  # vLLM 分配 GPU block 后，H2D 绑定 dst，D2H 绑定 src。
  graph.set_gpu_blocks(blocks([0, 1]))
  assert h2d.dst_block_ids.tolist() == [0, 1]
  assert d2h.src_block_ids.tolist() == [0, 1]

  # 非 GPU 方向的 H2DISK 不应被 set_gpu_blocks 修改。
  assert h2disk.src_block_ids.tolist() == [30]
  assert h2disk.dst_block_ids.tolist() == [100]

  # 传入的 GPU block id 也必须是一维 np.int64。
  assert_raises(ValueError, lambda: graph.set_gpu_blocks(np.array([0, 1], dtype=np.int32)))
  assert_raises(ValueError, lambda: graph.set_gpu_blocks(np.array([[0, 1]], dtype=np.int64)))

  short_graph = TransferOpGraph()
  short_h2d = make_op(short_graph, TransferType.H2D, [10, 11], [-1, -1])
  short_graph.add_transfer_op(short_h2d)
  assert_raises(ValueError, lambda: short_graph.set_gpu_blocks(blocks([0])))


def test_graph_id_and_explicit_override():
  first = TransferOpGraph()
  second = TransferOpGraph.create_empty_graph()

  # graph_id 是全局递增分配的；只比较不同即可，避免依赖固定起始值。
  assert first.graph_id != second.graph_id

  # set_graph_id 用于后续图合并或上层指定 id。
  second.set_graph_id(12345)
  assert second.graph_id == 12345


TEST_CASES = [
  ("TransferOp 参数校验和 CompletedOp 语义", test_transfer_op_validation_and_completed_op),
  ("单个 transfer op 的 ready/running/completed 生命周期", test_single_transfer_op_lifecycle),
  ("SSD 读回路径 DISK2H -> H2D 依赖", test_dependency_disk2h_then_h2d),
  ("写入路径 D2H -> H2DISK 依赖", test_dependency_d2h_then_h2disk),
  ("虚拟 op 等待多个前置 op 完成", test_virtual_op_waits_for_multiple_predecessors),
  ("add_dependency 错误路径校验", test_add_dependency_validation),
  ("set_gpu_blocks 绑定 H2D/D2H 的 GPU block", test_set_gpu_blocks_binds_h2d_and_d2h),
  ("graph_id 分配和显式覆盖", test_graph_id_and_explicit_override),
]


def run_all_tests():
  print("开始运行传输图测试")
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
  print(f"传输图测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

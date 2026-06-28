# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：TransferScheduler 的依赖调度：就绪 op 出队、依赖链、多图插入序、虚拟 op、未知 op 忽略、空图完成。

import numpy as np

from miniflex.common.transfer import TransferOp, TransferOpGraph, TransferOpStatus, TransferType
from miniflex.transfer.scheduler import TransferScheduler


def blocks(ids):
  return np.array(ids, dtype=np.int64)


def make_op(graph, transfer_type, src_ids, dst_ids):
  return TransferOp(
    transfer_type=transfer_type,
    graph_id=graph.graph_id,
    src_block_ids=blocks(src_ids),
    dst_block_ids=blocks(dst_ids),
  )


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def test_add_duplicate_graph_rejected_and_pending_status():
  scheduler = TransferScheduler()
  graph = TransferOpGraph()

  assert not scheduler.has_pending_graphs()
  scheduler.add_transfer_graph(graph)
  assert scheduler.has_pending_graphs()
  assert_raises(ValueError, lambda: scheduler.add_transfer_graph(graph))


def test_schedule_single_ready_op_and_complete_graph():
  scheduler = TransferScheduler()
  graph = TransferOpGraph()
  op = make_op(graph, TransferType.H2D, [10, 11], [0, 1])
  graph.add_transfer_op(op)
  scheduler.add_transfer_graph(graph)

  completed_graphs, next_ops = scheduler.schedule([])
  assert completed_graphs == []
  assert [ready_op.op_id for ready_op in next_ops] == [op.op_id]
  assert op.status == TransferOpStatus.RUNNING
  assert scheduler.has_pending_graphs()

  completed_graphs, next_ops = scheduler.schedule([op])
  assert completed_graphs == [graph.graph_id]
  assert next_ops == []
  assert op.status == TransferOpStatus.COMPLETED
  assert not scheduler.has_pending_graphs()


def test_schedule_dependency_chain():
  scheduler = TransferScheduler()
  graph = TransferOpGraph()
  disk2h = make_op(graph, TransferType.DISK2H, [100, 101], [10, 11])
  h2d = make_op(graph, TransferType.H2D, [10, 11], [0, 1])
  graph.add_transfer_op(disk2h)
  graph.add_transfer_op(h2d)
  graph.add_dependency(h2d.op_id, disk2h.op_id)
  scheduler.add_transfer_graph(graph)

  completed_graphs, next_ops = scheduler.schedule([])
  assert completed_graphs == []
  assert [op.op_id for op in next_ops] == [disk2h.op_id]
  assert disk2h.status == TransferOpStatus.RUNNING
  assert h2d.status == TransferOpStatus.PENDING

  completed_graphs, next_ops = scheduler.schedule([disk2h])
  assert completed_graphs == []
  assert [op.op_id for op in next_ops] == [h2d.op_id]
  assert h2d.status == TransferOpStatus.RUNNING

  completed_graphs, next_ops = scheduler.schedule([h2d])
  assert completed_graphs == [graph.graph_id]
  assert next_ops == []
  assert not scheduler.has_pending_graphs()


def test_schedule_multiple_graphs_keeps_insertion_order():
  scheduler = TransferScheduler()
  first_graph = TransferOpGraph()
  second_graph = TransferOpGraph()
  first_op = make_op(first_graph, TransferType.D2H, [0], [10])
  second_op = make_op(second_graph, TransferType.H2DISK, [11], [100])
  first_graph.add_transfer_op(first_op)
  second_graph.add_transfer_op(second_op)

  scheduler.add_transfer_graph(first_graph)
  scheduler.add_transfer_graph(second_graph)

  completed_graphs, next_ops = scheduler.schedule([])
  assert completed_graphs == []
  assert [op.op_id for op in next_ops] == [first_op.op_id, second_op.op_id]

  completed_graphs, next_ops = scheduler.schedule([second_op])
  assert completed_graphs == [second_graph.graph_id]
  assert next_ops == []
  assert scheduler.has_pending_graphs()

  completed_graphs, next_ops = scheduler.schedule([first_op])
  assert completed_graphs == [first_graph.graph_id]
  assert next_ops == []
  assert not scheduler.has_pending_graphs()


def test_schedule_virtual_op_completes_without_worker_execution():
  scheduler = TransferScheduler()
  graph = TransferOpGraph()
  d2h = make_op(graph, TransferType.D2H, [0], [10])
  h2disk = make_op(graph, TransferType.H2DISK, [10], [100])
  virtual = make_op(graph, TransferType.VIRTUAL, [], [])
  graph.add_transfer_op(d2h)
  graph.add_transfer_op(h2disk)
  graph.add_virtual_op(virtual)
  graph.add_dependency(virtual.op_id, d2h.op_id)
  graph.add_dependency(virtual.op_id, h2disk.op_id)
  scheduler.add_transfer_graph(graph)

  completed_graphs, next_ops = scheduler.schedule([])
  assert completed_graphs == []
  assert {op.op_id for op in next_ops} == {d2h.op_id, h2disk.op_id}
  assert virtual.status == TransferOpStatus.PENDING

  completed_graphs, next_ops = scheduler.schedule([d2h])
  assert completed_graphs == []
  assert next_ops == []
  assert virtual.status == TransferOpStatus.PENDING

  completed_graphs, next_ops = scheduler.schedule([h2disk])
  assert completed_graphs == [graph.graph_id]
  assert [op.op_id for op in next_ops] == [virtual.op_id]
  assert virtual.status == TransferOpStatus.COMPLETED
  assert not scheduler.has_pending_graphs()


def test_unknown_finished_op_is_ignored():
  scheduler = TransferScheduler()
  graph = TransferOpGraph()
  op = make_op(graph, TransferType.H2D, [10], [0])
  graph.add_transfer_op(op)

  completed_graphs, next_ops = scheduler.schedule([op])
  assert completed_graphs == []
  assert next_ops == []


def test_empty_graph_completes_on_schedule():
  scheduler = TransferScheduler()
  graph = TransferOpGraph.create_empty_graph()
  scheduler.add_transfer_graph(graph)

  completed_graphs, next_ops = scheduler.schedule([])
  assert completed_graphs == [graph.graph_id]
  assert next_ops == []
  assert not scheduler.has_pending_graphs()


TEST_CASES = [
  ("重复 graph 拒绝和 pending 状态", test_add_duplicate_graph_rejected_and_pending_status),
  ("单个 ready op 调度和 graph 完成", test_schedule_single_ready_op_and_complete_graph),
  ("依赖链调度", test_schedule_dependency_chain),
  ("多 graph 按插入顺序调度", test_schedule_multiple_graphs_keeps_insertion_order),
  ("virtual op 自动完成", test_schedule_virtual_op_completes_without_worker_execution),
  ("未知 finished op 忽略", test_unknown_finished_op_is_ignored),
  ("空 graph 调度即完成", test_empty_graph_completes_on_schedule),
]


def run_all_tests():
  print("开始运行 TransferScheduler 测试")
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
  print(f"TransferScheduler 测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

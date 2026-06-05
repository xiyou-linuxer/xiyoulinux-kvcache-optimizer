import tempfile
import traceback

import numpy as np

import miniflex.kvtask as kvtask_module
from miniflex.common.block import SequenceMeta
from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.request import KVRequest, KVRequestType, KVResponseStatus
from miniflex.common.transfer import CompletedOp, TransferType


def tokens(values):
  return np.array(values, dtype=np.int64)


def mask(values):
  return np.array(values, dtype=np.bool_)


def slots(values):
  return np.array(values, dtype=np.int64)


def seq(values, tokens_per_block=2, namespace=None):
  return SequenceMeta(tokens(values), tokens_per_block, namespace)


def sorted_ops(graph):
  return sorted(graph._op_map.values(), key=lambda op: op.op_id)


class FakeTransferManagerHandle:
  def __init__(self, model_config, cache_config, gpu_register_port, mode="process"):
    del model_config, cache_config, gpu_register_port, mode
    self.started = False
    self.ready = False
    self.shutdown_called = False
    self.submitted = []
    self.submitted_batches = []
    self._completed = []

  def start(self):
    self.started = True
    self.ready = True

  def is_ready(self):
    return self.ready

  def submit(self, transfer_graph):
    self.submitted.append(transfer_graph)

  def submit_batch(self, transfer_graphs):
    self.submitted_batches.append(list(transfer_graphs))

  def wait(self, timeout=None):
    del timeout
    completed = self._completed
    self._completed = []
    return completed

  def shutdown(self):
    self.shutdown_called = True
    self.ready = False

  def push_completed(self, *completed_ops):
    self._completed.extend(completed_ops)


def make_model():
  return ModelConfig(
    num_layers=2,
    num_kv_heads=1,
    head_size=2,
  )


def make_cache_config(tmp_dir, enable_ssd=True):
  return CacheConfig(
    tokens_per_block=2,
    enable_ssd=enable_ssd,
    num_cpu_blocks=8,
    num_ssd_blocks=8,
    ssd_cache_dir=tmp_dir if enable_ssd else None,
    ssd_file_prefix="kvtask_test",
    use_direct_io=False,
  )


def make_engine(tmp_dir, enable_ssd=True):
  old_handle_cls = kvtask_module.TransferManagerHandle
  kvtask_module.TransferManagerHandle = FakeTransferManagerHandle
  try:
    engine = kvtask_module.KVTaskEngine(
      model_config=make_model(),
      cache_config=make_cache_config(tmp_dir, enable_ssd=enable_ssd),
      gpu_register_port="ipc:///tmp/miniflex_kvtask_test.sock",
    )
  finally:
    kvtask_module.TransferManagerHandle = old_handle_cls
  return engine, engine._manager._transfer_manager


def fill_cpu_cache(engine, token_values, namespace=None, is_ready=True):
  request = seq(token_values, engine.cache_config.tokens_per_block, namespace)
  cache_engine = engine._manager._global_cache_engine
  block_ids = cache_engine.cpu_cache_engine.take(request.num_blocks)
  node = cache_engine.cpu_cache_engine.insert(request, block_ids, is_ready=is_ready)
  return request, node


def fill_ssd_cache(engine, token_values, namespace=None, is_ready=True):
  request = seq(token_values, engine.cache_config.tokens_per_block, namespace)
  cache_engine = engine._manager._global_cache_engine
  block_ids = cache_engine.ssd_cache_engine.take(request.num_blocks)
  node = cache_engine.ssd_cache_engine.insert(request, block_ids, is_ready=is_ready)
  return request, node


def assert_status(response, expected_status):
  assert response.status == expected_status, (
    f"expected status {expected_status}, got {response.status}"
  )


def test_get_match_launch_and_wait_uses_fake_slot_mapping_until_launch():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, handle = make_engine(tmp_dir, enable_ssd=False)
    try:
      assert not engine.is_ready()
      engine.start()
      assert engine.is_ready()

      fill_cpu_cache(engine, [1, 2, 3, 4, 5, 6])
      request = KVRequest(
        request_type=KVRequestType.GET,
        token_ids=tokens([1, 2, 3, 4, 5, 6]),
        token_mask=mask([False, False, True, True, True, True]),
      )
      task_id, return_mask = engine.get_match(request)

      task = engine._manager.tasks[task_id]
      assert task.task_status == kvtask_module.TaskStatus.UNREADY
      assert return_mask.tolist() == [False, False, True, True, True, True]

      response = engine.wait(task_id, timeout=0)[task_id]
      assert_status(response, KVResponseStatus.UNREADY)

      launched_ids = engine.launch_tasks([task_id], [slots([2, 3, 4, 5])])
      assert launched_ids == [task_id]
      assert len(handle.submitted_batches) == 1
      graph = handle.submitted_batches[0][0]
      ops = sorted_ops(graph)
      assert len(ops) == 1
      assert ops[0].transfer_type == TransferType.H2D
      assert ops[0].src_block_ids.tolist() == [1, 2]
      assert ops[0].dst_block_ids.tolist() == [1, 2]

      handle.push_completed(
        CompletedOp(graph.graph_id, ops[0].op_id),
        CompletedOp.completed_graph(graph.graph_id),
      )
      response = engine.wait(task_id, completely=True)[task_id]
      assert_status(response, KVResponseStatus.SUCCESS)
      assert response.get_mask().tolist() == [False, False, True, True, True, True]
      assert task_id not in engine._manager.tasks
    finally:
      engine.shutdown()


def test_put_async_try_wait_returns_success_after_end_op_before_graph_completion():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, handle = make_engine(tmp_dir, enable_ssd=True)
    try:
      request = KVRequest(
        request_type=KVRequestType.PUT,
        token_ids=tokens([11, 12, 13, 14]),
        slot_mapping=slots([0, 1, 2, 3]),
      )
      task_id, return_mask = engine.put_async(request)
      assert return_mask.tolist() == [True, True, True, True]
      assert len(handle.submitted) == 1

      graph = handle.submitted[0]
      ops = sorted_ops(graph)
      assert [op.transfer_type for op in ops] == [TransferType.D2H, TransferType.H2DISK]

      assert engine.try_wait(task_id) == {}

      handle.push_completed(CompletedOp(graph.graph_id, ops[0].op_id))
      response = engine.try_wait(task_id)[task_id]
      assert_status(response, KVResponseStatus.SUCCESS)
      assert response.get_mask().tolist() == [True, True, True, True]
      assert engine._manager.tasks[task_id].task_status == kvtask_module.TaskStatus.RUNNING

      cpu_match = engine._manager._global_cache_engine.cpu_cache_engine.match(seq([11, 12, 13, 14]))
      assert cpu_match.num_ready_matched_blocks == 2

      handle.push_completed(
        CompletedOp(graph.graph_id, ops[1].op_id),
        CompletedOp.completed_graph(graph.graph_id),
      )
      response = engine.wait(task_id, completely=True)[task_id]
      assert_status(response, KVResponseStatus.NOTFOUND)

      ssd_match = engine._manager._global_cache_engine.ssd_cache_engine.match(seq([11, 12, 13, 14]))
      assert ssd_match.num_ready_matched_blocks == 2
      assert task_id not in engine._manager.tasks
    finally:
      engine.shutdown()


def test_get_async_returns_success_after_h2d_end_op_before_graph_completion():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, handle = make_engine(tmp_dir, enable_ssd=True)
    try:
      fill_ssd_cache(engine, [21, 22, 23, 24])
      request = KVRequest(
        request_type=KVRequestType.GET,
        token_ids=tokens([21, 22, 23, 24]),
        slot_mapping=slots([0, 1, 2, 3]),
      )
      task_id, return_mask = engine.get_async(request)
      assert return_mask.tolist() == [True, True, True, True]
      assert len(handle.submitted) == 1

      graph = handle.submitted[0]
      ops = sorted_ops(graph)
      assert [op.transfer_type for op in ops] == [TransferType.DISK2H, TransferType.H2D]

      handle.push_completed(CompletedOp(graph.graph_id, ops[0].op_id))
      assert engine.try_wait(task_id) == {}

      cpu_match = engine._manager._global_cache_engine.cpu_cache_engine.match(seq([21, 22, 23, 24]))
      assert cpu_match.num_ready_matched_blocks == 2

      handle.push_completed(CompletedOp(graph.graph_id, ops[1].op_id))
      response = engine.wait(task_id, timeout=0, completely=False)[task_id]
      assert_status(response, KVResponseStatus.SUCCESS)
      assert engine._manager.tasks[task_id].task_status == kvtask_module.TaskStatus.RUNNING

      handle.push_completed(CompletedOp.completed_graph(graph.graph_id))
      response = engine.wait(task_id, completely=True)[task_id]
      assert_status(response, KVResponseStatus.NOTFOUND)
      assert task_id not in engine._manager.tasks
    finally:
      engine.shutdown()


def test_put_async_releases_task_after_graph_completion_even_if_success_was_returned_early():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, handle = make_engine(tmp_dir, enable_ssd=True)
    try:
      request = KVRequest(
        request_type=KVRequestType.PUT,
        token_ids=tokens([61, 62, 63, 64]),
        slot_mapping=slots([0, 1, 2, 3]),
      )
      task_id, _ = engine.put_async(request)
      graph = handle.submitted[0]
      ops = sorted_ops(graph)

      handle.push_completed(CompletedOp(graph.graph_id, ops[0].op_id))
      response = engine.try_wait(task_id)[task_id]
      assert_status(response, KVResponseStatus.SUCCESS)
      assert task_id in engine._manager.tasks

      handle.push_completed(
        CompletedOp(graph.graph_id, ops[1].op_id),
        CompletedOp.completed_graph(graph.graph_id),
      )
      engine._manager._update_tasks(timeout=0)
      assert task_id not in engine._manager.tasks

      response = engine.wait(task_id, completely=True)[task_id]
      assert_status(response, KVResponseStatus.NOTFOUND)
    finally:
      engine.shutdown()


def test_launch_tasks_as_batch_merges_get_graphs_and_waits_by_batch_task():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, handle = make_engine(tmp_dir, enable_ssd=False)
    try:
      fill_cpu_cache(engine, [1, 2, 3, 4])
      fill_cpu_cache(engine, [5, 6, 7, 8])

      first_task_id, _ = engine.get_match(
        KVRequest(KVRequestType.GET, tokens([1, 2, 3, 4]))
      )
      second_task_id, _ = engine.get_match(
        KVRequest(KVRequestType.GET, tokens([5, 6, 7, 8]))
      )
      launched_ids = engine.launch_tasks(
        [first_task_id, second_task_id],
        [slots([0, 1, 2, 3]), slots([4, 5, 6, 7])],
        as_batch=True,
      )

      assert len(launched_ids) == 1
      batch_id = launched_ids[0]
      assert len(handle.submitted_batches) == 1
      batch_graph = handle.submitted_batches[0][0]
      ops = sorted_ops(batch_graph)
      assert batch_graph.graph_id == batch_id
      assert len(ops) == 1
      assert ops[0].transfer_type == TransferType.H2D
      assert ops[0].src_block_ids.tolist() == [0, 1, 2, 3]
      assert ops[0].dst_block_ids.tolist() == [0, 1, 2, 3]
      assert batch_id in engine._manager.tasks
      assert first_task_id not in engine._manager.tasks
      assert second_task_id not in engine._manager.tasks

      handle.push_completed(
        CompletedOp(batch_graph.graph_id, ops[0].op_id),
        CompletedOp.completed_graph(batch_graph.graph_id),
      )
      responses = engine.wait(batch_id, completely=True)
      response = responses[batch_id]
      assert_status(response, KVResponseStatus.SUCCESS)
      assert isinstance(response.return_mask, list)
      assert len(response.return_mask) == 2
      assert response.get_mask(0).tolist() == [True, True, True, True]
      assert response.get_mask(1).tolist() == [True, True, True, True]
      assert batch_id not in engine._manager.tasks
    finally:
      engine.shutdown()


def test_launch_tasks_as_batch_merges_put_graphs_and_waits_after_batch_end_op():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, handle = make_engine(tmp_dir, enable_ssd=True)
    try:
      first_task_id, _ = engine.put_match(
        KVRequest(KVRequestType.PUT, tokens([31, 32, 33, 34]))
      )
      second_task_id, _ = engine.put_match(
        KVRequest(KVRequestType.PUT, tokens([41, 42, 43, 44]))
      )
      launched_ids = engine.launch_tasks(
        [first_task_id, second_task_id],
        [slots([0, 1, 2, 3]), slots([4, 5, 6, 7])],
        as_batch=True,
      )

      assert len(launched_ids) == 1
      batch_id = launched_ids[0]
      batch_graph = handle.submitted_batches[0][0]
      ops = sorted_ops(batch_graph)
      assert [op.transfer_type for op in ops] == [TransferType.D2H, TransferType.H2DISK]
      assert ops[1].depends_on == {ops[0].op_id}
      assert batch_id in engine._manager.tasks
      assert first_task_id not in engine._manager.tasks
      assert second_task_id not in engine._manager.tasks

      handle.push_completed(CompletedOp(batch_graph.graph_id, ops[0].op_id))
      responses = engine.try_wait(batch_id)
      response = responses[batch_id]
      assert_status(response, KVResponseStatus.SUCCESS)
      assert isinstance(response.return_mask, list)
      assert len(response.return_mask) == 2
      assert engine._manager.tasks[batch_id].task_status == kvtask_module.TaskStatus.RUNNING

      handle.push_completed(
        CompletedOp(batch_graph.graph_id, ops[1].op_id),
        CompletedOp.completed_graph(batch_graph.graph_id),
      )
      responses = engine.wait(batch_id, completely=True)
      response = responses[batch_id]
      assert_status(response, KVResponseStatus.NOTFOUND)

      first_request = seq([31, 32, 33, 34])
      second_request = seq([41, 42, 43, 44])
      cpu_cache = engine._manager._global_cache_engine.cpu_cache_engine
      ssd_cache = engine._manager._global_cache_engine.ssd_cache_engine
      assert cpu_cache.match(first_request).num_ready_matched_blocks == 2
      assert cpu_cache.match(second_request).num_ready_matched_blocks == 2
      assert ssd_cache.match(first_request).num_ready_matched_blocks == 2
      assert ssd_cache.match(second_request).num_ready_matched_blocks == 2
      assert batch_id not in engine._manager.tasks
    finally:
      engine.shutdown()


def test_cancel_tasks_releases_task_and_wait_reports_notfound():
  with tempfile.TemporaryDirectory() as tmp_dir:
    engine, _ = make_engine(tmp_dir, enable_ssd=False)
    try:
      fill_cpu_cache(engine, [51, 52, 53, 54])
      task_id, _ = engine.get_match(
        KVRequest(KVRequestType.GET, tokens([51, 52, 53, 54]))
      )
      assert task_id in engine._manager.tasks

      engine.cancel_tasks(task_id)
      assert task_id not in engine._manager.tasks

      response = engine.wait(task_id, timeout=0)[task_id]
      assert_status(response, KVResponseStatus.NOTFOUND)
    finally:
      engine.shutdown()


TEST_CASES = [
  ("GET match 使用 fake slot mapping，launch 后完成", test_get_match_launch_and_wait_uses_fake_slot_mapping_until_launch),
  ("PUT async 在 end op 后 try_wait 返回 success", test_put_async_try_wait_returns_success_after_end_op_before_graph_completion),
  ("GET async early success 后 graph 完成会自动清理", test_get_async_returns_success_after_h2d_end_op_before_graph_completion),
  ("PUT async early success 后 graph 完成会自动清理", test_put_async_releases_task_after_graph_completion_even_if_success_was_returned_early),
  ("batch GET merge 后按 batch 任务 wait", test_launch_tasks_as_batch_merges_get_graphs_and_waits_by_batch_task),
  ("batch PUT early success 后 graph 完成会自动清理", test_launch_tasks_as_batch_merges_put_graphs_and_waits_after_batch_end_op),
  ("cancel_tasks 释放任务并让 wait 返回 NOTFOUND", test_cancel_tasks_releases_task_and_wait_reports_notfound),
]


def run_all_tests():
  print("开始运行 KVTask 测试")
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
  print(f"KVTask 测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

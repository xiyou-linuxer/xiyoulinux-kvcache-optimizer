# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：SharedOpPool 共享内存 block-id buffer 的分配复用/释放、容量限制与跨进程可见性。

import traceback

import numpy as np
import torch
import torch.multiprocessing as mp

from miniflex.common.ring_buffer import SharedOpPool


def blocks(ids):
  return np.array(ids, dtype=np.int64)


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def _read_shared_buffer_worker(buffer, slot_id, valid_block_num, expected, queue):
  try:
    actual = buffer[slot_id, :valid_block_num].cpu().numpy().tolist()
    if actual != expected:
      raise AssertionError(f"expected {expected}, got {actual}")
    queue.put(("ok", None))
  except Exception:
    queue.put(("error", traceback.format_exc()))


def _write_shared_buffer_worker(buffer, slot_id, values, queue):
  try:
    buffer[slot_id, :len(values)] = torch.tensor(values, dtype=torch.int64)
    queue.put(("ok", None))
  except Exception:
    queue.put(("error", traceback.format_exc()))


def run_child_process(target, args):
  ctx = mp.get_context("spawn")
  queue = ctx.Queue()
  process = ctx.Process(target=target, args=(*args, queue))
  process.start()
  process.join(timeout=30)
  if process.is_alive():
    process.terminate()
    process.join(timeout=10)
    raise AssertionError("child process did not exit")
  if process.exitcode != 0:
    raise AssertionError(f"child process exited with code {process.exitcode}")
  if queue.empty():
    raise AssertionError("child process returned no result")
  status, payload = queue.get()
  if status != "ok":
    raise AssertionError(payload)


def test_allocate_reuse_free_and_status():
  pool = SharedOpPool(max_op_num=2, max_block_num=4)

  assert pool.get_buffer().is_shared()
  assert pool.get_buffer_size() == (2, 4)
  assert pool.status() == {
    "used_slots": 0,
    "free_slots": 2,
    "capacity": 2,
  }

  first_slot = pool.allocate_slot(blocks([1, 2, 3]))
  second_slot = pool.allocate_slot(blocks([1, 2, 3]))
  assert first_slot == second_slot
  assert pool.status()["used_slots"] == 1
  assert pool.slot_ref_count[first_slot] == 2
  assert pool.get_buffer()[first_slot, :3].cpu().numpy().tolist() == [1, 2, 3]

  pool.free_slot(first_slot)
  assert pool.status()["used_slots"] == 1
  assert pool.slot_ref_count[first_slot] == 1

  pool.free_slot(second_slot)
  assert pool.status()["used_slots"] == 0
  assert pool.slot_ref_count[first_slot] == 0
  assert_raises(RuntimeError, lambda: pool.free_slot(first_slot))


def test_capacity_and_prefix_behavior():
  pool = SharedOpPool(max_op_num=2, max_block_num=2)

  assert pool.allocate_slot(blocks([])) == -1
  assert pool.allocate_slot(blocks([1, 2, 3])) == -1

  cpu_slot = pool.allocate_slot(blocks([0, 1]), device_type_prefix=2)
  ssd_slot = pool.allocate_slot(blocks([0, 1]), device_type_prefix=3)
  assert cpu_slot != ssd_slot
  assert pool.status()["used_slots"] == 2

  assert pool.allocate_slot(blocks([9])) == -1


def test_multiprocess_worker_reads_parent_allocated_slot():
  pool = SharedOpPool(max_op_num=2, max_block_num=4)
  slot_id = pool.allocate_slot(blocks([10, 11, 12]))

  run_child_process(
    _read_shared_buffer_worker,
    (pool.get_buffer(), slot_id, 3, [10, 11, 12]),
  )


def test_multiprocess_worker_write_is_visible_to_parent():
  pool = SharedOpPool(max_op_num=2, max_block_num=4)
  slot_id = pool.allocate_slot(blocks([1, 2, 3]))

  run_child_process(
    _write_shared_buffer_worker,
    (pool.get_buffer(), slot_id, [7, 8, 9]),
  )

  assert pool.get_buffer()[slot_id, :3].cpu().numpy().tolist() == [7, 8, 9]


TEST_CASES = [
  ("分配、复用、释放和状态", test_allocate_reuse_free_and_status),
  ("容量限制和 device prefix 隔离", test_capacity_and_prefix_behavior),
  ("多进程 worker 读取父进程分配 slot", test_multiprocess_worker_reads_parent_allocated_slot),
  ("多进程 worker 写入父进程可见", test_multiprocess_worker_write_is_visible_to_parent),
]


def run_all_tests():
  print("开始运行 SharedOpPool 测试")
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
  print(f"SharedOpPool 测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

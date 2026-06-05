import threading
from collections import deque

import numpy as np
import torch

from miniflex.common.hash import Hasher


def _hash_array(array: np.ndarray, prefix: int = 0) -> int:
  hasher = Hasher()
  hasher.update_numpy(np.array([prefix], dtype=np.int64))
  hasher.update_numpy(array)
  return hasher.digest()


class SharedOpPool:
  def __init__(self, max_op_num: int, max_block_num: int, dtype=np.int64):
    self.max_op_num = max_op_num
    self.max_block_num = max_block_num
    self.dtype = dtype
    self.buffer_o = torch.empty((self.max_op_num, self.max_block_num), dtype=torch.int64)
    self.buffer = self.buffer_o.share_memory_()

    self.free_slots = deque(range(max_op_num))
    self.slot_map = {}

    self.slot_ref_count = np.zeros(max_op_num, dtype=np.int32)
    self.slot_hashes = [0] * max_op_num

    self.lock = threading.Lock()

  def allocate_slot(self, block_ids: np.ndarray, device_type_prefix: int = 0):
    num_blocks = block_ids.size
    if num_blocks > self.max_block_num or num_blocks == 0:
      return -1

    slot_hash = _hash_array(block_ids, device_type_prefix)
    reuse = False

    with self.lock:
      if slot_hash in self.slot_map:
        slot_id = self.slot_map[slot_hash]
        reuse = True
      else:
        if not self.free_slots:
          return -1
        slot_id = self.free_slots.popleft()
        self.slot_map[slot_hash] = slot_id
      self.slot_ref_count[slot_id] += 1
      self.slot_hashes[slot_id] = slot_hash

    if not reuse:
      self.buffer[slot_id, :num_blocks] = torch.from_numpy(block_ids).to(torch.int64)

    return slot_id

  def free_slot(self, slot_id: int):
    with self.lock:
      slot_hash = self.slot_hashes[slot_id]
      if slot_hash not in self.slot_map:
        raise RuntimeError(f"Slot {slot_id} is not in use, double free detected!")
      self.slot_ref_count[slot_id] -= 1
      assert self.slot_ref_count[slot_id] >= 0, f"Slot {slot_id} ref count is negative"
      if self.slot_ref_count[slot_id] == 0:
        self.free_slots.append(slot_id)
        del self.slot_map[slot_hash]

  def get_buffer(self):
    return self.buffer

  def get_buffer_size(self):
    return self.max_op_num, self.max_block_num

  def status(self):
    with self.lock:
      used = len(self.slot_map)
      free = self.max_op_num - used
      return {
        "used_slots": used,
        "free_slots": free,
        "capacity": self.max_op_num,
      }

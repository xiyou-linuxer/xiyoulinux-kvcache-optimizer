from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from miniflex.cache.cache_engine import CacheEngine
from miniflex.cache.radix_tree import MatchResult
from miniflex.common.block import SequenceMeta
from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.metrics import incr as _metrics_incr
from miniflex.common.transfer import DeviceType, TransferOp, TransferOpGraph, TransferType


class GlobalCacheEngine:
  def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
    self.cache_config = cache_config
    self.model_config = model_config
    self.tokens_per_block = self.cache_config.tokens_per_block
    self.cpu_cache_engine = None
    self.ssd_cache_engine = None
    self.cache_engines = {}

    self.evict_ratio = self.cache_config.evict_ratio
    self.evict_start_threshold = self.cache_config.evict_start_threshold
    self.hit_add_counts = self.cache_config.hit_add_counts
    self.protected_threshold = self.cache_config.protected_threshold

    if self.cache_config.enable_cpu:
      self.cpu_cache_engine = CacheEngine(
        DeviceType.CPU,
        self.cache_config.num_cpu_blocks,
        self.tokens_per_block,
        self.cache_config.eviction_policy,
        self.hit_add_counts,
        self.evict_ratio,
        self.evict_start_threshold,
        self.protected_threshold,
      )
      self.cache_engines[DeviceType.CPU] = self.cpu_cache_engine
    if self.cache_config.enable_ssd:
      self.ssd_cache_engine = CacheEngine(
        DeviceType.SSD,
        self.cache_config.num_ssd_blocks,
        self.tokens_per_block,
        self.cache_config.eviction_policy,
        self.hit_add_counts,
        self.evict_ratio,
        self.evict_start_threshold,
        self.protected_threshold,
      )
      self.cache_engines[DeviceType.SSD] = self.ssd_cache_engine

    self._empty_get_return: Callable[
      [int], Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]
    ] = lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, {}, 0)
    self._empty_put_return: Callable[
      [int], Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]
    ] = lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, {}, 0, 0)

  def reset(self):
    for cache_engine in self.cache_engines.values():
      cache_engine.reset()

  def match_all(self, sequence: SequenceMeta) -> Tuple[MatchResult, MatchResult]:
    cpu_match_result = MatchResult()
    ssd_match_result = MatchResult()
    if self.cpu_cache_engine is not None:
      cpu_match_result = self.cpu_cache_engine.match(sequence)
    if self.ssd_cache_engine is not None:
      ssd_match_result = self.ssd_cache_engine.match(sequence)
    return cpu_match_result, ssd_match_result

  def _check_input(self, request_id: int, token_ids: np.ndarray, token_mask: np.ndarray, slot_mapping: np.ndarray) -> None:
    if request_id < 0:
      raise ValueError(f"request_id must be greater than 0, got {request_id}")
    if token_ids.ndim != 1:
      raise ValueError(f"token_ids must be a 1D array, got {token_ids.ndim}D")
    if token_ids.dtype != np.int64:
      raise ValueError(f"token_ids must be a 1D array of int64, got {token_ids.dtype}")
    if token_mask.ndim != 1:
      raise ValueError(f"token_mask must be a 1D array, got {token_mask.ndim}D")
    if token_mask.dtype != np.bool_:
      raise ValueError(f"token_mask must be a 1D array of bool, got {token_mask.dtype}")
    if slot_mapping.ndim != 1:
      raise ValueError(f"slot_mapping must be a 1D array, got {slot_mapping.ndim}D")
    if slot_mapping.dtype != np.int64:
      raise ValueError(f"slot_mapping must be a 1D array of int64, got {slot_mapping.dtype}")
    if len(token_ids) != len(token_mask):
      raise ValueError(f"token_ids and token_mask must have the same length, got {len(token_ids)} and {len(token_mask)}")
    if slot_mapping.size != int(token_mask.sum()):
      raise ValueError(f"slot_mapping size must equal token_mask sum, got {slot_mapping.size} and {int(token_mask.sum())}")

  def _get_block_range(self, token_mask: np.ndarray) -> Tuple[int, int]:
    mask_idx = np.where(token_mask)[0]
    if len(mask_idx) == 0:
      return 0, 0
    start_idx = mask_idx[0].item() // self.tokens_per_block
    end_idx = mask_idx[-1].item() // self.tokens_per_block + 1
    return start_idx, end_idx

  def _check_block_aligned_mask(self, token_mask: np.ndarray) -> None:
    mask_idx = np.where(token_mask)[0]
    if len(mask_idx) == 0:
      return
    first = mask_idx[0].item()
    last = mask_idx[-1].item()
    if len(mask_idx) != last - first + 1:
      raise ValueError("token_mask must describe a contiguous token range")
    if first % self.tokens_per_block != 0 or (last + 1) % self.tokens_per_block != 0:
      raise ValueError("token_mask must be aligned to complete token blocks")

  @staticmethod
  def slot_mapping_to_block_ids(slot_mapping: np.ndarray, tokens_per_block: int) -> np.ndarray:
    return slot_mapping[::tokens_per_block] // tokens_per_block

  def get(
    self,
    request_id: int,
    token_ids: np.ndarray,
    token_mask: np.ndarray,
    slot_mapping: np.ndarray,
    namespace: Optional[List[str]] = None,
  ) -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
    self._check_input(request_id, token_ids, token_mask, slot_mapping)
    aligned_length = token_ids.shape[0] // self.tokens_per_block * self.tokens_per_block
    aligned_token_ids = token_ids[:aligned_length]
    effective_token_mask = token_mask.copy()
    effective_token_mask[aligned_length:] = False
    self._check_block_aligned_mask(effective_token_mask)
    block_start_idx, block_end_idx = self._get_block_range(effective_token_mask)
    gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping, self.tokens_per_block)[:block_end_idx - block_start_idx]
    sequence = SequenceMeta(aligned_token_ids, self.tokens_per_block, namespace)

    (
      transfer_graph,
      finished_ops_ids,
      node_to_unlock,
      op_node_to_ready,
      buffer_to_free,
      num_gpu_blocks_to_transfer,
    ) = self.get_impl(request_id, sequence, block_start_idx, block_end_idx, gpu_block_ids)

    transfer_graph, task_end_op_id = self._add_task_end_virtual_op(transfer_graph, finished_ops_ids)
    return_mask = np.zeros_like(token_mask, dtype=np.bool_)
    return_mask[
      block_start_idx * self.tokens_per_block:
      (block_start_idx + num_gpu_blocks_to_transfer) * self.tokens_per_block
    ] = True

    for device_type, (node, _ready_length) in node_to_unlock.items():
      self.cache_engines[device_type].pin(node)

    callback = lambda: self._transfer_callback(node_to_unlock, buffer_to_free)
    op_callback_dict = {}
    for op_id, (device_type, node, ready_length) in op_node_to_ready.items():
      op_callback_dict[op_id] = lambda device_type=device_type, node=node, ready_length=ready_length: self._op_callback(device_type, node, ready_length)

    return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

  def put(
    self,
    request_id: int,
    token_ids: np.ndarray,
    token_mask: np.ndarray,
    slot_mapping: np.ndarray,
    namespace: Optional[List[str]] = None,
  ) -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
    self._check_input(request_id, token_ids, token_mask, slot_mapping)
    aligned_length = token_ids.shape[0] // self.tokens_per_block * self.tokens_per_block
    aligned_token_ids = token_ids[:aligned_length]
    effective_token_mask = token_mask.copy()
    effective_token_mask[aligned_length:] = False
    self._check_block_aligned_mask(effective_token_mask)
    block_start_idx, block_end_idx = self._get_block_range(effective_token_mask)
    if block_start_idx != 0:
      raise ValueError("put token_mask must start from block 0")

    gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping, self.tokens_per_block)[:block_end_idx - block_start_idx]
    put_token_len = block_end_idx * self.tokens_per_block
    sequence = SequenceMeta(aligned_token_ids[:put_token_len], self.tokens_per_block, namespace)

    (
      transfer_graph,
      finished_ops_ids,
      node_to_unlock,
      op_node_to_ready,
      buffer_to_free,
      num_gpu_blocks_to_transfer,
      skipped_gpu_blocks,
    ) = self.put_impl(request_id, sequence, block_start_idx, block_end_idx, gpu_block_ids)

    transfer_graph, task_end_op_id = self._add_task_end_virtual_op(transfer_graph, finished_ops_ids)
    return_mask = np.zeros_like(token_mask, dtype=np.bool_)
    return_mask[
      (block_start_idx + skipped_gpu_blocks) * self.tokens_per_block:
      (block_start_idx + skipped_gpu_blocks + num_gpu_blocks_to_transfer) * self.tokens_per_block
    ] = True

    for device_type, (node, _ready_length) in node_to_unlock.items():
      self.cache_engines[device_type].pin(node)

    callback = lambda: self._transfer_callback(node_to_unlock, buffer_to_free)
    op_callback_dict = {}
    for op_id, (device_type, node, ready_length) in op_node_to_ready.items():
      op_callback_dict[op_id] = lambda device_type=device_type, node=node, ready_length=ready_length: self._op_callback(device_type, node, ready_length)

    return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

  def get_impl(
    self,
    request_id: int,
    sequence: SequenceMeta,
    start_idx: int,
    end_idx: int,
    gpu_blocks: np.ndarray,
  ) -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]:
    if self.cpu_cache_engine is None:
      raise RuntimeError("CPU cache engine is required for MiniFlex GET")
    if start_idx < 0 or end_idx < start_idx:
      raise ValueError(f"invalid block range [{start_idx}, {end_idx})")
    if gpu_blocks.ndim != 1 or gpu_blocks.dtype != np.int64:
      raise ValueError(f"gpu_blocks must be a 1D array of int64, got shape={gpu_blocks.shape}, dtype={gpu_blocks.dtype}")

    cpu_match_result, ssd_match_result = self.match_all(sequence)
    cpu_matched_blocks = cpu_match_result.physical_block_ids[:cpu_match_result.num_ready_matched_blocks][start_idx:end_idx]
    ssd_matched_blocks = ssd_match_result.physical_block_ids[:ssd_match_result.num_ready_matched_blocks][start_idx:end_idx]

    if len(cpu_matched_blocks) > len(ssd_matched_blocks):
      ssd_matched_blocks = np.array([], dtype=np.int64)

    fragment1_num_blocks = len(cpu_matched_blocks)
    fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
    fragment2_num_blocks = max(fragment12_num_blocks - fragment1_num_blocks, 0)

    if fragment12_num_blocks == 0:
      _metrics_incr("miniflex_get_miss_blocks", end_idx - start_idx)
      return self._empty_get_return(request_id)
    if fragment12_num_blocks > len(gpu_blocks):
      raise RuntimeError(f"not enough GPU blocks for GET, required {fragment12_num_blocks}, got {len(gpu_blocks)}")

    transfer_graph = TransferOpGraph()
    finished_ops_ids = []
    node_to_unlock = {}
    op_node_to_ready = {}
    buffer_to_free = {}

    fragment12_gpu_blocks = gpu_blocks[:fragment12_num_blocks]
    fragment1_cpu_blocks = cpu_matched_blocks[:fragment1_num_blocks]
    fragment2_ssd_blocks = ssd_matched_blocks[fragment1_num_blocks:fragment12_num_blocks]
    fragment2_cpu_blocks = np.array([], dtype=np.int64)

    if fragment1_num_blocks > 0 and cpu_match_result.last_ready_node is not None:
      if not cpu_match_result.last_ready_node.is_root():
        node_to_unlock[DeviceType.CPU] = (cpu_match_result.last_ready_node, cpu_match_result.last_ready_node.size())
    if len(ssd_matched_blocks) > 0 and ssd_match_result.last_ready_node is not None:
      if not ssd_match_result.last_ready_node.is_root():
        node_to_unlock[DeviceType.SSD] = (ssd_match_result.last_ready_node, ssd_match_result.last_ready_node.size())

    op_disk2h = None
    if fragment2_num_blocks > 0:
      fragment2_cpu_blocks = self.cpu_cache_engine.take(
        num_required_blocks=fragment2_num_blocks,
        protected_node=cpu_match_result.last_node,
        strict=False,
      )
      if len(fragment2_cpu_blocks) < fragment2_num_blocks:
        self.cpu_cache_engine.recycle(fragment2_cpu_blocks)
        return self._empty_get_return(request_id)

      op_disk2h = TransferOp(
        transfer_type=TransferType.DISK2H,
        graph_id=transfer_graph.graph_id,
        src_block_ids=fragment2_ssd_blocks,
        dst_block_ids=fragment2_cpu_blocks,
      )
      transfer_graph.add_transfer_op(op_disk2h)

      can_insert_cpu_node = (
        cpu_match_result.num_ready_matched_blocks >= start_idx and
        cpu_match_result.num_ready_matched_blocks == cpu_match_result.num_matched_blocks
      )
      if can_insert_cpu_node:
        insert_num_blocks = start_idx + fragment12_num_blocks
        insert_token_len = insert_num_blocks * self.tokens_per_block
        insert_sequence = SequenceMeta(
          sequence.token_ids[:insert_token_len],
          sequence.tokens_per_block,
          sequence.namespace,
        )
        cpu_node_to_ready = self.cpu_cache_engine.insert(
          insert_sequence,
          fragment2_cpu_blocks,
          is_ready=False,
          match_result=cpu_match_result,
        )
        if cpu_node_to_ready is not None:
          ready_length = cpu_node_to_ready.size()
          node_to_unlock[DeviceType.CPU] = (cpu_node_to_ready, ready_length)
          op_node_to_ready[op_disk2h.op_id] = (DeviceType.CPU, cpu_node_to_ready, ready_length)
        else:
          buffer_to_free[DeviceType.CPU] = fragment2_cpu_blocks
      else:
        buffer_to_free[DeviceType.CPU] = fragment2_cpu_blocks

    fragment12_cpu_blocks = np.concatenate([fragment1_cpu_blocks, fragment2_cpu_blocks])
    op_h2d = TransferOp(
      transfer_type=TransferType.H2D,
      graph_id=transfer_graph.graph_id,
      src_block_ids=fragment12_cpu_blocks,
      dst_block_ids=fragment12_gpu_blocks,
    )
    transfer_graph.add_transfer_op(op_h2d)
    if op_disk2h is not None:
      transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
    finished_ops_ids.append(op_h2d.op_id)

    _metrics_incr("miniflex_get_hit_cpu_blocks", fragment1_num_blocks)
    if fragment2_num_blocks > 0:
      _metrics_incr("miniflex_get_hit_ssd_blocks", fragment2_num_blocks)

    return (
      transfer_graph,
      finished_ops_ids,
      node_to_unlock,
      op_node_to_ready,
      buffer_to_free,
      len(fragment12_gpu_blocks),
    )

  def put_impl(
    self,
    request_id: int,
    sequence: SequenceMeta,
    start_idx: int,
    end_idx: int,
    gpu_blocks: np.ndarray,
  ) -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]:
    if self.cpu_cache_engine is None:
      raise RuntimeError("CPU cache engine is required for MiniFlex PUT")
    if start_idx != 0 or end_idx < start_idx:
      raise ValueError(f"invalid PUT block range [{start_idx}, {end_idx})")
    if gpu_blocks.ndim != 1 or gpu_blocks.dtype != np.int64:
      raise ValueError(f"gpu_blocks must be a 1D array of int64, got shape={gpu_blocks.shape}, dtype={gpu_blocks.dtype}")

    enable_ssd = self.ssd_cache_engine is not None
    cpu_match_result, ssd_match_result = self.match_all(sequence)
    cpu_matched_blocks = cpu_match_result.physical_block_ids[:cpu_match_result.num_matched_blocks][start_idx:end_idx]
    ssd_matched_blocks = ssd_match_result.physical_block_ids[:ssd_match_result.num_matched_blocks][start_idx:end_idx]

    num_skipped_blocks = len(cpu_matched_blocks)
    fragment12_num_blocks = len(gpu_blocks) - num_skipped_blocks
    if fragment12_num_blocks == 0:
      return self._empty_put_return(request_id)
    if fragment12_num_blocks < 0:
      raise RuntimeError(f"CPU matched more blocks than requested, matched {num_skipped_blocks}, requested {len(gpu_blocks)}")

    fragment2_num_blocks = len(gpu_blocks) - len(ssd_matched_blocks)
    if not enable_ssd:
      fragment2_num_blocks = 0

    fragment12_gpu_blocks = gpu_blocks[num_skipped_blocks:]
    fragment12_cpu_blocks = self.cpu_cache_engine.take(
      num_required_blocks=fragment12_num_blocks,
      protected_node=cpu_match_result.last_node,
      strict=False,
    )
    if enable_ssd:
      fragment2_ssd_blocks = self.ssd_cache_engine.take(
        num_required_blocks=fragment2_num_blocks,
        protected_node=ssd_match_result.last_node,
        strict=False,
      )
    else:
      fragment2_ssd_blocks = np.array([], dtype=np.int64)

    if len(fragment12_cpu_blocks) < fragment12_num_blocks or len(fragment2_ssd_blocks) < fragment2_num_blocks:
      self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
      if enable_ssd:
        self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
      return self._empty_put_return(request_id)

    transfer_graph = TransferOpGraph()
    finished_ops_ids = []
    node_to_unlock = {}
    op_node_to_ready = {}
    buffer_to_free = {}

    op_d2h = TransferOp(
      transfer_type=TransferType.D2H,
      graph_id=transfer_graph.graph_id,
      src_block_ids=fragment12_gpu_blocks,
      dst_block_ids=fragment12_cpu_blocks,
    )
    transfer_graph.add_transfer_op(op_d2h)
    finished_ops_ids.append(op_d2h.op_id)

    op_h2disk = None
    if fragment2_num_blocks > 0:
      if len(fragment12_cpu_blocks) < fragment2_num_blocks:
        num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
        fragment2_cpu_blocks = np.concatenate([
          cpu_matched_blocks[-num_needed_from_cpu_matched:],
          fragment12_cpu_blocks,
        ])
      else:
        fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
      op_h2disk = TransferOp(
        transfer_type=TransferType.H2DISK,
        graph_id=transfer_graph.graph_id,
        src_block_ids=fragment2_cpu_blocks,
        dst_block_ids=fragment2_ssd_blocks,
      )
      transfer_graph.add_transfer_op(op_h2disk)
      transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

    cpu_node_to_ready = self.cpu_cache_engine.insert(
      sequence,
      fragment12_cpu_blocks,
      is_ready=False,
      match_result=cpu_match_result,
    )
    if cpu_node_to_ready is not None:
      ready_length = cpu_node_to_ready.size()
      node_to_unlock[DeviceType.CPU] = (cpu_node_to_ready, ready_length)
      op_node_to_ready[op_d2h.op_id] = (DeviceType.CPU, cpu_node_to_ready, ready_length)
    else:
      self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
      if enable_ssd:
        self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
      return self._empty_put_return(request_id)

    if fragment2_num_blocks > 0:
      ssd_node_to_ready = self.ssd_cache_engine.insert(
        sequence,
        fragment2_ssd_blocks,
        is_ready=False,
        match_result=ssd_match_result,
      )
      if ssd_node_to_ready is not None:
        ready_length = ssd_node_to_ready.size()
        node_to_unlock[DeviceType.SSD] = (ssd_node_to_ready, ready_length)
        op_node_to_ready[op_h2disk.op_id] = (DeviceType.SSD, ssd_node_to_ready, ready_length)
      else:
        buffer_to_free[DeviceType.SSD] = fragment2_ssd_blocks

    _metrics_incr("miniflex_put_skip_cpu_blocks", num_skipped_blocks)
    if fragment12_num_blocks > 0:
      _metrics_incr("miniflex_put_d2h_blocks", fragment12_num_blocks)
    if fragment2_num_blocks > 0:
      _metrics_incr("miniflex_put_h2disk_blocks", fragment2_num_blocks)

    return (
      transfer_graph,
      finished_ops_ids,
      node_to_unlock,
      op_node_to_ready,
      buffer_to_free,
      len(fragment12_gpu_blocks),
      num_skipped_blocks,
    )

  def _add_task_end_virtual_op(self, transfer_graph: TransferOpGraph, finished_ops_ids: List[int]) -> Tuple[TransferOpGraph, int]:
    if len(finished_ops_ids) == 0:
      return transfer_graph, -1
    if len(finished_ops_ids) == 1:
      return transfer_graph, finished_ops_ids[0]

    virtual_op = TransferOp(
      transfer_type=TransferType.VIRTUAL,
      graph_id=transfer_graph.graph_id,
      src_block_ids=np.array([], dtype=np.int64),
      dst_block_ids=np.array([], dtype=np.int64),
    )
    transfer_graph.add_virtual_op(virtual_op)
    for op_id in finished_ops_ids:
      transfer_graph.add_dependency(virtual_op.op_id, op_id)
    return transfer_graph, virtual_op.op_id

  def _transfer_callback(self, node_to_unlock: Dict[DeviceType, object], buffer_to_free: Dict[DeviceType, np.ndarray]) -> None:
    for device_type, (node, ready_length) in node_to_unlock.items():
      self.cache_engines[device_type].unpin(node)
      self.cache_engines[device_type].set_ready(node, True, ready_length)
    for device_type, blocks in buffer_to_free.items():
      self.cache_engines[device_type].recycle(blocks)

  def _op_callback(self, device_type: DeviceType, node, ready_length: int) -> None:
    self.cache_engines[device_type].set_ready(node, True, ready_length)

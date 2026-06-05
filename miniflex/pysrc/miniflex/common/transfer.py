from dataclasses import dataclass,field
from enum import Enum,IntEnum
import numpy as np
import threading
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

class TransferType(Enum):
  D2H = "D2H"
  H2D = "H2D"
  H2DISK = "H2DISK"
  DISK2H = "DISK2H"
  VIRTUAL = "VIRTUAL"

class DeviceType(IntEnum):
  CPU = 0
  GPU = 1
  SSD = 2
  
class TransferOpStatus(Enum):
  PENDING = "PENDING"
  RUNNING = "RUNNING"
  COMPLETED = "COMPLETED"

@dataclass
class TransferOp:
  transfer_type: TransferType
  
  op_id: int = field(init = False)
  graph_id: int
  src_block_ids: np.ndarray
  dst_block_ids: np.ndarray
  src_slot_id: int = -1
  dst_slot_id: int = -1
  valid_block_num: int = 0
  
  depends_on: set[int] = field(default_factory=set)
  dependents: set[int] = field(default_factory=set)
  
  status: TransferOpStatus = TransferOpStatus.PENDING
  _next_op_id: ClassVar[int] = 0
  _lock: ClassVar[threading.Lock] = threading.Lock()

  def __post_init__(self):
    if self.src_block_ids.ndim != 1 or self.dst_block_ids.ndim != 1:
      raise ValueError("src_block_ids and dst_block_ids must be 1D arrays")
    if self.src_block_ids.dtype != np.int64 or self.dst_block_ids.dtype != np.int64:
      raise ValueError("src_block_ids and dst_block_ids must be 1D arrays of int64")
    if self.src_block_ids.size != self.dst_block_ids.size and self.transfer_type != TransferType.VIRTUAL:
      raise ValueError("src_block_ids and dst_block_ids must have the same size")
    if self.transfer_type not in [TransferType.D2H, TransferType.H2D, TransferType.H2DISK, TransferType.DISK2H, TransferType.VIRTUAL]:
      raise ValueError("invalid transfer type")
    if self.graph_id < 0:
      raise ValueError("graph_id must be non-negative")
    self.valid_block_num = self.src_block_ids.size
    with TransferOp._lock:
      self.op_id = TransferOp._next_op_id
      TransferOp._next_op_id += 1

@dataclass(frozen=True)
class CompletedOp:
  graph_id: int
  op_id: int
  
  def is_graph_completed(self) -> bool:
    return self.op_id == -1
  
  def to_tuple(self) -> Tuple[int, int]:
    return (self.graph_id, self.op_id)
  
  @classmethod
  def from_tuple(cls, data: Tuple[int, int]) -> "CompletedOp":
    return cls(graph_id=data[0], op_id=data[1])
  
  @classmethod
  def completed_graph(cls, graph_id: int) -> "CompletedOp":
    return cls(graph_id=graph_id, op_id=-1)
  
  
class TransferOpGraph:
  _next_graph_id: ClassVar[int] = 0
  _lock: ClassVar[threading.Lock] = threading.Lock()
  
  def __init__(self):
    self.graph_id = self._get_next_graph_id()
    self._op_map: Dict[int, TransferOp] = {}
    self._ready_ops: set[int] = set()
    self._gpu_transfer_ops: list[int] = []
    
  @classmethod
  def _get_next_graph_id(cls):
    with cls._lock:
      graph_id = cls._next_graph_id
      cls._next_graph_id += 1
      return graph_id
    
  def set_graph_id(self, graph_id: int):
    self.graph_id = graph_id
    
    
  @classmethod
  def create_empty_graph(cls) -> "TransferOpGraph":
    return cls()
  
  def add_virtual_op(self, op: TransferOp):
    if op.transfer_type != TransferType.VIRTUAL:
      raise ValueError("op must be a virtual op")
    op.graph_id = self.graph_id
    self._op_map[op.op_id] = op
    self._ready_ops.add(op.op_id)
    
  def add_transfer_op(self, op: TransferOp):
    op.graph_id = self.graph_id
    self._op_map[op.op_id] = op
    if op.transfer_type in [TransferType.D2H, TransferType.H2D]:
      self._gpu_transfer_ops.append(op.op_id)
    self._ready_ops.add(op.op_id)
    
  def add_dependency(self, op_id: int, dependency_op_id: int):
    if op_id not in self._op_map or dependency_op_id not in self._op_map:
      raise ValueError("op_id or dependency_op_id not found")
    if self._op_map[op_id].status != TransferOpStatus.PENDING:
      raise ValueError("op must be pending")
    op = self._op_map[op_id]
    dependency_op = self._op_map[dependency_op_id]
    op.depends_on.add(dependency_op_id)
    dependency_op.dependents.add(op_id)
    self._ready_ops.discard(op_id)
    
  def mark_completed(self, op_id: int):
    if op_id not in self._op_map:
      raise ValueError("op_id not found")
    op = self._op_map[op_id]
    if op.status != TransferOpStatus.RUNNING:
      raise ValueError("op must be running")
    if op_id not in self._ready_ops:
      raise ValueError("op_id not ready")
    op.status = TransferOpStatus.COMPLETED
    self._ready_ops.discard(op_id)
    for dependent_op_id in op.dependents:
      if dependent_op_id not in self._op_map:
        continue
      dependent_op = self._op_map[dependent_op_id]
      dependent_op.depends_on.discard(op_id)
      if not dependent_op.depends_on:
        self._ready_ops.add(dependent_op_id)
    
  def take_ready_ops(self) -> list[TransferOp]:
    ready_ops = []
    for op_id in self._ready_ops:
      if op_id not in self._op_map:
        continue
      op = self._op_map[op_id]
      if op.status != TransferOpStatus.PENDING:
        continue
      op.status = TransferOpStatus.RUNNING
      ready_ops.append(op)
    return ready_ops

  def all_transfer_ops_completed(self) -> bool:
    return all(op.status == TransferOpStatus.COMPLETED for op in self._op_map.values())
  
  @property
  def num_ops(self) -> int:
    return len(self._op_map)
  
  def set_gpu_blocks(self, blocks: np.ndarray):
    if blocks.ndim != 1 or blocks.dtype != np.int64:
      raise ValueError("blocks must be a 1D array of int64")
    for op_id in self._gpu_transfer_ops:
      transfer_type = self._op_map[op_id].transfer_type
      op = self._op_map[op_id]
      match transfer_type:
        case TransferType.H2D:
          op.dst_block_ids = blocks[:op.dst_block_ids.size]
        case TransferType.D2H:
          op.src_block_ids = blocks[:op.src_block_ids.size]
      if op.src_block_ids.size != op.dst_block_ids.size: 
        raise ValueError("src_block_ids and dst_block_ids must have the same size")


def _make_combined_callback(callbacks: List[Callable]) -> Callable:
  def combined_callback(*args, **kwargs):
    for callback in callbacks:
      callback(*args, **kwargs)
  return combined_callback


def _merge_ops(ops: List[TransferOp],
               transfer_type: TransferType,
               graph: TransferOpGraph,
               callbacks: List[Callable],
               op_callback_dict: Dict[int, Callable]) -> Optional[TransferOp]:
  if not ops:
    return None
  src_blocks = np.concatenate([op.src_block_ids for op in ops])
  dst_blocks = np.concatenate([op.dst_block_ids for op in ops])
  merged_op = TransferOp(
    transfer_type=transfer_type,
    graph_id=graph.graph_id,
    src_block_ids=src_blocks,
    dst_block_ids=dst_blocks,
  )
  graph.add_transfer_op(merged_op)
  if callbacks:
    if len(callbacks) == 1:
      op_callback_dict[merged_op.op_id] = callbacks[0]
    else:
      op_callback_dict[merged_op.op_id] = _make_combined_callback(callbacks)
  return merged_op


def merge_to_batch_graph(batch_id: int,
                         transfer_graphs: List[TransferOpGraph],
                         task_end_op_ids: List[int],
                         op_callback_dict: Dict[int, Callable]) -> Tuple[TransferOpGraph, int, Dict[int, Callable]]:
  if not transfer_graphs:
    empty_graph = TransferOpGraph()
    empty_graph.set_graph_id(batch_id)
    return empty_graph, -1, {}

  merged_graph = TransferOpGraph()
  merged_graph.set_graph_id(batch_id)

  supported_types = {TransferType.DISK2H, TransferType.H2D, TransferType.D2H, TransferType.H2DISK}
  ops_by_type: Dict[TransferType, List[TransferOp]] = {transfer_type: [] for transfer_type in supported_types}
  callbacks_by_type: Dict[TransferType, List[Callable]] = {transfer_type: [] for transfer_type in supported_types}

  for graph in transfer_graphs:
    for op_id, op in graph._op_map.items():
      if op.transfer_type == TransferType.VIRTUAL:
        continue
      if op.transfer_type not in supported_types:
        raise NotImplementedError(
          f'batch merge does not support transfer type: {op.transfer_type}'
        )
      ops_by_type[op.transfer_type].append(op)
      callback = op_callback_dict.get(op_id)
      if callback is not None:
        callbacks_by_type[op.transfer_type].append(callback)

  new_op_callback_dict: Dict[int, Callable] = {}

  merged_disk2h_op = _merge_ops(
    ops_by_type[TransferType.DISK2H],
    TransferType.DISK2H,
    merged_graph,
    callbacks_by_type[TransferType.DISK2H],
    new_op_callback_dict,
  )
  merged_h2d_op = _merge_ops(
    ops_by_type[TransferType.H2D],
    TransferType.H2D,
    merged_graph,
    callbacks_by_type[TransferType.H2D],
    new_op_callback_dict,
  )
  if merged_disk2h_op is not None and merged_h2d_op is not None:
    merged_graph.add_dependency(merged_h2d_op.op_id, merged_disk2h_op.op_id)

  merged_d2h_op = _merge_ops(
    ops_by_type[TransferType.D2H],
    TransferType.D2H,
    merged_graph,
    callbacks_by_type[TransferType.D2H],
    new_op_callback_dict,
  )
  merged_h2disk_op = _merge_ops(
    ops_by_type[TransferType.H2DISK],
    TransferType.H2DISK,
    merged_graph,
    callbacks_by_type[TransferType.H2DISK],
    new_op_callback_dict,
  )
  if merged_d2h_op is not None and merged_h2disk_op is not None:
    merged_graph.add_dependency(merged_h2disk_op.op_id, merged_d2h_op.op_id)

  original_end_types = []
  for graph, task_end_op_id in zip(transfer_graphs, task_end_op_ids):
    if task_end_op_id in graph._op_map:
      original_end_types.append(graph._op_map[task_end_op_id].transfer_type)

  merged_end_ops = {
    TransferType.H2D: merged_h2d_op,
    TransferType.DISK2H: merged_disk2h_op,
    TransferType.D2H: merged_d2h_op,
    TransferType.H2DISK: merged_h2disk_op,
  }
  batch_end_op_id = -1
  for transfer_type in original_end_types:
    merged_end_op = merged_end_ops.get(transfer_type)
    if merged_end_op is not None:
      batch_end_op_id = merged_end_op.op_id
      break

  if batch_end_op_id == -1:
    if merged_h2d_op is not None:
      batch_end_op_id = merged_h2d_op.op_id
    elif merged_disk2h_op is not None:
      batch_end_op_id = merged_disk2h_op.op_id
    elif merged_h2disk_op is not None:
      batch_end_op_id = merged_h2disk_op.op_id
    elif merged_d2h_op is not None:
      batch_end_op_id = merged_d2h_op.op_id

  return merged_graph, batch_end_op_id, new_op_callback_dict

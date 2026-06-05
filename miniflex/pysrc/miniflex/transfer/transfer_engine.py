import os
import queue
import selectors
import threading
import time
from typing import Dict, List, Optional, Union
import multiprocessing as mp

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.ring_buffer import SharedOpPool
from miniflex.common.storage import StorageHandle, StorageHandlerType
from miniflex.common.transfer import (
  CompletedOp,
  DeviceType,
  TransferOp,
  TransferOpGraph,
  TransferType,
)
from miniflex.transfer.scheduler import TransferScheduler
from miniflex.transfer.worker import GPUCPUTransferWorker, SSDCPUTransferWorker, WorkerHandle


class TransferEngine:
  def __init__(self,
               model_config: ModelConfig,
               cache_config: CacheConfig,
               gpu_handle: StorageHandle,
               cpu_handle: StorageHandle,
               ssd_handle: Optional[StorageHandle]) -> None:
    self.model_config = model_config
    self.cache_config = cache_config
    self._gpu_handle = gpu_handle
    self._cpu_handle = cpu_handle
    self._ssd_handle = ssd_handle

    if self._cpu_handle.handle_type not in (StorageHandlerType.TENSOR, StorageHandlerType.TENSOR_HANDLE):
      raise ValueError("cpu_handle must be tensor-backed")
    if self._gpu_handle.handle_type not in (StorageHandlerType.TENSOR, StorageHandlerType.TENSOR_HANDLE):
      raise ValueError("gpu_handle must be tensor-backed")
    compatible_layout_fields = ("num_layers", "tokens_per_block", "num_heads", "head_size", "use_mla")
    for field_name in compatible_layout_fields:
      if getattr(self._gpu_handle.kv_layout, field_name) != getattr(self._cpu_handle.kv_layout, field_name):
        raise ValueError(f"gpu_handle and cpu_handle layout mismatch: {field_name}")
    if self._gpu_handle.dtype != self._cpu_handle.dtype:
      raise ValueError("gpu_handle and cpu_handle dtype must match")

    if self.cache_config.enable_ssd:
      if self._ssd_handle is None:
        raise ValueError("ssd_handle is required when enable_ssd is True")
      if self._cpu_handle.handle_type != StorageHandlerType.TENSOR:
        raise ValueError("cpu_handle must be tensor-backed for SSD transfer")
      if self._ssd_handle.handle_type != StorageHandlerType.FILE:
        raise ValueError("ssd_handle must be file-backed")
      if self._ssd_handle.num_blocks_per_file is None:
        raise ValueError("ssd_handle must provide num_blocks_per_file")
      if len(self._ssd_handle.get_file_list()) * self._ssd_handle.num_blocks_per_file < self._ssd_handle.kv_layout.num_blocks:
        raise ValueError("ssd_handle files do not cover SSD layout blocks")
      for field_name in compatible_layout_fields:
        if getattr(self._cpu_handle.kv_layout, field_name) != getattr(self._ssd_handle.kv_layout, field_name):
          raise ValueError(f"cpu_handle and ssd_handle layout mismatch: {field_name}")
      if self._cpu_handle.dtype != self._ssd_handle.dtype:
        raise ValueError("cpu_handle and ssd_handle dtype must match")

    self._scheduler = TransferScheduler()

    self._mp_ctx = mp.get_context("spawn")
    self._task_queue = self._mp_ctx.Queue()
    self._completed_queue = self._mp_ctx.Queue()
    self._finished_op_queue = self._mp_ctx.Queue()
    self._op_id_to_op: Dict[int, TransferOp] = {}

    self.shutdown_read_fd, self.shutdown_write_fd = os.pipe()

    max_block_num = max(self.cache_config.num_cpu_blocks, self.cache_config.num_ssd_blocks)
    self._pin_buffer = SharedOpPool(2048, max_block_num)
    self._transfer_worker: Dict[TransferType, WorkerHandle] = {}
    self._running = False
    self._scheduler_thread: Optional[threading.Thread] = None

  def start(self) -> None:
    self._init_workers()

  def _init_workers(self) -> None:
    if self._running:
      return

    if self.cache_config.enable_cpu:
      self._h2d_worker = GPUCPUTransferWorker.create_worker(
        self._mp_ctx,
        self._finished_op_queue,
        self._pin_buffer.get_buffer(),
        self._gpu_handle,
        self._cpu_handle,
      )
      self._transfer_worker[TransferType.H2D] = self._h2d_worker
      self._d2h_worker = GPUCPUTransferWorker.create_worker(
        self._mp_ctx,
        self._finished_op_queue,
        self._pin_buffer.get_buffer(),
        self._gpu_handle,
        self._cpu_handle,
      )
      self._transfer_worker[TransferType.D2H] = self._d2h_worker

    if self.cache_config.enable_ssd:
      if self._ssd_handle is None:
        raise ValueError("ssd_handle is required when enable_ssd is True")
      self._h2disk_worker = SSDCPUTransferWorker.create_worker(
        self._mp_ctx,
        self._finished_op_queue,
        self._pin_buffer.get_buffer(),
        self._cpu_handle,
        self._ssd_handle,
        self.cache_config.use_direct_io,
      )
      self._transfer_worker[TransferType.H2DISK] = self._h2disk_worker
      self._disk2h_worker = SSDCPUTransferWorker.create_worker(
        self._mp_ctx,
        self._finished_op_queue,
        self._pin_buffer.get_buffer(),
        self._cpu_handle,
        self._ssd_handle,
        self.cache_config.use_direct_io,
      )
      self._transfer_worker[TransferType.DISK2H] = self._disk2h_worker

    if len(self._transfer_worker) == 0:
      raise ValueError("No transfer workers created")

    for worker in self._transfer_worker.values():
      if not worker.ready_event.wait(timeout=10):
        raise RuntimeError(f"transfer worker {worker.worker_id} did not become ready")

    self._running = True
    self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
    self._scheduler_thread.start()

  def _scheduler_loop(self) -> None:
    sel = selectors.DefaultSelector()
    sel.register(self._task_queue._reader, selectors.EVENT_READ, data=self._handle_task_queue)
    sel.register(self.shutdown_read_fd, selectors.EVENT_READ, data=self._handle_shutdown_fd)
    sel.register(self._finished_op_queue._reader, selectors.EVENT_READ, data=self._handle_finished_op_queue)

    while self._running:
      try:
        events = sel.select(timeout=None)

        new_graphs_num = 0
        finished_ops: List[TransferOp] = []
        should_shutdown = False
        for key, _mask in events:
          callback = key.data
          if not callable(callback):
            raise ValueError(f"Expected callable selector data, got {type(callback)}")
          callback_result = callback()
          if isinstance(callback_result, bool):
            should_shutdown = should_shutdown or callback_result
          elif isinstance(callback_result, int):
            new_graphs_num += callback_result
          elif isinstance(callback_result, list):
            finished_ops.extend(callback_result)
          else:
            raise ValueError(f"Expected bool, int, or list, got {type(callback_result)}")

        if should_shutdown:
          break
        if finished_ops or new_graphs_num > 0:
          completed_graphs, next_ops = self._scheduler.schedule(finished_ops)
          for op in next_ops:
            if op.transfer_type == TransferType.VIRTUAL:
              self._completed_queue.put(CompletedOp(op.graph_id, op.op_id))
            else:
              self._register_op_buffer(op)
              self._op_id_to_op[op.op_id] = op
              self._assign_op_to_worker(op)
          for graph_id in completed_graphs:
            self._completed_queue.put(CompletedOp.completed_graph(graph_id))
      except Exception as e:
        print(f"Error in scheduler loop: {type(e).__name__}: {e}")
        time.sleep(0.001)
    sel.close()

  def _handle_task_queue(self) -> int:
    new_graphs_num = 0
    while True:
      try:
        transfer_graph = self._task_queue.get_nowait()
        graphs = transfer_graph if isinstance(transfer_graph, list) else [transfer_graph]
        for graph in graphs:
          self._scheduler.add_transfer_graph(graph)
        new_graphs_num += len(graphs)
      except queue.Empty:
        break
    return new_graphs_num

  def _handle_shutdown_fd(self) -> bool:
    try:
      os.read(self.shutdown_read_fd, 1024)
    except OSError:
      pass
    return True

  def _handle_finished_op_queue(self) -> List[TransferOp]:
    finished_ops = []
    while True:
      try:
        op_id = self._finished_op_queue.get_nowait()
        op = self._op_id_to_op.pop(op_id)
        self._free_op_buffer(op)
        self._completed_queue.put(CompletedOp(op.graph_id, op.op_id))
        finished_ops.append(op)
      except queue.Empty:
        break
    return finished_ops

  def _assign_op_to_worker(self, op: TransferOp) -> None:
    if op.transfer_type not in self._transfer_worker:
      raise ValueError(f"Unsupported transfer type: {op.transfer_type}")
    self._transfer_worker[op.transfer_type].submit_transfer(op)

  def _register_op_buffer(self, op: TransferOp) -> None:
    transfer_devices = {
      TransferType.D2H: (DeviceType.GPU, DeviceType.CPU),
      TransferType.H2D: (DeviceType.CPU, DeviceType.GPU),
      TransferType.H2DISK: (DeviceType.CPU, DeviceType.SSD),
      TransferType.DISK2H: (DeviceType.SSD, DeviceType.CPU),
    }
    src_device, dst_device = transfer_devices.get(op.transfer_type, (0, 0))
    op.src_slot_id = self._pin_buffer.allocate_slot(op.src_block_ids, int(src_device))
    op.dst_slot_id = self._pin_buffer.allocate_slot(op.dst_block_ids, int(dst_device))

  def _free_op_buffer(self, op: TransferOp) -> None:
    if op.src_slot_id != -1:
      self._pin_buffer.free_slot(op.src_slot_id)
      op.src_slot_id = -1
    if op.dst_slot_id != -1:
      self._pin_buffer.free_slot(op.dst_slot_id)
      op.dst_slot_id = -1

  def submit_transfer_graph(self, transfer_graph: Union[TransferOpGraph, List[TransferOpGraph]]) -> None:
    if isinstance(transfer_graph, TransferOpGraph):
      self._task_queue.put(transfer_graph)
      return
    if isinstance(transfer_graph, list):
      if not all(isinstance(graph, TransferOpGraph) for graph in transfer_graph):
        raise ValueError("transfer_graph list must contain only TransferOpGraph")
      self._task_queue.put(transfer_graph)
      return
    raise ValueError("transfer_graph must be a TransferOpGraph or list[TransferOpGraph]")

  def get_completed_graphs_and_ops(self, timeout: Optional[float] = None) -> List[CompletedOp]:
    completed_ops: List[CompletedOp] = []
    try:
      if timeout is None or timeout == 0:
        first_completed_op = self._completed_queue.get_nowait()
      else:
        first_completed_op = self._completed_queue.get(timeout=timeout)
      completed_ops.append(first_completed_op)
    except queue.Empty:
      return completed_ops

    while True:
      try:
        completed_ops.append(self._completed_queue.get_nowait())
      except queue.Empty:
        break
    return completed_ops

  def shutdown(self) -> None:
    if not self._running:
      return
    self._running = False
    try:
      os.write(self.shutdown_write_fd, b"1")
    except OSError:
      pass
    if self._scheduler_thread is not None:
      self._scheduler_thread.join(timeout=5)

    for worker in self._transfer_worker.values():
      worker.shutdown()
    self._transfer_worker.clear()

    for fd in (self.shutdown_read_fd, self.shutdown_write_fd):
      try:
        os.close(fd)
      except OSError:
        pass

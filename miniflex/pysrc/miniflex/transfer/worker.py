from abc import ABC, abstractmethod
import ctypes
import logging
from multiprocessing.connection import Connection
import multiprocessing as mp
import threading
from typing import Any, Tuple
from miniflex.common.transfer import TransferType
import numpy as np
from miniflex.common.transfer import TransferOp
from multiprocessing import Queue as MPQueue
import torch
from miniflex.common.storage import KVCacheLayoutType, StorageHandle, StorageHandlerType
import miniflex._C as _C

logger = logging.getLogger(__name__)

_CUDA_HOST_REGISTER_PORTABLE = 1
_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
_CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
_cudart = None
_cudart_load_failed = False
_cudart_lock = threading.Lock()


def _get_cudart():
  global _cudart
  global _cudart_load_failed
  with _cudart_lock:
    if _cudart is not None:
      return _cudart
    if _cudart_load_failed:
      return None
    try:
      _cudart = ctypes.CDLL("libcudart.so")
    except OSError:
      _cudart_load_failed = True
      return None
    return _cudart


def cuda_host_register(tensor: torch.Tensor, *, required: bool = False) -> bool:
  if not isinstance(tensor, torch.Tensor):
    raise ValueError("tensor must be a torch.Tensor")
  if tensor.is_cuda or tensor.numel() == 0:
    return False
  cudart = _get_cudart()
  if cudart is None:
    if required:
      raise RuntimeError("failed to load libcudart.so for cudaHostRegister")
    logger.debug("skip cudaHostRegister because libcudart.so is unavailable")
    return False
  ptr = tensor.data_ptr()
  size = tensor.numel() * tensor.element_size()
  ret = cudart.cudaHostRegister(
    ctypes.c_void_p(ptr),
    ctypes.c_size_t(size),
    _CUDA_HOST_REGISTER_PORTABLE,
  )
  if ret == 0 or ret == _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
    return ret == 0
  if required:
    raise RuntimeError(f"cudaHostRegister failed with error code {ret}")
  logger.debug("cudaHostRegister skipped/failed with error code %s", ret)
  return False


def cuda_host_unregister(tensor: torch.Tensor) -> None:
  if not isinstance(tensor, torch.Tensor) or tensor.is_cuda or tensor.numel() == 0:
    return
  cudart = _get_cudart()
  if cudart is None:
    return
  ret = cudart.cudaHostUnregister(ctypes.c_void_p(tensor.data_ptr()))
  if ret not in (0, _CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED):
    logger.debug("cudaHostUnregister failed with error code %s", ret)


class WorkerTransferOp:
  transfer_op_id: int
  transfer_graph_id: int
  transfer_type: TransferType
  src_slot_id: int
  dst_slot_id: int
  valid_block_num: int
  src_block_ids: np.ndarray
  dst_block_ids: np.ndarray
  
  def __init__(self, transfer_op: TransferOp) -> None:
    self.transfer_op_id = transfer_op.op_id
    self.transfer_graph_id = transfer_op.graph_id
    self.transfer_type = transfer_op.transfer_type
    self.src_slot_id = transfer_op.src_slot_id
    self.dst_slot_id = transfer_op.dst_slot_id
    self.valid_block_num = transfer_op.valid_block_num
    if self.src_slot_id == -1 or self.dst_slot_id == -1:
      self.src_block_ids = transfer_op.src_block_ids
      self.dst_block_ids = transfer_op.dst_block_ids
    else:
      self.src_block_ids = np.empty(0, dtype=np.int64)
      self.dst_block_ids = np.empty(0, dtype=np.int64)
      
class WorkerHandle:
  def __init__(self,
               worker_id: int,
               transfer_conn: Connection,
               process: mp.Process,
               ready_event: Any):
    self.worker_id = worker_id
    self.transfer_conn = transfer_conn
    self.process = process
    self.ready_event = ready_event
    
  def submit_transfer(self, transfer_op: TransferOp) -> None:
    self.transfer_conn.send(WorkerTransferOp(transfer_op))
    
  def shutdown(self) -> None:
    try:
      self.transfer_conn.send(None)
      self.transfer_conn.close()
    except (BrokenPipeError, OSError):
      logger.exception("failed to send shutdown signal to worker %s", self.worker_id)
    self.process.join(timeout=5)
    if self.process.is_alive():
      self.process.terminate()
      self.process.join()
  
  def __del__(self) -> None:
    if self.process.is_alive():
      self.shutdown()

    
class TransferWorkerBase(ABC):
  _worker_id_counter = 0
  _worker_id_lock = threading.Lock()
  def __init__(self, 
               worker_id: int,
               transfer_conn: Connection,
               finished_ops_queue: MPQueue,
               op_buffer_tensor: torch.Tensor):
    self.worker_id = worker_id
    self.transfer_conn = transfer_conn
    self.finished_ops_queue = finished_ops_queue
    
    self.op_buffer_tensor = op_buffer_tensor
    self._cuda_registered_tensors: list[torch.Tensor] = []
    self._register_cuda_host_tensor(self.op_buffer_tensor)

  def _register_cuda_host_tensor(self, tensor: torch.Tensor, *, required: bool = False) -> None:
    if cuda_host_register(tensor, required=required):
      self._cuda_registered_tensors.append(tensor)

  def shutdown(self) -> None:
    for tensor in reversed(self._cuda_registered_tensors):
      cuda_host_unregister(tensor)
    self._cuda_registered_tensors.clear()
  
  @classmethod
  def _get_worker_id(cls) -> int:
    with cls._worker_id_lock:
      worker_id = cls._worker_id_counter
      cls._worker_id_counter += 1
      return worker_id
    
  def get_transfer_block_ids(self, transfer_op: WorkerTransferOp) -> Tuple[torch.Tensor, torch.Tensor]:
    src_slot_id = transfer_op.src_slot_id
    dst_slot_id = transfer_op.dst_slot_id
    valid_block_num = transfer_op.valid_block_num
    if src_slot_id == -1:
      src_block_ids = torch.from_numpy(transfer_op.src_block_ids).to(torch.int64)
    else:
      src_block_ids = self.op_buffer_tensor[src_slot_id, :valid_block_num]
    if dst_slot_id == -1:
      dst_block_ids = torch.from_numpy(transfer_op.dst_block_ids).to(torch.int64)
    else:
      dst_block_ids = self.op_buffer_tensor[dst_slot_id, :valid_block_num]
    if src_block_ids.shape != dst_block_ids.shape:
      raise ValueError("src_block_ids and dst_block_ids must have the same shape")
    if src_block_ids.dtype != torch.int64 or dst_block_ids.dtype != torch.int64:
      raise ValueError("src_block_ids and dst_block_ids must be int64")
    if src_block_ids.numel() != valid_block_num or dst_block_ids.numel() != valid_block_num:
      raise ValueError("src_block_ids and dst_block_ids must have the same size")
    return src_block_ids, dst_block_ids
  
  @abstractmethod
  def _transfer_impl(self,
                     src_block_ids: torch.Tensor,
                     dst_block_ids: torch.Tensor,
                     transfer_type: TransferType,
                     **kwargs: Any
                     ):
    pass
  
  @abstractmethod
  def launch_transfer(self,transfer_op: WorkerTransferOp):
    pass
  
  @classmethod
  def create_worker(cls,
                    mp_ctx: Any,
                    finished_op_queue: MPQueue,
                    op_buffer_tensor: torch.Tensor,
                    *args: Any,
                    **kwargs: Any) -> WorkerHandle:
    parent_conn, child_conn = mp_ctx.Pipe()
    ready_event = mp_ctx.Event()
    worker_id = cls._get_worker_id()
    
    process = mp_ctx.Process(
      target=cls._worker_process,
      args=(worker_id, child_conn, finished_op_queue, op_buffer_tensor, ready_event, *args,),
      kwargs=kwargs,
      daemon=True,
    )
    process.start()
    return WorkerHandle(worker_id, parent_conn, process, ready_event)
  
  
  
  @classmethod
  def _worker_process(cls,
                      worker_id: int,
                      transfer_conn: Connection,
                      finished_ops_queue: MPQueue,
                      op_buffer_tensor: torch.Tensor,
                      ready_event: Any,
                      *args: Any,
                      **kwargs: Any) -> None:
    worker = cls(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor, *args, **kwargs)
    ready_event.set()
    worker.run()
    
  
  def run(self) -> None:
    should_shutdown = False
    while True:
      try:
        if self.transfer_conn.poll(timeout=0.0001):
          op = self.transfer_conn.recv()
          if op is None:
            if hasattr(self, 'shutdown') and callable(self.shutdown):
              try:
                self.shutdown()
              except Exception:
                logger.exception("worker %s shutdown hook failed", self.worker_id)
            break
          batch_op = [op]
          while self.transfer_conn.poll(timeout=0):
            op = self.transfer_conn.recv()
            if op is None:
              should_shutdown = True
              break
            batch_op.append(op)
          for op in batch_op:
            transfer_status = False
            try:
              transfer_status = self.launch_transfer(op)
            except Exception:
              logger.exception(
                "worker %s failed to launch transfer op_id=%s graph_id=%s type=%s",
                self.worker_id,
                getattr(op, "transfer_op_id", None),
                getattr(op, "transfer_graph_id", None),
                getattr(op, "transfer_type", None),
              )
            if transfer_status:
              self.finished_ops_queue.put(op.transfer_op_id)
          if should_shutdown:
            if hasattr(self, 'shutdown') and callable(self.shutdown):
              try:
                self.shutdown()
              except Exception:
                logger.exception("worker %s shutdown hook failed", self.worker_id)
            break
        else:
          continue
      except EOFError:
        break
      except Exception:
        logger.exception("worker %s run loop failed", self.worker_id)
        continue


class GPUCPUTransferWorker(TransferWorkerBase):
  def __init__(self,
               worker_id: int,
               transfer_conn: Connection,
               finished_ops_queue: MPQueue,
               op_buffer_tensor: torch.Tensor,
               gpu_storage_handle: StorageHandle,
               cpu_storage_handle: StorageHandle):
    super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
    self.gpu_storage_handle = gpu_storage_handle
    self.cpu_storage_handle = cpu_storage_handle
    gpu_layout = gpu_storage_handle.kv_layout
    cpu_layout = cpu_storage_handle.kv_layout
    self.num_layers = gpu_layout.num_layers
    self.gpu_layout = gpu_layout
    self.cpu_layout = cpu_layout
    self.gpu_num_blocks = gpu_layout.num_blocks
    self.cpu_num_blocks = cpu_layout.num_blocks
    self.dtype = gpu_storage_handle.dtype
    self.gpu_device_id = gpu_storage_handle.gpu_device_id
    if gpu_layout.layout_type != KVCacheLayoutType.LAYERFIRST or cpu_layout.layout_type != KVCacheLayoutType.LAYERFIRST:
      raise RuntimeError("miniflex first version only supports layer-first layout")
    if gpu_layout.num_layers != cpu_layout.num_layers:
      raise ValueError("gpu and cpu storage must have the same num_layers")
    if gpu_layout.tokens_per_block != cpu_layout.tokens_per_block:
      raise ValueError("gpu and cpu storage must have the same tokens_per_block")
    if gpu_layout.num_heads != cpu_layout.num_heads:
      raise ValueError("gpu and cpu storage must have the same num_heads")
    if gpu_layout.head_size != cpu_layout.head_size:
      raise ValueError("gpu and cpu storage must have the same head_size")
    if gpu_layout.use_mla != cpu_layout.use_mla:
      raise ValueError("gpu and cpu storage must have the same use_mla")
    if gpu_storage_handle.dtype != cpu_storage_handle.dtype:
      raise ValueError("gpu and cpu storage must have the same dtype")
    self.gpu_tensors_list = self._get_tensor_list(self.gpu_storage_handle)
    self.cpu_tensors_list = self._get_tensor_list(self.cpu_storage_handle)
    if isinstance(self.cpu_storage_handle.data, torch.Tensor):
      self._register_cuda_host_tensor(self.cpu_storage_handle.data)
    elif isinstance(self.cpu_storage_handle.data, list):
      for tensor in self.cpu_storage_handle.get_tensor_list():
        self._register_cuda_host_tensor(tensor)
    self.gpu_stream = torch.cuda.Stream()
    
  @staticmethod
  def _get_tensor_list(storage_handle: StorageHandle) -> list[torch.Tensor]:
    layout = storage_handle.kv_layout
    if isinstance(storage_handle.data,list):
      tensors = storage_handle.get_tensor_list()
      if len(tensors) != layout.num_layers:
        raise ValueError("tensor list length must equal num_layers")
      layer_shape = layout.kv_shape[1:]
      layer_numel = layer_shape.numel()
      for tensor in tensors:
        if tensor.numel() != layer_numel:
          raise ValueError("layer tensor numel does not match kv layout")
      return [tensor.view(layer_shape) for tensor in tensors]
    elif isinstance(storage_handle.data,torch.Tensor):
      tensor = storage_handle.get_tensor()
      if tensor.numel() != layout.get_total_elements():
        raise ValueError("tensor numel does not match kv layout")
      tensor = tensor.view(layout.kv_shape)
      return [tensor[layer_id] for layer_id in range(layout.num_layers)]
    raise ValueError(f"invalid storage handle data type: {type(storage_handle.data).__name__}")
  
   
  def _transfer_impl(self,
                     src_block_ids: torch.Tensor,
                     dst_block_ids: torch.Tensor,
                     transfer_type: TransferType,
                     **kwargs: Any):
    if transfer_type not in [TransferType.D2H, TransferType.H2D]:
      raise ValueError(f"invalid transfer type: {transfer_type}")
    if src_block_ids.numel() != dst_block_ids.numel():
      raise ValueError("src_block_ids and dst_block_ids must have the same size")
    if src_block_ids.dim() != 1 or dst_block_ids.dim() != 1:
      raise ValueError("src_block_ids and dst_block_ids must be 1D tensors")
    if src_block_ids.numel() == 0:
      return
    
    if transfer_type == TransferType.D2H:
      src_tensors = self.gpu_tensors_list
      dst_tensors = self.cpu_tensors_list
      src_num_blocks = self.gpu_num_blocks
      dst_num_blocks = self.cpu_num_blocks
    else:
      src_tensors = self.cpu_tensors_list
      dst_tensors = self.gpu_tensors_list
      src_num_blocks = self.cpu_num_blocks
      dst_num_blocks = self.gpu_num_blocks

    if src_block_ids.min().item() < 0 or src_block_ids.max().item() >= src_num_blocks:
      raise ValueError(f"src_block_ids out of range [0, {src_num_blocks})")
    if dst_block_ids.min().item() < 0 or dst_block_ids.max().item() >= dst_num_blocks:
      raise ValueError(f"dst_block_ids out of range [0, {dst_num_blocks})")

    src_device = src_tensors[0].device
    dst_device = dst_tensors[0].device
    src_index = src_block_ids.to(device=src_device, dtype=torch.long,non_blocking=True)
    dst_index = dst_block_ids.to(device=dst_device, dtype=torch.long,non_blocking=True)
    
    for layer_id in range(self.num_layers):
      src_layer = src_tensors[layer_id]
      dst_layer = dst_tensors[layer_id]
      
      gathered = src_layer.index_select(dim=1, index=src_index)
      
      
      gathered_on_dst = gathered.to(device=dst_device, non_blocking=(transfer_type == TransferType.H2D))
      dst_layer.index_copy_(dim=1, index=dst_index, source=gathered_on_dst)
  
  '''
  def _transfer_impl(self,
                   src_block_ids: torch.Tensor,
                   dst_block_ids: torch.Tensor,
                   transfer_type: TransferType,
                   **kwargs: Any) -> None:
    if transfer_type not in (TransferType.D2H, TransferType.H2D):
        raise ValueError(f"invalid transfer type: {transfer_type}")
    if src_block_ids.numel() != dst_block_ids.numel():
        raise ValueError(
            f"size mismatch: src={src_block_ids.numel()}, dst={dst_block_ids.numel()}"
        )
    if src_block_ids.numel() == 0:
        return

    if transfer_type == TransferType.D2H:
        src_tensors = self.gpu_tensors_list
        dst_tensors = self.cpu_tensors_list
        src_num_blocks = self.gpu_num_blocks
        dst_num_blocks = self.cpu_num_blocks
    else:
        src_tensors = self.cpu_tensors_list
        dst_tensors = self.gpu_tensors_list
        src_num_blocks = self.cpu_num_blocks
        dst_num_blocks = self.gpu_num_blocks

    if src_block_ids.min().item() < 0 or src_block_ids.max().item() >= src_num_blocks:
        raise ValueError(f"src_block_ids out of range [0, {src_num_blocks})")
    if dst_block_ids.min().item() < 0 or dst_block_ids.max().item() >= dst_num_blocks:
        raise ValueError(f"dst_block_ids out of range [0, {dst_num_blocks})")

    # tolist 提到 layer 循环外，整个调用只算一次
    src_ids = src_block_ids.tolist()
    dst_ids = dst_block_ids.tolist()

    # Per-block 直传：view 零拷贝 + copy_ 真异步到 pinned 目标
    # 所有 cudaMemcpyAsync 排队到 stream，PCIe 流水线满载
    for layer_id in range(self.num_layers):
        src_layer = src_tensors[layer_id]
        dst_layer = dst_tensors[layer_id]
        for src_id, dst_id in zip(src_ids, dst_ids):
            dst_layer[:, dst_id].copy_(src_layer[:, src_id], non_blocking=True)
  '''
  
  def launch_transfer(self,transfer_op: WorkerTransferOp) -> bool:
    src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
    with torch.cuda.stream(self.gpu_stream):
      self._transfer_impl(src_block_ids, dst_block_ids, transfer_op.transfer_type)
    self.gpu_stream.synchronize()
    return True



class SSDCPUTransferWorker(TransferWorkerBase):
  def __init__(self,
               worker_id: int,
               transfer_conn: Connection,
               finished_ops_queue: MPQueue,
               op_buffer_tensor: torch.Tensor,
               cpu_storage_handle: StorageHandle,
               ssd_storage_handle: StorageHandle,
               use_direct_io: bool = True):
    super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
    self.cpu_storage_handle = cpu_storage_handle
    self.ssd_storage_handle = ssd_storage_handle
    cpu_layout = cpu_storage_handle.kv_layout
    ssd_layout = ssd_storage_handle.kv_layout
    self.num_layers = cpu_layout.num_layers
    self.kv_dim = cpu_layout.kv_dim
    self.cpu_num_blocks = cpu_layout.num_blocks
    self.ssd_num_blocks = ssd_layout.num_blocks
    self.slice_bytes = cpu_layout.get_chunk_size() * cpu_storage_handle.dtype.itemsize
    self.dtype = cpu_storage_handle.dtype
    self.use_direct_io = use_direct_io
    if cpu_storage_handle.handle_type != StorageHandlerType.TENSOR:
      raise ValueError("CPU storage handle must be tensor-backed")
    if ssd_storage_handle.handle_type != StorageHandlerType.FILE:
      raise ValueError("SSD storage handle must be file-backed")
    if cpu_layout.layout_type != KVCacheLayoutType.LAYERFIRST:
      raise ValueError("CPU storage layout must be LAYERFIRST for SSDCPUTransferWorker")
    if ssd_layout.layout_type != KVCacheLayoutType.BLOCKFIRST:
      raise ValueError("SSD storage layout must be BLOCKFIRST for SSDCPUTransferWorker")
    if cpu_layout.num_layers != ssd_layout.num_layers:
      raise ValueError("CPU and SSD storage must have the same num_layers")
    if cpu_layout.tokens_per_block != ssd_layout.tokens_per_block:
      raise ValueError("CPU and SSD storage must have the same tokens_per_block")
    if cpu_layout.num_heads != ssd_layout.num_heads:
      raise ValueError("CPU and SSD storage must have the same num_heads")
    if cpu_layout.head_size != ssd_layout.head_size:
      raise ValueError("CPU and SSD storage must have the same head_size")
    if cpu_layout.use_mla != ssd_layout.use_mla:
      raise ValueError("CPU and SSD storage must have the same use_mla")
    if cpu_storage_handle.dtype != ssd_storage_handle.dtype:
      raise ValueError("CPU and SSD storage must have the same dtype")
    if not isinstance(cpu_storage_handle.data, torch.Tensor):
      raise ValueError("CPU storage data must be a torch.Tensor")
    if not cpu_storage_handle.data.is_contiguous():
      raise ValueError("CPU tensor must be contiguous for SSDCPUTransferWorker")
    if cpu_storage_handle.data.numel() != cpu_layout.get_total_elements():
      raise ValueError("CPU tensor numel does not match CPU layout")
    if ssd_storage_handle.num_blocks_per_file is None:
      raise ValueError("SSD storage handle must provide num_blocks_per_file")
    self.file_paths = ssd_storage_handle.get_file_list()
    self.ssd_io_ctx = _C.SSDIOCTX(
      queue_depth=128,
      blocks_per_file=self.ssd_storage_handle.num_blocks_per_file,
      cpu_tensor=cpu_storage_handle.data,
      layer_num=self.num_layers,
      kv_dim=self.kv_dim,
      cpu_num_blocks=self.cpu_num_blocks,
      slice_bytes=self.slice_bytes,
      file_paths=self.file_paths,
      use_direct_io=self.use_direct_io,
    )
    
  def _transfer_impl(self,
                     src_block_ids: torch.Tensor,
                     dst_block_ids: torch.Tensor,
                     transfer_type: TransferType,
                     **kwargs: Any):
    if transfer_type not in [TransferType.H2DISK, TransferType.DISK2H]:
      raise ValueError(f"invalid transfer type: {transfer_type}")
    if src_block_ids.dtype != torch.int64 or dst_block_ids.dtype != torch.int64:
      raise ValueError("src_block_ids and dst_block_ids must be int64")
    if src_block_ids.numel() != dst_block_ids.numel():
      raise ValueError("src_block_ids and dst_block_ids must have the same size")
    if src_block_ids.dim() != 1 or dst_block_ids.dim() != 1:
      raise ValueError("src_block_ids and dst_block_ids must be 1D tensors")
    if src_block_ids.numel() == 0:
      return True
    if transfer_type == TransferType.H2DISK:
      src_num_blocks = self.cpu_num_blocks
      dst_num_blocks = self.ssd_num_blocks
      is_read = False
    else:
      src_num_blocks = self.ssd_num_blocks
      dst_num_blocks = self.cpu_num_blocks
      is_read = True

    if src_block_ids.min().item() < 0 or src_block_ids.max().item() >= src_num_blocks:
      raise ValueError(f"src_block_ids out of range [0, {src_num_blocks})")
    if dst_block_ids.min().item() < 0 or dst_block_ids.max().item() >= dst_num_blocks:
      raise ValueError(f"dst_block_ids out of range [0, {dst_num_blocks})")

    src_block_ids = src_block_ids.to(device="cpu", dtype=torch.int64).contiguous()
    dst_block_ids = dst_block_ids.to(device="cpu", dtype=torch.int64).contiguous()
    return self.ssd_io_ctx.transfer_blocks(src_block_ids, dst_block_ids, is_read)

  def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
    src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
    return bool(self._transfer_impl(src_block_ids, dst_block_ids, transfer_op.transfer_type))

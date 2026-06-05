from typing import Any, Callable, Optional, Tuple

import torch
import torch.multiprocessing.reductions as reductions


class TensorSharedHandle:
  rebuild_func: Callable
  rebuild_args: Tuple[Any, ...]
  device: torch.device
  
  # only ipc
  use_direct_ipc: bool
  ipc_handle: Optional[bytes] = None
  tensor_shape: Optional[Tuple[int, ...]] = None
  tensor_dtype: Optional[torch.dtype] = None
  tensor_numel: Optional[int] = None
  offset: int = 0
  
  def __init__(self, 
               data: torch.Tensor|bytes, 
               device_id: int = -1,
               force_direct_ipc: bool = False,
               *,
               tensor_shape: Optional[Tuple[int, ...]] = None,
               tensor_dtype: Optional[torch.dtype] = None,
               offset: int = 0):
    if isinstance(data, bytes):
      raise NotImplementedError("external CUDA IPC handle is not supported yet")
    if force_direct_ipc:
      raise NotImplementedError("direct CUDA IPC is not supported yet")
    if not isinstance(data, torch.Tensor):
      raise ValueError("data must be a torch.Tensor")
    if not data.is_cuda:
      raise ValueError("data must be on GPU")
    self._cached_tensor: Optional[torch.Tensor] = None
    self._init_from_tensor(data, device_id)
    
  def _init_from_tensor(self, tensor: torch.Tensor, device_id: int):
    if not isinstance(tensor, torch.Tensor):
      raise ValueError("tensor must be a torch.Tensor")
    if not tensor.is_cuda:
      raise ValueError("tensor must be on GPU")
    if device_id < -1:
      raise ValueError(f"device_id must be -1 or non-negative, got {device_id}")

    tensor_device_id = tensor.device.index
    if device_id != -1 and tensor_device_id is not None and tensor_device_id != device_id:
      raise ValueError(
        f"device_id mismatch: tensor is on cuda:{tensor_device_id}, got cuda:{device_id}"
      )

    rebuild_func, rebuild_args, device = self._export_tensor_handle(tensor)
    if device_id == -1:
      self.device = device
    else:
      self.device = torch.device(f"cuda:{device_id}")

    self.rebuild_func = rebuild_func
    self.rebuild_args = rebuild_args
    self.use_direct_ipc = False
    self.ipc_handle = None
    self.tensor_shape = tuple(tensor.shape)
    self.tensor_dtype = tensor.dtype
    self.tensor_numel = tensor.numel()
    self.offset = 0
  
  @staticmethod
  def _export_tensor_handle(tensor: torch.Tensor) -> Tuple[Callable, Tuple[Any, ...], torch.device]:
    device = tensor.device
    if device.type == "cuda" and device.index is not None:
      torch.cuda.set_device(device.index)
    rebuild_func, rebuild_args = reductions.reduce_tensor(tensor)
    return rebuild_func, rebuild_args, device
  
  @staticmethod
  def _import_tensor_handle(rebuild_func: Callable, rebuild_args: Tuple[Any, ...], device: torch.device) -> torch.Tensor:
    tensor = rebuild_func(*rebuild_args)
    if not isinstance(tensor, torch.Tensor):
      raise ValueError("rebuilt tensor must be a torch.Tensor")
    if tensor.device != device:
      raise ValueError(f"rebuilt tensor must be on {device}, got {tensor.device}")
    return tensor

  
  def get_tensor(self) -> torch.Tensor:
    if self._cached_tensor is None:
      if self.device.type == "cuda" and self.device.index is not None:
        torch.cuda.set_device(self.device.index)
      tensor = self._import_tensor_handle(self.rebuild_func, self.rebuild_args, self.device)
      self._cached_tensor = tensor
    return self._cached_tensor

  def __getstate__(self) -> dict[str,Any]:
    state = self.__dict__.copy()
    state["_cached_tensor"] = None
    return state

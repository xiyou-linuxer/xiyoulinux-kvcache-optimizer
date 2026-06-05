import os
from abc import ABC, abstractmethod
from typing import Any, BinaryIO

import torch
import torch.cuda as cuda

from miniflex.common.memory_handle import TensorSharedHandle
from miniflex.common.storage import KVCacheLayout, StorageHandle, StorageHandlerType


class BaseStorageAllocator(ABC):
  @classmethod
  @abstractmethod
  def allocate(cls, layout: KVCacheLayout, dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    pass
  
  @classmethod
  @abstractmethod
  def free(cls, handle: StorageHandle) -> None:
    pass
  
  @classmethod
  @abstractmethod
  def from_raw_data(cls, data: Any, layout: KVCacheLayout, dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    pass
  

class GPUStorageAllocator(BaseStorageAllocator):
  @classmethod
  def allocate(cls, layout: KVCacheLayout, dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    device_id = kwargs.get("device_id", cuda.current_device())
    if device_id < 0:
      raise ValueError(f"device_id must be non-negative, got {device_id}")
    device = torch.device(f"cuda:{device_id}")
    num_chunks = kwargs.get("num_chunks", 1)
    if num_chunks <= 0:
      raise ValueError(f"num_chunks must be a positive int, got {num_chunks}")

    total_size = layout.get_total_elements()
    if total_size % num_chunks != 0:
      raise ValueError(f"total elements {total_size} must be divisible by num_chunks {num_chunks}")

    total_size_per_chunk = total_size // num_chunks
    physical_chunks = [
      torch.empty(size=(total_size_per_chunk,), dtype=dtype, device=device)
      for _ in range(num_chunks)
    ]

    return StorageHandle(
      handle_type=StorageHandlerType.TENSOR,
      data=physical_chunks,
      kv_layout=layout,
      dtype=dtype,
      gpu_device_id=device_id,
    )
  
  @classmethod
  def free(cls, handle: StorageHandle) -> None:
    pass
  
  @classmethod
  def from_raw_data(cls, data: Any, layout: KVCacheLayout, dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    if not isinstance(data, list) or len(data) == 0:
      raise ValueError("data must be a non-empty list[torch.Tensor] or list[TensorSharedHandle]")

    device_id = kwargs.get("device_id")
    if device_id is None:
      raise ValueError("device_id is required for GPUStorageAllocator.from_raw_data")
    if device_id < 0:
      raise ValueError(f"device_id must be non-negative, got {device_id}")

    if all(isinstance(item, torch.Tensor) for item in data):
      handle_type = StorageHandlerType.TENSOR
    elif all(isinstance(item, TensorSharedHandle) for item in data):
      handle_type = StorageHandlerType.TENSOR_HANDLE
    else:
      raise ValueError("data must contain only torch.Tensor or only TensorSharedHandle")

    return StorageHandle(
      handle_type=handle_type,
      data=data,
      kv_layout=layout,
      dtype=dtype,
      gpu_device_id=device_id,
    )
    
class CPUStorageAllocator(BaseStorageAllocator):
  @classmethod
  def allocate(cls,layout: KVCacheLayout,dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    total_size = layout.get_total_elements()
    physical_tensor = torch.empty(size=(total_size,), dtype=dtype, device="cpu",pin_memory=True)
    return StorageHandle(
      handle_type=StorageHandlerType.TENSOR,
      data=physical_tensor,
      kv_layout=layout,
      dtype=dtype,
    )
  
  @classmethod
  def free(cls,handle: StorageHandle) -> None:
    pass
  
  @classmethod
  def from_raw_data(cls,data: Any,layout: KVCacheLayout,dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    if not isinstance(data, torch.Tensor):
      raise ValueError("data must be a torch.Tensor")
    return StorageHandle(
      handle_type=StorageHandlerType.TENSOR,
      data=data,
      kv_layout=layout,
      dtype=dtype,
    )
    
class SSDStorageAllocator(BaseStorageAllocator):
  @classmethod
  def allocate(cls,layout: KVCacheLayout,dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    cache_dir = kwargs.get("cache_dir")
    file_prefix = kwargs.get("file_prefix", "miniflex_ssd_cache")
    max_file_size_gb = kwargs.get("max_file_size_gb", -1)

    if cache_dir is None:
      raise ValueError("cache_dir is required for SSDStorageAllocator")
    if isinstance(cache_dir, str):
      cache_dirs = [cache_dir]
    elif isinstance(cache_dir, list) and len(cache_dir) > 0 and all(isinstance(item, str) for item in cache_dir):
      cache_dirs = cache_dir
    else:
      raise ValueError("cache_dir must be a path string or a non-empty list[str]")
    if not isinstance(file_prefix, str):
      raise ValueError("file_prefix must be a string")
    if max_file_size_gb != -1 and max_file_size_gb <= 0:
      raise ValueError(f"max_file_size_gb must be positive or -1, got {max_file_size_gb}")

    for directory in cache_dirs:
      os.makedirs(directory, exist_ok=True)
      if not os.path.isdir(directory):
        raise ValueError(f"cache_dir must be a directory, got {directory}")

    num_ssd_devices = len(cache_dirs)
    if layout.num_blocks % num_ssd_devices != 0:
      raise ValueError(
        f"num_blocks ({layout.num_blocks}) must be a multiple of num_ssd_devices ({num_ssd_devices})"
      )

    total_blocks_per_device = layout.num_blocks // num_ssd_devices
    block_size = layout.get_elements_per_block() * dtype.itemsize
    if block_size <= 0:
      raise ValueError(f"block_size must be positive, got {block_size}")

    if max_file_size_gb == -1:
      max_blocks_per_file = total_blocks_per_device
    else:
      max_blocks_per_file = int(max_file_size_gb * 1024 * 1024 * 1024 // block_size)
      if max_blocks_per_file <= 0:
        raise ValueError(
          f"max_file_size_gb={max_file_size_gb} is too small for block_size={block_size}"
        )

    fs_blocks_per_file = cls.get_file_size_limit(cache_dirs[0]) // block_size
    if fs_blocks_per_file <= 0:
      raise RuntimeError(f"not enough available space in {cache_dirs[0]} for one block")
    num_blocks_per_file = min(fs_blocks_per_file, max_blocks_per_file, total_blocks_per_device)
    if num_blocks_per_file <= 0:
      raise RuntimeError("num_blocks_per_file must be positive")

    num_files_per_device = (total_blocks_per_device + num_blocks_per_file - 1) // num_blocks_per_file
    file_size = num_blocks_per_file * block_size
    file_paths: list[str] = []

    for device_idx, directory in enumerate(cache_dirs):
      for file_idx in range(num_files_per_device):
        file_path = os.path.join(directory, f"{file_prefix}_{device_idx}_{file_idx}.bin")
        with open(file_path, "wb+", buffering=0) as file:
          cls._create_file(file, file_size)
        file_paths.append(file_path)

    if len(file_paths) * num_blocks_per_file < layout.num_blocks:
      raise RuntimeError("created SSD cache files do not cover the requested layout capacity")

    return StorageHandle(
      handle_type=StorageHandlerType.FILE,
      data=file_paths,
      kv_layout=layout,
      dtype=dtype,
      num_blocks_per_file=num_blocks_per_file,
    )

  @classmethod
  def _create_file(cls, file: BinaryIO, file_size: int) -> None:
    try:
      os.posix_fallocate(file.fileno(), 0, file_size)
    except AttributeError:
      try:
        os.truncate(file.fileno(), file_size)
      except OSError as exc:
        raise RuntimeError(f"failed to initialize SSD cache file: {exc}") from exc
    except OSError as exc:
      raise RuntimeError(f"failed to preallocate SSD cache file: {exc}") from exc
    file.flush()
    os.fsync(file.fileno())
  
  @classmethod
  def free(cls,handle: StorageHandle) -> None:
    pass
  
  @classmethod
  def from_raw_data(cls,data: Any,layout: KVCacheLayout,dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    num_blocks_per_file = kwargs.get("num_blocks_per_file")
    if isinstance(data, str):
      file_paths = [data]
    elif isinstance(data, list) and len(data) > 0 and all(isinstance(item, str) for item in data):
      file_paths = data
    else:
      raise ValueError("data must be a path string or a non-empty list[str]")
    if num_blocks_per_file is None:
      raise ValueError("num_blocks_per_file is required for SSDStorageAllocator.from_raw_data")
    if num_blocks_per_file <= 0:
      raise ValueError(f"num_blocks_per_file must be positive, got {num_blocks_per_file}")
    if len(file_paths) * num_blocks_per_file < layout.num_blocks:
      raise ValueError("file paths do not cover the requested layout capacity")

    return StorageHandle(
      handle_type=StorageHandlerType.FILE,
      data=file_paths,
      kv_layout=layout,
      dtype=dtype,
      num_blocks_per_file=num_blocks_per_file,
    )

  @staticmethod
  def get_file_size_limit(path: str) -> int:
    stat = os.statvfs(path)
    return stat.f_frsize * stat.f_bavail

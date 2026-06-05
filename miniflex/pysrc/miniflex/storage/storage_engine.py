from typing import Any, Dict, Tuple, Optional

import torch

from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.memory_handle import TensorSharedHandle
from miniflex.common.storage import KVCacheLayout, KVCacheLayoutType, StorageHandle
from miniflex.common.transfer import DeviceType
from miniflex.storage.allocator import (
  CPUStorageAllocator,
  GPUStorageAllocator,
  SSDStorageAllocator,
)


class StorageEngine:
  def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
    self.cache_config = cache_config
    self.model_config = model_config

    self._storage_handles: Dict[Tuple[DeviceType, int], StorageHandle] = {}
    if self.cache_config.enable_cpu:
      self._storage_handles[(DeviceType.CPU, 0)] = self._allocate_cpu(
        self._create_cpu_layout(),
        self.model_config.dtype,
      )
    if self.cache_config.enable_ssd:
      self._storage_handles[(DeviceType.SSD, 0)] = self._allocate_ssd(
        self._create_ssd_layout(),
        self.model_config.dtype,
        cache_dir=self.cache_config.ssd_cache_dir,
        file_prefix=self.cache_config.ssd_file_prefix,
        max_file_size_gb=self.cache_config.ssd_max_file_size_gb,
      )

  def _create_cpu_layout(self) -> KVCacheLayout:
    return KVCacheLayout(
      layout_type=self.cache_config.cpu_layout_type,
      num_layers=self.model_config.num_layers,
      num_blocks=self.cache_config.num_cpu_blocks,
      tokens_per_block=self.cache_config.tokens_per_block,
      num_heads=self.model_config.num_kv_heads,
      head_size=self.model_config.head_size,
      use_mla=self.model_config.use_mla,
    )

  def _create_ssd_layout(self) -> KVCacheLayout:
    return KVCacheLayout(
      layout_type=self.cache_config.ssd_layout_type,
      num_layers=self.model_config.num_layers,
      num_blocks=self.cache_config.num_ssd_blocks,
      tokens_per_block=self.cache_config.tokens_per_block,
      num_heads=self.model_config.num_kv_heads,
      head_size=self.model_config.head_size,
      use_mla=self.model_config.use_mla,
    )

  def _allocate_cpu(self, layout: KVCacheLayout, dtype: torch.dtype) -> StorageHandle:
    return CPUStorageAllocator.allocate(layout, dtype)

  def _allocate_ssd(self, layout: KVCacheLayout, dtype: torch.dtype, **kwargs: Any) -> StorageHandle:
    return SSDStorageAllocator.allocate(layout, dtype, **kwargs)

  def register_gpu_blocks(
      self,
      gpu_blocks: list[torch.Tensor] | list[TensorSharedHandle],
      gpu_layout: KVCacheLayout,
      device_id: int = 0,
      dtype: torch.dtype | None = None,
  ) -> None:
    if dtype is None:
      dtype = self.model_config.dtype
    self._validate_gpu_registration(gpu_blocks, gpu_layout, device_id, dtype)
    self._storage_handles[(DeviceType.GPU, device_id)] = GPUStorageAllocator.from_raw_data(
      gpu_blocks,
      gpu_layout,
      dtype,
      device_id=device_id,
    )

  def _validate_gpu_registration(
      self,
      gpu_blocks: list[torch.Tensor] | list[TensorSharedHandle],
      gpu_layout: KVCacheLayout,
      device_id: int,
      dtype: torch.dtype,
  ) -> None:
    if not isinstance(gpu_layout, KVCacheLayout):
      raise ValueError(f"gpu_layout must be KVCacheLayout, got {type(gpu_layout).__name__}")
    if not isinstance(dtype, torch.dtype):
      raise ValueError(f"dtype must be torch.dtype, got {dtype}")
    if device_id != 0:
      raise ValueError(f"MiniFlex first version only supports single GPU device_id=0, got {device_id}")
    if (DeviceType.GPU, device_id) in self._storage_handles:
      raise ValueError(f"GPU storage handle already registered for device_id={device_id}")
    
    expected_layout_fields = {
      "num_layers": self.model_config.num_layers,
      "tokens_per_block": self.cache_config.tokens_per_block,
      "num_heads": self.model_config.num_kv_heads,
      "head_size": self.model_config.head_size,
      "use_mla": self.model_config.use_mla,
    }
    for field_name, expected_value in expected_layout_fields.items():
      actual_value = getattr(gpu_layout, field_name)
      if actual_value != expected_value:
        raise ValueError(
          f"gpu_layout.{field_name} must be {expected_value}, got {actual_value}"
        )
    if dtype != self.model_config.dtype:
      raise ValueError(f"GPU dtype must be {self.model_config.dtype}, got {dtype}")


  def get_storage_handle(self, device_type: DeviceType, device_id: int = 0) -> Optional[StorageHandle]:
    key = (device_type, device_id)
    if key not in self._storage_handles:
      return None
    return self._storage_handles[key]

  def has_storage_handle(self, device_type: DeviceType, device_id: int = 0) -> bool:
    return (device_type, device_id) in self._storage_handles

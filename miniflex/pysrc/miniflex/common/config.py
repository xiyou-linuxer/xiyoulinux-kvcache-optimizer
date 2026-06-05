from dataclasses import dataclass
import torch
from typing import Optional, Union
from miniflex.common.storage import KVCacheLayoutType

@dataclass
class ModelConfig:
  num_layers: int = 1
  num_kv_heads: int = 1
  head_size: int = 1
  use_mla: bool = False
  dtype: torch.dtype = torch.bfloat16
  tp_size: int = 1
  dp_size: int = 1
  
  def __post_init__(self):
    if self.num_layers <= 0:
      raise ValueError(f"num_layers must be greater than 0, got {self.num_layers}")
    if self.num_kv_heads <= 0:
      raise ValueError(f"num_kv_heads must be greater than 0, got {self.num_kv_heads}")
    if self.head_size <= 0:
      raise ValueError(f"head_size must be greater than 0, got {self.head_size}")
    if self.use_mla and self.num_kv_heads != 1:
      raise ValueError(f"use_mla must be False when num_kv_heads != 1")
    if self.tp_size <= 0 or self.dp_size <= 0:
      raise ValueError(f"tp_size and dp_size must be greater than 0, got {self.tp_size} and {self.dp_size}")
    
  @property
  def token_bytes(self) -> int:
    kv_dim = 1 if self.use_mla else 2
    return self.num_layers * self.num_kv_heads * self.head_size * kv_dim * self.dtype.itemsize
  

@dataclass
class CacheConfig:
  tokens_per_block: int
  enable_cpu: bool = True
  enable_ssd: bool = False
  num_cpu_blocks: int = 1024
  num_ssd_blocks: int = 1024
  ssd_cache_dir: Optional[Union[str, list[str]]] = None
  ssd_file_prefix: str = "miniflex_ssd_cache"
  ssd_max_file_size_gb: float = -1
  cpu_layout_type: KVCacheLayoutType = KVCacheLayoutType.LAYERFIRST
  ssd_layout_type: KVCacheLayoutType = KVCacheLayoutType.BLOCKFIRST
  eviction_policy: str = "lru"
  evict_ratio: float = 0.1
  evict_start_threshold: float = 1.0
  hit_add_counts: float = 0.0
  protected_threshold: int = 2
  use_direct_io: bool = True
  
  def __post_init__(self):
    if self.tokens_per_block <= 0 or (self.tokens_per_block & (self.tokens_per_block - 1)) != 0:
      raise ValueError(f"tokens_per_block must be greater than 0 and a power of 2, got {self.tokens_per_block}")
    if not self.enable_cpu:
      raise ValueError(f"enable_cpu  must be True, got {self.enable_cpu} and {self.enable_ssd}")
    if self.enable_cpu and self.num_cpu_blocks <= 0:
      raise ValueError(f"num_cpu_blocks must be greater than 0, got {self.num_cpu_blocks}")
    if not isinstance(self.cpu_layout_type, KVCacheLayoutType):
      raise ValueError(f"invalid cpu_layout_type, got {self.cpu_layout_type}")
    if not isinstance(self.ssd_layout_type, KVCacheLayoutType):
      raise ValueError(f"invalid ssd_layout_type, got {self.ssd_layout_type}")
    if self.enable_ssd:
      if self.num_ssd_blocks <= 0:
        raise ValueError(f"num_ssd_blocks must be greater than 0, got {self.num_ssd_blocks}")
      if self.ssd_cache_dir is None:
        raise ValueError("ssd_cache_dir is required when enable_ssd is True")
      if isinstance(self.ssd_cache_dir, str):
        if self.ssd_cache_dir == "":
          raise ValueError("ssd_cache_dir must be a non-empty path string or a non-empty list[str]")
      elif isinstance(self.ssd_cache_dir, list):
        if len(self.ssd_cache_dir) == 0 or not all(isinstance(item, str) and item != "" for item in self.ssd_cache_dir):
          raise ValueError("ssd_cache_dir must be a non-empty path string or a non-empty list[str]")
      else:
        raise ValueError("ssd_cache_dir must be a path string or a non-empty list[str]")
      if not isinstance(self.ssd_file_prefix, str) or self.ssd_file_prefix == "":
        raise ValueError("ssd_file_prefix must be a non-empty string")
      if self.ssd_max_file_size_gb != -1 and self.ssd_max_file_size_gb <= 0:
        raise ValueError(f"ssd_max_file_size_gb must be positive or -1, got {self.ssd_max_file_size_gb}")
    if self.eviction_policy not in ["lru", "lfu", "slru", "fifo"]:
      raise ValueError(f"invalid eviction policy, got {self.eviction_policy}")
    if self.evict_ratio < 0 or self.evict_ratio >= 1:
      raise ValueError(f"evict_ratio must be between 0 and 1, got {self.evict_ratio}")
    if self.evict_start_threshold <= 0 or self.evict_start_threshold > 1:
      raise ValueError(f"evict_start_threshold must be between 0 and 1, got {self.evict_start_threshold}")
    if self.protected_threshold <= 0:
      raise ValueError(f"protected_threshold must be greater than 0, got {self.protected_threshold}")
    

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

import torch

from miniflex.common.memory_handle import TensorSharedHandle


class StorageHandlerType(Enum):
  TENSOR = auto()
  TENSOR_HANDLE = auto()
  FILE = auto()

  
class KVCacheLayoutType(Enum):
  LAYERFIRST = "LAYERFIRST"
  BLOCKFIRST = "BLOCKFIRST"
  LAYERBLOCK = "LAYERBLOCK"

  
@dataclass
class KVCacheLayout:
  layout_type: KVCacheLayoutType
  num_layers: int
  num_blocks: int
  tokens_per_block: int
  num_heads: int
  head_size: int
  use_mla: bool
  _kv_shape: Optional[torch.Size] = None
  
  def __post_init__(self):
    if not isinstance(self.layout_type, KVCacheLayoutType):
      raise ValueError(f"invalid layout type, got {self.layout_type}")
    if self.num_layers <= 0:
      raise ValueError(f"num_layers must be greater than 0, got {self.num_layers}")
    if self.num_blocks < 0:
      raise ValueError(f"num_blocks must be greater than or equal to 0, got {self.num_blocks}")
    if self.tokens_per_block <= 0:
      raise ValueError(f"tokens_per_block must be greater than 0, got {self.tokens_per_block}")
    if self.num_heads <= 0:
      raise ValueError(f"num_heads must be greater than 0, got {self.num_heads}")
    if self.head_size <= 0:
      raise ValueError(f"head_size must be greater than 0, got {self.head_size}")
    if self.use_mla and self.num_heads != 1:
      raise ValueError("num_heads must be 1 when use_mla is True")
    self._compute_kv_shape()

  def __eq__(self, other: "KVCacheLayout") -> bool:
    if not isinstance(other, KVCacheLayout):
      return NotImplemented
    return (self.layout_type == other.layout_type and
            self.num_layers == other.num_layers and
            self.num_blocks == other.num_blocks and
            self.tokens_per_block == other.tokens_per_block and
            self.num_heads == other.num_heads and
            self.head_size == other.head_size and
            self.use_mla == other.use_mla and
            self._kv_shape == other._kv_shape)

  @property
  def _kv_dim(self) -> int:
    return 1 if self.use_mla else 2

  @property
  def kv_dim(self) -> int:
    return self._kv_dim

  def _compute_kv_shape(self):
    if self.layout_type == KVCacheLayoutType.LAYERFIRST:
      self._kv_shape = torch.Size([self.num_layers,
                                   self._kv_dim, 
                                   self.num_blocks,
                                   self.tokens_per_block,
                                   self.num_heads,
                                   self.head_size])
    elif self.layout_type == KVCacheLayoutType.BLOCKFIRST:
      self._kv_shape = torch.Size([self.num_blocks,
                                   self.num_layers,
                                   self._kv_dim,
                                   self.tokens_per_block,
                                   self.num_heads,
                                   self.head_size])
    elif self.layout_type == KVCacheLayoutType.LAYERBLOCK:
      self._kv_shape = torch.Size([self.num_layers,
                                   self.num_blocks,
                                   self._kv_dim,
                                   self.tokens_per_block,
                                   self.num_heads,
                                   self.head_size])
    else: 
      raise ValueError(f"invalid layout type, got {self.layout_type}")
    
  @property
  def kv_shape(self) -> torch.Size:
    if self._kv_shape is None:
      self._compute_kv_shape()
    if self._kv_shape is None:
      raise ValueError("kv_shape is not computed yet")
    return self._kv_shape

  def with_num_blocks(self, num_blocks: int) -> "KVCacheLayout":
    if num_blocks < 0:
      raise ValueError(f"num_blocks must be greater than or equal to 0, got {num_blocks}")
    return KVCacheLayout(
      layout_type=self.layout_type,
      num_layers=self.num_layers,
      num_blocks=num_blocks,
      tokens_per_block=self.tokens_per_block,
      num_heads=self.num_heads,
      head_size=self.head_size,
      use_mla=self.use_mla
    )
  
  def div_blocks(self, num_chunk: int, padding: bool = False) -> "KVCacheLayout":
    if num_chunk <= 0:
      raise ValueError(f"num_chunk must be greater than 0, got {num_chunk}")
    if padding:
      num_blocks = (self.num_blocks + num_chunk - 1) // num_chunk
    else:
      if self.num_blocks % num_chunk != 0:
        raise ValueError(f"num_blocks must be divisible by {num_chunk}, got {self.num_blocks}")
      num_blocks = self.num_blocks // num_chunk
    return self.with_num_blocks(num_blocks)
    
  def div_layers(self, num_chunk: int) -> "KVCacheLayout":
    if num_chunk <= 0:
      raise ValueError(f"num_chunk must be greater than 0, got {num_chunk}")
    if self.num_layers % num_chunk != 0:
      raise ValueError(f"num_layers must be divisible by {num_chunk}, got {self.num_layers}")
    num_layers = self.num_layers // num_chunk
    return KVCacheLayout(
      layout_type=self.layout_type,
      num_layers=num_layers,
      num_blocks=self.num_blocks,
      tokens_per_block=self.tokens_per_block,
      num_heads=self.num_heads,
      head_size=self.head_size,
      use_mla=self.use_mla
    )

  def div_heads(self, num_chunk: int) -> "KVCacheLayout":
    if num_chunk <= 0:
      raise ValueError(f"num_chunk must be greater than 0, got {num_chunk}")
    if self.num_heads % num_chunk != 0:
      raise ValueError(f"num_heads must be divisible by {num_chunk}, got {self.num_heads}")
    num_heads = self.num_heads // num_chunk
    return KVCacheLayout(
      layout_type=self.layout_type,
      num_layers=self.num_layers,
      num_blocks=self.num_blocks,
      tokens_per_block=self.tokens_per_block,
      num_heads=num_heads,
      head_size=self.head_size,
      use_mla=self.use_mla
    )

  def get_chunk_size(self) -> int:
    return self.tokens_per_block * self.num_heads * self.head_size

  def get_layer_stride(self) -> int:
    if self.layout_type == KVCacheLayoutType.LAYERFIRST:
      return self.kv_shape[1:].numel()
    if self.layout_type == KVCacheLayoutType.BLOCKFIRST:
      return self.kv_shape[2:].numel()
    if self.layout_type == KVCacheLayoutType.LAYERBLOCK:
      return self.kv_shape[1:].numel()
    raise ValueError(f"invalid layout type, got {self.layout_type}")

  def get_block_stride(self) -> int:
    if self.layout_type == KVCacheLayoutType.LAYERFIRST:
      return self.kv_shape[3:].numel()
    if self.layout_type == KVCacheLayoutType.BLOCKFIRST:
      return self.kv_shape[1:].numel()
    if self.layout_type == KVCacheLayoutType.LAYERBLOCK:
      return self.kv_shape[2:].numel()
    raise ValueError(f"invalid layout type, got {self.layout_type}")

  def get_kv_stride(self) -> int:
    if self.layout_type == KVCacheLayoutType.LAYERFIRST:
      return self.kv_shape[2:].numel()
    if self.layout_type == KVCacheLayoutType.BLOCKFIRST:
      return self.kv_shape[3:].numel()
    if self.layout_type == KVCacheLayoutType.LAYERBLOCK:
      return self.kv_shape[3:].numel()
    raise ValueError(f"invalid layout type, got {self.layout_type}")

  def get_total_elements(self) -> int:
    return self.kv_shape.numel()

  def get_elements_per_block(self) -> int:
    return self.num_layers * self.kv_dim * self.tokens_per_block * self.num_heads * self.head_size

@dataclass
class StorageHandle:
  handle_type: StorageHandlerType
  data: Union[list[torch.Tensor], torch.Tensor, list[TensorSharedHandle], list[str]]
  kv_layout: KVCacheLayout
  dtype: torch.dtype
  num_blocks_per_file: Optional[int] = None
  gpu_device_id: Optional[int] = None

  def __post_init__(self):
    if not isinstance(self.handle_type, StorageHandlerType):
      raise ValueError(f"invalid storage handle type, got {self.handle_type}")
    if not isinstance(self.kv_layout, KVCacheLayout):
      raise ValueError(f"kv_layout must be KVCacheLayout, got {type(self.kv_layout).__name__}")
    if not isinstance(self.dtype, torch.dtype):
      raise ValueError(f"dtype must be torch.dtype, got {self.dtype}")
    if self.num_blocks_per_file is not None and self.num_blocks_per_file <= 0:
      raise ValueError(f"num_blocks_per_file must be greater than 0, got {self.num_blocks_per_file}")
    if self.gpu_device_id is not None and self.gpu_device_id < 0:
      raise ValueError(f"gpu_device_id must be greater than or equal to 0, got {self.gpu_device_id}")

    if self.handle_type == StorageHandlerType.TENSOR:
      if isinstance(self.data, torch.Tensor):
        if self.data.dtype != self.dtype:
          raise ValueError(f"tensor dtype must be {self.dtype}, got {self.data.dtype}")
      elif isinstance(self.data, list) and len(self.data) > 0 and all(isinstance(item, torch.Tensor) for item in self.data):
        bad_dtype = [item.dtype for item in self.data if item.dtype != self.dtype]
        if bad_dtype:
          raise ValueError(f"all tensors must have dtype {self.dtype}, got {bad_dtype[0]}")
      else:
        raise ValueError("tensor storage handle data must be a torch.Tensor or a non-empty list[torch.Tensor]")
    elif self.handle_type == StorageHandlerType.TENSOR_HANDLE:
      if not isinstance(self.data, list) or len(self.data) == 0 or not all(isinstance(item, TensorSharedHandle) for item in self.data):
        raise ValueError("tensor handle storage data must be a non-empty list[TensorSharedHandle]")
      bad_dtype = [item.tensor_dtype for item in self.data if item.tensor_dtype != self.dtype]
      if bad_dtype:
        raise ValueError(f"all tensor handles must have dtype {self.dtype}, got {bad_dtype[0]}")
    elif self.handle_type == StorageHandlerType.FILE:
      if not isinstance(self.data, list) or len(self.data) == 0 or not all(isinstance(item, str) for item in self.data):
        raise ValueError("file storage handle data must be a non-empty list[str]")
    else:
      raise ValueError(f"invalid storage handle type, got {self.handle_type}")

  def get_tensor(self) -> torch.Tensor:
    if self.handle_type != StorageHandlerType.TENSOR:
      raise ValueError(f"invalid handle type: {self.handle_type}, expected Tensor")
    if not isinstance(self.data, torch.Tensor):
      raise ValueError("handle data must be a torch.Tensor")
    return self.data

  def get_tensor_list(self) -> list[torch.Tensor]:
    if not isinstance(self.data, list) or not (
      all(isinstance(item, torch.Tensor) for item in self.data) or
      all(isinstance(item, TensorSharedHandle) for item in self.data)
    ):
      raise ValueError("handle data must be list[torch.Tensor] or list[TensorSharedHandle]")
    if self.handle_type == StorageHandlerType.TENSOR:
      if not all(isinstance(item, torch.Tensor) for item in self.data):
        raise ValueError("all elements must be torch.Tensor for TENSOR type")
      return self.data
    if self.handle_type == StorageHandlerType.TENSOR_HANDLE:
      if not all(isinstance(item, TensorSharedHandle) for item in self.data):
        raise ValueError("all elements must be TensorSharedHandle for TENSOR_HANDLE type")
      return [item.get_tensor() for item in self.data]
    raise ValueError(f"invalid handle type: {self.handle_type}, expected TENSOR or TENSOR_HANDLE")

  def get_tensor_handle_list(self) -> list[TensorSharedHandle]:
    if not isinstance(self.data, list) or not (
      all(isinstance(item, torch.Tensor) for item in self.data) or
      all(isinstance(item, TensorSharedHandle) for item in self.data)
    ):
      raise ValueError("handle data must be list[torch.Tensor] or list[TensorSharedHandle]")
    if self.handle_type == StorageHandlerType.TENSOR_HANDLE:
      if not all(isinstance(item, TensorSharedHandle) for item in self.data):
        raise ValueError("all elements must be TensorSharedHandle for TENSOR_HANDLE type")
      return self.data
    if self.handle_type == StorageHandlerType.TENSOR:
      if not all(isinstance(item, torch.Tensor) for item in self.data):
        raise ValueError("all elements must be torch.Tensor for TENSOR type")
      return [TensorSharedHandle(item) for item in self.data]
    raise ValueError(f"invalid handle type: {self.handle_type}, expected TENSOR_HANDLE or TENSOR")

  def get_file_list(self) -> list[str]:
    if self.handle_type != StorageHandlerType.FILE:
      raise ValueError(f"invalid handle type: {self.handle_type}, expected File")
    if not isinstance(self.data, list) or not all(isinstance(item, str) for item in self.data):
      raise ValueError("handle data must be a list[str]")
    return self.data

from dataclasses import dataclass
from typing import List


from miniflex.common.memory_handle import TensorSharedHandle
from miniflex.common.storage import KVCacheLayout


@dataclass(frozen=True)
class RegisterGPUBlocksRequest:
  handles: List[TensorSharedHandle]
  gpu_layout: KVCacheLayout
  device_id: int = 0
  dp_rank: int = 0
  tp_rank: int = 0

  def __post_init__(self) -> None:
    if not isinstance(self.gpu_layout, KVCacheLayout):
      raise ValueError(f"gpu_layout must be KVCacheLayout, got {type(self.gpu_layout).__name__}")
    if not isinstance(self.handles, list) or len(self.handles) == 0:
      raise ValueError("handles must be a non-empty list[TensorSharedHandle]")
    if not  all(isinstance(handle, TensorSharedHandle) for handle in self.handles) :
      raise ValueError("handles must contain only TensorSharedHandle items")
    if self.device_id < 0:
      raise ValueError(f"device_id must be non-negative, got {self.device_id}")
    if self.dp_rank < 0:
      raise ValueError(f"dp_rank must be non-negative, got {self.dp_rank}")
    if self.tp_rank < 0:
      raise ValueError(f"tp_rank must be non-negative, got {self.tp_rank}")

  def validate_single_gpu(self) -> None:
    if self.device_id != 0 or self.dp_rank != 0 or self.tp_rank != 0:
      raise ValueError(
        "MiniFlex first version only supports single GPU registration "
        f"with device_id=0, dp_rank=0, tp_rank=0; got "
        f"device_id={self.device_id}, dp_rank={self.dp_rank}, tp_rank={self.tp_rank}"
      )




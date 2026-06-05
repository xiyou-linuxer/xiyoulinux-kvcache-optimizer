from typing import List

import torch
import zmq

from miniflex.common.memory_handle import TensorSharedHandle
from miniflex.common.storage import KVCacheLayout
from miniflex.server.request import RegisterGPUBlocksRequest
from miniflex.server.utils import get_zmq_socket


class MiniFlexGPURegisterClient:
  def __init__(
    self,
    gpu_register_port: str,
    device_id: int = 0,
    dp_rank: int = 0,
    tp_rank: int = 0,
  ):
    if not isinstance(gpu_register_port, str) or gpu_register_port == "":
      raise ValueError("gpu_register_port must be a non-empty string")
    if device_id < 0:
      raise ValueError(f"device_id must be non-negative, got {device_id}")
    if dp_rank < 0:
      raise ValueError(f"dp_rank must be non-negative, got {dp_rank}")
    if tp_rank < 0:
      raise ValueError(f"tp_rank must be non-negative, got {tp_rank}")

    self.gpu_register_port = gpu_register_port
    self.device_id = device_id
    self.dp_rank = dp_rank
    self.tp_rank = tp_rank

    self._context = zmq.Context(2)
    self._send_socket = get_zmq_socket(
      self._context,
      zmq.SocketType.PUSH,
      gpu_register_port,
      bind=False,
    )

  def register_to_server(
    self,
    gpu_blocks: List[torch.Tensor],
    gpu_layout: KVCacheLayout,
  ) -> None:
    if not isinstance(gpu_layout, KVCacheLayout):
      raise ValueError(f"gpu_layout must be KVCacheLayout, got {type(gpu_layout).__name__}")
    if not isinstance(gpu_blocks, list) or len(gpu_blocks) == 0:
      raise ValueError("gpu_blocks must be a non-empty list[torch.Tensor]")

    handles: List[TensorSharedHandle] = []
    for tensor in gpu_blocks:
      if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"gpu_blocks must contain torch.Tensor, got {type(tensor).__name__}")
      if not tensor.is_cuda:
        raise ValueError("gpu_blocks must be CUDA tensors")
      handles.append(TensorSharedHandle(tensor, self.device_id))

    request = RegisterGPUBlocksRequest(
      handles=handles,
      gpu_layout=gpu_layout,
      device_id=self.device_id,
      dp_rank=self.dp_rank,
      tp_rank=self.tp_rank,
    )
    self._send_socket.send_pyobj(request, flags=zmq.NOBLOCK)

  def close(self) -> None:
    self._send_socket.close(0)
    self._context.term()

  def __del__(self) -> None:
    send_socket = getattr(self, "_send_socket", None)
    context = getattr(self, "_context", None)
    if send_socket is not None:
      send_socket.close(0)
      self._send_socket = None
    if context is not None:
      context.term()
      self._context = None

"""vLLM V1 KV connector 入口薄壳。

这是 vLLM 通过 ``kv_connector`` + ``kv_connector_module_path`` 真正加载并
实例化的类——唯一继承 ``KVConnectorBase_V1`` 的地方。它本身不含业务逻辑，
只负责：
  1. 满足 vLLM 0.21.0 的 3 参构造约定，并调用 ``super().__init__`` 让基类
     初始化元数据绑定机制；
  2. 持有一个 :class:`MiniFlexConnectorV1Impl`，把所有钩子方法转发给它。

启动示例（vLLM CLI）::

    --kv-transfer-config '{"kv_connector":"MiniFlexConnectorV1",
      "kv_connector_module_path":"miniflex.integration.vllm.connector",
      "kv_role":"kv_both"}'
    --disable-hybrid-kv-cache-manager
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
  KVConnectorBase_V1,
  KVConnectorMetadata,
  KVConnectorRole,
)

from miniflex.integration.vllm.vllm_v1_adapter import MiniFlexConnectorV1Impl

if TYPE_CHECKING:
  from vllm.config import VllmConfig
  from vllm.forward_context import ForwardContext
  from vllm.v1.core.sched.output import SchedulerOutput
  from vllm.v1.core.kv_cache_manager import KVCacheBlocks
  from vllm.v1.request import Request
  from vllm.v1.outputs import KVConnectorOutput
  from vllm.distributed.kv_events import KVCacheEvent
  try:
    from vllm.v1.attention.backend import AttentionMetadata
  except ImportError:
    from vllm.attention.backends.abstract import AttentionMetadata


class MiniFlexConnectorV1(KVConnectorBase_V1):
  def __init__(
    self,
    vllm_config: "VllmConfig",
    role: "KVConnectorRole",
    kv_cache_config=None,
  ):
    # vLLM 0.21.0 要求 3 参构造；KVConnectorFactory 还会用
    # supports_kw(cls, "kv_cache_config") 校验，所以 kv_cache_config 必须带
    # 默认值并透传。super().__init__ 负责设置 _role/_vllm_config/
    # _kv_transfer_config/_kv_cache_config/_connector_metadata。
    super().__init__(vllm_config, role)
    self._impl = MiniFlexConnectorV1Impl(vllm_config, role)

  # ============================================================
  # Scheduler-side
  # ============================================================
  def get_num_new_matched_tokens(
    self,
    request: "Request",
    num_computed_tokens: int,
  ) -> Tuple[int, bool]:
    return self._impl.get_num_new_matched_tokens(request, num_computed_tokens)

  def update_state_after_alloc(
    self,
    request: "Request",
    blocks: "KVCacheBlocks",
    num_external_tokens: int,
  ) -> None:
    self._impl.update_state_after_alloc(request, blocks, num_external_tokens)

  def build_connector_meta(
    self,
    scheduler_output: "SchedulerOutput",
  ) -> "KVConnectorMetadata":
    return self._impl.build_connector_meta(scheduler_output)

  def request_finished(
    self,
    request: "Request",
    block_ids: List[int],
  ) -> Tuple[bool, Optional[Dict[str, Any]]]:
    return self._impl.request_finished(request, block_ids)

  def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
    self._impl.update_connector_output(connector_output)

  def take_events(self) -> Iterable["KVCacheEvent"]:
    return self._impl.take_events()

  def get_block_ids_with_load_errors(self) -> set[int]:
    return self._impl.get_block_ids_with_load_errors()

  # ============================================================
  # Worker-side
  # ============================================================
  def register_kv_caches(self, kv_caches: Dict[str, torch.Tensor]) -> None:
    self._impl.register_kv_caches(kv_caches)

  def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
    self._impl.start_load_kv(forward_context, **kwargs)

  def wait_for_layer_load(self, layer_name: str) -> None:
    self._impl.wait_for_layer_load(layer_name)

  def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,
    attn_metadata: "AttentionMetadata",
    **kwargs,
  ) -> None:
    self._impl.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

  def wait_for_save(self) -> None:
    self._impl.wait_for_save()

  def get_finished(
    self,
    finished_req_ids: set[str],
  ) -> Tuple[Optional[set[str]], Optional[set[str]]]:
    return self._impl.get_finished(finished_req_ids)

  # ============================================================
  # Both
  # ============================================================
  def get_kv_connector_stats(self):
    return self._impl.get_kv_connector_stats()

  def shutdown(self) -> None:
    self._impl.shutdown()

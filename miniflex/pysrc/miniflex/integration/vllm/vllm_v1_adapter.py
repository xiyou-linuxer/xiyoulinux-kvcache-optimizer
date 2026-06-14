


import os
import time
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Literal, TYPE_CHECKING, Optional, List, Tuple

from miniflex.common.request import KVRequest, KVRequestType, KVResponseStatus
from miniflex.common.metrics import dump_json_if_missing as _metrics_dump
from miniflex.common.metrics import incr as _metrics_incr
from miniflex.common.storage import KVCacheLayout, KVCacheLayoutType
import numpy as np
import torch

from miniflex.integration.config import MiniFlexConfig
from miniflex.kvtask import KVTaskEngine
from miniflex.server.client import MiniFlexGPURegisterClient

import sys as _sys
def _dbg(*a):
  if os.getenv("MINIFLEX_DEBUG"):
    print("[MINIFLEX]", *a, file=_sys.stderr, flush=True)

try:
  from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
  )
except ImportError:
  KVConnectorMetadata = None
  KVConnectorRole = None

try:
  from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats

  class _MiniFlexWorkerSentinelStats(KVConnectorStats):

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
      return other

    def reduce(self) -> dict:
      return {}

    def reset(self) -> None:
      pass

    def is_empty(self) -> bool:
      return True
except ImportError:
  KVConnectorStats = None
  _MiniFlexWorkerSentinelStats = None

try:
  from vllm.v1.outputs import KVConnectorOutput as _KVConnectorOutput
  _HAS_KV_CONNECTOR_OUTPUT = True
except ImportError:
  _HAS_KV_CONNECTOR_OUTPUT = False

try:
  from vllm.v1.engine import FinishReason
  # 只对正常完成（STOP / LENGTH）的请求 PUT 其 KV；
  # ABORT / ERROR / REPETITION 不保存。
  _PUTTABLE_FINISH_REASONS = (FinishReason.STOP, FinishReason.LENGTH)
except ImportError:
  FinishReason = None
  _PUTTABLE_FINISH_REASONS = ()

if TYPE_CHECKING:
  from vllm.config import VllmConfig
  from vllm.forward_context import ForwardContext
  from vllm.v1.core.sched.output import SchedulerOutput
  try:
    from vllm.v1.attention.backend import AttentionMetadata
  except ImportError:
    from vllm.attention.backends.abstract import AttentionMetadata
  from vllm.distributed.kv_events import KVCacheEvent
  from vllm.v1.core.kv_cache_manager import KVCacheBlocks
  from vllm.v1.request import Request
  if _HAS_KV_CONNECTOR_OUTPUT:
    from vllm.v1.outputs import KVConnectorOutput
  
  

def cdiv(a: int, b: int) -> int:
    return -(a // -b)
  
@dataclass
class MiniFlexKVTask(ABC):
  task_id: int
  request: "Request" = 0
  
  slot_mapping: Optional[np.ndarray] = None
  block_ids: Optional[List[int]] = None
  
@dataclass(kw_only=True)
class MiniFlexGetTask(MiniFlexKVTask):
  num_computed_tokens: int
  num_new_matched_tokens: int

  @property
  def task_type(self):
    return "get"

@dataclass(kw_only=True)
class MiniFlexPutTask(MiniFlexKVTask):
  num_matched_tokens: int
  num_unmatched_tokens: int
  
  @property
  def task_type(self):
    return "put"

@dataclass
class MiniFlexResponse:
  task_id: int
  task_type: Literal["get","put"]
  request: "Request"
  success: bool
  
  
class MiniFlexSchedulerConnector:
  def __init__(self, config: MiniFlexConfig):
    self.gpu_register_port = config.gpu_register_port
    self.tp_size = config.model_config.tp_size
    self.dp_size = config.model_config.dp_size
    self.block_size = config.cache_config.tokens_per_block
    self.model_config = config.model_config
    self.cache_config = config.cache_config

    self._kvengine = KVTaskEngine(
      model_config=self.model_config,
      cache_config=self.cache_config,
      gpu_register_port=self.gpu_register_port,
    )
    self._kvengine.start()

    self.req_id_to_task_dict: dict[str, int] = {}
    self.get_tasks: dict[int, list[MiniFlexGetTask]] = {}
    self.put_tasks: dict[int, list[MiniFlexPutTask]] = {}
    self.tasks_to_launch: dict[int, MiniFlexKVTask] = {}
    self.tasks_to_cancel: dict[int, MiniFlexKVTask] = {}

    self.failed_block_ids: set[int] = set()
    self.enable_batch = config.enable_batch
    self.sync_get = config.sync_get
    # 仅 sync_get 模式需要：暂存同步等待中已完成的 GET req_id，交给
    # query_finished_tasks 上报 finished_recving；异步默认模式下保持 None。
    self._pending_finished_recving: Optional[set[str]] = set() if self.sync_get else None
    self.init_time = time.perf_counter()

  def is_ready(self) -> bool:
    return self._kvengine.is_ready()

  def shutdown(self) -> None:
    self._kvengine.shutdown()

  def _extract_namespace(self, request: "Request") -> Optional[List[str]]:
    namespace_info: List[str] = []

    lora_request = getattr(request, "lora_request", None)
    if lora_request is not None:
      lora_name = getattr(lora_request, "lora_name", None)
      if lora_name is not None:
        namespace_info.append(str(lora_name))

    cache_salt = getattr(request, "cache_salt", None)
    if cache_salt is not None:
      namespace_info.append(str(cache_salt))

    user_namespace = getattr(request, "namespace_info", None)
    if user_namespace is not None:
      if isinstance(user_namespace, list):
        namespace_info.extend(str(item) for item in user_namespace)
      else:
        namespace_info.append(str(user_namespace))

    if not namespace_info:
      return None
    return namespace_info

  def _get_match(self,
                 request: "Request",
                 num_computed_tokens: int = 0) -> Tuple[int, int]:
    num_token_to_get = (request.num_tokens // self.block_size) * self.block_size
    token_ids = request.all_token_ids[:num_token_to_get]
    
    if num_token_to_get == num_computed_tokens:
      return -1,0
    
    np_token_ids = np.array(token_ids, dtype=np.int64)
    np_token_mask = np.ones_like(np_token_ids,dtype=bool)
    np_token_mask[:num_computed_tokens] = False
    namespace = self._extract_namespace(request)
    kv_request = KVRequest(
      request_type=KVRequestType.GET,
      token_ids=np_token_ids,
      token_mask=np_token_mask,
      namespace=namespace,
    )
    task_id, return_mask = self._kvengine.get_match(kv_request)
    num_new_matched_tokens = return_mask.sum().item()
    _dbg(f"GET_MATCH req={request.request_id} num_token_to_get={num_token_to_get} matched={num_new_matched_tokens} task_id={task_id}")

    if num_new_matched_tokens > 0:
      _metrics_incr("miniflex_get_matched_tokens", num_new_matched_tokens)
      self.req_id_to_task_dict[request.request_id] = task_id
      self.tasks_to_cancel[task_id] = MiniFlexGetTask(
        task_id=task_id,
        request=request,
        num_computed_tokens=num_computed_tokens,
        num_new_matched_tokens=num_new_matched_tokens,
      )
    else:
      _metrics_incr("miniflex_get_miss_tokens", num_token_to_get - num_computed_tokens)
      self._kvengine.cancel_tasks([task_id])
    return task_id, num_new_matched_tokens
  
  def get_num_new_matched_tokens(self, 
                                 request: "Request",
                                 num_computed_tokens: int) -> Tuple[int,bool]:
    task_id, num_new_matched_tokens = self._get_match(request, num_computed_tokens)
    if not self._need_to_get(
      num_prompt_tokens=request.num_tokens,
      num_computed_tokens=num_computed_tokens,
      num_new_matched_tokens=num_new_matched_tokens,
    ):
      return 0, False

    return num_new_matched_tokens, True

  def _need_to_get(self,
                   num_prompt_tokens: int,
                   num_computed_tokens: int,
                   num_new_matched_tokens: int,) -> bool:
    return num_new_matched_tokens > 0
    
    
  def update_state_after_alloc(self,
                               request: "Request",
                               blocks: "KVCacheBlocks",
                               num_new_matched_tokens: int) -> None:
    if num_new_matched_tokens == 0:
      return
    task_id = self.req_id_to_task_dict[request.request_id]
    task = self.tasks_to_cancel.pop(task_id)
    self.tasks_to_launch[task_id] = task
    
    num_computed_blocks = task.num_computed_tokens // self.block_size
    num_blocks_to_get = num_new_matched_tokens // self.block_size
    all_block_ids = blocks.get_block_ids()[0]
    block_ids_to_get = all_block_ids[num_computed_blocks:num_computed_blocks + num_blocks_to_get]
    task.block_ids = block_ids_to_get
    task.slot_mapping = np.array(block_ids_to_get).repeat(self.block_size) * self.block_size
    
  def _put_match(self,
                 request:"Request") -> tuple[int,int,int]:
    num_token_to_put = (cdiv(request.num_tokens,self.block_size) - 1)*self.block_size
    token_ids = request.all_token_ids[:num_token_to_put]
    if num_token_to_put == 0:
      return -1,0,0
    
    np_token_ids = np.array(token_ids, dtype=np.int64)
    namespace = self._extract_namespace(request)
    kv_request = KVRequest(
      request_type=KVRequestType.PUT,
      token_ids=np_token_ids,
      namespace=namespace,
    )
    task_id, return_mask = self._kvengine.put_match(kv_request)

    num_unmatched_tokens = return_mask.sum().item()
    num_matched_tokens = num_token_to_put - num_unmatched_tokens
    _dbg(f"PUT_MATCH req={request.request_id} num_token_to_put={num_token_to_put} unmatched={num_unmatched_tokens} matched={num_matched_tokens} task_id={task_id}")

    _metrics_incr("miniflex_put_matched_tokens", num_matched_tokens)
    _metrics_incr("miniflex_put_unmatched_tokens", num_unmatched_tokens)

    if num_unmatched_tokens > 0:
      self.req_id_to_task_dict[request.request_id] = task_id
      self.tasks_to_cancel[task_id] = MiniFlexPutTask(
        task_id=task_id,
        request=request,
        num_matched_tokens=num_matched_tokens,
        num_unmatched_tokens=num_unmatched_tokens,
      )
    else:
      self._kvengine.cancel_tasks([task_id])
    return task_id, num_matched_tokens, num_unmatched_tokens

  def _need_to_put(self,
                   num_all_tokens: int,
                   num_matched_tokens: int,
                   num_unmatched_tokens: int) -> bool:
    return num_unmatched_tokens > 0
  
  def request_finished(self,
                       request:"Request",
                       block_ids: List[int]) -> bool:
    _dbg(f"REQUEST_FINISHED req={request.request_id} num_tokens={request.num_tokens} block_ids={block_ids} finished={request.is_finished()} reason={request.get_finished_reason()}")
    if request.request_id in self.req_id_to_task_dict:
      _dbg(f"  -> already tracked, returning True")
      return True
    if not (request.is_finished() and request.get_finished_reason() in _PUTTABLE_FINISH_REASONS):
      return False
    task_id, num_matched_tokens, num_unmatched_tokens = self._put_match(request)
    
    if not self._need_to_put(
      num_all_tokens=request.num_tokens,
      num_matched_tokens=num_matched_tokens,
      num_unmatched_tokens=num_unmatched_tokens,
    ):
      return False
    
    task: MiniFlexPutTask = self.tasks_to_cancel.pop(task_id)
    self.tasks_to_launch[task_id] = task
    
    num_matched_blocks = task.num_matched_tokens // self.block_size
    num_unmatched_blocks = task.num_unmatched_tokens // self.block_size
    block_ids_to_put = block_ids[num_matched_blocks:num_matched_blocks + num_unmatched_blocks]
    task.block_ids = block_ids_to_put
    task.slot_mapping = np.array(block_ids_to_put).repeat(self.block_size) * self.block_size
    
    return True
  
  
  def cancel_tasks(self) -> None:
    if len(self.tasks_to_cancel) == 0:
      return
    for task in self.tasks_to_cancel.values():
      del self.req_id_to_task_dict[task.request.request_id]
    self._kvengine.cancel_tasks(list(self.tasks_to_cancel.keys()))
    self.tasks_to_cancel.clear()
    
  def launch_tasks(self) -> None:
    if len(self.tasks_to_launch) == 0:
      return
    _dbg(f"LAUNCH_TASKS count={len(self.tasks_to_launch)} ids={list(self.tasks_to_launch.keys())}")

    get_task_ids: List[int] = []
    get_slot_mappings: List[np.ndarray] = []
    get_tasks_to_launch: List[MiniFlexGetTask] = []
    put_task_ids: List[int] = []
    put_slot_mappings: List[np.ndarray] = []
    put_tasks_to_launch: List[MiniFlexPutTask] = []
    
    for task_id, task in self.tasks_to_launch.items():
      if task.slot_mapping is None:
        raise ValueError(f"Task {task_id} has no slot_mapping")
      if isinstance(task, MiniFlexGetTask):
        get_task_ids.append(task_id)
        get_slot_mappings.append(task.slot_mapping)
        get_tasks_to_launch.append(task)
      else:
        put_task_ids.append(task_id)
        put_slot_mappings.append(task.slot_mapping)
        put_tasks_to_launch.append(task)

    if get_task_ids:
      launched_task_ids = self._kvengine.launch_tasks(
        get_task_ids,
        get_slot_mappings,
        as_batch=self.enable_batch,
      )
      if len(launched_task_ids) == 1 and len(get_tasks_to_launch) > 1:
        self.get_tasks[launched_task_ids[0]] = get_tasks_to_launch
      elif len(launched_task_ids) == len(get_tasks_to_launch):
        for launched_task_id, task in zip(launched_task_ids, get_tasks_to_launch):
          self.get_tasks[launched_task_id] = [task]
      else:
        raise ValueError("KVTaskEngine returned unexpected get task ids")

    if put_task_ids:
      launched_task_ids = self._kvengine.launch_tasks(
        put_task_ids,
        put_slot_mappings,
        as_batch=self.enable_batch,
      )
      if len(launched_task_ids) == 1 and len(put_tasks_to_launch) > 1:
        self.put_tasks[launched_task_ids[0]] = put_tasks_to_launch
      elif len(launched_task_ids) == len(put_tasks_to_launch):
        for launched_task_id, task in zip(launched_task_ids, put_tasks_to_launch):
          self.put_tasks[launched_task_id] = [task]
      else:
        raise ValueError("KVTaskEngine returned unexpected put task ids")

    self.tasks_to_launch.clear()

  def query_finished_tasks(self) -> Tuple[set[str], set[str]]:
    if len(self.get_tasks) == 0 and len(self.put_tasks) == 0 and not self._pending_finished_recving:
      return set(), set()

    task_ids = list(self.get_tasks.keys()) + list(self.put_tasks.keys())
    responses = self._kvengine.try_wait(task_ids)
    _dbg(f"QUERY_FINISHED polling={task_ids} responses={list(responses.keys())}")
    finished_sending: set[str] = set()
    finished_recving: set[str] = set()

    for task_id, response in responses.items():
      success = response.status == KVResponseStatus.SUCCESS
      if task_id in self.get_tasks:
        tasks = self.get_tasks.pop(task_id)
        finished_recving.update(task.request.request_id for task in tasks)
      elif task_id in self.put_tasks:
        tasks = self.put_tasks.pop(task_id)
        finished_sending.update(task.request.request_id for task in tasks)
      else:
        continue

      for task in tasks:
        self.req_id_to_task_dict.pop(task.request.request_id, None)
        if not success and isinstance(task, MiniFlexGetTask) and task.block_ids is not None:
          self.failed_block_ids.update(task.block_ids)

    if self._pending_finished_recving:
      finished_recving |= self._pending_finished_recving
      self._pending_finished_recving = set()

    _metrics_dump("/tmp/miniflex_metrics.json")
    return finished_sending, finished_recving

  def get_and_clear_failed_block_ids(self) -> set[int]:
    failed = self.failed_block_ids
    self.failed_block_ids = set()
    return failed
  
  def handle_preemption(self,preemption_req_ids: set[str]):
    for req_id in preemption_req_ids:
      if req_id not in self.req_id_to_task_dict:
        continue
      task_id = self.req_id_to_task_dict[req_id]
      if task_id in self.tasks_to_launch:
        task = self.tasks_to_launch.pop(task_id)
        self.tasks_to_cancel[task_id] = task
        
  def _blocking_wait_for_tasks(self,
                               tasks: Dict[int, List[MiniFlexKVTask]]) -> List[MiniFlexResponse]:
    if len(tasks) == 0:
      return []

    task_ids = list(tasks.keys())
    responses = self._kvengine.wait(task_ids)
    responses_to_return: List[MiniFlexResponse] = []

    for task_id, response in responses.items():
      success = response.status == KVResponseStatus.SUCCESS
      task_list = tasks.pop(task_id)
      for task in task_list:
        self.req_id_to_task_dict.pop(task.request.request_id, None)
        if not success and isinstance(task, MiniFlexGetTask) and task.block_ids is not None:
          self.failed_block_ids.update(task.block_ids)
        responses_to_return.append(
          MiniFlexResponse(
            task_id=task.task_id,
            task_type=task.task_type,
            request=task.request,
            success=success,
          )
        )

    return responses_to_return

  def sync_wait_for_get_tasks(self) -> None:
    # sync_get 模式：阻塞等已 launch 的 GET 完成，把完成的 req_id 暂存，
    # 由 query_finished_tasks 照常上报 finished_recving，避免直接 pop 导致通知丢失。
    for response in self.wait_for_all_get_tasks():
      self._pending_finished_recving.add(response.request.request_id)
    _metrics_dump("/tmp/miniflex_metrics.json")

  def wait_for_all_get_tasks(self) -> List[MiniFlexResponse]:
    return self._blocking_wait_for_tasks(self.get_tasks)

  def wait_for_all_put_tasks(self) -> List[MiniFlexResponse]:
    return self._blocking_wait_for_tasks(self.put_tasks)

  def wait_for_all_tasks(self) -> List[MiniFlexResponse]:
    responses: List[MiniFlexResponse] = []
    responses.extend(self.wait_for_all_get_tasks())
    responses.extend(self.wait_for_all_put_tasks())
    return responses

class MiniFlexWorkerConnector:
  def __init__(
    self,
    config: MiniFlexConfig,
    dp_rank: int = 0,
    tp_rank: int = 0,
    device_id: Optional[int] = None,
  ):
    if not isinstance(config, MiniFlexConfig):
      raise ValueError(f"config must be MiniFlexConfig, got {type(config).__name__}")
    if dp_rank < 0:
      raise ValueError(f"dp_rank must be non-negative, got {dp_rank}")
    if tp_rank < 0:
      raise ValueError(f"tp_rank must be non-negative, got {tp_rank}")
    if device_id is not None and device_id < 0:
      raise ValueError(f"device_id must be non-negative, got {device_id}")

    self.config = config
    self.gpu_register_port = config.gpu_register_port
    self.model_config = config.model_config
    self.cache_config = config.cache_config
    self.dp_rank = dp_rank
    self.tp_rank = tp_rank

    if device_id is None:
      device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    self.device_id = int(device_id)

    self.gpu_register_client = MiniFlexGPURegisterClient(
      gpu_register_port=self.gpu_register_port,
      device_id=self.device_id,
      dp_rank=self.dp_rank,
      tp_rank=self.tp_rank,
    )
    self.registered = False

  def register_to_server(self,
                         kv_caches: Dict[str, torch.Tensor]) -> None:
    if self.registered:
      raise ValueError("kv_caches have already been registered")
    if not isinstance(kv_caches, dict) or len(kv_caches) == 0:
      raise ValueError("kv_caches must be a non-empty dict[str, torch.Tensor]")

    gpu_blocks = list(kv_caches.values())
    num_layers = len(gpu_blocks)
    if num_layers != self.model_config.num_layers:
      raise ValueError(
        "kv_caches layer count must equal model_config.num_layers, "
        f"got {num_layers} and {self.model_config.num_layers}"
      )

    first_block = gpu_blocks[0]
    if not isinstance(first_block, torch.Tensor):
      raise ValueError(f"kv_caches values must be torch.Tensor, got {type(first_block).__name__}")

    for name, tensor in kv_caches.items():
      if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"kv_caches[{name!r}] must be torch.Tensor, got {type(tensor).__name__}")
      if not tensor.is_cuda:
        raise ValueError(f"kv_caches[{name!r}] must be a CUDA tensor")
      if tensor.device.index is not None and tensor.device.index != self.device_id:
        raise ValueError(
          f"kv_caches[{name!r}] is on cuda:{tensor.device.index}, "
          f"but worker device_id is cuda:{self.device_id}"
        )
      if tensor.dtype != self.model_config.dtype:
        import warnings
        warnings.warn(
          f"kv_caches[{name!r}] dtype {tensor.dtype} != model_config.dtype "
          f"{self.model_config.dtype}; 可能是 fp8/量化 KV cache。请确认 "
          f"CacheConfig 与底层存储按实际 KV dtype 计算 token_bytes。"
        )
      if tensor.shape != first_block.shape:
        raise ValueError(
          f"all kv_caches tensors must have the same shape, "
          f"got {tuple(tensor.shape)} and {tuple(first_block.shape)}"
        )

    if self.model_config.use_mla:
      if first_block.ndim != 3:
        raise ValueError(f"MLA kv_cache tensor must be 3D, got shape={tuple(first_block.shape)}")
      num_blocks = int(first_block.shape[0])
      block_size = int(first_block.shape[1])
      num_heads = 1
      head_size = int(first_block.shape[2])
    else:
      if first_block.ndim != 5:
        raise ValueError(f"kv_cache tensor must be 5D, got shape={tuple(first_block.shape)}")
      kv_dim = int(first_block.shape[0])
      if kv_dim != 2:
        raise ValueError(f"non-MLA kv_cache tensor first dim must be 2, got {kv_dim}")
      num_blocks = int(first_block.shape[1])
      block_size = int(first_block.shape[2])
      num_heads = int(first_block.shape[3])
      head_size = int(first_block.shape[4])

    if block_size != self.cache_config.tokens_per_block:
      raise ValueError(
        "kv_cache block size must equal cache_config.tokens_per_block, "
        f"got {block_size} and {self.cache_config.tokens_per_block}"
      )
    if num_heads != self.model_config.num_kv_heads:
      raise ValueError(
        "kv_cache num_heads must equal model_config.num_kv_heads, "
        f"got {num_heads} and {self.model_config.num_kv_heads}"
      )
    if head_size != self.model_config.head_size:
      raise ValueError(
        "kv_cache head_size must equal model_config.head_size, "
        f"got {head_size} and {self.model_config.head_size}"
      )

    gpu_layout = KVCacheLayout(
      layout_type=KVCacheLayoutType.LAYERFIRST,
      num_layers=num_layers,
      num_blocks=num_blocks,
      tokens_per_block=block_size,
      num_heads=num_heads,
      head_size=head_size,
      use_mla=self.model_config.use_mla,
    )
    self.gpu_register_client.register_to_server(gpu_blocks, gpu_layout)
    self.registered = True

  def register_kv_caches(self, kv_caches: Dict[str, torch.Tensor]) -> None:
    self.register_to_server(kv_caches)

  def get_finished(
    self,
    finished_req_ids: set[str],
  ) -> Tuple[Optional[set[str]], Optional[set[str]]]:
    return None, None

  def start_load_kv(self, *args, **kwargs) -> None:
    return None

  def wait_for_layer_load(self, layer_name: str) -> None:
    return None

  def save_kv_layer(self, *args, **kwargs) -> None:
    return None

  def wait_for_save(self) -> None:
    return None

  def close(self) -> None:
    self.gpu_register_client.close()

  def __del__(self) -> None:
    client = getattr(self, "gpu_register_client", None)
    if client is not None:
      client.close()
      self.gpu_register_client = None

class MiniFlexConnectorV1Impl:
  def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole"):
    if KVConnectorRole is None:
         raise ImportError("vLLM KVConnectorRole is required")
    self.role = role
    miniflex_config = MiniFlexConfig.from_env()
    miniflex_config.post_init_from_vllm_config(vllm_config)
    
    match role:
      case KVConnectorRole.SCHEDULER:
        self.scheduler = MiniFlexSchedulerConnector(miniflex_config)
        self.previous_scheduler_req_ids: set[str] = set()
      case KVConnectorRole.WORKER:
        self.worker = MiniFlexWorkerConnector(miniflex_config)
      case _:
        raise ValueError(f"Unrecongnized KVConnectorRole: {role}")
      

  def shutdown(self) -> None:
    if self.role == KVConnectorRole.SCHEDULER:
      self.scheduler.shutdown()

  def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
    if self.role == KVConnectorRole.SCHEDULER:
      raise ValueError("Scheduler does not support start_load_kv")
    self.worker.start_load_kv(forward_context, **kwargs)

  def wait_for_layer_load(self, layer_name: str) -> None:
    if self.role == KVConnectorRole.SCHEDULER:
      raise ValueError("Scheduler does not support wait_for_layer_load")
    self.worker.wait_for_layer_load(layer_name)

  def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,
    attn_metadata: "AttentionMetadata",
    **kwargs,
  ) -> None:
    if self.role == KVConnectorRole.SCHEDULER:
      raise ValueError("Scheduler does not support save_kv_layer")
    self.worker.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
    
  def wait_for_save(self) -> None:
    if self.role == KVConnectorRole.SCHEDULER:
      raise ValueError("Scheduler does not support wait_for_save")
    self.worker.wait_for_save()

  def get_finished(
    self,
    finished_req_ids: set[str],
  ) -> Tuple[Optional[set[str]], Optional[set[str]]]:
    if self.role == KVConnectorRole.SCHEDULER:
      raise ValueError("Scheduler does not support get_finished")
    return self.worker.get_finished(finished_req_ids)

  def register_kv_caches(self, kv_caches: Dict[str, torch.Tensor]) -> None:
    if self.role == KVConnectorRole.SCHEDULER:
      raise ValueError("Scheduler does not support register_kv_caches")
    self.worker.register_kv_caches(kv_caches)

  def get_num_new_matched_tokens(
    self,
    request: "Request",
    num_computed_tokens: int,
  ) -> Tuple[int, bool]:
    if self.role == KVConnectorRole.WORKER:
      raise ValueError("Worker does not support get_num_new_matched_tokens")
    return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

  def update_state_after_alloc(
    self,
    request: "Request",
    blocks: "KVCacheBlocks",
    num_external_tokens: int,
  ) -> None:
    if self.role == KVConnectorRole.WORKER:
      raise ValueError("Worker does not support update_state_after_alloc")
    self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

  def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> "KVConnectorMetadata":
    if self.role == KVConnectorRole.WORKER:
      raise ValueError("Worker does not support build_connector_meta")
    preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", None)
    if preempted_req_ids is None:
      current_req_ids = set()
      for req in scheduler_output.scheduled_new_reqs:
        current_req_ids.add(req.req_id)
      if scheduler_output.scheduled_cached_reqs:
        current_req_ids.update(scheduler_output.scheduled_cached_reqs.req_ids)
      finished_req_ids = scheduler_output.finished_req_ids
      preempted_req_ids = self.previous_scheduler_req_ids - current_req_ids - finished_req_ids
      self.previous_scheduler_req_ids = current_req_ids
    
    if preempted_req_ids:
      self.scheduler.handle_preemption(preempted_req_ids)
    
    self.scheduler.cancel_tasks()
    self.scheduler.launch_tasks()


    if self.scheduler.sync_get:
      self.scheduler.sync_wait_for_get_tasks()

    return KVConnectorMetadata()
  
  def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
    if self.role == KVConnectorRole.WORKER:
      raise ValueError("Worker does not support update_connector_output")
    if not _HAS_KV_CONNECTOR_OUTPUT:
         return
    _dbg("UPDATE_CONNECTOR_OUTPUT called")
    finished_sending, finished_recving = self.scheduler.query_finished_tasks()
    connector_output.finished_sending = finished_sending
    connector_output.finished_recving = finished_recving

  def request_finished(
    self,
    request: "Request",
    block_ids: List[int],
  ) -> Tuple[bool, Optional[Dict[str, object]]]:
    if self.role == KVConnectorRole.WORKER:
      raise ValueError("Worker does not support request_finished")
    return self.scheduler.request_finished(request, block_ids), None

  def take_events(self):
    return []

  def get_kv_connector_stats(self) -> Optional["KVConnectorStats"]:
    if KVConnectorStats is None:
      return None
    if self.role != KVConnectorRole.SCHEDULER:
      return _MiniFlexWorkerSentinelStats(data={})
    return KVConnectorStats(data={})

  def get_block_ids_with_load_errors(self) -> set[int]:
    if self.role == KVConnectorRole.SCHEDULER:
      return self.scheduler.get_and_clear_failed_block_ids()
    return set()
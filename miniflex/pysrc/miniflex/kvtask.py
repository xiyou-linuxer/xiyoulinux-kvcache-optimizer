


from dataclasses import dataclass
from enum import Enum
import threading
import time
from miniflex.cache.global_cache_engine import GlobalCacheEngine
from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.metrics import incr as _metrics_incr
from miniflex.common.metrics import observe as _metrics_observe
from miniflex.common.request import KVRequest, KVRequestType, KVResponse, KVResponseStatus
from miniflex.common.transfer import CompletedOp, TransferOpGraph, merge_to_batch_graph
from miniflex.common.storage import KVCacheLayoutType
from miniflex.transfer_manager import TransferManagerHandle
import numpy as np
from typing import Optional,Union,List,Callable,Dict,Tuple
class TaskStatus(Enum):
  UNREADY = "unready"
  READY = "ready"
  RUNNING = "running"
  COMPLETED = "completed"
  CANCELED = "canceled"
  FAILED = "failed"


class TaskType(Enum):
  GET = "get"
  PUT = "put"
  BATCH_GET = "batch_get"
  BATCH_PUT = "batch_put"

TASK_STATUS_TO_RESPONSE_STATUS = {
  TaskStatus.COMPLETED: KVResponseStatus.SUCCESS,
  TaskStatus.CANCELED: KVResponseStatus.CANCELED,
  TaskStatus.FAILED: KVResponseStatus.FAILED,
  TaskStatus.RUNNING: KVResponseStatus.SUCCESS,
}


def convert_to_response_status(task_status: TaskStatus) -> KVResponseStatus:
  return TASK_STATUS_TO_RESPONSE_STATUS[task_status]

@dataclass
class KVTask:
  task_id: int
  task_type: TaskType
  task_status: TaskStatus
  task_end_op_id: int
  task_end_op_finished: bool
  
  
  graph: Optional[TransferOpGraph]
  return_mask: Union[np.ndarray, List[np.ndarray]]
  callback: Optional[Union[Callable, List[Callable]]]
  op_callback_dict: Dict[int, Callable]

  request_returned: bool = False

  is_batch_task: bool = False
  sub_task: Optional[List[int]] = None

  # metrics: timestamp set when the task is submitted to transfer engine
  submit_time: float = 0.0
  
  def is_completed(self) -> bool:
    return self.task_status in [TaskStatus.COMPLETED, TaskStatus.CANCELED, TaskStatus.FAILED]
  
  def shed_heavy_resource(self) -> None:
    self.graph = None
    self.callback = None 

class KVTaskManager:
  def __init__(self,
               model_config: ModelConfig,
               cache_config: CacheConfig,
               gpu_register_port: str
              ):
    self._check_config(model_config, cache_config)
    self.model_config = model_config
    self.cache_config = cache_config
    self.gpu_register_port = gpu_register_port
    
    self._global_cache_engine = GlobalCacheEngine(
      model_config=model_config,
      cache_config=cache_config
    )
    self._transfer_manager = TransferManagerHandle(
      model_config=model_config,
      cache_config=cache_config,
      gpu_register_port=gpu_register_port,
      mode="process"
    )
    
    self.tasks: Dict[int, KVTask] = {}
    self.graph_to_task: Dict[int, int] = {}
    
    self.task_id_counter: int = 0
    self.task_id_lock: threading.Lock = threading.Lock()

  def _check_config(self, model_config: ModelConfig, cache_config: CacheConfig) -> None:
    if not isinstance(model_config, ModelConfig):
      raise ValueError(f"model_config must be ModelConfig, got {type(model_config).__name__}")
    if not isinstance(cache_config, CacheConfig):
      raise ValueError(f"cache_config must be CacheConfig, got {type(cache_config).__name__}")

    if not cache_config.enable_cpu:
      raise ValueError("MiniFlex KVTaskManager requires enable_cpu=True")
    if model_config.tp_size != 1 or model_config.dp_size != 1:
      raise ValueError(
        "MiniFlex KVTaskManager only supports single GPU with tp_size=1 and dp_size=1, "
        f"got tp_size={model_config.tp_size}, dp_size={model_config.dp_size}"
      )
    if model_config.use_mla and model_config.num_kv_heads != 1:
      raise ValueError("use_mla=True requires num_kv_heads=1")
    if cache_config.tokens_per_block <= 0 or (cache_config.tokens_per_block & (cache_config.tokens_per_block - 1)) != 0:
      raise ValueError(
        "tokens_per_block must be greater than 0 and a power of 2, "
        f"got {cache_config.tokens_per_block}"
      )
    if cache_config.num_cpu_blocks <= 0:
      raise ValueError(f"num_cpu_blocks must be greater than 0, got {cache_config.num_cpu_blocks}")
    if cache_config.cpu_layout_type != KVCacheLayoutType.LAYERFIRST:
      raise ValueError(
        "MiniFlex GPUCPUTransferWorker currently requires CPU layout LAYERFIRST, "
        f"got {cache_config.cpu_layout_type}"
      )
    if cache_config.enable_ssd:
      if cache_config.num_ssd_blocks <= 0:
        raise ValueError(f"num_ssd_blocks must be greater than 0, got {cache_config.num_ssd_blocks}")
      if cache_config.ssd_layout_type != KVCacheLayoutType.BLOCKFIRST:
        raise ValueError(
          "MiniFlex SSDCPUTransferWorker currently requires SSD layout BLOCKFIRST, "
          f"got {cache_config.ssd_layout_type}"
        )
      if cache_config.ssd_cache_dir is None:
        raise ValueError("ssd_cache_dir is required when enable_ssd=True")
    if cache_config.eviction_policy not in ["lru", "lfu", "slru", "fifo"]:
      raise ValueError(f"invalid eviction policy, got {cache_config.eviction_policy}")
    if cache_config.evict_ratio < 0 or cache_config.evict_ratio >= 1:
      raise ValueError(f"evict_ratio must be between 0 and 1, got {cache_config.evict_ratio}")
    if cache_config.evict_start_threshold <= 0 or cache_config.evict_start_threshold > 1:
      raise ValueError(
        "evict_start_threshold must be between 0 and 1, "
        f"got {cache_config.evict_start_threshold}"
      )
    if cache_config.protected_threshold <= 0:
      raise ValueError(
        "protected_threshold must be greater than 0, "
        f"got {cache_config.protected_threshold}"
      )
    
  def start(self):
    self._transfer_manager.start()
    
  def is_ready(self):
    return self._transfer_manager.is_ready()
  
  def shutdown(self):
    self._transfer_manager.shutdown()
    
  def __del__(self):
    try:
      self.shutdown()
    except Exception:
      pass
    
  def _get_new_task_id(self) -> int:
    with self.task_id_lock:
      task_id = self.task_id_counter
      self.task_id_counter += 1
      return task_id
    
  def create_get_task(self,
                      task_id: int,
                      token_ids: np.ndarray,
                      slot_mapping: np.ndarray,
                      token_mask: Optional[np.ndarray],
                      is_fake_slot_mapping: bool = False,
                      namespace: Optional[List[str]] = None) -> None:
    if task_id in self.tasks:
      raise ValueError(f"Task with id {task_id} already exists")
    if token_mask is None:
      token_mask = np.ones_like(token_ids, dtype=np.bool_)
    graph,return_mask,callback,op_callback_dict,task_end_op_id = self._global_cache_engine.get(
      request_id=task_id,
      token_ids=token_ids,
      token_mask=token_mask,
      slot_mapping=slot_mapping,
      namespace=namespace
    )
    self.tasks[task_id] = KVTask(
      task_id=task_id,
      task_type=TaskType.GET,
      task_status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
      task_end_op_id=task_end_op_id,
      task_end_op_finished=False,
      graph=graph,
      return_mask=return_mask,
      callback=callback,
      op_callback_dict=op_callback_dict,
    )
    self.graph_to_task[graph.graph_id] = task_id
    
  def create_put_task(self,
                      task_id: int,
                      token_ids: np.ndarray,
                      slot_mapping: np.ndarray,
                      token_mask: Optional[np.ndarray],
                      is_fake_slot_mapping: bool = False,
                      namespace: Optional[List[str]] = None) -> None:
    if task_id in self.tasks:
      raise ValueError(f"Task with id {task_id} already exists")
    if token_mask is None:
      token_mask = np.ones_like(token_ids, dtype=np.bool_)
    graph,return_mask,callback,op_callback_dict,task_end_op_id = self._global_cache_engine.put(
      request_id=task_id,
      token_ids=token_ids,
      token_mask=token_mask,
      slot_mapping=slot_mapping,
      namespace=namespace
    )
    self.tasks[task_id] = KVTask(
      task_id=task_id,
      task_type=TaskType.PUT,
      task_status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
      task_end_op_id=task_end_op_id,
      task_end_op_finished=False,
      graph=graph,
      return_mask=return_mask,
      callback=callback,
      op_callback_dict=op_callback_dict
    )
    self.graph_to_task[graph.graph_id] = task_id
    
  def set_slot_mapping(self,task_ids: List[int], slot_mappings: List[np.ndarray]) -> None:
    for task_id, slot_mapping in zip(task_ids, slot_mappings):
      self._set_slot_mapping(task_id, slot_mapping)
  
  def _set_slot_mapping(self,task_id: int, slot_mapping: np.ndarray) -> None:
    if task_id not in self.tasks:
      raise ValueError(f"Task with id {task_id} not found")
    task = self.tasks[task_id]
    if task.task_status != TaskStatus.UNREADY:
      return
    if task.graph is None:
      raise ValueError(f"Task with id {task_id} has no graph")
    graph_id = self._global_cache_engine.slot_mapping_to_block_ids(slot_mapping, 
                                                                   self.cache_config.tokens_per_block)
    task.graph.set_gpu_blocks(graph_id)
    task.task_status = TaskStatus.READY
  
  def check_task_ready(self,task_id: int)-> Optional[TransferOpGraph]:
    if task_id not in self.tasks:
      raise ValueError(f"Task with id {task_id} not found")
    task = self.tasks[task_id]
    if task.is_completed():
      return None
    if task.task_status != TaskStatus.READY:
      raise ValueError(f"Task with id {task_id} status is {task.task_status}, cannot launch")
    task.task_status = TaskStatus.RUNNING
    return task.graph
  
  def _release_task(self,task_id: int) -> None:
    if task_id not in self.tasks:
      return 
    task = self.tasks[task_id]
    if task.graph is not None:
      self.graph_to_task.pop(task.graph.graph_id, None)
    self.tasks.pop(task_id, None)
        
  def _mark_completed(self,task_id: int) -> None:
    if task_id not in self.tasks:
      raise ValueError(f"Task with id {task_id} not found")
    task = self.tasks[task_id]
    if task.is_completed():
      return

    # Record transfer latency *before* callback so metrics survive
    # callback exceptions.
    if task.submit_time > 0:
      elapsed = time.perf_counter() - task.submit_time
      _metrics_incr("miniflex_transfer_completed_count")
      _metrics_observe("miniflex_transfer_latency_sec", elapsed)

    if task.callback:
      if isinstance(task.callback, list):
        for callback in task.callback:
          if callback is not None:
            callback()
      elif callable(task.callback):
        task.callback()
      else:
        raise ValueError(f"Invalid callback type, got {type(task.callback).__name__}")
    task.task_status = TaskStatus.COMPLETED
    task.task_end_op_finished = True
    if task.graph is not None:
      self.graph_to_task.pop(task.graph.graph_id, None)
    task.shed_heavy_resource()
    if task.request_returned:
      self._release_task(task_id)
      
  def _process_empty_graph(self,task_id: int) -> None:
    if task_id not in self.tasks:
      raise ValueError(f"Task with id {task_id} not found")
    task = self.tasks[task_id]
    if task.graph is None:
      return
    if task.graph.num_ops == 0:
      self._mark_completed(task_id)

  def _get_completed_ops(self, timeout: Optional[float] = None) -> List[CompletedOp]:
    result: List[CompletedOp] = []
    completed_ops = self._transfer_manager.wait(timeout)
    for completed_op in completed_ops:
      if not isinstance(completed_op, CompletedOp):
        raise ValueError(f"expected CompletedOp, got {type(completed_op).__name__}")
      if completed_op.graph_id not in self.graph_to_task:
        continue
      result.append(completed_op)
    return result

  def check_completed(self, task_id: int, completely: bool = False) -> bool:
    if task_id not in self.tasks:
      raise ValueError(f"Task with id {task_id} not found")
    task = self.tasks[task_id]
    self._process_empty_graph(task_id)
    if completely:
      return task.is_completed()
    return task.is_completed() or task.task_end_op_finished

  def _cancel_task(self,task_id: int) -> None:
    if task_id not in self.tasks:
      raise ValueError(f"Task with id {task_id} not found")
    task = self.tasks[task_id]
    if not task.is_completed():
      task.task_status = TaskStatus.CANCELED
    self._release_task(task_id)
    
  def _update_tasks(self, timeout: Optional[float] = None) -> None:
    completed_ops = self._get_completed_ops(timeout)
    for completed_op in completed_ops:
      if completed_op.graph_id not in self.graph_to_task:
        continue
      task_id = self.graph_to_task[completed_op.graph_id]
      if task_id not in self.tasks:
        continue
      task = self.tasks[task_id]

      if completed_op.is_graph_completed():
        self._mark_completed(task_id)
      elif completed_op.op_id == task.task_end_op_id:
        task.task_end_op_finished = True

      if completed_op.op_id in task.op_callback_dict:
        task.op_callback_dict[completed_op.op_id]()

  def _launch_task(self,task_id: int) -> None:
    transfer_graph = self.check_task_ready(task_id)
    if transfer_graph is None:
      return
    if transfer_graph.num_ops > 0:
      self.tasks[task_id].submit_time = time.perf_counter()
      self._transfer_manager.submit(transfer_graph)
      


class KVTaskEngine:
  def __init__(self,
               model_config: ModelConfig,
               cache_config: CacheConfig,
               gpu_register_port: str):
    self.model_config = model_config
    self.cache_config = cache_config
    self.gpu_register_port = gpu_register_port
    self._manager = KVTaskManager(model_config, cache_config, gpu_register_port)
  
  def start(self):
    self._manager.start()
  
  def is_ready(self):
    return self._manager.is_ready()

  def shutdown(self):
    self._manager.shutdown()
    
  def get_new_task_id(self) -> int:
    return self._manager._get_new_task_id()
  
  def _get_impl(self,
                      task_id: int,
                      token_ids: np.ndarray,
                      slot_mapping: np.ndarray,
                      is_fake_slot_mapping: bool = False,
                      token_mask: Optional[np.ndarray] = None,
                      namespace: Optional[List[str]] = None
                      ) -> Tuple[int, np.ndarray]:
    self._manager.create_get_task(
      task_id=task_id,
      token_ids=token_ids,
      slot_mapping=slot_mapping,
      token_mask=token_mask,
      is_fake_slot_mapping=is_fake_slot_mapping,
      namespace=namespace
    )
    self._manager._process_empty_graph(task_id)
    return task_id, self._manager.tasks[task_id].return_mask
  
  def get_async(self,
                 request: KVRequest) -> Tuple[int, np.ndarray]:
    if request.request_type != KVRequestType.GET:
      raise ValueError(f"Invalid request type, got {request.request_type}")
    task_id = request.request_id if request.request_id != -1 else self.get_new_task_id()
    if task_id in self._manager.tasks:
      raise ValueError(f"Task with id {task_id} already exists")
    if request.slot_mapping is None:
      raise ValueError("in func: get_async, slot_mapping is required")
    slot_mapping = request.slot_mapping
    token_ids = request.token_ids
    namespace = request.namespace
    token_mask = request.token_mask
    if token_mask is None:
      token_mask = np.ones_like(request.token_ids, dtype=np.bool_)
    task_id,return_mask = self._get_impl(
      task_id=task_id,
      token_ids=token_ids,
      slot_mapping=slot_mapping,
      token_mask=token_mask,
      is_fake_slot_mapping=False,
      namespace=namespace
    )
    self._manager._launch_task(task_id)
    return task_id, return_mask
  
  def get_match(self,
                request: KVRequest) -> Tuple[int, np.ndarray]:
    if request.request_type != KVRequestType.GET:
      raise ValueError(f"Invalid request type, got {request.request_type}")
    task_id = request.request_id if request.request_id != -1 else self.get_new_task_id()
    if task_id in self._manager.tasks:
      raise ValueError(f"Task with id {task_id} already exists")
    slot_mapping = request.slot_mapping
    token_mask = request.token_mask
    if token_mask is None:
      token_mask = np.ones_like(request.token_ids, dtype=np.bool_)
    if slot_mapping is None:
      slot_mapping = np.zeros(int(token_mask.sum()), dtype=np.int64)
    token_ids = request.token_ids
    namespace = request.namespace
    task_id,return_mask = self._get_impl(
      task_id=task_id,
      token_ids=token_ids,
      slot_mapping=slot_mapping,
      token_mask=token_mask,
      is_fake_slot_mapping=True,
      namespace=namespace
    )
    return task_id, return_mask

  def _put_impl(self,
                task_id: int,
                token_ids: np.ndarray,
                slot_mapping: np.ndarray,
                is_fake_slot_mapping: bool = False,
                token_mask: Optional[np.ndarray] = None,
                namespace: List[str] = None
                ) -> Tuple[int, np.ndarray]:
    self._manager.create_put_task(
      task_id=task_id,
      token_ids=token_ids,
      slot_mapping=slot_mapping,
      token_mask=token_mask,
      is_fake_slot_mapping=is_fake_slot_mapping,
      namespace=namespace
    )
    self._manager._process_empty_graph(task_id)
    return task_id, self._manager.tasks[task_id].return_mask
  
  def put_async(self,
                request: KVRequest) -> Tuple[int, np.ndarray]:
    if request.request_type != KVRequestType.PUT:
      raise ValueError(f"Invalid request type, got {request.request_type}")
    task_id = request.request_id if request.request_id != -1 else self.get_new_task_id()
    if task_id in self._manager.tasks:
      raise ValueError(f"Task with id {task_id} already exists")
    if request.slot_mapping is None:
      raise ValueError("in func: put_async, slot_mapping is required")
    slot_mapping = request.slot_mapping
    token_ids = request.token_ids
    namespace = request.namespace
    token_mask = request.token_mask
    if token_mask is None:
      token_mask = np.ones_like(request.token_ids, dtype=np.bool_)
    task_id,return_mask = self._put_impl(
      task_id=task_id,
      token_ids=token_ids,
      slot_mapping=slot_mapping,
      token_mask=token_mask,
      is_fake_slot_mapping=False,
      namespace=namespace
    )
    self._manager._launch_task(task_id)
    return task_id, return_mask
  
  def put_match(self,
               request: KVRequest) -> Tuple[int, np.ndarray]:
    if request.request_type != KVRequestType.PUT:
      raise ValueError(f"Invalid request type, got {request.request_type}")
    task_id = request.request_id if request.request_id != -1 else self.get_new_task_id()
    if task_id in self._manager.tasks:
      raise ValueError(f"Task with id {task_id} already exists")
    slot_mapping = request.slot_mapping
    token_mask = request.token_mask
    if token_mask is None:
      token_mask = np.ones_like(request.token_ids, dtype=np.bool_)
    if slot_mapping is None:
      slot_mapping = np.zeros(int(token_mask.sum()), dtype=np.int64)
    token_ids = request.token_ids
    namespace = request.namespace
    task_id,return_mask = self._put_impl(
      task_id=task_id,
      token_ids=token_ids,
      slot_mapping=slot_mapping,
      token_mask=token_mask,
      is_fake_slot_mapping=True,
      namespace=namespace
    )
    return task_id, return_mask
  
  def _wait_impl(self,
                 task_ids: List[int],
                 timeout: float = 20.0,
                 completely: bool = False,
                 only_return_finished: bool = False) -> Dict[int, KVResponse]:
    return_responses: Dict[int, KVResponse] = {}
    start_time = time.time()
    is_timeout = timeout == 0.0

    self._manager._update_tasks(timeout=0)

    for task_id in task_ids:
      while True:
        if task_id not in self._manager.tasks:
          return_responses[task_id] = KVResponse(
            status=KVResponseStatus.NOTFOUND,
            task_id=task_id,
            return_mask=None,
          )
          break

        task = self._manager.tasks[task_id]
        if task.task_status == TaskStatus.UNREADY:
          print(f"[DBG-KVTSK] task={task_id} UNREADY, only_return_finished={only_return_finished}", flush=True)
          return_responses[task_id] = KVResponse(
            status=KVResponseStatus.UNREADY,
            task_id=task_id,
            return_mask=None,
          )
          break

        if self._manager.check_completed(task_id, completely=completely):
          return_responses[task_id] = KVResponse(
            status=convert_to_response_status(task.task_status),
            task_id=task_id,
            return_mask=task.return_mask,
          )
          if task.is_completed():
            self._manager._release_task(task_id)
          else:
            task.request_returned = True
          break

        if only_return_finished:
          break

        if time.time() - start_time > timeout:
          is_timeout = True
        if is_timeout:
          return_responses[task_id] = KVResponse(
            status=KVResponseStatus.TIMEOUT,
            task_id=task_id,
            return_mask=None,
          )
          break

        self._manager._update_tasks(timeout=0.001)

    return return_responses

  def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
    if isinstance(task_ids, int):
      task_ids = [task_ids]
    return self._wait_impl(task_ids, completely=False, only_return_finished=True)

  def wait(self,
           task_ids: Union[int, List[int]],
           timeout: float = 20.0,
           completely: bool = False) -> Dict[int, KVResponse]:
    if isinstance(task_ids, int):
      task_ids = [task_ids]
    return self._wait_impl(task_ids, timeout, completely=completely)

  def set_slot_mappings(self,
                       task_ids: List[int],
                      slot_mappings: List[np.ndarray]) -> None:
    if len(task_ids) != len(slot_mappings):
      raise ValueError("in func: set_slot_mapping, task_ids and slot_mappings must have the same length")
    self._manager.set_slot_mapping(task_ids, slot_mappings)
    
  def merge_to_batch_kvtask(self,
                            batch_id: int,
                            task_ids: List[int],
                            batch_task_type: TaskType) -> TransferOpGraph:
    if batch_id in self._manager.tasks:
      raise ValueError(f"Task with id {batch_id} already exists")

    op_callback_dict: Dict[int, Callable] = {}
    task_end_op_ids: List[int] = []
    callbacks: List[Optional[Callable]] = []
    transfer_graphs: List[TransferOpGraph] = []
    return_masks: List[np.ndarray] = []
    expected_type = TaskType.GET if batch_task_type == TaskType.BATCH_GET else TaskType.PUT

    for task_id in task_ids:
      if task_id not in self._manager.tasks:
        raise ValueError(f"Task with id {task_id} not found")
      task = self._manager.tasks[task_id]
      if task.task_type != expected_type:
        raise ValueError(
          f"only {expected_type.value} task can be launched as {batch_task_type.value}"
        )
      return_masks.append(task.return_mask)
      transfer_graph = self._manager.check_task_ready(task_id)
      if transfer_graph is not None and transfer_graph.num_ops > 0:
        transfer_graphs.append(transfer_graph)
        op_callback_dict.update(task.op_callback_dict)
        task_end_op_ids.append(task.task_end_op_id)
        callbacks.append(task.callback)
        

    batch_task_graph, task_end_op_id, merged_op_callback_dict = merge_to_batch_graph(
      batch_id, transfer_graphs, task_end_op_ids, op_callback_dict
    )
    self._manager.tasks[batch_id] = KVTask(
      task_id=batch_id,
      task_type=batch_task_type,
      task_status=TaskStatus.READY,
      task_end_op_id=task_end_op_id,
      task_end_op_finished=False,
      graph=batch_task_graph,
      return_mask=return_masks,
      callback=callbacks,
      op_callback_dict=merged_op_callback_dict,
      is_batch_task=True,
      sub_task=task_ids
    )
    self._manager.graph_to_task[batch_task_graph.graph_id] = batch_id
    self._manager._process_empty_graph(batch_id)

    for task_id in task_ids:
      child_task = self._manager.tasks[task_id]
      if child_task.graph is not None:
        self._manager.graph_to_task.pop(child_task.graph.graph_id, None)
      self._manager.tasks.pop(task_id, None)
    return batch_task_graph

  def launch_tasks(self,
                   task_ids: List[int],
                   slot_mappings: List[np.ndarray],
                   as_batch: bool = False,
                   batch_id: int = -1) -> List[int]:
    if len(task_ids) != len(slot_mappings):
      raise ValueError("in func: launch_tasks, task_ids and slot_mappings must have the same length")
    if len(task_ids) == 0:
      return []

    self.set_slot_mappings(task_ids, slot_mappings)

    all_get = all(self._manager.tasks[task_id].task_type == TaskType.GET for task_id in task_ids)
    all_put = all(self._manager.tasks[task_id].task_type == TaskType.PUT for task_id in task_ids)

    if len(task_ids) > 1 and as_batch and (all_get or all_put):
      if batch_id == -1:
        batch_id = self.get_new_task_id()
      batch_task_type = TaskType.BATCH_GET if all_get else TaskType.BATCH_PUT
      batch_graph = self.merge_to_batch_kvtask(batch_id, task_ids, batch_task_type)
      if batch_graph.num_ops > 0:
        self._manager.tasks[batch_id].task_status = TaskStatus.RUNNING
        self._manager.tasks[batch_id].submit_time = time.perf_counter()
        self._manager._transfer_manager.submit_batch([batch_graph])
      return [batch_id]

    transfer_graphs: List[TransferOpGraph] = []
    for task_id in task_ids:
      transfer_graph = self._manager.check_task_ready(task_id)
      if transfer_graph is not None and transfer_graph.num_ops > 0:
        self._manager.tasks[task_id].submit_time = time.perf_counter()
        transfer_graphs.append(transfer_graph)

    if transfer_graphs:
      self._manager._transfer_manager.submit_batch(transfer_graphs)
    return task_ids

  def cancel_tasks(self, task_ids: Union[int, List[int]]) -> None:
    if isinstance(task_ids, int):
      task_ids = [task_ids]
    for task_id in task_ids:
      self._manager._cancel_task(task_id)

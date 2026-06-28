# 本测试文件由 Claude (Anthropic) 编写。
# 测试内容：vLLM connector 适配：prefill GET 与 request_finished PUT 生命周期、batch GET、preemption 取消、worker no-op、no-match 取消、部分命中只写未命中、abort 不 PUT、失败上报、SYNC_GET，以及真实 PUT→GET 经 CPU 的端到端往返。

import os
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import miniflex.integration.vllm.vllm_v1_adapter as adapter_module
from miniflex.common.config import CacheConfig, ModelConfig
from miniflex.common.request import KVResponse, KVResponseStatus
from miniflex.integration.config import MiniFlexConfig
from miniflex.server.utils import normalize_zmq_endpoint


def tokens(values):
  return list(values)


class FakeKVTaskEngine:
  next_task_id = 100
  next_batch_id = 1000
  instances = []

  def __init__(self, model_config, cache_config, gpu_register_port):
    self.model_config = model_config
    self.cache_config = cache_config
    self.gpu_register_port = gpu_register_port
    self.started = False
    self.shutdown_called = False
    self.cancelled = []
    self.launched = []
    self.completed = {}
    # 可配置：get_match 返回 0 命中（no-match）；put_match 只有末尾 N 个 token 未命中（部分命中）
    self.get_match_returns_zero = False
    self.put_unmatched = None
    FakeKVTaskEngine.instances.append(self)

  def start(self):
    self.started = True

  def is_ready(self):
    return self.started and not self.shutdown_called

  def shutdown(self):
    self.shutdown_called = True

  def get_match(self, request):
    task_id = FakeKVTaskEngine.next_task_id
    FakeKVTaskEngine.next_task_id += 1
    if self.get_match_returns_zero:
      return_mask = np.zeros_like(request.token_mask)
    else:
      return_mask = request.token_mask.copy()
    return task_id, return_mask

  def put_match(self, request):
    task_id = FakeKVTaskEngine.next_task_id
    FakeKVTaskEngine.next_task_id += 1
    if self.put_unmatched is None:
      return_mask = np.ones_like(request.token_ids, dtype=np.bool_)
    else:
      # 前缀已命中、末尾 put_unmatched 个未命中（需写入），与真实引擎语义一致
      return_mask = np.zeros_like(request.token_ids, dtype=np.bool_)
      return_mask[-self.put_unmatched:] = True
    return task_id, return_mask

  def cancel_tasks(self, task_ids):
    self.cancelled.extend(task_ids)

  def launch_tasks(self, task_ids, slot_mappings, as_batch=False):
    task_ids = list(task_ids)
    slot_mappings = [slot_mapping.copy() for slot_mapping in slot_mappings]
    if as_batch and len(task_ids) > 1:
      launched_ids = [FakeKVTaskEngine.next_batch_id]
      FakeKVTaskEngine.next_batch_id += 1
    else:
      launched_ids = task_ids
    self.launched.append({
      "task_ids": task_ids,
      "slot_mappings": slot_mappings,
      "as_batch": as_batch,
      "launched_ids": launched_ids,
    })
    return launched_ids

  def try_wait(self, task_ids):
    responses = {}
    for task_id in list(task_ids):
      if task_id in self.completed:
        responses[task_id] = self.completed.pop(task_id)
    return responses

  def wait(self, task_ids, timeout=20.0, completely=False):
    del timeout, completely
    return self.try_wait(task_ids)

  def complete(self, task_id, status=KVResponseStatus.SUCCESS):
    self.completed[task_id] = KVResponse(
      status=status,
      task_id=task_id,
      return_mask=None,
    )


class FakeGPURegisterClient:
  instances = []

  def __init__(self, gpu_register_port, device_id=0, dp_rank=0, tp_rank=0):
    self.gpu_register_port = gpu_register_port
    self.device_id = device_id
    self.dp_rank = dp_rank
    self.tp_rank = tp_rank
    self.registered = []
    self.closed = False
    FakeGPURegisterClient.instances.append(self)

  def register_to_server(self, gpu_blocks, gpu_layout):
    self.registered.append((gpu_blocks, gpu_layout))

  def close(self):
    self.closed = True


@dataclass
class FakeParallelConfig:
  tensor_parallel_size: int = 1
  data_parallel_size: int = 1
  data_parallel_rank: int = 0


class FakeModelConfig:
  dtype = torch.float32
  is_deepseek_mla = False

  def get_num_layers(self, parallel_config):
    del parallel_config
    return 2

  def get_head_size(self):
    return 2

  def get_total_num_kv_heads(self):
    return 1


@dataclass
class FakeCacheConfig:
  block_size: int = 2


@dataclass
class FakeVllmConfig:
  model_config: FakeModelConfig
  cache_config: FakeCacheConfig
  parallel_config: FakeParallelConfig


class FakeRequest:
  def __init__(
      self,
      request_id: str,
      token_ids,
      finished: bool = False,
      finished_reason: int = 1,
      namespace_info: Optional[list[str]] = None,
  ):
    self.request_id = request_id
    self.req_id = request_id
    self.all_token_ids = tokens(token_ids)
    self.num_tokens = len(self.all_token_ids)
    self._finished = finished
    self._finished_reason = finished_reason
    self.namespace_info = namespace_info
    self.cache_salt = None
    self.lora_request = None

  def is_finished(self):
    return self._finished

  def get_finished_reason(self):
    return self._finished_reason


class FakeKVCacheBlocks:
  def __init__(self, block_ids):
    self._block_ids = list(block_ids)

  def get_block_ids(self):
    return [self._block_ids]


class FakeScheduledCachedReqs:
  def __init__(self, req_ids):
    self.req_ids = set(req_ids)


class FakeSchedulerOutput:
  def __init__(
      self,
      scheduled_new_reqs=None,
      scheduled_cached_req_ids=None,
      finished_req_ids=None,
      preempted_req_ids=None,
      expose_preempted=True,
  ):
    self.scheduled_new_reqs = scheduled_new_reqs or []
    self.scheduled_cached_reqs = (
      FakeScheduledCachedReqs(scheduled_cached_req_ids)
      if scheduled_cached_req_ids else None
    )
    self.finished_req_ids = set(finished_req_ids or [])
    if expose_preempted:
      self.preempted_req_ids = set(preempted_req_ids or [])


class FakeConnectorOutput:
  def __init__(self):
    self.finished_sending = None
    self.finished_recving = None


@contextmanager
def patched_adapter(enable_batch=False, sync_get=False):
  old_engine = adapter_module.KVTaskEngine
  old_client = adapter_module.MiniFlexGPURegisterClient
  old_env = {
    "MINIFLEX_ENABLE_BATCH": os.environ.get("MINIFLEX_ENABLE_BATCH"),
    "MINIFLEX_SYNC_GET": os.environ.get("MINIFLEX_SYNC_GET"),
    "MINIFLEX_GPU_REGISTER_PORT": os.environ.get("MINIFLEX_GPU_REGISTER_PORT"),
    "MINIFLEX_NUM_CPU_BLOCKS": os.environ.get("MINIFLEX_NUM_CPU_BLOCKS"),
    "MINIFLEX_ENABLE_SSD": os.environ.get("MINIFLEX_ENABLE_SSD"),
  }
  FakeKVTaskEngine.next_task_id = 100
  FakeKVTaskEngine.next_batch_id = 1000
  FakeKVTaskEngine.instances.clear()
  FakeGPURegisterClient.instances.clear()
  adapter_module.KVTaskEngine = FakeKVTaskEngine
  adapter_module.MiniFlexGPURegisterClient = FakeGPURegisterClient
  os.environ["MINIFLEX_ENABLE_BATCH"] = "1" if enable_batch else "0"
  os.environ["MINIFLEX_SYNC_GET"] = "1" if sync_get else "0"
  os.environ["MINIFLEX_GPU_REGISTER_PORT"] = "ipc:///tmp/miniflex_vllm_adapter_test.sock"
  os.environ["MINIFLEX_NUM_CPU_BLOCKS"] = "16"
  os.environ["MINIFLEX_ENABLE_SSD"] = "0"
  try:
    yield
  finally:
    adapter_module.KVTaskEngine = old_engine
    adapter_module.MiniFlexGPURegisterClient = old_client
    for key, value in old_env.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value


def make_vllm_config():
  return FakeVllmConfig(
    model_config=FakeModelConfig(),
    cache_config=FakeCacheConfig(block_size=2),
    parallel_config=FakeParallelConfig(),
  )


def make_scheduler_connector(enable_batch=False):
  if adapter_module.KVConnectorRole is None:
    raise RuntimeError("vLLM KVConnectorRole is unavailable")
  connector = adapter_module.MiniFlexConnectorV1Impl(
    make_vllm_config(),
    adapter_module.KVConnectorRole.SCHEDULER,
  )
  assert connector.scheduler.enable_batch == enable_batch
  assert len(FakeKVTaskEngine.instances) == 1
  return connector, FakeKVTaskEngine.instances[-1]


def test_single_card_prefill_get_lifecycle_reports_finished_recving():
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    request = FakeRequest("req-get", [1, 2, 3, 4, 5, 6], namespace_info=["tenant-a"])

    num_new_matched_tokens, need_get = connector.get_num_new_matched_tokens(
      request,
      num_computed_tokens=2,
    )
    assert need_get is True
    assert num_new_matched_tokens == 4
    assert list(connector.scheduler.req_id_to_task_dict) == ["req-get"]

    connector.update_state_after_alloc(
      request,
      FakeKVCacheBlocks([10, 11, 12]),
      num_external_tokens=num_new_matched_tokens,
    )
    connector.build_connector_meta(
      FakeSchedulerOutput(scheduled_new_reqs=[request])
    )

    assert len(engine.launched) == 1
    launch = engine.launched[0]
    assert launch["task_ids"] == [100]
    assert launch["as_batch"] is False
    assert launch["slot_mappings"][0].tolist() == [22, 22, 24, 24]

    engine.complete(100)
    output = FakeConnectorOutput()
    connector.update_connector_output(output)
    assert output.finished_recving == {"req-get"}
    assert output.finished_sending == set()
    assert connector.scheduler.req_id_to_task_dict == {}

    connector.shutdown()
    assert engine.shutdown_called


def test_single_card_request_finished_put_lifecycle_reports_finished_sending():
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    request = FakeRequest("req-put", [7, 8, 9, 10, 11, 12], finished=True)

    should_hold_blocks, transfer_params = connector.request_finished(
      request,
      block_ids=[20, 21, 22],
    )
    assert should_hold_blocks is True
    assert transfer_params is None

    connector.build_connector_meta(
      FakeSchedulerOutput(finished_req_ids={"req-put"})
    )
    assert len(engine.launched) == 1
    launch = engine.launched[0]
    assert launch["task_ids"] == [100]
    assert launch["as_batch"] is False
    assert launch["slot_mappings"][0].tolist() == [40, 40, 42, 42]

    engine.complete(100)
    output = FakeConnectorOutput()
    connector.update_connector_output(output)
    assert output.finished_sending == {"req-put"}
    assert output.finished_recving == set()
    assert connector.scheduler.req_id_to_task_dict == {}
    connector.shutdown()


def test_single_card_batch_get_reports_all_child_requests_finished():
  with patched_adapter(enable_batch=True):
    connector, engine = make_scheduler_connector(enable_batch=True)
    first = FakeRequest("req-a", [1, 2, 3, 4])
    second = FakeRequest("req-b", [5, 6, 7, 8])

    assert connector.get_num_new_matched_tokens(first, 0) == (4, True)
    connector.update_state_after_alloc(first, FakeKVCacheBlocks([0, 1]), 4)
    assert connector.get_num_new_matched_tokens(second, 0) == (4, True)
    connector.update_state_after_alloc(second, FakeKVCacheBlocks([2, 3]), 4)

    connector.build_connector_meta(
      FakeSchedulerOutput(scheduled_new_reqs=[first, second])
    )
    assert len(engine.launched) == 1
    launch = engine.launched[0]
    assert launch["task_ids"] == [100, 101]
    assert launch["as_batch"] is True
    assert launch["launched_ids"] == [1000]
    assert launch["slot_mappings"][0].tolist() == [0, 0, 2, 2]
    assert launch["slot_mappings"][1].tolist() == [4, 4, 6, 6]
    assert list(connector.scheduler.get_tasks.keys()) == [1000]

    engine.complete(1000)
    output = FakeConnectorOutput()
    connector.update_connector_output(output)
    assert output.finished_recving == {"req-a", "req-b"}
    assert output.finished_sending == set()
    assert connector.scheduler.get_tasks == {}
    connector.shutdown()


def test_preempted_request_before_launch_is_cancelled():
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    request = FakeRequest("req-preempt", [1, 2, 3, 4])

    assert connector.get_num_new_matched_tokens(request, 0) == (4, True)
    connector.update_state_after_alloc(request, FakeKVCacheBlocks([3, 4]), 4)
    connector.build_connector_meta(
      FakeSchedulerOutput(
        scheduled_new_reqs=[],
        preempted_req_ids={"req-preempt"},
      )
    )

    assert engine.cancelled == [100]
    assert engine.launched == []
    assert connector.scheduler.req_id_to_task_dict == {}
    assert connector.scheduler.tasks_to_launch == {}
    assert connector.scheduler.tasks_to_cancel == {}
    connector.shutdown()


def test_worker_side_noop_interfaces_and_stats_sentinel():
  with patched_adapter(enable_batch=False):
    if adapter_module.KVConnectorRole is None:
      raise RuntimeError("vLLM KVConnectorRole is unavailable")
    connector = adapter_module.MiniFlexConnectorV1Impl(
      make_vllm_config(),
      adapter_module.KVConnectorRole.WORKER,
    )

    assert connector.take_events() == []
    assert connector.get_finished({"already-finished"}) == (None, None)
    connector.start_load_kv(object())
    connector.wait_for_layer_load("model.layers.0.self_attn")
    connector.save_kv_layer("model.layers.0.self_attn", torch.empty(0), object())
    connector.wait_for_save()

    stats = connector.get_kv_connector_stats()
    if adapter_module.KVConnectorStats is not None:
      assert stats is not None
      stats.reset()
      assert stats.reduce() == {}
      assert stats.aggregate(adapter_module.KVConnectorStats(data={"x": 1})).data == {"x": 1}

    connector.worker.close()
    assert FakeGPURegisterClient.instances[-1].closed


def test_no_match_get_cancels_and_returns_not_needed():
  """GET 未命中（matched=0）：立即取消任务，返回 (0, False)，不留下任何待处理状态。"""
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    engine.get_match_returns_zero = True
    request = FakeRequest("req-nomatch", [1, 2, 3, 4, 5, 6])

    assert connector.get_num_new_matched_tokens(request, num_computed_tokens=0) == (0, False)
    assert engine.cancelled == [100]
    assert connector.scheduler.req_id_to_task_dict == {}
    assert connector.scheduler.tasks_to_cancel == {}
    assert connector.scheduler.tasks_to_launch == {}

    connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[request]))
    assert engine.launched == []
    connector.shutdown()


def test_partial_match_put_only_writes_unmatched_blocks():
  """部分命中的 PUT：前缀已在缓存，只写入末尾未命中的 block。"""
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    engine.put_unmatched = 4  # 待存 6 token，末尾 4 个未命中 -> 命中 2(1 block)，写 2 block
    request = FakeRequest("req-partial", [1, 2, 3, 4, 5, 6, 7, 8], finished=True)

    should_hold, params = connector.request_finished(request, block_ids=[20, 21, 22, 23])
    assert should_hold is True
    assert params is None

    connector.build_connector_meta(FakeSchedulerOutput(finished_req_ids={"req-partial"}))
    assert len(engine.launched) == 1
    # num_token_to_put=(cdiv(8,2)-1)*2=6；matched=2(1 block)，unmatched=4(2 block)
    # 写入 block_ids[1:3]=[21,22] -> slot=[21,22].repeat(2)*2
    assert engine.launched[0]["slot_mappings"][0].tolist() == [42, 42, 44, 44]

    engine.complete(100)
    output = FakeConnectorOutput()
    connector.update_connector_output(output)
    assert output.finished_sending == {"req-partial"}
    connector.shutdown()


def test_empty_put_below_one_block_is_skipped():
  """不足一个 block 的 PUT（num_token_to_put==0）：直接跳过，不建任务、不取消。"""
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    request = FakeRequest("req-tiny", [1, 2], finished=True)  # 2 token -> num_token_to_put=0

    should_hold, params = connector.request_finished(request, block_ids=[30])
    assert should_hold is False
    assert params is None
    assert connector.scheduler.req_id_to_task_dict == {}

    connector.build_connector_meta(FakeSchedulerOutput(finished_req_ids={"req-tiny"}))
    assert engine.launched == []
    assert engine.cancelled == []
    connector.shutdown()


def test_aborted_request_is_not_put():
  """非正常结束(ABORT 等 reason>=2)的请求不保存 KV：request_finished 返回 (False, None)。"""
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    # reason=2 (ABORT)，足够长本可 PUT，但因结束原因异常应被跳过
    request = FakeRequest("req-abort", [1, 2, 3, 4, 5, 6], finished=True, finished_reason=2)

    should_hold, params = connector.request_finished(request, block_ids=[40, 41, 42])
    assert should_hold is False
    assert params is None
    assert connector.scheduler.req_id_to_task_dict == {}

    connector.build_connector_meta(FakeSchedulerOutput(finished_req_ids={"req-abort"}))
    assert engine.launched == []
    connector.shutdown()


def test_failed_get_task_records_failed_block_ids():
  """GET 任务失败：仍上报 finished_recving，失败 block 进入 failed_block_ids 供回收且取出即清空。"""
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    request = FakeRequest("req-fail", [1, 2, 3, 4])

    assert connector.get_num_new_matched_tokens(request, 0) == (4, True)
    connector.update_state_after_alloc(request, FakeKVCacheBlocks([3, 4]), 4)
    connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[request]))
    assert len(engine.launched) == 1

    engine.complete(100, status=KVResponseStatus.FAILED)
    output = FakeConnectorOutput()
    connector.update_connector_output(output)
    assert output.finished_recving == {"req-fail"}
    assert connector.get_block_ids_with_load_errors() == {3, 4}
    assert connector.get_block_ids_with_load_errors() == set()  # 取出即清空
    connector.shutdown()


def test_preemption_via_state_diff_fallback_cancels_pending_task():
  """preemption 的 state-diff 回退分支：scheduler_output 不带 preempted_req_ids 时，
  用 previous-current-finished 推断出被抢占请求，并在 launch 前取消其任务。"""
  with patched_adapter(enable_batch=False):
    connector, engine = make_scheduler_connector(enable_batch=False)
    request = FakeRequest("req-fb", [1, 2, 3, 4])

    # step 1：请求被调度但还没有 miniflex 任务，仅用于 seed previous_scheduler_req_ids
    connector.build_connector_meta(
      FakeSchedulerOutput(scheduled_new_reqs=[request], expose_preempted=False)
    )
    assert connector.previous_scheduler_req_ids == {"req-fb"}

    # step 2：建立 GET 任务(进入 tasks_to_launch)，随后该请求从调度中消失
    assert connector.get_num_new_matched_tokens(request, 0) == (4, True)
    connector.update_state_after_alloc(request, FakeKVCacheBlocks([5, 6]), 4)
    connector.build_connector_meta(
      FakeSchedulerOutput(scheduled_new_reqs=[], expose_preempted=False)
    )

    assert engine.cancelled == [100]
    assert engine.launched == []
    assert connector.scheduler.req_id_to_task_dict == {}
    assert connector.scheduler.tasks_to_launch == {}
    assert connector.scheduler.tasks_to_cancel == {}
    connector.shutdown()


def test_sync_get_blocks_then_reports_finished_recving():
  """MINIFLEX_SYNC_GET=1：build_connector_meta 内同步等待 GET 完成，
  且该请求仍要通过 finished_recving 通知 vLLM（否则会卡在 WAITING_FOR_REMOTE_KVS）。"""
  with patched_adapter(enable_batch=False, sync_get=True):
    connector, engine = make_scheduler_connector(enable_batch=False)
    assert connector.scheduler.sync_get is True
    request = FakeRequest("req-sync", [1, 2, 3, 4])

    assert connector.get_num_new_matched_tokens(request, 0) == (4, True)
    connector.update_state_after_alloc(request, FakeKVCacheBlocks([7, 8]), 4)
    engine.complete(100)  # 预先标记完成，使同步 wait 立即返回
    connector.build_connector_meta(FakeSchedulerOutput(scheduled_new_reqs=[request]))
    assert connector.scheduler.get_tasks == {}  # 同步等待已消费该 GET 任务

    output = FakeConnectorOutput()
    connector.update_connector_output(output)
    assert output.finished_recving == {"req-sync"}
    connector.shutdown()


def _build_real_config(gpu_register_port: str) -> MiniFlexConfig:
  """构造真实端到端测试用的小配置：CPU-only、2 层、每块 2 token。"""
  model = ModelConfig(num_layers=2, num_kv_heads=1, head_size=2, dtype=torch.float32)
  cache = CacheConfig(
    tokens_per_block=2,
    enable_cpu=True,
    enable_ssd=False,
    num_cpu_blocks=16,
  )
  return MiniFlexConfig(
    enable_miniflex=True,
    gpu_register_port=gpu_register_port,
    cache_config=cache,
    model_config=model,
  )


def _make_gpu_layer_tensors(model: ModelConfig, cache: CacheConfig, num_blocks: int):
  """按 vLLM 非 MLA 的 5D 分层布局创建 GPU KV tensor（全零）。"""
  layer_shape = (2, num_blocks, cache.tokens_per_block, model.num_kv_heads, model.head_size)
  return [
    torch.zeros(layer_shape, device="cuda:0", dtype=model.dtype)
    for _ in range(model.num_layers)
  ]


def _fill_gpu_blocks(tensors, block_ids):
  """给指定 block 写入按 (layer, block) 唯一的可校验值。"""
  for layer_id, tensor in enumerate(tensors):
    for block_id in block_ids:
      tensor[:, block_id].fill_(float(1000 + layer_id * 100 + block_id))


def _wait_until_ready(scheduler, timeout: float = 20.0) -> None:
  deadline = time.time() + timeout
  while time.time() < deadline:
    if scheduler.is_ready():
      return
    time.sleep(0.05)
  raise AssertionError("scheduler 未在超时时间内 ready")


def _wait_for_finished(scheduler, kind: str, req_id: str, timeout: float = 30.0) -> None:
  """轮询 query_finished_tasks，直到 req_id 出现在 sending/recving 集合里。"""
  deadline = time.time() + timeout
  while time.time() < deadline:
    finished_sending, finished_recving = scheduler.query_finished_tasks()
    done = finished_recving if kind == "recv" else finished_sending
    if req_id in done:
      return
    time.sleep(0.02)
  raise AssertionError(f"等待 {req_id} 的 {kind} 完成超时")


def _cleanup_ipc_endpoint(gpu_register_port: str) -> None:
  endpoint = normalize_zmq_endpoint(gpu_register_port)
  if endpoint.startswith("ipc://"):
    try:
      os.unlink(endpoint[len("ipc://"):])
    except FileNotFoundError:
      pass


def test_end_to_end_real_put_then_get_roundtrips_kv_through_cpu():
  """真实传输：PUT 把 GPU block[0,1] 存到 CPU，GET 再读回到 GPU block[4,5]，逐字节校验。"""
  if not torch.cuda.is_available():
    print("跳过：CUDA 不可用，真实 GPU 传输测试需要 CUDA")
    return

  torch.cuda.set_device(0)
  gpu_register_port = "ipc:///tmp/miniflex_vllm_adapter_e2e.sock"
  config = _build_real_config(gpu_register_port)
  model, cache = config.model_config, config.cache_config
  num_blocks = 8

  tensors = _make_gpu_layer_tensors(model, cache, num_blocks)
  _fill_gpu_blocks(tensors, block_ids=(0, 1))  # 源 block，PUT 读取
  originals = [tensor.clone() for tensor in tensors]
  kv_caches = {f"layers.{i}": tensor for i, tensor in enumerate(tensors)}

  token_ids = [101, 102, 103, 104, 105, 106]  # 6 token -> 3 block，PUT 存前 2 block

  scheduler = adapter_module.MiniFlexSchedulerConnector(config)
  worker = adapter_module.MiniFlexWorkerConnector(config)
  try:
    worker.register_to_server(kv_caches)
    _wait_until_ready(scheduler)

    # ---- PUT：GPU block[0,1] -> CPU ----
    put_request = FakeRequest("e2e-put", token_ids, finished=True, finished_reason=1)
    should_hold = scheduler.request_finished(put_request, block_ids=[0, 1, 2])
    assert should_hold is True
    scheduler.cancel_tasks()
    scheduler.launch_tasks()
    _wait_for_finished(scheduler, "send", "e2e-put")

    # ---- GET：CPU -> GPU block[4,5]（这两块当前为全零）----
    get_request = FakeRequest("e2e-get", token_ids)
    num_new_matched_tokens, need_get = scheduler.get_num_new_matched_tokens(get_request, 0)
    assert need_get is True
    assert num_new_matched_tokens == 4  # CPU 里只有前 2 block（4 token）
    scheduler.update_state_after_alloc(
      get_request,
      FakeKVCacheBlocks([4, 5, 6]),
      num_new_matched_tokens,
    )
    scheduler.cancel_tasks()
    scheduler.launch_tasks()
    _wait_for_finished(scheduler, "recv", "e2e-get")

    # ---- 校验：GPU block4==原 block0，block5==原 block1（逐层逐字节）----
    torch.cuda.synchronize()
    for layer_id, tensor in enumerate(tensors):
      assert torch.equal(
        tensor[:, 4].cpu(), originals[layer_id][:, 0].cpu()
      ), f"layer={layer_id} block4 应等于原 block0"
      assert torch.equal(
        tensor[:, 5].cpu(), originals[layer_id][:, 1].cpu()
      ), f"layer={layer_id} block5 应等于原 block1"

    assert scheduler.req_id_to_task_dict == {}
    assert scheduler.get_tasks == {} and scheduler.put_tasks == {}
  finally:
    worker.close()
    scheduler.shutdown()
    _cleanup_ipc_endpoint(gpu_register_port)


TEST_CASES = [
  ("单机单卡 prefill GET：命中、分配 block、launch、finished_recving", test_single_card_prefill_get_lifecycle_reports_finished_recving),
  ("单机单卡 request_finished PUT：异步保存、finished_sending", test_single_card_request_finished_put_lifecycle_reports_finished_sending),
  ("单机单卡 batch GET：一个 batch task 映射多个 request", test_single_card_batch_get_reports_all_child_requests_finished),
  ("单机单卡 preemption：launch 前取消任务", test_preempted_request_before_launch_is_cancelled),
  ("worker 侧 no-op 接口和 stats sentinel", test_worker_side_noop_interfaces_and_stats_sentinel),
  ("GET 未命中：取消任务并返回 (0, False)", test_no_match_get_cancels_and_returns_not_needed),
  ("部分命中 PUT：只写未命中的 block", test_partial_match_put_only_writes_unmatched_blocks),
  ("空 PUT 边界：不足一个 block 直接跳过", test_empty_put_below_one_block_is_skipped),
  ("ABORT 结束的请求不 PUT", test_aborted_request_is_not_put),
  ("失败 GET 任务：记录 failed_block_ids 供回收", test_failed_get_task_records_failed_block_ids),
  ("preemption state-diff 回退分支：launch 前取消", test_preemption_via_state_diff_fallback_cancels_pending_task),
  ("MINIFLEX_SYNC_GET：同步等待并仍上报 finished_recving", test_sync_get_blocks_then_reports_finished_recving),
  ("真实传输 PUT->GET：KV 经 CPU 往返并逐字节校验", test_end_to_end_real_put_then_get_roundtrips_kv_through_cpu),
]


def run_all_tests():
  print("开始运行 MiniFlex vLLM V1 adapter 单机单卡模拟测试")
  passed = 0
  total = len(TEST_CASES)
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{total}] 开始：{name}")
    try:
      test_fn()
    except Exception as exc:
      print(f"[{index}/{total}] 失败：{name}，错误：{type(exc).__name__}: {exc}")
      traceback.print_exc()
      raise
    passed += 1
    print(f"[{index}/{total}] 通过：{name}")
  print(f"MiniFlex vLLM V1 adapter 测试完成：通过 {passed}/{total}")


if __name__ == "__main__":
  run_all_tests()

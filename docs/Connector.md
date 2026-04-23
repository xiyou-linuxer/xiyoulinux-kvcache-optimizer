# KV Connector 基类（`KVConnectorBase_V1`）说明

面向 vLLM **v1** 分布式 KV 传输。源码：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`。  
类型别名：`KVConnectorBase` / `KVConnectorBaseType` 均等价于 `KVConnectorBase_V1`（`vllm/distributed/kv_transfer/kv_connector/base.py`）。

---

## 1. 架构：两个角色、两个实例

同一连接器 **类** 会创建 **两个实例**，由构造函数中的 **`KVConnectorRole`** 区分：

| 角色 | 所在位置 | 职责 |
|------|-----------|------|
| `SCHEDULER` | 与 `Scheduler` 同进程（`Scheduler.connector`） | 决定外部 KV 可匹配长度、block 分配后更新状态、生成本步 `KVConnectorMetadata` |
| `WORKER` | 与 GPU Worker 同进程（`get_kv_transfer_group()` 全局单例） | 根据 metadata 加载/保存分页 KV、与每层 attention 同步、回报完成/错误 |

调度器在 `schedule()` 末尾调用 **`build_connector_meta`**，将结果写入 **`SchedulerOutput.kv_connector_metadata`**，经 executor 传到 Worker；Worker 在 forward 前 **`bind_connector_metadata`**。

---

## 2. 子类必须重写的 7 个抽象方法（`@abstractmethod`）

以下 7 个方法在 `KVConnectorBase_V1` 上为抽象方法，**任意具体连接器都必须实现**（简单场景可实现为空函数，但不能不实现）。

### 2.1 Worker 侧（4 个）

#### `start_load_kv(self, forward_context, **kwargs) -> None`

- **何时调用**：已进入 `set_forward_context` 之后、主模型 **`_model_forward` 之前**（`KVConnectorModelRunnerMixin._get_kv_connector_output` 的 `__enter__` 中）。
- **做什么**：根据已绑定的 metadata，将外部 KV 写入 vLLM **分页 KV 缓冲**；可异步发起传输以与计算重叠。
- **注意**：若按层异步加载，需与 **`wait_for_layer_load`** 一致（例如每层 Event）。

#### `wait_for_layer_load(self, layer_name: str) -> None`

- **何时调用**：在 **`maybe_transfer_kv_layer`** 装饰的 attention 路径上，**该层 `impl.forward` 之前**（`model_executor/layers/attention/kv_transfer_utils.py`）。
- **做什么**：阻塞直到 **该层** 分页 KV 对本步读取安全。若 `start_load_kv` 已同步完成全部加载，可为空操作。

#### `save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None`

- **何时调用**：同上装饰器，**该层 attention forward 之后**。
- **做什么**：将该层分页 KV `kv_layer` 写回 connector 侧（可异步）；`attn_metadata` 用于定位本步有效区域。

#### `wait_for_save(self) -> None`

- **何时调用**：本步退出 `_get_kv_connector_output` 的 `finally`（默认 **`wait_for_save=True` 且未 `defer_finalize`**）；若启用 speculative 且推迟 finalize，则在 **`finalize_kv_connector()`** 中补调。
- **做什么**：等待 **`save_kv_layer`** 发起的所有异步保存完成，避免下一步覆盖 GPU 分页 KV。

---

### 2.2 调度器侧（3 个）

#### `get_num_new_matched_tokens(self, request, num_computed_tokens) -> tuple[int | None, bool]`

- **何时调用**：仅在 **`Scheduler.schedule()`** 中，且 **`request.num_computed_tokens == 0`** 的分支（新请求从 waiting 被考虑调度时）。
- **返回**：`(额外可匹配的外部 token 数 | None, load_kv_async)`。  
  - 第一个为 **`None`**：本步不调度该请求。  
  - 第一个为 **`0`**：第二个必须为 **`False`**。  
- **`load_kv_async=True`**：请求可进入 **`WAITING_FOR_REMOTE_KVS`**，本步可能不算子，仅占位 block。

#### `update_state_after_alloc(self, request, blocks, num_external_tokens)`

- **何时调用**：**`allocate_slots` 成功之后**，且调度器上 `connector` 非空。
- **做什么**：根据 **`KVCacheBlocks`**（按 kv_cache_group 的块列表）与 **`num_external_tokens`** 更新连接器内部状态，供 **`build_connector_meta`** 使用。

#### `build_connector_meta(self, scheduler_output) -> KVConnectorMetadata`

- **何时调用**：每步 **`SchedulerOutput` 即将返回前**。
- **做什么**：将本步调度信息编码为子类自定义的 **`KVConnectorMetadata`**；**不得修改** `scheduler_output` 的字段；实现中可重置连接器步内临时状态（见基类注释）。

---

## 3. 非抽象但常用的钩子（对照读代码）

基类中带默认空实现、生产连接器常覆盖的包括：

- `register_kv_caches` / `register_cross_layers_kv_cache`：Worker 注册 KV 显存。
- `handle_preemptions`：`execute_model` 中、大 `with` 之前。
- `get_finished`、`request_finished` / `request_finished_all_groups`（HMA）、`update_connector_output`：异步传输与 block 延迟释放。
- `get_block_ids_with_load_errors`：外部块加载失败。
- `build_connector_worker_meta`：Worker 回传，调度器在 `update_connector_output` 中合并。

---

## 4. 推理中接口调用流程（主路径）

以下以 **`GPUModelRunner` + `KVConnectorModelRunnerMixin`** 为准。未配置 KV transfer 时 **`has_kv_transfer_group()`** 为假，Worker 侧 connector 上下文为 `nullcontext`，下列 Worker 调用不发生。

### 4.1 初始化（一次）

1. 调度器：`KVConnectorFactory.create_connector(..., SCHEDULER)`。  
2. Worker：`ensure_kv_transfer_initialized` → `create_connector(..., WORKER)`。  
3. Worker：`initialize_kv_cache_tensors` 后 `register_kv_caches` 或 `register_cross_layers_kv_cache`，以及可选 `set_host_xfer_buffer_ops`。

### 4.2 每个 `EngineCore.step`

**（1）`scheduler.schedule()`（调度器）**

1. 条件满足时：`get_num_new_matched_tokens`。  
2. `allocate_slots`（不直接调用上述 7 个抽象方法）。  
3. `update_state_after_alloc`。  
4. `build_connector_meta` → 写入 `scheduler_output.kv_connector_metadata`。

**（2）`execute_model`（Worker）**

1. `handle_preemptions(metadata)`（在 `set_forward_context` 的大 `with` 之外）。  
2. `with set_forward_context(...), maybe_get_kv_connector_output(...):`  
   - `bind_connector_metadata`  
   - `start_load_kv`  
   - `_model_forward`：对走 `maybe_transfer_kv_layer` 的每层  
     - `wait_for_layer_load` → attention → `save_kv_layer`  
3. 退出上述 context 的 `finally`：  
   - 若未推迟：`wait_for_save`  
   - `get_finished`、`get_block_ids_with_load_errors`、stats、events、`build_connector_worker_meta`  
   - 若未推迟：`clear_connector_metadata`  

**Speculative decoding**：`defer_finalize=True` 时 **`finally` 中跳过 `wait_for_save` 与 `clear_connector_metadata`**，改在 **`sample_tokens` 末尾 `finalize_kv_connector()`** 中执行。

**本步 `total_num_scheduled_tokens == 0`**：可走 `kv_connector_no_forward`，仍执行 bind + start_load + `finally` 收集，但 **`wait_for_save=False`**。

**（3）`scheduler.update_from_output()`（调度器）**

1. 处理 `invalid_block_ids`。  
2. `_update_from_kv_xfer_finished`：`update_connector_output`；按 `finished_recving` / `finished_sending` 更新状态并可能 `_free_blocks`。  
3. `connector.take_events()`；与 worker stats 做 aggregate（若存在）。

**（4）请求结束（调度器）**

- `_connector_finished` → `request_finished` 或 `request_finished_all_groups`；若异步未完成则延迟释放 block，直至 `get_finished` 路径释放。

---

## 5. 七抽象方法在流程中的位置（速查表）

| 方法 | 进程 | 调用阶段 |
|------|------|----------|
| `get_num_new_matched_tokens` | 调度器 | `schedule()`，新请求且 `num_computed_tokens==0` |
| `update_state_after_alloc` | 调度器 | `allocate_slots` 成功后 |
| `build_connector_meta` | 调度器 | `schedule()` 返回前 |
| `start_load_kv` | Worker | bind 之后、`_model_forward` 之前 |
| `wait_for_layer_load` | Worker | 每层 attention 前（装饰器路径） |
| `save_kv_layer` | Worker | 每层 attention 后（装饰器路径） |
| `wait_for_save` | Worker | 每步 `finally` 或 `finalize_kv_connector` |

---

## 6. 与其它文件的对应关系（便于跳转）

| 主题 | 文件 |
|------|------|
| 基类定义 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` |
| Worker 生命周期封装 | `vllm/v1/worker/kv_connector_model_runner_mixin.py` |
| 每层 wait/save 装饰器 | `vllm/model_executor/layers/attention/kv_transfer_utils.py` |
| 调度器侧调用 | `vllm/v1/core/sched/scheduler.py` |
| 主模型 execute | `vllm/v1/worker/gpu_model_runner.py` |
| 引擎 step | `vllm/v1/engine/core.py` |

---

*文档描述与具体行号可能随上游 vLLM 版本变化，以当前仓库源码为准。*
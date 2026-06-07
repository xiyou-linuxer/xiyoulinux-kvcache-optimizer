# MiniFlexKV 项目结构

MiniFlexKV 是一个从 0 编写的**单机单卡** KV Cache 插件，作为 vLLM **V1** KV connector
使用，外部缓存只含 **CPU + SSD** 两层。本文是按当前代码整理的架构说明（as-built）。
安装与使用见 [usage.md](usage.md)，验证结论见 [validation.md](validation.md)。

## 目标范围

- 作为 vLLM V1 KV connector 插件使用。
- 只支持单机单卡；外部 KV Cache 只含 CPU 和 SSD 两层。
- 不涉及分布式元数据、Redis、P2P、Remote Cache、GDS、Mooncake、TP、DP、多实例 server-client。
- `tp_size` / `dp_size` 仅作配置占位，实现假设始终只有一个 GPU。
- 核心设计：**逻辑缓存管理**与**物理数据搬运**分离。

## 总体分层

```text
vLLM
└─ MiniFlexConnectorV1                          integration/vllm/connector.py
   （继承 KVConnectorBase_V1，唯一入口，纯转发）
   └─ MiniFlexConnectorV1Impl                   integration/vllm/vllm_v1_adapter.py
      （按 role 分发，不含业务逻辑）
      ├─[SCHEDULER] MiniFlexSchedulerConnector
      │   └─ KVTaskEngine（内含 KVTaskManager）  kvtask.py
      │      ├─ GlobalCacheEngine                cache/global_cache_engine.py
      │      │    ├─ CacheEngine(CPU)            cache/cache_engine.py
      │      │    └─ CacheEngine(SSD)
      │      └─ TransferManagerHandle            transfer_manager.py
      │         └─ TransferManager（独立进程）
      │            ├─ StorageEngine              storage/storage_engine.py
      │            └─ TransferEngine             transfer/transfer_engine.py
      │               └─ TransferWorker(s)       transfer/worker.py
      └─[WORKER] MiniFlexWorkerConnector
          └─ MiniFlexGPURegisterClient          server/client.py
             （ZMQ 把 GPU KV cache 注册到 TransferManager）
```

## 核心边界

- `CacheEngine`：只管逻辑缓存状态——前缀树、mempool、block 分配、淘汰、ready/pending。
- `GlobalCacheEngine`：统一调度 CPU / SSD 两层 `CacheEngine`。
- `StorageEngine`：物理存储句柄——GPU KV cache handle、CPU buffer、SSD file。
- `TransferEngine`：按 `TransferOpGraph` 执行实际搬运，底层是若干 `TransferWorker`。
- `KVTaskEngine` / `KVTaskManager`：连接"逻辑缓存规划"和"传输执行"。
- `MiniFlexSchedulerConnector` / `MiniFlexWorkerConnector`：只适配 vLLM 生命周期接口。

## 模块清单

| 模块 | 主要类 | 职责 |
|---|---|---|
| `integration/vllm/connector.py` | `MiniFlexConnectorV1` | 继承 `KVConnectorBase_V1` 的入口薄壳，转发给 Impl |
| `integration/vllm/vllm_v1_adapter.py` | `MiniFlexConnectorV1Impl`、`MiniFlexSchedulerConnector`、`MiniFlexWorkerConnector`、`MiniFlex{Get,Put}Task`、`MiniFlexResponse` | vLLM scheduler/worker 适配与任务编排 |
| `integration/config.py` | `MiniFlexConfig` | 集成层配置，`from_env` + `post_init_from_vllm_config` |
| `kvtask.py` | `KVTaskEngine`、`KVTaskManager`、`KVTask` | 任务编排：match / launch / 完成查询 |
| `cache/` | `GlobalCacheEngine`、`CacheEngine`、`RadixTree`、`Mempool` | 逻辑缓存：前缀匹配、block 分配、淘汰 |
| `storage/` | `StorageEngine`、`{GPU,CPU,SSD}StorageAllocator` | 物理存储句柄与分配 |
| `transfer/` | `TransferEngine`、`TransferScheduler`、`GPUCPUTransferWorker`、`SSDCPUTransferWorker` | 数据搬运执行 |
| `transfer_manager.py` | `TransferManager`、`TransferManagerHandle` | 传输管理（支持独立进程模式），worker 经 ZMQ 注册 GPU |
| `common/` | `ModelConfig`、`CacheConfig`、`KVCacheLayout`、`StorageHandle`、`TransferOpGraph`、`SequenceMeta`、`Hasher` 等 | 跨层的配置、布局、传输图、hash 等基础类型 |
| `server/` | `MiniFlexGPURegisterClient`、`RegisterGPUBlocksRequest` | worker 向 TransferManager 注册 GPU KV 的 ZMQ 客户端/协议 |

## vLLM 集成与数据流

connector 分 SCHEDULER / WORKER 两个 role，由 `MiniFlexConnectorV1Impl` 按 role 创建对应实现。

### Scheduler 侧
负责规划与编排，关键钩子：

- `get_num_new_matched_tokens`：对 prompt 做前缀匹配（`_get_match`），命中则返回
  `(matched_tokens, True)`，请求进入 vLLM 的"等待远程 KV"状态。
- `update_state_after_alloc`：拿到 vLLM 分配的 block，计算 slot_mapping，任务入待 launch 队列。
- `build_connector_meta`：处理 preemption → 取消任务 → `launch_tasks` 提交 GET/PUT。
- `update_connector_output`：`query_finished_tasks` 轮询完成，上报 `finished_recving` / `finished_sending`。
- `request_finished`：请求正常结束（STOP / LENGTH）时做 `_put_match`，把新增 KV 存入缓存。
- `get_block_ids_with_load_errors`：回收加载失败的 block。

### Worker 侧
只做一件实事：`register_kv_caches` 通过 `MiniFlexGPURegisterClient` 把本地 GPU KV cache
注册到 scheduler 侧的 `TransferManager`。`start_load_kv` / `save_kv_layer` / `wait_*` 均为
no-op——实际搬运由 scheduler 侧的 `TransferManager` 进程**直接读写已注册的 GPU 显存**完成，
不经过 worker 的逐层钩子。

### 两条主链路
- **GET（prefill 命中）**：match 命中 → 分配 block → launch GET 任务（CPU→GPU）→ 轮询完成
  → 上报 `finished_recving` → vLLM 释放请求继续前向。
- **PUT（请求结束保存）**：`request_finished` → `_put_match` 找出未命中部分 → launch PUT 任务
  （GPU→CPU）→ 上报 `finished_sending`。

### 两个行为开关
- `MINIFLEX_ENABLE_BATCH`：把多个任务合并成一个 batch task 提交。
- `MINIFLEX_SYNC_GET`：在 `build_connector_meta` 内同步阻塞等待 GET 加载完成（默认关闭，
  仅调试/特殊低频场景用，详见 usage.md）。

## 测试

`test/` 下每个核心模块都有对应单测，覆盖 cache（radix tree / mempool / cache engine）、
storage、transfer（engine / worker / manager）、kvtask，以及 vLLM 适配
（`test/vllm_v1_adapter_test.py`，含 fake 引擎的生命周期用例和真实 GPU↔CPU 传输的逐字节校验）。

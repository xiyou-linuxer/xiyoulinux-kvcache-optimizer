# MiniFlexKV 项目结构

MiniFlexKV 是一个从 0 编写的**单机单卡** KV Cache 插件，作为 vLLM **V1** KV connector
使用，外部缓存只含 **CPU + SSD** 两层。本文按当前代码整理（as-built），
安装与使用见 [usage.md](usage.md)，实测结论见 [validation.md](validation.md)。

## 目标范围

- 作为 vLLM V1 KV connector 插件使用。
- 只支持单机单卡；外部 KV Cache 只含 CPU 和 SSD 两层。
- 不涉及分布式元数据、Redis、P2P、Remote Cache、GDS、Mooncake、TensorRT-LLM、TP、DP、
  多实例 server-client。
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
      │         └─ TransferManager（可独立进程）
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
- `KVTaskEngine` / `KVTaskManager`：连接“逻辑缓存规划”和“传输执行”。
- `MiniFlexSchedulerConnector` / `MiniFlexWorkerConnector`：只适配 vLLM 生命周期接口。

## 目录结构

```text
pysrc/miniflex/
  integration/
    config.py            MiniFlexConfig（集成层配置）
    vllm/
      connector.py       MiniFlexConnectorV1（入口薄壳）
      vllm_v1_adapter.py scheduler/worker connector + 任务/响应对象
  kvtask.py              KVTaskEngine / KVTaskManager / KVTask
  cache/
    global_cache_engine.py  GlobalCacheEngine（统一 CPU/SSD）
    cache_engine.py         CacheEngine（单层逻辑缓存）
    radix_tree.py           RadixTree / RadixTreeNode / MatchResult
    mempool.py              Mempool（物理 block-id 池）
  storage/
    storage_engine.py    StorageEngine
    allocator.py         Base/GPU/CPU/SSD StorageAllocator
  transfer/
    transfer_engine.py   TransferEngine
    scheduler.py         TransferScheduler
    worker.py            TransferWorkerBase / GPUCPU / SSDCPU worker
  transfer_manager.py    TransferManager + 三种 Handle
  server/
    client.py            MiniFlexGPURegisterClient（ZMQ 客户端）
    request.py           RegisterGPUBlocksRequest（注册协议）
    utils.py             ZMQ 端点/参数工具
  common/
    config.py            ModelConfig / CacheConfig
    storage.py           KVCacheLayout(Type) / StorageHandle / StorageHandlerType
    transfer.py          TransferOp(Graph) / TransferType / DeviceType
    block.py             SequenceMeta（block 切分 + namespace hash）
    hash.py              Hasher（blake2b 64-bit）
    request.py           KVRequest / KVResponse（任务请求/响应协议）
    memory_handle.py     TensorSharedHandle（跨进程 CUDA tensor 句柄）
    ring_buffer.py       SharedOpPool（共享 block-id buffer）
    metrics.py           轻量线程安全指标
csrc/
  transfer.cu / .cuh     GPUCPUTransferCTX（GPU<->CPU stride 化拷贝）
  ssd_io_uring.cpp / .h  SSDIOCTX（SSD io_uring backend）
  bindings.cpp           pybind11 -> _C
```

## 模块清单

| 模块 | 主要类 | 职责 |
|---|---|---|
| `integration/vllm/connector.py` | `MiniFlexConnectorV1` | 继承 `KVConnectorBase_V1` 的入口薄壳，转发给 Impl |
| `integration/vllm/vllm_v1_adapter.py` | `MiniFlexConnectorV1Impl`、`MiniFlexSchedulerConnector`、`MiniFlexWorkerConnector`、`MiniFlexKVTask`/`MiniFlexGetTask`/`MiniFlexPutTask`、`MiniFlexResponse` | vLLM scheduler/worker 适配与任务编排 |
| `integration/config.py` | `MiniFlexConfig` | 集成层配置，`from_env` + `post_init_from_vllm_config` |
| `kvtask.py` | `KVTaskEngine`、`KVTaskManager`、`KVTask`、`TaskType`、`TaskStatus` | 任务编排：match / launch / 完成查询 / cancel |
| `cache/` | `GlobalCacheEngine`、`CacheEngine`、`RadixTree`、`RadixTreeNode`、`MatchResult`、`Mempool` | 逻辑缓存：前缀匹配、block 分配、淘汰 |
| `storage/` | `StorageEngine`、`BaseStorageAllocator`、`{GPU,CPU,SSD}StorageAllocator` | 物理存储句柄与分配 |
| `transfer/` | `TransferEngine`、`TransferScheduler`、`TransferWorkerBase`、`GPUCPUTransferWorker`、`SSDCPUTransferWorker`、`WorkerTransferOp`、`WorkerHandle` | 数据搬运执行 |
| `transfer_manager.py` | `TransferManager`、`TransferManagerHandle`（+ Intra/Inter 进程 handle） | 传输管理（支持独立进程模式），worker 经 ZMQ 注册 GPU |
| `server/` | `MiniFlexGPURegisterClient`、`RegisterGPUBlocksRequest`、ZMQ 工具 | worker 向 TransferManager 注册 GPU KV 的 ZMQ 客户端/协议 |
| `common/config.py` | `ModelConfig`、`CacheConfig` | 模型结构与缓存调优配置 |
| `common/storage.py` | `KVCacheLayout`、`KVCacheLayoutType`、`StorageHandle`、`StorageHandlerType` | KV 物理布局元数据与存储句柄 |
| `common/transfer.py` | `TransferOpGraph`、`TransferOp`、`TransferType`、`DeviceType`、`CompletedOp` | 搬运 DAG 与算子类型 |
| `common/block.py` | `SequenceMeta` | 按 `tokens_per_block` 切块、namespace/salt 隔离 |
| `common/hash.py` | `Hasher` | 增量 64-bit blake2b，用作 block hash |
| `common/request.py` | `KVRequest`、`KVResponse`、`KVRequestType`、`KVResponseStatus` | 任务请求/响应协议（GET/PUT） |
| `common/memory_handle.py` | `TensorSharedHandle` | PyTorch reducer/IPC 的跨进程 CUDA tensor 访问句柄 |
| `common/ring_buffer.py` | `SharedOpPool` | 共享内存 block-id buffer（进程间传 op 的 block 列表） |
| `common/metrics.py` | `Counter`、`Histogram`、`MetricsRegistry` | 轻量线程安全指标（命中率、block 数等） |

## vLLM 集成与数据流

connector 分 SCHEDULER / WORKER 两个 role，由 `MiniFlexConnectorV1Impl` 按 role 创建对应实现。

### Scheduler 侧
负责规划与编排，关键钩子：

- `get_num_new_matched_tokens`：对 prompt 做前缀匹配（`_get_match`），命中则返回
  `(matched_tokens, True)`，请求进入 vLLM 的“等待远程 KV”状态。
- `update_state_after_alloc`：拿到 vLLM 分配的 block，计算 slot_mapping，任务入待 launch 队列。
- `build_connector_meta`：处理 preemption → 取消任务 → `launch_tasks` 提交 GET/PUT。
- `update_connector_output`：`query_finished_tasks` 轮询完成，上报 `finished_recving` / `finished_sending`。
- `request_finished`：请求正常结束（STOP / LENGTH）时做 `_put_match`，把新增 KV 存入缓存。
- `get_block_ids_with_load_errors`：回收加载失败的 block。

### Worker 侧
只做一件实事：`register_kv_caches` 通过 `MiniFlexGPURegisterClient` 把本地 GPU KV cache
注册到 scheduler 侧的 `TransferManager`。注册前会**按张量形状自动探测 GPU 布局**
（非 MLA 5D：`shape[1]==2` → vLLM 0.23 的 `LAYERBLOCK`，`shape[0]==2` → 旧版 `LAYERFIRST`；
MLA 3D → `LAYERFIRST`），见下文「KV 布局自适配」。`start_load_kv` / `save_kv_layer` / `wait_*`
均为 no-op——实际搬运由 scheduler 侧的 `TransferManager` 进程**直接读写已注册的 GPU 显存**完成，
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

## 各层内部细节

### 逻辑缓存层（`cache/`）
逻辑层只跟踪“哪些 block 在缓存里、能不能复用、该淘汰谁”，从不碰真实显存/内存数据。

- **`RadixTree`**：前缀树，key 是 **block hash**（`SequenceMeta` 按 `tokens_per_block` 切块后
  用 `Hasher` 逐块求 64-bit hash）。`match_prefix(sequence)` 走最长公共前缀，返回 `MatchResult`
  （命中节点链 + 物理 block id + 命中长度）；`insert(...)` 把新 block 挂上去，带 `is_ready`
  区分“已就绪 / 传输中”，`set_ready` 在传输完成后翻转。
- **淘汰**：`evict(n)` 用最小堆按优先级挑可淘汰叶子（`is_evictable()` 要求 ready 且无锁），
  默认 `lru`，叠加 `hit_add_counts`（命中加权）和 `protected_threshold`（命中超阈值受保护）。
- **`Mempool`**：纯物理 block-id 池（`allocate` / `free` / `num_free_blocks`），不认识前缀语义。
- **`CacheEngine`** = 一棵 `RadixTree` + 一个 `Mempool`，代表**单层**（CPU 或 SSD）。
  **`GlobalCacheEngine`** 统一持有两层：`match_all(sequence)` 同时匹配并返回 `(cpu_match, ssd_match)`；
  `get` / `put`（`get_impl` / `put_impl`）据此规划层间搬运（如 SSD 命中先回 CPU 再上 GPU）。

### 物理存储层（`storage/` + `common/storage.py`）
- **`KVCacheLayout`** 是布局元数据：给定 `layout_type` 推出 `kv_shape`，并派生
  `get_block_stride()` / `get_kv_stride()` / `get_chunk_size()` 等跨距（详见布局表）。
  它**只描述布局**，不分配 tensor、不建文件、不搬数据。
- **`StorageEngine`** 持有三类物理句柄：注册进来的 **GPU** KV handle、自己分配的 **CPU** pinned
  buffer、**SSD** file handle，分别由 `GPUStorageAllocator` / `CPUStorageAllocator` /
  `SSDStorageAllocator` 创建；`register_gpu_blocks()` 做薄校验（layout/dtype/单 GPU `device_id=0`/
  禁止重复注册）。
- 约定：**CPU 固定 `LAYERFIRST`**、**SSD 固定 `BLOCKFIRST`**、**GPU 布局由注册时按形状探测**
  （`LAYERFIRST` 或 `LAYERBLOCK`）；三者不要求一致——差异全部由各自 layout 的 stride 吸收。

### 传输层（`transfer/` + `csrc/`）
- **`TransferOpGraph`**（`common/transfer.py`）：把一次搬运表达成 `TransferOp` 的 DAG，
  `add_dependency(op, dep)` 描述先后约束（如“先 DISK2H 再 H2D”）。`TransferType` 给方向
  （`D2H` / `H2D` / `H2DISK` / `DISK2H`），`DeviceType` 给设备。
- **`TransferEngine`** 接收 graph（`submit_transfer_graph`），内部 **`TransferScheduler`**
  的 `schedule(finished_ops)` 按依赖解出可执行 op；借 `SharedOpPool` 把 src/dst block id 写入
  共享 buffer 后交给 worker，worker 只回报 `op_id`。
- **`GPUCPUTransferWorker`**：负责 `D2H` / `H2D`，CE 路径把逐块 `cudaMemcpy2DAsync` 下放到
  C++ `_C.GPUCPUTransferCTX`（见下文布局自适配）。
- **`SSDCPUTransferWorker`**：负责 `H2DISK` / `DISK2H`，布局组合 **CPU `LAYERFIRST` +
  SSD `BLOCKFIRST`**，初始化时把 CPU tensor、shape、SSD file list、`num_blocks_per_file`、
  `use_direct_io` 传给 C++ `_C.SSDIOCTX`（io_uring 后端，可选 `O_DIRECT`，按
  `num_blocks_per_file` 把 block 映射到多文件）。
- **`csrc/`**：`transfer.cu/.cuh`、`ssd_io_uring.cpp/.h`、`bindings.cpp`（pybind11 暴露
  `GPUCPUTransferCTX` / `SSDIOCTX`），产物 `_C`，链接 `-luring`。

### 任务编排（`kvtask.py`）
连接“逻辑缓存规划”和“传输执行”的中间层。

- **`KVTaskEngine`**：`match` 调 `GlobalCacheEngine` 得到命中规划，生成 `KVTask`
  （`TaskType` = `GET` / `PUT` / `BATCH_GET` / `BATCH_PUT`）；`launch_tasks` 把任务编译成
  `TransferOpGraph` 提交 `TransferEngine`；上层轮询完成；`cancel_tasks` 在 preemption
  或 match/put 条件不满足时**立刻取消并回收 block**（防泄漏）。
- **`KVTaskManager`**：维护任务状态机（`TaskStatus` ↔ `KVResponseStatus` 映射）。

### 跨进程：TransferManager 与 ZMQ 注册
- **`TransferManager`** 可跑在**独立进程**（`TransferManagerHandleInterProcessHandle`）或
  同进程（`...IntraProcessHandle`），由 `TransferManagerHandle` 统一对外。
- worker 侧 `register_kv_caches` → `MiniFlexGPURegisterClient` 经 **ZMQ** 发
  `RegisterGPUBlocksRequest`（含 `TensorSharedHandle` + `gpu_layout`）给 `TransferManager`；
  此后 `TransferManager` **直接读写已注册的 GPU 显存**，不再经 worker 逐层钩子。
- 进程间用 `SharedOpPool` 的共享内存 tensor 传 block-id 列表，避免大数组 pickle 开销。

## KV 布局自适配（LAYERBLOCK，兼容 vLLM ≤0.21 与 0.23+）

vLLM 0.23 改了**非 MLA** GPU KV cache 的物理布局（权威来源
`FlashAttentionBackend.get_kv_cache_shape()`）：

| 版本 | 单层 GPU KV 形状 | MiniFlex 布局名 |
|---|---|---|
| ≤0.21 | `(kv=2, num_blocks, block, heads, head_dim)` | `LAYERFIRST` |
| 0.23+ | `(num_blocks, kv=2, block, heads, head_dim)` | `LAYERBLOCK` |

差别只在 `kv` 维和 `num_blocks` 维谁在前。MiniFlex 用三步吸收，**传输 C++ 层不需要知道布局**：

1. **布局枚举**：`common/storage.py` 的 `KVCacheLayoutType` 含 `LAYERBLOCK`，
   `KVCacheLayout` 为它给出对应的 `kv_shape` / `get_block_stride()` / `get_kv_stride()`。
2. **注册时探测**：worker 注册 GPU KV 时按 `first_block` 形状判定布局
   （非 MLA 5D 且 `shape[1]==2` → `LAYERBLOCK`，`shape[0]==2` → `LAYERFIRST`；MLA 3D → `LAYERFIRST`），
   不依赖 vLLM 版本号。
3. **stride 驱动搬运**：`GPUCPUTransferWorker` 的 CE 路径把逐块 `cudaMemcpy2DAsync` 下放到
   C++ `_C.GPUCPUTransferCTX`，CPU/GPU 均以 per-layer 指针 + 字节跨距传入
   （`slice_bytes`、`*_block_step`、`*_kv_pitch`）；布局差异完全由 Python 侧
   `KVCacheLayout.get_block_stride()/get_kv_stride()` 表达，C++ 只按 stride 拷贝。
   CPU 侧固定 `LAYERFIRST`，GPU 侧接受 `LAYERFIRST` 或 `LAYERBLOCK`。

各布局派生跨距（以元素计，`T`=tokens_per_block、`H`=num_heads、`D`=head_size、`B`=num_blocks、`kv`=2）：

| 布局 | `kv_shape`（单层） | `block_step` | `kv_pitch` |
|---|---|---|---|
| `LAYERFIRST` | `(L, kv, B, T, H, D)` | `T·H·D` | `B·T·H·D` |
| `LAYERBLOCK` | `(L, B, kv, T, H, D)` | `kv·T·H·D` | `T·H·D` |

C++ 每块发一次 `cudaMemcpy2DAsync`：`width=slice_bytes`（=`T·H·D·itemsize`）、`height=kv_dim`、
行间距=`kv_pitch·itemsize`、块间步进=`block_step·itemsize`，只靠不同 pitch 即同时支持两布局。
（原 `index_select -> index_copy_` 的 gather 实现已在 `worker.py` 注释保留备查，不再使用。）

## 测试

`test/` 下每个核心模块都有对应单测，可直接 `python test/<file>.py` 运行：

- 逻辑缓存：`mempool_test`、`radix_tree_test`、`cache_engine_test`、`global_cache_engine_test`。
- 基础类型：`hash_test`、`block_test`、`storage_test`、`transfer_test`。
- 存储 / 传输：`storage_engine_test`、`scheduler_test`、`ring_buffer_test`、
  `GPUCPUTransferWorker_test`（含 `LAYERBLOCK` 往返逐字节校验）、`worker_test`、
  `ssd_cpu_worker_test`、`transfer_engine_test`、`transfer_manager_test`。
- C++：`ssd_io_uring_test.cpp`（真实临时文件覆盖 file IO / O_DIRECT / io_uring chunk）。
- 任务 / 集成：`kvtask_test`、`vllm_v1_adapter_test`（fake 引擎生命周期 + 真实 GPU↔CPU 逐字节校验）。

## 当前规模

截至当前状态（vLLM 0.21 与 0.23 均已在真机跑通，见 [validation.md](validation.md)）：

- Python 源码约 6100 行、Python 测试约 6200 行、C++（`csrc/`）约 700 行，合计约 1.3 万行。

目标范围（vLLM V1 connector、单机单卡、CPU + SSD 双介质）已全部落地并有单测覆盖：
逻辑缓存层、存储层、传输层（含 C++ stride 化 GPU↔CPU 拷贝与 io_uring SSD 后端）、任务层、
vLLM 适配（含 vLLM 0.23 非 MLA 的 `LAYERBLOCK` 布局自适配）。

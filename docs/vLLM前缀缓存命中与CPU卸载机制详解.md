# vLLM 前缀缓存与 CPU Offload 机制

> 基于 vLLM v1 架构，介绍前缀缓存的命中流程、淘汰策略、KV 数据存入 CPU 内存的完整链路，以及不同插件的对接方式。

---

## 目录

- [vLLM 前缀缓存与 CPU Offload 机制](#vllm-前缀缓存与-cpu-offload-机制)
  - [目录](#目录)
  - [1. 核心概念](#1-核心概念)
  - [2. 前缀缓存命中流程](#2-前缀缓存命中流程)
    - [2.1 GPU 顺序查找](#21-gpu-顺序查找)
    - [2.2 CPU 二级查找（启用 Offload 时）](#22-cpu-二级查找启用-offload-时)
    - [2.3 为什么必须顺序查找](#23-为什么必须顺序查找)
  - [3. GPU 侧淘汰策略：隐式 LRU](#3-gpu-侧淘汰策略隐式-lru)
  - [4. CPU 侧淘汰策略：可插拔 CachePolicy](#4-cpu-侧淘汰策略可插拔-cachepolicy)
    - [4.1 策略抽象层](#41-策略抽象层)
    - [4.2 LRU 策略](#42-lru-策略)
    - [4.3 ARC 策略](#43-arc-策略)
    - [4.4 策略选择与注册](#44-策略选择与注册)
  - [5. 缓存释放 vs 缓存淘汰](#5-缓存释放-vs-缓存淘汰)
  - [6. KV 数据存入 CPU 的完整链路](#6-kv-数据存入-cpu-的完整链路)
    - [6.1 总览](#61-总览)
    - [6.2 调度器侧：决定存哪些 block](#62-调度器侧决定存哪些-block)
    - [6.3 Worker 侧：物理数据传输](#63-worker-侧物理数据传输)
    - [6.4 传输完成确认](#64-传输完成确认)
  - [7. KV Connector 插件体系](#7-kv-connector-插件体系)
    - [7.1 两层插件工厂](#71-两层插件工厂)
    - [7.2 OffloadingConnector 的内部分工](#72-offloadingconnector-的内部分工)
    - [7.3 已注册的 Connector 插件](#73-已注册的-connector-插件)
  - [8. 关键代码文件索引](#8-关键代码文件索引)

---

## 1. 核心概念

KV Cache 按固定大小的 **block** 管理（默认 16 token/block）。每个 block 有一个 **BlockHash**（链式哈希，依赖前面所有 block 内容），作为前缀缓存的键。全局维护一张哈希表 `{BlockHash → KVCacheBlock}`。

block 写满后注册到哈希表（`cache_full_blocks`），后续请求可通过相同 hash 直接复用。

---

## 2. 前缀缓存命中流程

### 2.1 GPU 顺序查找

**代码**: `single_type_kv_cache_manager.py` → `find_longest_cache_hit()`

调度器对每个新请求，按 block hash 顺序在全局哈希表中查找：

```
request.block_hashes = [h0, h1, h2, h3, h4, h5]

查找过程:
  h0 → 哈希表 → ✓ 命中 block_10
  h1 → 哈希表 → ✓ 命中 block_23
  h2 → 哈希表 → ✓ 命中 block_7
  h3 → 哈希表 → ✗ 未命中 → 停止

结果: GPU 命中 3 block = 48 token
命中的 block 执行 touch: ref_cnt +1，从 free 队列移除（防止被淘汰）
```

### 2.2 CPU 二级查找（启用 Offload 时）

**代码**: `offloading/scheduler.py` → `get_num_new_matched_tokens()`

GPU 未命中的部分继续在 CPU 缓存的 `CachePolicy` 中查找：

```
GPU 查找: h0 ✓  h1 ✓  h2 ✓  h3 ✗ (停止)
CPU 查找 (从 h3 开始):    h3 ✓  h4 ✓  h5 ✗ (停止)

最终: GPU 命中 3 block + CPU 命中 2 block = 80 token 无需重算
      CPU 命中的 2 block 需要异步搬运到 GPU
```

CPU 查找的内部实现（`CPUOffloadingManager.lookup()`）同样是**顺序匹配**，遇到未命中或 `is_ready=False`（正在传输）的 block 就停止。

### 2.3 为什么必须顺序查找

Attention 计算要求所有前缀 token 的 KV 数据**连续存在于 GPU**。如果中间有一块缺失（`[命中][缺失][命中]`），后面的 block 无法使用。所以查找从头开始，一旦 miss 就停止。

---

## 3. GPU 侧淘汰策略：隐式 LRU

**代码**: `kv_cache_utils.py` → `FreeKVCacheBlockQueue`

GPU 使用一个 **LRU 双向链表** 管理空闲 block：

```
HEAD (最久未用) ←→ ... ←→ block_A ←→ block_B ←→ TAIL (最近释放)
  优先被淘汰 ←                                    → 最后被淘汰
```

| 事件 | 操作 |
|------|------|
| 请求结束释放 block | `ref_cnt` -1，归零后插入队列**尾部** |
| 缓存命中复用 block | 从队列中**移除**（`ref_cnt` +1，不在 free 队列中） |
| 需要新 block | 从队列**头部**取出最旧的，如果它有 `block_hash` 则清除映射 |

淘汰是隐式发生的：分配新 block 时，如果取到的是缓存 block（`block_hash != None`），清除其 hash 注册，物理内存在后续 forward 中被覆盖。

---

## 4. CPU 侧淘汰策略：可插拔 CachePolicy

### 4.1 策略抽象层

**代码**: `vllm/v1/kv_offload/cpu/policies/abstract.py`

CPU 侧的淘汰策略通过 `CachePolicy` 抽象基类定义，所有策略实现以下接口：

```python
class CachePolicy(ABC):
    def get(key) -> BlockStatus | None     # 查找 block
    def insert(key, block)                  # 插入新 block
    def remove(key)                         # 移除 block
    def touch(keys)                         # 标记最近使用
    def evict(n, protected) -> list | None  # 淘汰 n 个 block
```

其中 `BlockStatus` 包含两个字段：
- `block_id`：CPU 物理缓存槽位编号
- `ref_cnt`：引用计数，`-1` 表示数据还没写完（`is_ready=False`），`0` 表示就绪

`evict()` 方法接受一个 `protected` 集合，正在被使用（加载/存储中）的 block 不会被淘汰。

### 4.2 LRU 策略

**代码**: `vllm/v1/kv_offload/cpu/policies/lru.py`

内部只有一个 `OrderedDict`，`touch()` 将 block 移到末尾，`evict()` 从头部取 `ref_cnt==0` 且不在 `protected` 中的 block：

```python
class LRUCachePolicy(CachePolicy):
    blocks: OrderedDict[OffloadKey, BlockStatus]

    def touch(self, keys):
        for key in reversed(list(keys)):
            self.blocks.move_to_end(key)     # 移到末尾 = 最近使用

    def evict(self, n, protected):
        # 从头部（最旧）开始找可淘汰的 block
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((key, block))
```

### 4.3 ARC 策略

**代码**: `vllm/v1/kv_offload/cpu/policies/arc.py`

ARC（Adaptive Replacement Cache）维护四个列表，自适应平衡"最近使用"和"最常使用"：

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│   B1    │    │   T1    │    │   T2    │    │   B2    │
│ 幽灵列表 │    │ 访问1次  │    │ 访问≥2次 │    │ 幽灵列表 │
│(只记hash)│    │(有数据)  │    │(有数据)  │    │(只记hash)│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
 T1淘汰记录      recency        frequency      T2淘汰记录
```

**`touch()` 的自适应学习逻辑**：

```python
def touch(self, keys):
    for key in reversed(list(keys)):
        if key in self.t1:
            # 第二次被访问 → 从 T1 晋升到 T2（从"最近"变为"频繁"）
            block = self.t1.pop(key)
            self.t2[key] = block

        elif key in self.t2:
            # 已经在 T2 → 移到末尾（刷新位置）
            self.t2.move_to_end(key)

        elif key in self.b1:
            # 在 B1 幽灵列表中被命中
            # → 说明 T1（recency）空间不够，被过早淘汰了
            # → 增大 target_t1_size，给 T1 更多空间
            delta = max(1, len(self.b2) / len(self.b1))
            self.target_t1_size = min(self.target_t1_size + delta, capacity)

        elif key in self.b2:
            # 在 B2 幽灵列表中被命中
            # → 说明 T2（frequency）空间不够
            # → 减小 target_t1_size，给 T2 更多空间
            delta = max(1, len(self.b1) / len(self.b2))
            self.target_t1_size = max(self.target_t1_size - delta, 0)
```

**`evict()` 的淘汰决策**：

```python
def evict(self, n, protected):
    for _ in range(n):
        if len(self.t1) >= target_t1_size:
            # T1 超过目标大小 → 从 T1 淘汰，记入 B1
            victim = T1 中 ref_cnt==0 且不在 protected 中的最旧 block
            del self.t1[victim_key]
            self.b1[victim_key] = None   # 记入幽灵列表
        else:
            # T1 未满 → 从 T2 淘汰，记入 B2
            victim = T2 中 ref_cnt==0 且不在 protected 中的最旧 block
            del self.t2[victim_key]
            self.b2[victim_key] = None   # 记入幽灵列表

    # 修剪幽灵列表，不超过 cache_capacity
    for ghost in (self.b1, self.b2):
        while len(ghost) > cache_capacity:
            ghost.popitem(last=False)
```

**ARC 的优势**：能自动适应混合访问模式。高频重复的系统提示词（frequency）和低频的用户文本（recency）不会互相"污染"缓存。

### 4.4 策略选择与注册

**代码**: `vllm/v1/kv_offload/cpu/manager.py`

`CPUOffloadingManager` 在初始化时通过字符串选择策略：

```python
_CACHE_POLICIES = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}

class CPUOffloadingManager(OffloadingManager):
    def __init__(self, block_size, num_blocks, cache_policy="lru"):
        policy_cls = _CACHE_POLICIES[cache_policy]
        self._policy = policy_cls(cache_capacity=num_blocks)
```

`CPUOffloadingManager` 负责所有共性逻辑（引用计数、block 池管理、事件记录），策略特定的决策（如何组织 block、谁该被淘汰）完全委托给 `CachePolicy` 实现。

---

## 5. 缓存释放 vs 缓存淘汰

| | 释放（free） | 淘汰（evict） |
|---|---|---|
| 触发时机 | 请求结束 / 抢占 | 分配新 block 时空间不够 |
| 操作 | `ref_cnt` -1，block 进入 free 队列 | 从 free 队列取出（GPU）或从 policy 删除（CPU），清除 hash |
| hash 状态 | **保留** hash，仍可被复用 | **删除** hash，缓存失效 |
| 数据状态 | 数据仍在显存/内存中 | 数据被新的 KV 覆盖 |

释放后 block 仍然是有效缓存，新请求可以复用。只有被淘汰时缓存才真正失效。

---

## 6. KV 数据存入 CPU 的完整链路

### 6.1 总览

```
model.forward() 计算出 KV 数据，写入 GPU block
        ↓
schedule() 结束时，build_connector_meta()
        ↓
调度器侧：_get_reqs_to_store() 检查哪些 block 新写满了
        ↓  生成 store 指令 (GPU block ID → CPU block ID)
        ↓
CPUOffloadingManager.prepare_store()
  - 过滤已存储的 block
  - 如果 CPU 空间不够 → CachePolicy.evict() 淘汰旧 block
  - 分配 CPU block 槽位
  - CachePolicy.insert() 注册新 block（此时 is_ready=False）
        ↓  打包成 SchedulerOutput.kv_connector_metadata
        ↓
Worker 收到 metadata
        ↓
Worker 侧：start_kv_transfers() → transfer_async()
  - 在独立 CUDA stream 上执行 ops.swap_blocks_batch()
  - GPU → CPU 异步 DMA（不阻塞推理）
        ↓
Worker 侧：get_finished() 检查 CUDA event 完成
        ↓
调度器侧：complete_store(success=True)
  - block.ref_cnt 设为 0，is_ready=True
  - 此后该 block 可被 lookup() 命中
```

### 6.2 调度器侧：决定存哪些 block

**代码**: `offloading/scheduler.py` → `_get_reqs_to_store()`

```python
def _get_reqs_to_store(self, scheduler_output):
    for req_id, new_block_ids, preempted in yield_req_data(scheduler_output):
        # 计算本步后该请求有多少个完整 block
        new_tokens = scheduler_output.num_scheduled_tokens[req_id]
        total_tokens = req.num_computed_tokens + new_tokens
        num_blocks = total_tokens // block_size

        # 上次已存到第几个 block → 新增了几个
        start = group_state.next_stored_block_idx
        num_new_blocks = num_blocks - start

        if num_new_blocks <= 0:
            continue  # 没有新写满的 block

        # 调用 CPUOffloadingManager.prepare_store()
        # 内部会：过滤已存的 → 淘汰旧的 → 分配CPU槽位 → insert到policy
        store_output = self.manager.prepare_store(new_offload_keys)

        # 更新记录：下次从 num_blocks 开始
        group_state.next_stored_block_idx = num_blocks

        # 构造 GPU→CPU 的传输指令
        src_spec = GPULoadStoreSpec(gpu_block_ids)
        dst_spec = store_output.store_spec  # CPULoadStoreSpec(cpu_block_ids)
        reqs_to_store[req_id] = (src_spec, dst_spec)
```

关键逻辑：
1. 遍历所有被调度的请求
2. 算出本步 forward 后有多少个**完整 block**（`total_tokens // block_size`）
3. 减去上次已存储的数量，得到**新写满的 block 数**
4. 调用 `prepare_store()`，内部触发 CPU 侧的淘汰和空间分配
5. 构造 `(GPU block IDs, CPU block IDs)` 的传输指令对

### 6.3 Worker 侧：物理数据传输

**代码**: `vllm/v1/kv_offload/worker/cpu_gpu.py` → `SingleDirectionOffloadingHandler.transfer_async()`

```python
def transfer_async(self, job_id, transfer_spec):
    src_blocks, dst_blocks = transfer_spec  # GPU IDs, CPU IDs

    # 计算每个 block 的源/目标内存地址
    all_src[i] = gpu_tensor.data_ptr() + src_block_id * block_size_bytes
    all_dst[i] = cpu_tensor.data_ptr() + dst_block_id * block_size_bytes

    # 获取一个独立的 CUDA stream（不阻塞推理的 default stream）
    stream = self._stream_pool.pop() or torch.cuda.Stream()

    # GPU→CPU 时，等待当前 forward 计算完成
    stream.wait_stream(torch.cuda.current_stream())

    # 在独立 stream 上执行批量内存拷贝
    with torch.cuda.stream(stream):
        ops.swap_blocks_batch(batch_src, batch_dst, batch_sizes)
        end_event.record(stream)  # 记录完成事件
```

关键细节：
- 使用**独立 CUDA stream**，不阻塞模型推理
- GPU→CPU 时先 `wait_stream` 等 forward 完成，确保数据有效
- 使用 `ops.swap_blocks_batch` 做批量 DMA 拷贝（一次调用传多个 block）
- 记录 CUDA event，后续通过 `event.query()` 检查是否完成

### 6.4 传输完成确认

Worker 侧 `get_finished()` 轮询 CUDA event：

```python
def get_finished(self):
    while self._transfers and self._transfers[0].end_event.query():
        transfer = self._transfers.popleft()  # 完成了，出队
        # 返回 (job_id, success, transfer_size, transfer_time)
```

调度器侧收到完成通知后调用 `complete_store(success=True)`：

```python
def complete_store(self, keys, success=True):
    if success:
        for key in keys:
            block = self._policy.get(key)
            block.ref_cnt = 0    # 从 -1 → 0，标记为 is_ready=True
    else:
        for key in keys:
            self._policy.remove(key)   # 失败则清理
            self._free_block(block)
```

`ref_cnt` 从 -1 变为 0 后，`is_ready` 变为 `True`，该 block 就能被 `lookup()` 命中了。

---

## 7. KV Connector 插件体系

### 7.1 两层插件工厂

vLLM 有两层插件注册机制：

**第一层：KVConnectorFactory**（`kv_connector/factory.py`）

注册各种 KV Connector（CPU offload、分布式传输等）。Connector 负责**调度器-Worker 之间的桥接**：

```python
KVConnectorFactory.register_connector("OffloadingConnector", ...)
KVConnectorFactory.register_connector("LMCacheConnectorV1", ...)
KVConnectorFactory.register_connector("MooncakeConnector", ...)
# ... 等
```

**第二层：OffloadingSpecFactory**（`kv_offload/factory.py`）

注册 Offloading 的具体实现规格。目前只有 CPU offload：

```python
OffloadingSpecFactory.register_spec("CPUOffloadingSpec", ...)
```

这层的存在是为了将来支持其他 offload 介质（如 SSD）。

### 7.2 OffloadingConnector 的内部分工

`OffloadingConnector` 是内置的 CPU offload 插件。它把工作分为三层：

```
OffloadingConnector (桥梁层)
  ├── 调度器侧:
  │     └── CPUOffloadingManager (管理层)
  │           └── CachePolicy (策略层, LRU/ARC)
  │
  └── Worker 侧:
        └── CpuGpuOffloadingHandlers (传输层)
              ├── store_handler (GPU→CPU)
              └── load_handler  (CPU→GPU)
```

| 层 | 文件 | 职责 |
|----|------|------|
| 桥梁层 | `offloading/scheduler.py` + `offloading/worker.py` | 连接调度器和 Worker，打包/解析 metadata |
| 管理层 | `kv_offload/cpu/manager.py` | block 池管理、引用计数、协调 store/load |
| 策略层 | `kv_offload/cpu/policies/lru.py` / `arc.py` | 纯淘汰决策，可替换 |
| 传输层 | `kv_offload/worker/cpu_gpu.py` | 独立 CUDA stream 上的异步 DMA |

### 7.3 已注册的 Connector 插件

| 插件名 | 用途 |
|--------|------|
| `OffloadingConnector` | 内置 CPU offload（本文主要内容） |
| `SimpleCPUOffloadConnector` | 简化版 CPU offload |
| `LMCacheConnectorV1` / `LMCacheMPConnector` | LMCache 外部缓存系统 |
| `MooncakeConnector` | Mooncake 分布式 KV 传输 |
| `NixlConnector` | NIXL 高性能 KV 传输 |
| `P2pNcclConnector` | 基于 NCCL 的 P2P KV 传输 |
| `HF3FSKVConnector` | 基于 HuggingFace 3FS 文件系统 |
| `MoRIIOConnector` | MoRIIO 远程 I/O |
| `FlexKVConnectorV1` | FlexKV 灵活 KV 管理 |

这些插件都实现 `KVConnectorBase_V1` 接口，调度器通过统一的 `get_num_new_matched_tokens()` / `update_state_after_alloc()` / `build_connector_meta()` 与它们交互，不感知底层传输细节。

---

## 8. 关键代码文件索引

| 文件 | 作用 |
|------|------|
| `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock、BlockHash、FreeKVCacheBlockQueue（GPU LRU） |
| `vllm/v1/core/block_pool.py` | GPU block 分配/释放/前缀缓存映射/隐式淘汰 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | GPU 前缀缓存查找 `find_longest_cache_hit()` |
| `vllm/v1/core/sched/scheduler.py` | `schedule()` 中调用缓存查找和 block 分配 |
| `vllm/v1/kv_offload/abstract.py` | `OffloadingManager` 抽象接口 |
| `vllm/v1/kv_offload/cpu/manager.py` | `CPUOffloadingManager`，CPU block 池 + 策略调度 |
| `vllm/v1/kv_offload/cpu/policies/abstract.py` | `CachePolicy` 抽象基类 |
| `vllm/v1/kv_offload/cpu/policies/lru.py` | LRU 淘汰策略 |
| `vllm/v1/kv_offload/cpu/policies/arc.py` | ARC 淘汰策略 |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | 调度器侧：CPU 查找 + 存储决策（`_get_reqs_to_store`） |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py` | Worker 侧：提交传输任务 + 完成检查 |
| `vllm/v1/kv_offload/worker/cpu_gpu.py` | GPU↔CPU 异步 DMA（`transfer_async`） |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | KVConnector 插件注册工厂 |
| `vllm/v1/kv_offload/factory.py` | OffloadingSpec 注册工厂 |

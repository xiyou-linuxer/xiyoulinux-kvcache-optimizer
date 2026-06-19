# MiniFlexKV 项目结构

MiniFlexKV 是一个从 0 开始编写的单机单卡 KV Cache 插件项目，只保留单机单卡 vLLM 推理所需的 CPU + SSD 双层缓存路径。

## 目标范围

- 作为 vLLM V1 KV connector 插件使用。
- 只支持单机单卡。
- 外部 KV Cache 只包含 CPU 和 SSD 两层。
- 不支持分布式元数据、Redis、P2P、Remote Cache、GDS、Mooncake、TensorRT-LLM、TP、DP、server-client 多实例逻辑。
- 不实现多卡推理。`tp_size` 和 `dp_size` 可以作为配置占位字段存在，但实现假设始终只有一个 GPU。
- 核心分层：逻辑缓存管理和物理数据搬运分离。

## 总体分层

```text
SchedulerConnector
  -> KVManager
    -> KVTaskEngine(KVTaskManager)
      -> GlobalCacheEngine
        -> CacheEngine(CPU)
        -> CacheEngine(SSD)
      -> TransferManagerHandle
        -> TransferManagerInterProcessHandle
          -> TransferManager
            -> StorageEngine
            -> TransferEngine
              -> TransferWorker(s)
```

核心边界：

- `CacheEngine` 只负责逻辑缓存状态：前缀树、mempool、block 分配、淘汰、ready/pending 状态。
- `StorageEngine` 负责物理存储：GPU KV cache handle、CPU buffer、SSD file。
- `TransferEngine` 根据 `TransferOpGraph` 执行数据搬运。
- `KVTaskEngine` 连接逻辑缓存规划和传输执行。
- `SchedulerConnector` 和 `WorkerConnector` 只适配 vLLM 的生命周期接口。

## 当前实际目录结构

```text
miniflex/
  pyproject.toml
  setup.py

  docs/
    project_structure.md
    notes/
      mempool.md
      radix_tree.md

  csrc/
    ssd_io_uring.h
    ssd_io_uring.cpp

  pysrc/
    miniflex/
      __init__.py
      common/
        __init__.py
        config.py
        hash.py
        block.py
        memory_handle.py
        ring_buffer.py
        storage.py
        transfer.py
      cache/
        __init__.py
        mempool.py
        radix_tree.py
        cache_engine.py
        global_cache_engine.py
      storage/
        allocator.py
        storage_engine.py
      transfer/
        __init__.py
        scheduler.py
        worker.py
        transfer_engine.py
      server/
        request.py
        utils.py
      transfer_manager.py

  test/
    hash_test.py
    block_test.py
    storage_test.py
    mempool_test.py
    radix_tree_test.py
    transfer_test.py
    ring_buffer_test.py
    scheduler_test.py
    GPUCPUTransferWorker_test.py
    worker_test.py
    ssd_cpu_worker_test.py
    transfer_engine_test.py
    transfer_manager_test.py
    cache_engine_test.py
    global_cache_engine_test.py
    storage_engine_test.py
```

规划中的后续模块尚未完整落地，包括 `engine/` 和 `integration/vllm/`。`common/config.py` 已有 dataclass 配置层；`common/storage.py` 已实现 KV cache layout 与 handle 元数据层；`common/memory_handle.py` 已有 PyTorch reducer 路径的 CUDA tensor shared handle；`common/ring_buffer.py` 已实现共享 block-id buffer；`storage/allocator.py` 已实现 CPU/GPU/SSD allocator；`storage/storage_engine.py` 已实现 CPU/SSD 物理存储初始化和 GPU KV cache 注册；`cache/global_cache_engine.py` 已实现第一版 CPU/SSD GET/PUT 规划、返回 mask 和 callback 绑定；`transfer/scheduler.py`、`transfer/worker.py`、`transfer/transfer_engine.py`、`server/request.py`、`server/utils.py`、`transfer_manager.py` 和 `csrc/ssd_io_uring.*` 已有第一版调度器、GPU<->CPU worker、SSD<->CPU worker、TransferEngine、ZMQ GPU 注册入口、TransferManager 独立进程 handle 和 C++ SSD IO backend。

## 模块职责

### Python hash foundation

MiniFlex 当前已经稳定 Python hash/block 语义，并已有 `common/config.py`、`common/storage.py` 的早期 layout 元数据层、`mempool.py`、`radix_tree.py`、`common/transfer.py`、`cache_engine.py` 和第一版 `global_cache_engine.py`。radix tree 的 key 是 block hash；索引、缓存匹配和测试都基于当前 `SequenceMeta` 契约。

MiniFlex 当前核心文件：

- `pysrc/miniflex/common/hash.py`
- `pysrc/miniflex/common/block.py`
- `pysrc/miniflex/common/config.py`
- `pysrc/miniflex/common/memory_handle.py`
- `pysrc/miniflex/common/ring_buffer.py`
- `pysrc/miniflex/common/storage.py`
- `pysrc/miniflex/common/transfer.py`
- `pysrc/miniflex/cache/mempool.py`
- `pysrc/miniflex/cache/radix_tree.py`
- `pysrc/miniflex/cache/cache_engine.py`
- `pysrc/miniflex/cache/global_cache_engine.py`
- `pysrc/miniflex/storage/allocator.py`
- `pysrc/miniflex/storage/storage_engine.py`
- `pysrc/miniflex/transfer/scheduler.py`
- `pysrc/miniflex/transfer/worker.py`
- `csrc/bindings.cpp`
- `csrc/ssd_io_uring.h`
- `csrc/ssd_io_uring.cpp`
- `test/hash_test.py`
- `test/block_test.py`
- `test/storage_test.py`
- `test/mempool_test.py`
- `test/radix_tree_test.py`
- `test/transfer_test.py`
- `test/ring_buffer_test.py`
- `test/scheduler_test.py`
- `test/GPUCPUTransferWorker_test.py`
- `test/ssd_cpu_worker_test.py`
- `test/ssd_io_uring_test.cpp`
- `test/cache_engine_test.py`

保留的核心行为：

- 使用 64-bit block hash。MiniFlex 当前使用 `hashlib.blake2b(digest_size=8)`，不强制先做 C++ `xxhash` 扩展。
- `Hasher` 是增量状态，每个完整 token block 更新同一个 hasher 后生成一个 block hash。
- `num_blocks = len(token_ids) // tokens_per_block`，尾部不足一个完整 block 的 token 不参与 hash。
- `SequenceMeta` 在生成 block hash 前先写入 namespace/model salt。
- namespace 使用 `"\x00"` 拼接后编码为 UTF-8，避免不同 namespace 列表拼接出相同字节序列。
- 对同一个 namespace/model salt，短请求的 block hashes 必须等于长请求对应 token 前缀的 block hash 前缀。

明确不引入的内容：

- CUDA transfer extension。
- GDS。
- P2P/Redis/distributed metadata。
- PCFS/Mooncake/remote cache。
- TensorRT-LLM/SGLang backend 分支。
- Cython 编译路径。

### `setup.py` / C++ extension path

当前 hash 路径不依赖 C++ 扩展。`pysrc/miniflex/common/hash.py` 是实际使用的 hash 实现。

当前状态：

- `setup.py` 声明 `torch.utils.cpp_extension.CppExtension(name="miniflex._C", sources=["csrc/bindings.cpp", "csrc/ssd_io_uring.cpp"], libraries=["uring"])`。
- `csrc/bindings.cpp` 通过 pybind11 暴露 `SSDIOCTX`。
- `csrc/ssd_io_uring.h` 和 `csrc/ssd_io_uring.cpp` 已实现第一版 CPU<->SSD IO context。
- 当前 `SSDIOCTX` 向 Python 暴露单一 `transfer_blocks(src_block_ids, dst_block_ids, is_read)` 接口；`is_read=False` 表示 CPU -> SSD，`is_read=True` 表示 SSD -> CPU。
- `SSDIOCTX` 在类内保存 CPU tensor 和必要 shape 信息，假设 CPU tensor 是 LAYERFIRST `[layer, kv, cpu_block, token, head, dim]`，SSD 文件按 BLOCKFIRST 块布局存放。
- C++ 内部支持 io_uring `readv` / `writev`，并有同步 `preadv` / `pwritev` fallback；`O_DIRECT` 由 Python 用户通过 `use_direct_io` 选择，并受当前对齐条件约束。
- 如果后续需要 C++ hash extension，应先补齐最小 CPU-only xxhash sources，并同步 `pyproject.toml` / `setup.py`。
- 不要在 MiniFlex 当前阶段加入 CUDAExtension、GDS、P2P 或 distributed transfer extension。

### `common/hash.py`

Python hash 基础层。

预期类型/函数：

- `Hasher`
- `gen_block_hashes(token_ids, tokens_per_block, namespace=None, hasher=None)`

职责：

- 使用 `hashlib.blake2b(digest_size=8)` 生成 64-bit hash。
- 保证输入为 1D `np.int64` token ids。
- 输出 `np.uint64` block hashes。
- 只 hash 完整 token block，不要求 token 长度能被 `tokens_per_block` 整除。
- `HashType = NewType("HashType", int)` 不是当前必需项；普通 Python `int` 足够。
- 不依赖 radix tree、cache engine 或 vLLM。

### `common/config.py`

定义单机单卡 CPU + SSD 路径需要的配置。当前已经有 dataclass 实现，并在 storage/cache 测试中覆盖主要配置边界。

当前类型：

- `ModelConfig`
- `CacheConfig`

当前尚未实现：

- `GlobalConfig`
- 从环境变量或配置文件加载配置的函数
- `test/config_test.py`

当前 `ModelConfig` 字段：

- `num_layers`
- `num_kv_heads`
- `head_size`
- `use_mla`
- `dtype`
- `tp_size`
- `dp_size`

当前 `CacheConfig` 字段：

- `tokens_per_block`
- `enable_cpu`
- `enable_ssd`
- `num_cpu_blocks`
- `num_ssd_blocks`
- `ssd_cache_dir`
- `ssd_file_prefix`
- `ssd_max_file_size_gb`
- `cpu_layout_type`
- `ssd_layout_type`
- `eviction_policy`
- `evict_ratio`
- `evict_start_threshold`
- `hit_add_counts`
- `protected_threshold`

MiniFlex 配置方向：

- `tokens_per_block` 属于 `CacheConfig`，不属于 `ModelConfig`。
- `ModelConfig` 只描述物理 KV cache shape 和模型并行占位字段。
- `ModelConfig.token_bytes` 当前返回每 token 的 KV cache 字节数；后续可以改名为 `token_size_in_bytes`，语义更清楚。
- `use_mla=True` 时应要求 `num_kv_heads == 1`。MLA 的 `head_size` 必须是真实 GPU KV cache tensor 的物理宽度，不是普通 attention head dim。
- 当前实现允许 `tp_size` / `dp_size` 为正数作为占位字段，但 MiniFlex 不实现多卡推理；正式 vLLM adapter 或 storage/transfer 入口必须拒绝非 1 配置。
- `enable_cpu == True`
- `enable_remote == False`
- `enable_gds == False`
- `enable_ssd` 默认应为 `False`，显式开启 SSD 时才使用 `num_ssd_blocks` 和 `ssd_cache_dir`。
- `ssd_cache_dir` 支持路径字符串或非空 `list[str]`，`enable_ssd=True` 时必须提供。
- `ssd_file_prefix` 必须是非空字符串。
- `ssd_max_file_size_gb` 必须是 `-1` 或正数。
- 当前 SSD 数据路径使用 CPU LAYERFIRST 和 SSD BLOCKFIRST。`CacheConfig` 和 `StorageEngine` 允许 CPU/SSD layout type 不同，SSD 默认是 `KVCacheLayoutType.BLOCKFIRST`。
- GPU layout 来自 vLLM 实际 KV cache 注册，不强制和 CPU/SSD layout type 一致。

### `common/block.py`

定义 token block 的元数据和哈希逻辑。

预期类型/函数：

- `SequenceMeta`
- `hash_token`
- 对 `common/hash.py` 的轻量调用

hash 应包含 namespace/model salt，避免不同模型或不同 namespace 之间发生逻辑冲突。

`SequenceMeta` 预期行为：

- 接收 1D `np.int64` token ids。
- 接收 `tokens_per_block > 0`。
- `num_blocks = len(token_ids) // tokens_per_block`。
- 只对完整 token block 生成 hash。
- 初始化时生成并缓存 `block_hashes`。
- `block_hashes` 存储为 `np.ndarray(dtype=np.uint64)`。
- `get_hash(block_id)` 返回对应 block hash 的 Python `int`，越界返回 `None`。
- `has_hashed()` 返回 hash 是否已生成。
- `namespace=None` 和 `namespace=[]` 等价。
- 非空 namespace/model salt 先进入 hasher，再进入 token block。
- 同 namespace/model salt 下保持 prefix hash property。

### `common/storage.py`

定义 KV cache 物理布局的元数据层，位置低于后续 `storage/storage_engine.py`。

当前类型：

- `StorageHandlerType`
- `KVCacheLayoutType`
- `KVCacheLayout`
- `StorageHandle`

当前状态：

- `StorageHandlerType` 区分 raw tensor、tensor shared handle 和 file handle 的粗粒度类型。
- `KVCacheLayoutType` 保留 FlexKV 风格的 `LAYERFIRST` 与 `BLOCKFIRST`。
- `KVCacheLayout` 已实现 layout 参数校验、`kv_shape`、stride、元素数和 layout 派生 helper。
- `StorageHandle` 已实现 tensor、tensor shared handle、file 三类 handle 的基础类型校验和 getter。
- 当前文件只描述布局和可访问 handle，不分配 CPU/GPU tensor，不创建 SSD 文件，也不执行传输。
- `test/storage_test.py` 覆盖 `KVCacheLayout` 的 shape、stride、切分、校验，以及 `StorageHandle` 的 tensor/tensor-handle/file getter、基础参数校验和多进程 CUDA tensor shared handle 行为。

对照 FlexKV 后的边界：

- FlexKV 的 `StorageHandle` 本身也是薄封装，主要提供 `get_tensor()`、`get_tensor_list()`、`get_file_list()` 和 GPU shared-handle getter。
- FlexKV 的真实分配发生在 `storage/allocator.py`，`StorageHandle` 只携带 `kv_layout`、`dtype`、`gpu_device_id`、`num_blocks_per_file` 等元数据。
- MiniFlex 不实现远程、GDS、P2P、多 GPU shared handle；因此当前 `StorageHandle` 不需要 `GDS_MANAGER` 或 remote config。
- MiniFlex 保留 `TENSOR_HANDLE`，因为 vLLM connector/transfer manager 之间可能跨进程传递 GPU tensor 访问能力。当前实现先使用 PyTorch reducer/pickle 路径，不直接手写 CUDA IPC handle。
- MiniFlex 后续 SSD 路径仍需要固定 file handle 约束：File handle 应由 allocator 填入 `num_blocks_per_file`，并保证文件数量覆盖 `kv_layout.num_blocks`。
- Tensor 容量、SSD 文件创建和 block 到文件 offset 的映射应放到 `storage/allocator.py` / `storage/storage_engine.py` / transfer executor 中，除非后续决定把部分轻量校验提前放进 `StorageHandle`。

### `common/memory_handle.py`

定义跨进程传递 CUDA tensor 访问能力的轻量 handle。

当前类型：

- `TensorSharedHandle`

当前状态：

- 当前实现使用 PyTorch `torch.multiprocessing.reductions.reduce_tensor()` 导出可 pickle 的重建函数和参数。
- `get_tensor()` 在目标进程中通过 PyTorch reducer 重建 CUDA tensor，并缓存重建结果。
- `__getstate__()` 会清除 `_cached_tensor`，避免把本进程缓存 tensor 一起序列化。
- 只支持从 CUDA `torch.Tensor` 构造。
- 预留了 direct CUDA IPC 的构造形态：`data: bytes`、`force_direct_ipc`、`tensor_shape`、`tensor_dtype`、`offset`。这些参数当前只作为未来扩展点；直接 IPC 尚未实现时应明确抛出。

设计边界：

- MiniFlex 当前优先使用 PyTorch reducer 路径，因为它已经覆盖同机多进程 CUDA tensor 共享场景，和 ZMQ `send_pyobj` / `recv_pyobj` 能自然组合。
- 后续如果需要绕过 PyTorch reducer，再补 direct CUDA IPC bytes 的 export/import 逻辑，并保持 `StorageHandle` 上层接口不变。
- 不在此层实现远程传输、跨机器句柄注册或多 GPU 共享策略。

### `common/transfer.py`

定义传输图相关的数据结构。

支持的设备：

- `CPU`
- `GPU`
- `SSD`

支持的传输类型：

- `D2H`
- `H2D`
- `H2DISK`
- `DISK2H`
- `VIRTUAL`

不支持的传输类型：

- `REMOTE2H`
- `H2REMOTE`
- `PEERH2H`
- `PEERSSD2H`
- `D2DISK`
- `DISK2D`

预期类型：

- `DeviceType`
- `TransferType`
- `TransferOpStatus`
- `TransferOp`
- `TransferOpGraph`
- `CompletedOp`

当前实现：

- 已实现 `DeviceType`、`TransferType`、`TransferOpStatus`。
- 已实现 `TransferOp` 的基本字段、`op_id` 分配和输入校验。
- 已实现 `CompletedOp`。
- 已实现 `TransferOpGraph` 的 `graph_id` 分配和基础容器。
- 已实现 `TransferOpGraph.add_virtual_op()`、`add_transfer_op()`、`add_dependency()`。
- 已实现 `TransferOpGraph.mark_completed()`、`take_ready_ops()` 和 `all_transfer_ops_completed()`。
- 已实现 `TransferOpGraph.set_gpu_blocks()`，用于在 vLLM 分配 GPU block 后绑定 `H2D` / `D2H` 的 GPU block id。
- `TransferOp` 和 `TransferOpGraph` 都使用类级 `ClassVar` 计数器与锁来分配全局 id。
- `TransferOpGraph` 当前使用 MiniFlex 简化状态机，不完全照搬 FlexKV 内部推进位置。
- MiniFlex 当前默认每个 `TransferOp` 搬运全量 KV 层，不携带 FlexKV 的 `layer_id` / `layer_granularity` 字段。

当前 `TransferOp` 字段：

- `transfer_type`
- `op_id`
- `graph_id`
- `src_block_ids`
- `dst_block_ids`
- `src_slot_id`
- `dst_slot_id`
- `valid_block_num`
- `depends_on`
- `dependents`
- `status`

当前 `TransferOpGraph` 字段：

- `graph_id`
- `_op_map`
- `_ready_ops`
- `_gpu_transfer_ops`

当前 `TransferOpGraph` 接口：

- `set_graph_id(graph_id)`
- `create_empty_graph()`
- `add_virtual_op(op)`
- `add_transfer_op(op)`
- `add_dependency(op_id, dependency_op_id)`
- `mark_completed(op_id)`
- `take_ready_ops()`
- `all_transfer_ops_completed()`
- `set_gpu_blocks(blocks)`

当前状态机：

- `_ready_ops` 同时保存 PENDING ready op 和 RUNNING in-flight op。
- `take_ready_ops()` 返回 `list[TransferOp]`，并把 PENDING ready op 标记为 RUNNING。
- `mark_completed()` 要求 op 为 RUNNING 且仍在 `_ready_ops` 中，然后标记为 COMPLETED。
- `mark_completed()` 会从 dependent op 的 `depends_on` 中移除已完成 op；当 dependent op 依赖清空后，把它加入 `_ready_ops`。
- 依赖图应先完整构建，再开始调度。当前 `add_dependency()` 要求被添加依赖的 op 仍是 `PENDING`。

字段设计约定：

- op 索引使用 `_op_map`，语义是 `op_id -> TransferOp`。
- ready 集合和依赖集合应使用 set，因为它们需要去重、快速 membership 检查和删除。
- GPU late-binding op id 清单可以使用 list，因为它主要按加入顺序遍历。
- 如果继续参考 FlexKV 的 DAG scheduler，依赖字段建议命名为 `predecessors` 和 `successors`。

当前测试：

- `test/transfer_test.py` 覆盖 op 参数校验、`CompletedOp`、单 op 生命周期、`DISK2H -> H2D`、`D2H -> H2DISK`、虚拟 op 多依赖汇聚、`add_dependency()` 错误路径、`set_gpu_blocks()` 和 `graph_id` 行为。

### `cache/mempool.py`

简单的物理 block id 分配器。

职责：

- 分配物理 block id。
- 回收物理 block id。
- 汇报 free/used block 数量。
- 不感知 token、tensor、CPU buffer 或 SSD file。

当前实现：

- 使用 `_free_blocks` 作为 free mask。
- 使用 `_free_ids` 和 `_free_offset` 作为空闲 block id 快照。
- 使用 `_is_dirty` 做 lazy recycle 折中：回收时不立即重建 `_free_ids`，下一次分配看到 dirty 后再刷新快照。
- 对回收输入做一维 `np.int64`、范围、重复回收过滤检查。

### `cache/radix_tree.py`

前缀索引实现。

当前状态：

- 已实现 `RadixTreeNode`。
- 已实现 `MatchResult`。
- 已实现 `RadixTree`。
- 已添加 `test/radix_tree_test.py`。

当前 `RadixTreeNode` 字段：

- `block_hashes`
- `physical_block_ids`
- `parent`
- `children`
- `_pin_count`
- `_is_ready`
- `hit_count`
- `create_time`
- `last_access_time`
- `grace_time`

当前 `RadixTreeNode` 接口：

- `size()`
- `head_hash()`
- `is_root()`
- `is_leaf()`
- `is_ready()`
- `is_in_use()`
- `is_evictable()`
- `num_children()`
- `split(prefix_length)`
- `shrink(shrink_length)`
- `merge_single_child()`
- `set_child(hash, child)`
- `get_child(hash)`

当前 `RadixTree` 接口：

- `reset()`
- `match_prefix(sequence)`
- `num_matched(sequence)`
- `insert(sequence, physical_block_ids, match_result=None, is_ready=True)`
- `evict(num_evict_blocks)`
- `is_empty()`
- `pin(node)`
- `unpin(node)`
- `set_ready(node, ready)`
- `total_cached_blocks()`
- `total_node_size()`

职责：

- 根据 block hash 匹配最长前缀。
- 插入 block hash 和 physical block id。
- 维护 ready/pending 状态。
- transfer 进行期间通过 pin/unpin 保护节点。
- 按 eviction policy 淘汰 leaf 节点。
- `children` key 使用 Python `int`，与 `SequenceMeta.get_hash()` 返回值一致。
- node 内部的 `block_hashes` 使用 `np.uint64` 数组。

MiniFlex-specific radix tree 语义：

- root 是空节点，始终 ready，但不算 leaf，永远不进入 `leaf_nodes`。
- `leaf_nodes` 只记录非 root leaf。
- `insert()` 默认插入全部未命中后缀，不支持 `num_insert_blocks`。
- `physical_block_ids` 是未命中后缀的物理 block id，不是整条序列的物理 block id。
- split 时新建 shared-prefix parent，原 node 缩成 suffix child。
- 当前不在 insert/evict 后自动合并 single-child parent。
- pending 节点和 pinned 节点都不可淘汰。
- `evict()` 当前只返回 evicted physical block ids；如果后续需要事件发布，再补 evicted block hashes。
- `total_cached_blocks()` 统计 block 数；`total_node_size()` 统计非 root 节点数量。

### `cache/cache_engine.py`

已实现的可复用逻辑缓存引擎。CPU 和 SSD 都实例化同一个 `CacheEngine` 类：

```python
cpu_cache_engine = CacheEngine(DeviceType.CPU, ...)
ssd_cache_engine = CacheEngine(DeviceType.SSD, ...)
```

当前实现：

- 只接受 `DeviceType.CPU` 和 `DeviceType.SSD`，不接受 `GPU`。
- 拥有一个 `RadixTree` 和一个 `Mempool`。
- 实现 `match(sequence_meta)`。
- 实现 `insert(sequence_meta, physical_block_ids, is_ready=True, match_result=None)`。
- 实现 `take(num_required_blocks, protected_node=None, strict=True)`。
- 实现 `recycle(physical_blocks)`。
- 实现 `set_ready(node, ready=True)`。
- 实现 `pin(node)` / `unpin(node)`。
- `take()` 会在 free block 不足或利用率超过 `evict_start_threshold` 时触发 radix-tree eviction，并把 evicted physical block ids 回收到 mempool 后再分配。
- `protected_node` 会在 `take()` 期间临时 pin，避免当前节点被驱逐。
- `strict=False` 时如果可用 block 不足会返回当前可分配数量；`strict=True` 时不足会抛 `RuntimeError`。

禁止职责：

- 不分配 tensor。
- 不读写 SSD 文件。
- 不拷贝 GPU 数据。
- 不感知 vLLM 的 request 对象。

当前测试：

- `test/cache_engine_test.py` 覆盖初始化校验、take/recycle、共享前缀 insert、pending/ready、evict/recycle、protected node、pin/unpin 和 reset。

### `cache/global_cache_engine.py`

已实现第一版跨 CPU/SSD 的逻辑缓存规划层。它根据 `CacheConfig` 创建 CPU/SSD 两个 `CacheEngine`，并为 GET/PUT 请求生成 `TransferOpGraph`、返回 mask 和 callback；仍然不执行真实 tensor/SSD 数据搬运。

当前已实现：

- `GlobalCacheEngine.__init__(cache_config, model_config)`
- `reset()`
- `match_all(sequence_meta)`
- `get(request_id, token_ids, token_mask, slot_mapping, namespace=None)`
- `put(request_id, token_ids, token_mask, slot_mapping, namespace=None)`
- `get_impl(request_id, sequence, start_idx, end_idx, gpu_blocks)`
- `put_impl(request_id, sequence, start_idx, end_idx, gpu_blocks)`
- `_transfer_callback(...)`
- `_op_callback(...)`
- `_check_input(...)`
- `_check_block_aligned_mask(...)`
- `_get_block_range(...)`
- `slot_mapping_to_block_ids(...)`
- `_add_task_end_virtual_op(...)`
- 从 `CacheConfig` 读取 `tokens_per_block`、eviction 参数和 CPU/SSD 容量。
- 当 `enable_cpu=True` 时创建 `CacheEngine(DeviceType.CPU, ...)`。
- 当 `enable_ssd=True` 时创建 `CacheEngine(DeviceType.SSD, ...)`。
- `cache_engines: dict[DeviceType, CacheEngine]`。

GET 本地路径：

```text
CPU hit: H2D
SSD hit beyond CPU ready prefix: DISK2H -> H2D
miss: 缺失后缀不生成传输图
CPU pending overlap: 使用临时 CPU buffer，callback 回收
```

PUT 本地路径：

```text
new GPU blocks: D2H
optional SSD insert/write: D2H 之后执行 H2DISK
already cached CPU prefix: 跳过已缓存前缀，只写入 miss suffix
not enough logical blocks: 回收部分申请并返回空 graph
```

职责：

- `get(...) -> TransferOpGraph, return_mask, callbacks`。
- `put(...) -> TransferOpGraph, return_mask, callbacks`。
- `match_all(...)`。
- 构建用于更新 ready 状态和回收临时 block 的 callback。
- 不直接拷贝 tensor。

当前测试：

- `test/global_cache_engine_test.py` 覆盖输入 helper、GET miss、CPU hit、SSD hit 回填 CPU、CPU 前缀 + SSD 后缀、非 0 起始 GET、pending CPU 临时 buffer、空间不足回滚、PUT CPU-only、PUT CPU+SSD、跳过已缓存前缀、复用 SSD 前缀、尾部不完整 block、mask 校验和 callback 行为。

### `storage/allocator.py`

各类物理存储介质的分配辅助类。

当前类型：

- `BaseStorageAllocator`
- `GPUStorageAllocator`
- `CPUStorageAllocator`
- `SSDStorageAllocator`

当前状态：

- `GPUStorageAllocator.allocate()` 按 `KVCacheLayout.get_total_elements()` 分配一个或多个 flat CUDA tensor，并返回 `StorageHandlerType.TENSOR` 的 `StorageHandle`。
- `GPUStorageAllocator.from_raw_data()` 用于注册已有 GPU tensor 或 `TensorSharedHandle` 列表，并返回 `TENSOR` 或 `TENSOR_HANDLE` handle。
- `CPUStorageAllocator.allocate()` 当前返回一个 flat CPU tensor handle。
- `CPUStorageAllocator.from_raw_data()` 当前把已有 CPU tensor 包成 `StorageHandle`。
- `SSDStorageAllocator.allocate()` 根据 `KVCacheLayout`、dtype、`cache_dir`、`file_prefix` 和 `max_file_size_gb` 创建 SSD cache 文件，返回 file handle。
- `SSDStorageAllocator.from_raw_data()` 可以把已有文件路径注册成 file handle，并要求 `num_blocks_per_file` 覆盖布局容量。
- `SSDStorageAllocator` 支持单个 cache dir 字符串或多个 cache dir 的 `list[str]`。
- `free()` 当前都是 no-op，由 Python 引用生命周期和后续 storage engine 管理释放边界。

设计边界：

- 对照 FlexKV，allocator 是内部薄封装，不应该重复承担所有 `StorageHandle` 已有的类型、dtype、列表一致性校验。
- allocator 层应保留会造成静默错误的检查，例如 `device_id` 这类必需元数据、`num_chunks` 容量切分不能截断、SSD 文件容量必须覆盖 block 数。
- 具体 tensor/file 的结构校验继续交给 `StorageHandle` 和测试覆盖。
- CPU allocator 后续可以在 transfer worker 落地后再决定是否启用 pinned memory。
- SSD allocator 负责文件创建和布局容量，不负责缓存匹配。

### `storage/storage_engine.py`

管理物理存储 handle。

职责：

- 初始化时分配 CPU 存储。
- `enable_ssd=True` 时初始化 SSD cache 文件存储。
- 注册 vLLM GPU KV cache tensor 或 `TensorSharedHandle`。
- 根据 `(DeviceType, device_id)` 返回 storage handle。

单卡假设：

- 只有一个 GPU storage handle，`device_id` 必须是 `0`。
- 没有 TP grouping。
- 没有 DP grouping。

当前实现：

- CPU layout 从 `ModelConfig` 和 `CacheConfig.num_cpu_blocks/tokens_per_block/cpu_layout_type` 创建。
- SSD layout 从 `ModelConfig` 和 `CacheConfig.num_ssd_blocks/tokens_per_block/ssd_layout_type` 创建。
- 当前 SSD worker 需要 CPU LAYERFIRST 和 SSD BLOCKFIRST；`CacheConfig` / `StorageEngine` 已允许 CPU/SSD layout type 不同。
- GPU layout 由 vLLM 侧真实 KV cache 注册时传入，不由 `StorageEngine` 推导，也不强制和 CPU/SSD layout type 一致。
- `register_gpu_blocks()` 做薄校验：`gpu_layout` 类型、dtype 类型、单 GPU `device_id=0`、禁止重复注册、layout 字段匹配 `ModelConfig` / `CacheConfig`、dtype 匹配 `ModelConfig.dtype`。
- tensor list 的类型和 dtype 仍主要交给 `GPUStorageAllocator` / `StorageHandle` 校验；`StorageEngine` 不做 tensor shape 校验。
- `test/storage_engine_test.py` 覆盖 CPU/SSD 初始化、SSD 多目录、SSD 配置校验、CPU/SSD layout 不同、GPU tensor/tensor-handle 注册、GPU 注册参数校验和重复注册。

### `transfer/scheduler.py`

从 `TransferOpGraph` 中调度 ready 的 transfer op。

职责：

- 遵守 op 依赖关系。
- 取出 ready op。
- 标记 completed op。
- 所有 op 完成后标记 graph 完成。

当前状态：

- `TransferScheduler` 已实现。
- `add_transfer_graph()` 拒绝重复 `graph_id`。
- `schedule(finished_ops)` 接收 `list[TransferOp]`，不是 op id。
- `schedule()` 会完成虚拟 op，并返回 `(completed_graph_ids, next_ops)`。
- `test/scheduler_test.py` 覆盖重复 graph、依赖链、多 graph、虚拟 op、未知完成 op 和空 graph。

### `common/ring_buffer.py`

为 worker 进程共享 transfer op 的 block id 列表。

当前状态：

- `SharedOpPool` 已实现。
- 内部使用 shared-memory `torch.int64` tensor，形状为 `(max_op_num, max_block_num)`。
- `allocate_slot(block_ids, device_type_prefix=0)` 会按 hash 复用相同 block-id 列表。
- `free_slot()` 使用引用计数释放 slot。
- `test/ring_buffer_test.py` 覆盖复用、容量限制和多进程可见性。

### `transfer/worker.py`

实现实际的数据拷贝原语。

当前已实现：

- `WorkerTransferOp`
- `WorkerHandle`
- `TransferWorkerBase.create_worker(...)`
- `TransferWorkerBase._worker_process(...)`
- `TransferWorkerBase.run()`
- `GPUCPUTransferWorker`，负责 `D2H` 和 `H2D`
- `SSDCPUTransferWorker`，负责 `H2DISK` 和 `DISK2H`

当前限制和待修：

- `GPUCPUTransferWorker` 只支持 `LAYERFIRST`。
- CPU/GPU per-layer tensor 形状为 `[kv_dim, num_blocks, tokens_per_block, num_heads, head_size]`，block 维度是 `dim=1`。
- `_transfer_impl()` 使用 `index_select -> to(dst_device) -> index_copy_`。
- D2H gathered `.to(cpu)` 必须保持同步；H2D gathered movement 可以保持 `non_blocking=True`。
- `TransferWorkerBase.run()` 当前会把第一次 `recv()` 的 op 放入 `batch_op`，再 drain 当前可用的后续 op，执行成功后只把 `transfer_op_id` 放进 `finished_ops_queue`。
- `worker.py` 仍有一段 triple-quoted 旧 `_transfer_impl`，下一次 worker cleanup 应删除。
- `SSDCPUTransferWorker` 使用 CPU LAYERFIRST + SSD BLOCKFIRST 的布局组合，初始化时把 CPU tensor、shape、SSD file list、`num_blocks_per_file` 和 `use_direct_io` 传给 `_C.SSDIOCTX`。
- SSD 路径的 Python 层做轻量输入校验，C++ 层保持接近数据搬运实现本身，不做厚重防御性校验。

当前测试：

- `test/GPUCPUTransferWorker_test.py` 直接绕过 `__init__` 测 `_transfer_impl()`。
- 覆盖 D2H、H2D、非连续 block、空传输、错误参数和简单性能输出。
- `test/ssd_cpu_worker_test.py` 用 fake `_C.SSDIOCTX` 覆盖 SSDCPU worker 初始化、轻量校验、方向映射和 block-id 参数校验。
- `test/ssd_io_uring_test.cpp` 用真实临时文件覆盖 file IO、O_DIRECT、io_uring chunk 和 io_uring+O_DIRECT chunk，并做写入、清零、读回后的 payload 校验。

删除的 worker：

- TP GPU/CPU worker。
- GDS worker。
- peer/remote worker。
- Mooncake/PCFS worker。

### `transfer/transfer_engine.py`

消费 transfer graph，并把 op 分发给 worker 执行。

当前状态：

- 第一版 GPUCPU + SSDCPU TransferEngine 已实现，执行 `D2H` / `H2D` / `H2DISK` / `DISK2H`。
- TransferEngine 拥有 `TransferScheduler`、`op_id -> TransferOp` 映射、finished queue、completed queue、worker handle 和 scheduler thread。
- TransferEngine 拥有 `SharedOpPool`，把 TransferOp 的 src/dst block id 写入共享 block-id buffer 后交给 worker。
- worker 只上报 `op_id`；TransferEngine 把 op id 映射回 `TransferOp` 后再调用 `TransferScheduler.schedule(...)`。
- `D2H` / `H2D` 分发给 `GPUCPUTransferWorker`，`H2DISK` / `DISK2H` 分发给 `SSDCPUTransferWorker`。
- `submit_transfer_graph(...)` 接受单个 `TransferOpGraph` 或 `list[TransferOpGraph]`，不接受 tuple 或混合列表。
- `start()` 是显式生命周期入口；submit 只入队，不隐式启动 worker。
- `get_completed_graphs_and_ops(...)` 返回 `CompletedOp` 列表，空队列时直接返回空列表。

保留事项：

- `worker.py` 仍有一段旧的 triple-quoted `_transfer_impl` 代码块，可作为后续清理项，但不阻塞 `TransferManager`。

### `transfer_manager.py`

当前已经实现第一版 `TransferManager` 生命周期边界，供后续 `KVTaskEngine` 调用。MiniFlex 这里保留了 FlexKV 的核心形态：manager 可以运行在独立进程中，worker 侧通过 ZMQ 注册 GPU KV cache handle，scheduler/任务侧通过 handle 提交 transfer graph 并等待完成结果。

当前类型：

- `TransferManager`
- `TransferManagerHandleBase`
- `TransferManagerHandleIntraProcessHandle`
- `TransferManagerHandleInterProcessHandle`
- `TransferManagerHandle`
- `RegisterGPUBlocksRequest`

当前行为：

- `TransferManager` 持有 `StorageEngine`，启动前通过 ZMQ `PULL` 接收 `RegisterGPUBlocksRequest`。
- GPU 注册请求携带 `list[TensorSharedHandle]` 和 `KVCacheLayout`，由 `StorageEngine.register_gpu_blocks(...)` 构建 GPU `StorageHandle`。
- SSD 未启用时 `_ssd_handle` 保持 `None`；启用 SSD 时必须能从 `StorageEngine` 取到 SSD handle。
- `initialize_transfer_engine()` 在 GPU handle 注册后创建 `TransferEngine`。
- `start()`、`submit()`、`submit_batch()`、`wait()`、`shutdown()` 委托给 `TransferEngine`。
- process handle 使用 `spawn` 创建独立 manager 进程；父进程在 `start()` 期间临时设置并恢复 `MPI4PY_RC_INITIALIZE=false`。
- 独立进程内部用 selector 同时监听 command pipe 和 `TransferEngine` completed queue。
- `submit` / `submit_batch` 请求通过 multiprocessing pipe 发送；completed ops 通过 result pipe 回传。
- `TransferManagerHandle` 是上层薄封装，默认 `mode="process"`，也支持 `mode="thread"`。未显式传入 `gpu_register_port` 时会生成默认 IPC endpoint。
- `server/utils.py` 默认把裸路径解释为 IPC endpoint；`tcp://...` endpoint 保留给后续 TP/DP/分离式扩展。

当前测试：

- `test/transfer_manager_test.py` 启动真实独立进程 `TransferManagerHandle`，通过 ZMQ 注册真实 CUDA `TensorSharedHandle`，启动真实 GPUCPU/SSDCPU workers，提交 `D2H -> H2DISK -> DISK2H -> H2D` graph，并校验 GPU 目标 block 数据真实往返。

当前不实现：

- `TransferManagerOnRemote`。
- multi-node / multi-GPU / DP / TP grouping。
- remote query path。
- cache match、task status、return mask、op callback、final callback。

### `engine/kvtask.py`

任务状态机。

预期类型：

- `TaskStatus`
- `TaskType`
- `KVTask`
- `KVTaskManager`
- `KVTaskEngine`

职责：

- 通过 `GlobalCacheEngine` 创建 GET/PUT task。
- 在 vLLM 分配 GPU block 后设置 slot mapping。
- 通过 `TransferManagerHandle` launch transfer graph。
- 轮询 completed ops。
- 执行 graph/op callback。
- 暴露 `get_match`、`put_match`、`launch_tasks`、`cancel_tasks`、`try_wait`、`wait`。

### `engine/kvmanager.py`

提供给 vLLM adapter 使用的轻量公共 API。

职责：

- 持有 `KVTaskEngine`。
- start/shutdown。
- `get_match`。
- `put_match`。
- `launch`。
- `cancel`。
- `try_wait`。
- `wait`。

删除：

- server-client mode。
- DP client。
- Redis metadata 初始化。
- multi-instance id 逻辑。
- MPS 管理，除非后续明确需要。

### `integration/vllm/vllm_v1_adapter.py`

面向 vLLM 的 scheduler/worker 实现。

预期类：

- `MiniFlexKVSchedulerConnector`
- `MiniFlexKVWorkerConnector`
- `MiniFlexKVConnectorV1Impl`

Scheduler 侧方法：

- `get_num_new_matched_tokens`
- `update_state_after_alloc`
- `build_connector_meta`
- `request_finished`
- `update_connector_output`
- `get_kv_connector_stats`
- `shutdown`

Worker 侧方法：

- `register_kv_caches`
- `start_load_kv`
- `wait_for_layer_load`
- `save_kv_layer`
- `wait_for_save`
- `get_finished`

### `integration/vllm/connector.py`

继承 vLLM `KVConnectorBase_V1` 的轻量包装类。

这个文件只保留 vLLM import 和对 `MiniFlexKVConnectorV1Impl` 的委托。

### `integration/vllm/utils.py`

vLLM 版本兼容和对象解析工具。

职责：

- 从 vLLM `Request` 中提取 token ids。
- 提取 request id。
- 提取 namespace/cache salt。
- 把 vLLM block 对象转换成 physical block ids。
- 校验单机单卡 parallel config。

## 测试规划

测试按层次推进：

1. `mempool`
2. `radix_tree`
3. `cache_engine`
4. `global_cache_engine`
5. `storage_engine`
6. `transfer_graph` - 已有 `test/transfer_test.py`
7. `ring_buffer` - 已有 `test/ring_buffer_test.py`
8. `transfer_scheduler` - 已有 `test/scheduler_test.py`
9. `gpu_cpu_worker` - 已有 `test/GPUCPUTransferWorker_test.py`
10. `transfer_engine` - 已有 `test/transfer_engine_test.py`
11. `transfer_manager` - 已有 `test/transfer_manager_test.py`
12. `kvtask`
13. `vllm_adapter`

第一批测试应避免依赖 vLLM。`transfer_manager` 的真实集成测试需要 CUDA，因为它验证跨进程 CUDA tensor handle 注册和 GPU<->CPU worker。vLLM connector 测试先用 fake request/block 对象，后续再接真实 vLLM 集成测试。

测试文件约定：

- 每个测试文件都必须可以直接用 `python path/to/test_file.py` 运行。
- 每个测试文件中的每个函数都必须有中文注释或 docstring，说明该函数的作用或该测试函数覆盖的测试内容。
- 直接运行时必须打印中文逐项日志，能看到每个测试用例的开始、通过、失败状态。
- 失败日志应包含异常类型和异常信息，然后重新抛出异常，保留原始 traceback。
- 测试函数仍保持普通 `test_...` 形式，方便后续接入 pytest。

## 当前规模和进度估算

截至当前状态：

- Python 源码约 3300 行。
- Python 测试约 4700 行。
- 源码 + 测试合计约 8000 行。

目标范围是 vLLM V1 connector、单机单卡、CPU + SSD 双介质。

粗略估算：

- 可跑 demo 版：总量约 6000-9000 行，当前约 65%-97%。
- 稳定 vLLM 对接版：总量约 8500-13500 行，当前约 43%-68%。
- 只按源码估算，当前约占 demo 版源码目标 29%-44%，约占稳定版源码目标 20%-31%。

实际工程进度约 60%-65%。逻辑缓存基础层、第一版跨层 GET/PUT 规划、storage metadata、allocator、StorageEngine、TransferScheduler、GPU<->CPU worker、SSD<->CPU worker、C++ SSD IO backend、第一版 TransferEngine 和独立进程 TransferManager 已经完成；下一步是任务层和 vLLM 对接。

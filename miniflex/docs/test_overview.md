# 测试说明（AI 协助开发）

本项目 `test/` 下的测试代码由 **Claude (Anthropic)** 协助编写，作者负责审阅、运行与结果校验。
每个测试文件头部都标注了「由 Claude (Anthropic) 编写」及该文件的测试内容；本文是全部测试
文件的覆盖汇总，既作为**测试索引**，也作为 **AI 使用说明**。

- 运行方式：每个测试文件都可单独运行 `PYTHONPATH=pysrc python test/<file>.py`；
  C++ 测试 `test/ssd_io_uring_test.cpp` 随 `_C` 一并编译运行。
- 约定：每个测试函数都有中文注释/docstring，直接运行会打印逐项中文日志（开始/通过/失败），
  失败时打印异常类型与信息并重新抛出，保留原始 traceback。
- 模块与分层见 [project_structure.md](project_structure.md)，验证结论见 [validation.md](validation.md)。

## 覆盖总览

| 层 | 测试文件 | 被测模块 |
|---|---|---|
| 基础类型 | `block_test.py`、`hash_test.py`、`storage_test.py`、`transfer_test.py` | `common/` 的 block 切分、hash、布局/句柄、传输图 |
| 逻辑缓存 | `mempool_test.py`、`radix_tree_test.py`、`cache_engine_test.py`、`global_cache_engine_test.py` | `cache/` 物理池、前缀树、单层与跨层缓存引擎 |
| 物理存储 | `storage_engine_test.py` | `storage/` CPU/SSD 初始化与 GPU 注册 |
| 传输 | `scheduler_test.py`、`ring_buffer_test.py`、`GPUCPUTransferWorker_test.py`、`ssd_cpu_worker_test.py`、`worker_test.py`、`transfer_engine_test.py`、`transfer_manager_test.py`、`ssd_io_uring_test.cpp` | `transfer/` 调度、worker、引擎、独立进程管理、C++ IO |
| 任务 / 集成 | `kvtask_test.py`、`vllm_v1_adapter_test.py` | `kvtask.py` 任务编排、vLLM connector 适配 |

---

## 一、基础类型（`common/`）

### `block_test.py`
- `test_block`：`SequenceMeta` 按 `tokens_per_block` 切块，namespace / cache salt 参与哈希以做缓存隔离。

### `hash_test.py`
- `test_hash`：`Hasher`（blake2b、64-bit）增量哈希的正确性与同输入稳定性。

### `storage_test.py`
- 布局：`LAYERFIRST` / `BLOCKFIRST` / MLA 三种 `KVCacheLayout` 的 `kv_shape`、stride 与派生量；非法参数校验。
- 句柄：`StorageHandle` 的 tensor / tensor-list / tensor-handle（FlexKV 风格 getter）/ file 四类的 getter 与基础校验。
- 分配器：`SSDStorageAllocator` 建缓存文件、多目录与 `from_raw_data`、配置校验。
- 共享句柄：`TensorSharedHandle` 的 CPU 侧安全校验、CUDA tensor 元数据与转换，以及**多进程共享同一 CUDA tensor**。

### `transfer_test.py`
- `TransferOp` / `CompletedOp` 校验与单 op 生命周期。
- 依赖：`DISK2H→H2D`、`D2H→H2DISK`、虚拟 op 等待多前驱。
- `add_dependency` 校验、`set_gpu_blocks` 绑定 H2D/D2H、graph id 生成与显式覆盖。

## 二、逻辑缓存（`cache/`）

### `mempool_test.py`
- `test_mempool`：物理 block-id 池的分配、回收、空闲/占用计数与非法输入校验。

### `radix_tree_test.py`
- 节点校验与叶语义；空树匹配、插入与前缀查询。
- 插入分裂已有叶子并能匹配两条分支；插入校验失败不破坏树结构。
- pending/ready 状态对「ready 匹配」与淘汰的影响；`ready_length` 标记分裂出的 pending 前缀。
- `pin` / `unpin` 控制淘汰；`evict` 的 0/负数/部分/超额淘汰；分裂树淘汰后父节点升为叶子；`reset`。
- **淘汰策略优先级**：`lru` / `lfu` / `slru`（热块保护段）/ `fifo` 各自的排序行为。

### `cache_engine_test.py`
- 初始化校验与基础字段；取 0/负数与回收；前缀匹配、插入与共享前缀。
- pending/ready 状态与 `set_ready`；淘汰回收与严格模式；**protected 节点临时阻止淘汰**；pin/unpin 控制淘汰；`reset` 清空索引与 mempool。

### `global_cache_engine_test.py`（跨层 GET/PUT 规划）
- 输入 helper 与校验；GET miss 返回空图。
- GET 命中：CPU 命中构 `H2D`；SSD 命中构 `DISK2H→H2D` 并回填 CPU 缓存；CPU 前缀 + SSD 续接的分段读；忽略尾部不完整块；非零 block 区间的 CPU 命中 / SSD 回填；CPU pending 回退到临时 buffer；CPU 空间不足返回空图；拒绝非对齐/非连续 mask。
- PUT：仅 CPU 构 `D2H` 并回填；CPU+SSD 构 `D2H→H2DISK`；跳过已存在的 CPU/SSD 前缀只写后缀；完全命中返回空图；空间不足返回空图并回收；插入失败回收（含 SSD 回调回收）；重叠 PUT 分裂保持 ready 前缀连续；用 mask 覆盖的完整前缀；忽略尾部不完整块；拒绝非零/非对齐 mask。

## 三、物理存储（`storage/`）

### `storage_engine_test.py`
- CPU 存储初始化；SSD 初始化（含文件前缀、`cache_dir` 列表多目录）；SSD 配置校验；CPU/SSD 布局可不同。
- GPU block 注册：tensor 列表、tensor-handle 列表；参数与布局/张量校验；**拒绝重复注册**。

## 四、传输（`transfer/` + `csrc/`）

### `scheduler_test.py`
- 拒绝重复加入 graph 与 pending 状态；单个就绪 op 调度并完成 graph；依赖链调度；多 graph 保持插入序；虚拟 op 无需 worker 执行即完成；未知 finished op 被忽略；空 graph 调度即完成。

### `ring_buffer_test.py`
- `SharedOpPool` 的分配复用/释放与状态；容量与前缀行为；**多进程**子进程读父进程分配的 slot、子进程写对父进程可见。

### `GPUCPUTransferWorker_test.py`
- `D2H` / `H2D` 往返逐字节相等；不同数值保持不混淆；空传输 no-op；越界 / 尺寸不符 / 非法 transfer 类型报错；性能（吞吐）报告；**`LAYERBLOCK`（vLLM 0.23 布局）GPU↔CPU 往返**。

### `ssd_cpu_worker_test.py`
- 初始化按期望 shape 构造 `SSDIOCTX` 且 direct IO 生效；拒绝错误布局与 dtype；`H2DISK` / `DISK2H` 到 C++ 方向映射；输入参数校验。

### `worker_test.py`
- `TransferWorkerBase` worker 进程：提交 `D2H` / `H2D` 更新对应张量；共享 op slot 取 block id（GPU↔CPU 与 SSD↔CPU）；shutdown 前排空已提交/多个排队任务后退出；无任务时正常退出。
- `H2DISK` / `DISK2H` 真实文件往返；GPU↔CPU 与 SSD↔CPU 的端到端吞吐基准。

### `transfer_engine_test.py`
- submit 需先 `start`，接受单个 graph；接受 graph 列表并拒绝非法输入；完成查询遵守超时；初始化拒绝明显的 handle 不匹配；op buffer 按设备前缀复用相同 block id；**`H2DISK`/`DISK2H` 真实文件往返**。

### `transfer_manager_test.py`
- `test_inter_process_manager_real_gpu_cpu_ssd_roundtrip`：**独立进程** `TransferManager` 的真实 GPU↔CPU↔SSD 往返集成测试（需 CUDA）。

### `ssd_io_uring_test.cpp`（C++）
- SSD io_uring 后端：普通 file IO、`O_DIRECT`、io_uring 分块、io_uring + `O_DIRECT` 分块；用真实临时文件做写入 / 清零 / 读回的 payload 校验。

## 五、任务层与 vLLM 集成

### `kvtask_test.py`
- GET 任务 match→launch→wait（launch 前用 fake slot_mapping）；PUT/GET 异步在 end-op 后、graph 完成前即返回成功；提前返回成功后 graph 完成仍正确释放任务；batch 合并 GET / PUT 图并按 batch 任务等待；`cancel_tasks` 释放任务且 wait 报 not-found。

### `vllm_v1_adapter_test.py`（connector 适配 + 端到端）
- 单卡 prefill GET 生命周期上报 `finished_recving`；`request_finished` PUT 生命周期上报 `finished_sending`；batch GET 上报全部子请求完成。
- preemption（launch 前 / 状态 diff 回退）取消 pending 任务；worker 侧 no-op 接口与 stats sentinel。
- no-match GET 取消并返回「不需要」；部分命中只写未命中 block；不足一个 block 的 PUT 跳过；**aborted 请求不 PUT**；失败 GET 记录失败 block id；`MINIFLEX_SYNC_GET` 同步等待后上报 `finished_recving`。
- `test_end_to_end_real_put_then_get_roundtrips_kv_through_cpu`：**真实 PUT→GET 经 CPU 的端到端往返**（逐字节校验）。

---

*测试代码由 Claude (Anthropic) 协助编写，作者审阅与校验。*

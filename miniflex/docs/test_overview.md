# 测试说明与 AI 交互记录

本项目 `test/` 下的测试代码由 **Claude (Anthropic)** 协助编写，作者负责审阅、运行与结果校验。
每个测试文件头部都标注了「由 Claude (Anthropic) 编写」及该文件的测试内容；本文是全部测试
文件的覆盖汇总，既作为**测试索引**，也作为 **AI 使用说明 / AI 交互记录**。

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

## AI 使用范围说明

结合当前项目实际情况，AI 工具主要参与了以下几类工作：

1. **测试代码辅助编写**：围绕 `test/` 目录下的模块测试、集成测试和端到端回归测试进行辅助生成；
2. **测试说明整理**：帮助归纳测试文件覆盖范围、测试目标和运行方式；
3. **比赛文档辅助整理**：帮助梳理设计开发文档结构、补充章节组织、整理合规披露口径；
4. **答辩材料辅助整理**：帮助归纳 PPT 素材来源、参考材料边界和展示结构建议；
5. **合规说明辅助整理**：帮助汇总非本队来源说明、开源协议提示、AI 使用披露口径。

需要说明的是：AI 工具主要起到**辅助整理、辅助生成、辅助归纳**的作用，最终内容均需要由本队成员结合代码实现、测试结果和比赛要求进行人工复核。

## AI 工具清单

### 1. Claude (Anthropic)

主要用于：

- 协助编写 `test/` 目录下的测试代码；
- 协助补充测试覆盖说明和测试索引；
- 辅助梳理模块级、链路级和端到端测试场景。

人工复核方式：

- 由项目成员逐个检查测试目标是否与被测模块一致；
- 实际运行测试脚本，核对输出是否符合预期；
- 对失败或异常结果进行人工定位与修正。

### 2. Codex CLI / OpenAI 助手

主要用于：

- 辅助整理 `miniflex/docs/设计开发文档.md` 的比赛文档结构；
- 辅助梳理 `README.md`、设计文档、PPT 之间的内容分工；
- 辅助补充非本队来源说明、PPT 素材来源说明、AI 使用说明；
- 辅助总结 `MiniFlex` 当前实现路径、模块关系和 GET / PUT 链路；
- 辅助核对 `projects/` 目录中的 PPT 源素材、风格参考和输出工程目录。

人工复核方式：

- 由项目成员对照真实仓库结构、源码实现和 PPT 工程目录逐项核查；
- 对 AI 生成的文档性描述进行人工删改，避免与项目实际情况不符；
- 对涉及 FlexKV 参考关系、开源协议、素材来源等敏感内容进行单独确认。

---

## 补充 AI 交互记录（文档与比赛材料阶段）

以下记录用于补充说明当前阶段 AI 工具的使用场景，便于后续在比赛材料中统一披露。

### 交互记录汇总表

| 记录编号 | 阶段 | 使用工具 | 目标 | 主要产出 | 人工复核情况 |
| --- | --- | --- | --- | --- | --- |
| 1 | 测试整理阶段 | Claude (Anthropic) | 梳理 `test/` 覆盖范围 | `test_overview.md` 测试索引主体 | 已由项目成员结合测试文件逐项核对 |
| 2 | 文档整理阶段 | Codex CLI / OpenAI 助手 | 重构比赛主文档结构 | `设计开发文档.md` 章节重组与补充 | 已结合真实仓库和实现人工修改 |
| 3 | 架构讲解整理阶段 | Codex CLI / OpenAI 助手 | 从源码提炼架构链路 | Scheduler/Worker、GET/PUT 链路说明 | 已对照源码类名与模块边界复核 |
| 4 | 合规整理阶段 | Codex CLI / OpenAI 助手 | 补充引用、协议、AI 披露 | 非本队来源说明、协议提示、AI 说明 | 已结合当前材料人工核查 |
| 5 | PPT 材料整理阶段 | Codex CLI / OpenAI 助手 | 整理答辩素材来源 | PPT 源素材、风格参考与工程目录说明 | 已结合 `projects/` 目录人工核查 |
| 6 | 参考关系排查阶段 | Codex CLI / OpenAI 助手 | 排查与 FlexKV 的关系说明是否完备 | “仍需补充增量贡献说明”的结论 | 已由项目成员确认 |

### 记录一：测试体系整理

- 目标：梳理 `test/` 目录下所有测试文件的覆盖范围，形成统一测试说明文档；
- AI 辅助内容：按模块分类整理测试点、生成说明性文字、归纳运行方式；
- 人工工作：逐个核对测试文件、测试对象和说明内容是否一致；
- 产出：当前文件 `test_overview.md` 的主体测试索引部分。

### 记录二：设计开发文档重构

- 目标：将原始说明文档改写为更接近比赛提交材料的结构；
- AI 辅助内容：提出章节重组建议，补充“项目目标与规划、方案设计与分析、项目详细推进情况、合规披露”等章节框架；
- 人工工作：结合真实实现逐节修改，补充与实际项目一致的内容；
- 相关文档：`miniflex/docs/设计开发文档.md`。

### 记录三：项目实现路径讲解整理

- 目标：把 `MiniFlex` 当前实现从“代码存在”整理成“可用于答辩讲解”的架构说明；
- AI 辅助内容：帮助从 `MiniFlexSchedulerConnector`、`MiniFlexWorkerConnector`、`KVTaskEngine`、`GlobalCacheEngine`、`TransferManager` 等核心类中抽取结构关系；
- 人工工作：核对类名、模块边界、运行路径和 GET / PUT 链路是否与真实代码一致；
- 相关文档：`miniflex/docs/设计开发文档.md`、`miniflex/docs/project_structure.md`。

### 记录四：合规披露材料整理

- 目标：满足比赛对非本队来源、开源协议和 AI 使用记录的披露要求；
- AI 辅助内容：帮助归纳披露口径、整理引用说明章节、标出仍需补充的风险点；
- 人工工作：核查当前仓库 `LICENSE`、参考文档、PPT 工程目录和已有调研材料；
- 相关文档：`miniflex/docs/设计开发文档.md`。

### 记录五：答辩 PPT 素材来源整理

- 目标：把 `projects/project21_miniflex_update_ppt169_20260627/` 中涉及的 PPT 源素材和风格参考纳入说明；
- AI 辅助内容：帮助识别 `sources/`、`analysis/`、`spec_lock.md`、参考 PPT/Markdown 的作用，并生成材料来源说明；
- 人工工作：核对源文件目录、参考 PPT 名称、用途边界和最终披露口径；
- 相关文档：`miniflex/docs/设计开发文档.md`。

### 记录六：FlexKV 参考关系排查

- 目标：确认当前提交文档中是否已明确写出“基于 FlexKV 的参考关系与增量贡献”；
- AI 辅助内容：帮助全局搜索 `docs/` 与 `miniflex/docs/` 中关于 `FlexKV` 的现有描述；
- 人工工作：确认 `Implementation_Investigation/` 中已有“主参考 FlexKV”的调研内容，但主提交文档仍需进一步转写；
- 当前结论：比赛主文档中仍需要单独增加“与 FlexKV 的关系说明 / 增量贡献说明”。

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

## 六、当前阶段说明

1. 当前文件已经同时承担**测试索引**与**AI 使用记录**两类作用；
2. 若后续继续使用 AI 参与代码开发、PPT 生成、视频脚本撰写或结果分析，应继续在本文件或单独附录中按时间顺序补记；
3. 若比赛要求提交更细粒度的 AI 原始对话记录，建议再额外整理一份“时间线式 AI 交互附录”，保留每次提问目的、输出内容类型和人工采纳情况。
4. 当前记录重点覆盖**测试整理、文档整理、合规整理与 PPT 材料整理阶段**；若后续需要补“代码实现阶段”的 AI 使用明细，应以真实 commit、聊天记录和文件修改记录为依据继续补充。

---

*测试代码由 Claude (Anthropic) 协助编写；文档整理与比赛材料完善阶段使用了 Codex CLI / OpenAI 助手辅助；最终内容由项目成员审阅、核对与决定是否采纳。*

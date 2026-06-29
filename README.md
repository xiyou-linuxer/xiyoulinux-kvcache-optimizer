# MiniFlex：面向 LLM 推理负载的 KV Cache 分层管理与优化系统

| 项目 | 内容 |
| --- | --- |
| 项目名称 | 面向 LLM 推理负载的 KV Cache 分层管理与优化系统（MiniFlex） |
| 参赛队伍 | 古法编程继承人 |
| 团队成员 | 井光成、徐子淳、祁馨叶 |

## 目录

- [一、项目概述](#一项目概述)
- [二、项目亮点](#二项目亮点)
- [三、系统设计与实现内容](#三系统设计与实现内容)
- [四、阶段性结果](#四阶段性结果)
- [五、快速启动](#五快速启动)
- [六、测试与演示](#六测试与演示)
- [七、仓库结构](#七仓库结构)
- [八、文档索引](#八文档索引)
- [九、核心文件说明](#九核心文件说明)
- [十一、视频展示](#十一视频展示)
- [十二、当前边界与后续规划](#十二当前边界与后续规划)
- [十三、合规与说明](#十三合规与说明)

## 一、项目概述

MiniFlex 是一个面向大语言模型推理场景的 KV Cache 分层管理系统，重点解决 **长上下文、显存受限、工作集超过 GPU 容量** 时推理效率快速下降的问题。项目以 vLLM V1 KV connector 为接入边界，在尽量少改动推理框架主流程的前提下，实现 KV Cache 在 **GPU / CPU / SSD** 多层介质之间的组织、迁移、回填与复用。

项目当前聚焦以下目标：

- 在长上下文场景下降低 prefill 重算开销；
- 在超显存容量场景下维持更稳定的命中能力；
- 通过透明接入方式降低工程改造成本；
- 提供可复现的测试、验证与展示材料；
- 形成适合比赛提交的设计文档、使用说明和验证记录。

从问题定位上看，MiniFlex 围绕 **KV Cache 生命周期管理** 这一关键瓶颈，构建一个可插拔、可扩展、可验证的分层缓存系统。

## 二、项目亮点

### 2.1 透明接入现有推理框架

MiniFlex 基于 vLLM 的 KV connector 机制实现接入，尽量保留原有推理主流程，不要求大规模侵入式改造，便于工程集成与后续演示。

### 2.2 面向操作系统问题的分层存储设计

项目围绕 GPU、CPU、SSD 三层存储介质构建统一缓存视图，将“显存不足”问题转化为“跨层存储调度与回填”问题，突出操作系统中的资源管理、分层存储、缓存替换与 I/O 路径优化能力。

### 2.3 兼顾性能与容量稳定性

MiniFlex 的核心价值不只是追求容量内最快，其在追求 **越过 GPU 显存边界之后退化更平稳**。在工作集超出 GPU KV 容量时，相比原生 APC 路径更能维持稳定命中和较低时延。

### 2.4 具备完整工程化材料

项目提供：

- 启动脚本；
- 一键演示脚本；
- 长上下文 / 超容量 / 混合负载基准；
- 使用说明、结构说明、验证文档；
- 测试总览与 AI 工具使用记录。

## 三、系统设计与实现内容

### 3.1 分层缓存管理

- 支持将 KV Cache 作为 block 级资源进行组织；
- 在 GPU / CPU / SSD 之间维护统一缓存管理视图；
- 支持缓存命中、写入、回填与状态管理；
- 支持跨层 GET / PUT 路径规划与缓存状态维护。

### 3.2 透明接入 vLLM

- 基于 vLLM V1 connector 接入；
- 尽量保留原有推理主流程；
- 通过 scheduler / worker 两侧协同完成 KV 生命周期接管；
- 兼容 OpenAI 接口风格的服务启动与验证方式。

### 3.3 底层传输与工程化支持

- 提供 GPU ↔ CPU 的 C++ / CUDA 传输实现；
- 提供 CPU ↔ SSD 的 `io_uring` 传输后端；
- 提供启动脚本、演示脚本和基准测试脚本；
- 支持以环境变量方式进行缓存容量和行为调优。

### 3.4 场景化验证

- 长上下文加速验证；
- 超容量工作集验证；
- 混合负载验证；
- 功能正确性与稳定性验证。

### 3.5 当前完成情况

| 预期方向 | 当前状态 | 说明 |
| :-- | :-- | :-- |
| vLLM 接入路径 | 已完成 | 已打通基于 connector 的透明接入 |
| CPU 缓存层 | 已完成 | 已支持 CPU 层缓存命中与回填 |
| SSD 缓存层 | 已完成 | 已支持 SSD 侧存储与读取路径 |
| GPU ↔ CPU 传输 | 已完成 | 已提供 C++ / CUDA 扩展实现 |
| CPU ↔ SSD 传输 | 已完成 | 已提供 `io_uring` 后端 |
| 结构文档与使用说明 | 已完成 | 已整理结构、使用与验证文档 |
| 长上下文 / 容量测试 | 已完成 | 已形成阶段性实验结果 |



## 四、阶段性结果

### 4.1 长上下文加速

在当前实验结果中，随着上下文长度增长，MiniFlex 命中后的回填路径可以显著减少 prefill 重算：

- `~1k` 上下文：约 `1.63×` 加速；
- `~8k` 上下文：约 `4.06×` 加速；
- `~30k` 上下文：约 `9.32×` 加速。

代表性结果：`~30k` 上下文下，冷启动约 `3806 ms`，命中回填后约 `408 ms`。

### 4.2 超容量场景优势

与 vLLM 原生 APC 相比，MiniFlex 在“工作集超出 GPU KV 容量”后表现更稳定：

| 工作集 | APC | MiniFlex |
| --- | --- | --- |
| `~45k` | `95 ms / 0% miss` | `199 ms / 0% miss` |
| `~75k` | `684 ms / 100% miss` | `191 ms / 0% miss` |
| `~105k` | `700 ms / 100% miss` | `190 ms / 0% miss` |

这说明 MiniFlex 的核心价值不在于容量内峰值最快，而在于 **越过显存边界后退化更平稳**。

## 五、快速启动

更完整的环境说明、配置说明与常见问题见 `docs/usage.md`。

如果只是想快速拉起项目并验证功能，建议按下面步骤进行。

### 5.1 环境前提

推荐环境：

- Linux 环境；
- Python `>= 3.10`；
- 已正确安装 PyTorch；
- 使用 vLLM connector 时安装 `vllm`；
- 系统安装 `liburing-dev`（SSD I/O 后端需要）。

安装系统依赖：

```bash
sudo apt install liburing-dev
```

安装项目：

```bash
cd miniflex

# 在已装好 torch 的环境里安装
pip install --no-build-isolation .

# 若需要通过 vLLM connector 启动
pip install --no-build-isolation '.[vllm]'
```

### 5.2 方式一：使用封装脚本启动

最推荐直接使用仓库内已封装好的启动脚本：

```bash
cd miniflex
bash run_vllm_miniflex.sh
```

该脚本会完成：

- 清理上一次残留的 vLLM / EngineCore 进程；
- 清理旧的 IPC socket；
- 设置 MiniFlex 所需环境变量；
- 启动带 MiniFlex connector 的 vLLM 服务。

可用环境变量覆盖常见参数：

```bash
MODEL=Qwen/Qwen3-8B \
PORT=8000 \
MINIFLEX_MAX_MODEL_LEN=32768 \
MINIFLEX_GPU_MEM_UTIL=0.80 \
bash run_vllm_miniflex.sh
```

### 5.3 方式二：手动启动 vLLM + MiniFlex

如果需要手动调试，可以直接运行：

```bash
cd miniflex

export ENABLE_MINIFLEX=1
export PYTHONPATH=pysrc
export MINIFLEX_GPU_REGISTER_PORT=ipc:///tmp/miniflex.sock

vllm serve <你的模型名> \
  --served-model-name qwen3-8b \
  --kv-transfer-config '{"kv_connector":"MiniFlexConnectorV1","kv_connector_module_path":"miniflex.integration.vllm.connector","kv_role":"kv_both"}' \
  --disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000
```

注意事项：

- `--kv-transfer-config` 的 JSON 必须写在同一行；
- 单机单卡场景下必须加 `--disable-hybrid-kv-cache-manager`；
- 启动目录建议为 `miniflex/`，并确保 `PYTHONPATH=pysrc`。

### 5.4 服务验证

服务启动成功后，可通过 OpenAI 兼容接口验证：

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

### 5.5 常见入口

- 启动脚本：`run_vllm_miniflex.sh`
- 一键演示：`demo.sh`
- 使用说明：`docs/usage.md`
- 结构说明：`docs/project_structure.md`
- 验证报告：`docs/validation.md`

## 六、测试与演示

### 6.1 一键演示

项目提供自包含演示脚本，可自动完成功能验证、长上下文测试、容量交叉测试和混合负载测试：

```bash
cd miniflex
bash demo.sh
```

若不需要每幕暂停：

```bash
cd miniflex
PAUSE=0 bash demo.sh
```

### 6.2 单项基准测试

长上下文 TTFT：

```bash
cd miniflex
PYTHONPATH=pysrc python bench_ttft.py --url http://localhost:8000 --model qwen3-8b --body-repeat 1000 --runs 3
```

超容量工作集：

```bash
cd miniflex
PYTHONPATH=pysrc python bench_overflow.py --url http://localhost:8000 --tag miniflex --num-prefixes 10
```

混合负载：

```bash
cd miniflex
PYTHONPATH=pysrc python bench_mixed.py --url http://localhost:8000 --tag both
```

### 6.3 测试材料位置

- 测试总览与 AI 交互记录：`docs/test_overview.md`
- 验证报告：`docs/validation.md`
- 测试代码目录：`test/`

## 七、仓库结构

```text
miniflex/
├── README.md
├── csrc/                      # C++ / CUDA 扩展
├── docs/                      # 设计文档、结构说明、使用说明、验证报告
├── pysrc/miniflex/            # Python 主体实现
├── test/                      # 单元测试与功能验证
├── bench_ttft.py              # 长上下文 TTFT 测试
├── bench_overflow.py          # 超容量工作集测试
├── bench_mixed.py             # 混合负载测试
├── demo.sh                    # 一键演示脚本
├── run_vllm_miniflex.sh       # 启动 vLLM + MiniFlex 脚本
├── pyproject.toml
└── setup.py
```

### 7.1 核心代码说明

- `pysrc/miniflex/integration/vllm/connector.py`：MiniFlex 接入 vLLM 的入口；
- `pysrc/miniflex/kvtask.py`：KV 任务组织与编排；
- `pysrc/miniflex/cache/`：逻辑缓存管理；
- `pysrc/miniflex/storage/`：物理存储与分配；
- `pysrc/miniflex/transfer/`：传输调度与 worker；
- `csrc/transfer.cu`：GPU ↔ CPU 传输实现；
- `csrc/ssd_io_uring.cpp`：SSD I/O 后端实现。

## 八、文档索引

为了便于比赛提交与答辩材料整理，项目文档按用途拆分如下：

- `docs/设计开发文档.md`：比赛主说明文档；
- `docs/project_structure.md`：项目结构与模块边界说明；
- `docs/usage.md`：安装、启动与配置说明；
- `docs/validation.md`：阶段性验证与性能结果；
- `docs/test_overview.md`：测试覆盖总览、AI 使用说明与交互记录。

如果你是第一次阅读本项目，推荐顺序为：

1. 先看本 `README.md` 了解项目全貌；
2. 再看 `docs/usage.md` 完成启动；
3. 接着看 `docs/project_structure.md` 理解实现；
4. 最后结合 `docs/validation.md` 与 `docs/设计开发文档.md` 准备比赛材料。

## 九、核心文件说明

### 文档内容

- [设计开发文档.md](./设计开发文档.md)：比赛主设计开发文档；
- [初赛PPT.pptx](./初赛PPT.pptx)：初赛答辩 PPT；
- [miniflex演示视频.mp4](./miniflex演示视频.mp4)：项目演示视频。

### 代码内容

- [Connector.md](./docs/Connector.md)：`vLLM` connector 接入说明；
- [KV_Cache工作机制详解.md](./docs/KV_Cache工作机制详解.md)：`KV Cache` 工作机制说明；
- [KvCache_token复用机制详解.md](./docs/KvCache_token复用机制详解.md)：token 复用机制说明；
- [vLLM前缀缓存命中与CPU卸载机制详解.md](./docs/vLLM前缀缓存命中与CPU卸载机制详解.md)：前缀缓存和 CPU 卸载说明；
- [1.flexkv实现方案.md](./docs/Implementation_Investigation/1.flexkv实现方案.md)：FlexKV 实现方案；
- [2.lmcache实现方案.md](./docs/Implementation_Investigation/2.lmcache实现方案.md)：LMCache 实现方案；
- [3.mooncake实现方案.md](./docs/Implementation_Investigation/3.mooncake实现方案.md)：Mooncake 实现方案；
- [比赛方案参考路线.md](./docs/Implementation_Investigation/比赛方案参考路线.md)：方案参考路线；
- [基线测试报告.md](./docs/test/基线测试报告.md)：基线测试报告；
- [比赛要求.md](./docs/test/supporting/比赛要求.md)：比赛要求整理；
- [完整测试执行流程_方案B.md](./docs/test/supporting/完整测试执行流程_方案B.md)：测试执行流程；
- [测试数据集设计方案.md](./docs/test/supporting/测试数据集设计方案.md)：测试数据集设计方案。

## 十、视频展示

> 如果你的 Markdown 预览器支持本地视频，下面会直接显示；否则至少可以点击链接打开。

<video src="./miniflex演示视频.mp4" controls width="100%"></video>

[点击打开演示视频](./miniflex演示视频.mp4)

## 十一、当前边界与后续规划

### 11.1 当前版本边界

- 当前版本仅支持 **单机单卡**；
- 当前外部缓存层以 **CPU + SSD** 为主；
- 不涉及分布式远端 KV 服务；
- 不覆盖 TP、DP、多实例协同等复杂部署模式；
- 当前更关注功能可行性、路径正确性与容量退化稳定性。

### 11.2 后续可扩展方向

- 多卡 / 分布式 KV 协同；
- 更细粒度的缓存替换与热度管理策略；
- 更高性能的跨层异步调度；
- 更完备的自动化评测与可视化展示。

## 十二、合规与说明

### 12.1 参考项目研究与社区协作

我们不仅复现和学习参考项目，还持续参与其问题修复与技术讨论，围绕 KV layout 适配、批处理生命周期、并发安全、内存回收与边界检查等问题，累计形成 8 项公开开发记录，其中包括 6 项已合并 PR、1 项待合并 PR 和 1 项 Issue 讨论，体现了我们对项目底层机制的持续投入。

相关记录主要覆盖以下问题类型：

- vLLM 0.23+ 非 MLA 场景下的 GPU KV layout 兼容适配；
- batch task 合并后的子任务生命周期与资源回收；
- match / put 异常路径下的 KVTask 及时取消与内存泄漏规避；
- slot mapping 更新缺失导致的批量任务映射错误；
- 共享资源分配中的临界区竞态问题；
- TransferWorker 批处理循环中的 shutdown 清理路径；
- Mempool 回收路径中的 block id 边界校验；
- `FLEXKV_SYNC_GET=1` 同步路径下的任务字典清理问题讨论。

比赛提交要求中的设计文档、非本队来源说明、开源协议状态说明、AI 工具使用说明等内容，统一整理在以下文档中：

- `docs/设计开发文档.md`
- `docs/test_overview.md`

当前 `README` 主要承担仓库入口、项目概览、启动说明和文档导航的职责；更完整的比赛材料请以 `docs/` 目录中的文档为准。

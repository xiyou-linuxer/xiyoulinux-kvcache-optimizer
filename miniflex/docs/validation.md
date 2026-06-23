# MiniFlexKV 验证报告

记录 MiniFlexKV vLLM connector 的正确性 / 性能验证结论与复现方法。
本轮在 **vLLM 0.23 + RTX 5090 + Qwen3-8B** 上完成，含与 vLLM 原生前缀缓存（APC）的对照。
布局自适配原理见 [project_structure.md](project_structure.md) 的「KV 布局自适配」。

## 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5090 32 GiB |
| 模型 | Qwen3-8B（bf16） |
| 推理框架 | vLLM 0.23.0 |
| 底层 | torch 2.11.0+cu130 / triton 3.6.0 / CUDA 13.0 / Python 3.12 |
| MiniFlex | 含 vLLM 0.23 KV 布局自适配（`LAYERBLOCK`，非 MLA） |
| 服务参数（三方对齐） | `--gpu-memory-utilization 0.80`、`--enforce-eager`、`--max-model-len 32768` |
| GPU KV cache 容量 | ~68,368 tokens |
| MiniFlex CPU 池 | 8192 blocks（≈131k tokens） |

三种服务配置：**裸跑**（关缓存）、**APC**（vLLM 原生显存前缀缓存）、**MiniFlex**
（CPU/SSD 多级缓存）；另含 **APC + MiniFlex** 同时开。

> **基准两条正确性前提**（脚本内已保证）：
> 1. 每条请求**读完整个流**再结束——否则首 token 后关连接 = server `abort` = MiniFlex 不 PUT。
> 2. 命中路径先**预热 + pump**——让 MiniFlex 异步 PUT commit、随后 GET 才能命中。

## 1. 功能与正确性 ✅

| 检查项 | 结论 |
|---|---|
| 启动 | 8B + vLLM 0.23 + MiniFlex connector 启动成功，`register_kv_caches`（`LAYERBLOCK` 路径）正常 |
| 补全正确 | `"The capital of France is"` → `" Paris. The capital of Italy is Rome…"` |
| KV 传输 | PUT 把 GPU block 存到 CPU、GET 读回，`torch.equal` **逐字节相等** |
| 布局自适配 | GPU=`LAYERBLOCK` ↔ CPU=`LAYERFIRST` 的 D2H/H2D 往返（含 block 重排）**逐字节相等**，见 `test/GPUCPUTransferWorker_test.py` |
| 输出一致性 | 命中后输出与冷算**非逐字相同**，是**良性浮点差异**（原生 APC 冷/热也不一致——chunked-prefill 浮点累加顺序不同）；单测已证 KV 字节本身无误 |

## 2. 长上下文加速（冷 vs 热：MiniFlex 命中 vs 全量重算）

冷 = 全新 prompt 全量重算 prefill；热 = 预热到确认命中后从 CPU 加载。TTFT 中位数：

| 上下文 | 加速比 | 省下的重算 |
|---|---|---|
| ~1k  | 1.63× | 39% |
| ~2k  | 2.02× | 50% |
| ~4k  | 2.76× | 64% |
| ~8k  | 4.06× | 75% |
| ~15k | 6.81× | 85% |
| ~23k | 7.71× | 87% |
| ~30k | 9.32× | 89% |

> 锚点：~30k 上下文 **冷 3806 ms → 热 408 ms**。
> 加速随上下文变长单调放大——重算量随上下文线性涨，而 KV 搬运是带宽瓶颈、增长更慢。

## 3. 容量优势 vs vLLM 原生 APC（核心结论）

工作集逐步增大、轮询访问；GPU KV 容量 ~68k tokens。median TTFT 与 miss 率：

| 工作集 | APC（median / miss） | MiniFlex（median / miss） |
|---|---|---|
| ~45k（装得进 GPU） | **95 ms** / 0% | 199 ms / 0% |
| ~75k（略超容量） | 684 ms / **100%** | **191 ms** / 0% |
| ~105k（远超容量） | 700 ms / **100%** | **190 ms** / 0% |

> 两段式结论，交叉点 ≈ GPU KV 容量（~68k）：
> - **装得进 GPU**：APC 更快（命中在显存、零拷贝；MiniFlex 命中走 PCIe）。
> - **越过 GPU 容量**：APC 全驱逐 → 每次重算（~700 ms、100% miss）；MiniFlex 摊到 CPU →
>   **全命中、稳定 ~190 ms**。此时 MiniFlex 快 **3.6×** 且 miss 率 0%。
> MiniFlex 把 TTFT 与工作集解耦：APC 随溢出崩盘，MiniFlex 基本持平。

## 4. APC + MiniFlex 叠加（混合负载：热数据 + 超容量长尾）

少量高频"热" prompt（留在 GPU）+ 超容量长尾前缀；分别量热 / 长尾 / 总体 TTFT 中位数：

| 配置 | hot | tail | overall |
|---|---|---|---|
| APC | **60 ms** | 707 ms | 515 ms |
| MiniFlex | 103 ms | **177 ms** | **153 ms** |
| **APC + MiniFlex** | 67 ms | 188 ms | 165 ms |

> - APC：热数据最快（60 ms），但长尾溢出 → 重算（707 ms），拖垮 overall。
> - MiniFlex：长尾 / overall 最优，热数据略慢于 APC。
> - **两者同开**：热数据拿到接近 APC 的速度（67 ms），长尾拿到 MiniFlex 的容量（188 ms）——
>   **两个维度都没有短板**，是面向真实混合负载的稳妥组合（APC 吃 GPU 热层、MiniFlex 兜溢出层）。

## 5. 行为说明：异步 commit "慢一拍"

PUT 的最终 commit 只在引擎"有请求在 step"时推进。因此**空闲后立刻重复同一请求**，
其 GET 可能早于上一个 PUT 的 commit 而 miss；**连续流量下会自愈**（中间请求会把引擎 step
起来、把之前的 PUT 落地）。这是异步设计的固有取舍，不是 bug。低频场景若要稳定命中，
可用 `MINIFLEX_SYNC_GET=1`（牺牲性能，见 usage.md）。基准脚本用"预热 + pump"消除这一影响。

稳定性：长上下文加速（②）、容量溢出（③，含 ~105k 工作集）、混合并发（④）三轮 sweep
**全程无请求挂死 / 无超时**——"只剩等待远程 KV、无其他可调度 token"的边界场景被正确处理。

## 6. 复现方法

5090 环境一次性准备（cu13 / Blackwell sm_120）：

```bash
git clone <repo> && cd miniflex
sudo apt install -y liburing-dev                      # _C 链接 -luring
export CPATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/include
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")')
TORCH_CUDA_ARCH_LIST="12.0" python setup.py build_ext --inplace   # 编译 _C
```

一键演示（自包含脚本：内联启动 vLLM 的 miniflex/apc/both 三种配置 + 调用通用 `bench_*.py`，
覆盖 ①功能 ②加速 ③容量交叉 ④叠加）：

```bash
bash demo.sh            # 每幕停顿（便于录制 / 配字幕）
PAUSE=0 bash demo.sh    # 连续跑
```

单项基准（分别对 baseline / apc / miniflex / both 服务各跑一次）：

```bash
PYTHONPATH=pysrc python bench_ttft.py     --url http://localhost:8000 --model qwen3-8b --body-repeat 1000 --runs 3   # 冷/热
PYTHONPATH=pysrc python bench_overflow.py --url http://localhost:8000 --tag <apc|miniflex> --num-prefixes <6|10|14>  # 容量交叉
PYTHONPATH=pysrc python bench_mixed.py    --url http://localhost:8000 --tag <apc|miniflex|both>                      # 混合负载
```

公共参数：`MINIFLEX_MAX_MODEL_LEN=32768`、`MINIFLEX_NUM_CPU_BLOCKS=8192`、`MODEL=Qwen/Qwen3-8B`。

---

*评测日期：2026-06 · vLLM 0.23.0 + MiniFlex（LAYERBLOCK）· RTX 5090 32G · Qwen3-8B*

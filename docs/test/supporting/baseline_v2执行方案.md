# baseline v2 执行手册

## 1. 这份文档是干什么的

这是一份“按顺序照着跑就行”的 baseline v2 执行手册。

你现在要做的事情只有一句话：

> 用统一的 token 口径，重新跑一套原始 vLLM baseline，给后面的优化版当对照组。

这份手册会明确写清楚：

- 你要用哪个 `py` 文件
- 终端里具体输入什么命令
- 每一步跑完会产出什么文件
- 你下一步该做什么
- 什么时候需要给 AI 看，什么时候不需要

这份手册默认你是在云服务器上执行，默认路径和环境尽量模仿旧版 `方案 B`：

- 模型路径：`/root/autodl-tmp/Qwen/Qwen3-8B`
- LongBench 本地目录：`/root/autodl-tmp/longbench_local`
- 服务地址：`http://127.0.0.1:8000`
- 运行方式：云服务器本机部署、本机调用

## 2. 你会用到哪几个文件

这次 baseline v2 只会反复用到这 3 个脚本：

### 2.1 `tests/vllm_baseline_runner.py`

作用：

- 真正执行测试
- 发请求给 vLLM
- 采集 `/metrics`
- 采集 GPU / CPU 信息
- 生成 `results.jsonl`、`summary.json`

你可以把它理解成：

> 主测试脚本

### 2.2 `tests/prepare_baseline_manifest.py`

作用：

- 只给 `longbench` 和 `needlebench` 用
- 先把开源数据集按 token 长度筛出来
- 固定成一份 `sample_manifest.json`

你可以把它理解成：

> 数据准备脚本

### 2.3 `tests/summarize_baseline_runs.py`

作用：

- 把所有 run 的 `summary.json` 汇总成一张 CSV 表

你可以把它理解成：

> 汇总脚本

## 3. 先记住：6 个场景怎么分

这次 baseline v2 里有 6 个场景。

### 3.0 最重要的：前缀缓存策略（跑之前先看清楚）

vLLM 0.6+ 默认开启自动前缀缓存（Automatic Prefix Caching）。你的测试里 `repeated_text()` 生成的合成文本由同一段 chunk 反复重复构造，即使关了跨请求共享，请求内部的大量重复块也会被 vLLM 自动缓存命中（"自命中"），导致 latency 被人为压低。

因此，不同场景需要**不同的前缀缓存策略**：

| 策略 | 场景 | vLLM 启动参数 | 原因 |
|------|------|-------------|------|
| **关前缀缓存** | `short_synth`、`ultra_long_synth`、`longbench`、`needlebench`、`mixed_pressure` | `--no-enable-prefix-caching` | 测的是纯长度/真实数据压力，前缀缓存命中是噪声 |
| **开前缀缓存** | `shared_prefix` | 不加（或显式 `--enable-prefix-caching`） | 这个场景的目的就是测共享前缀的缓存收益 |

**为什么 `shared_prefix` 单独开着？**

因为比赛要求的四个核心指标之一是"KV 缓存命中率"。`shared_prefix` 场景所有请求共享同一段前缀，是验证缓存复用最直接的场景。如果关了，这个场景就失去意义了。

**为什么 `mixed_pressure` 也要关？**

`mixed_pressure` 虽然 25% 的请求是 shared_prefix 桶，但它的主要定位是"混合压力并发测试"，不是"缓存命中测试"。如果开着前缀缓存，短/中/长桶的合成文本也会产生内部自命中，污染并发压力数据。

**注意**：这意味着 `shared_prefix` 和其余 5 个场景**必须用不同的 vLLM 启动参数分别跑**。

按推荐顺序：

1. 先用 `--no-enable-prefix-caching` 启动 vLLM，跑完除 `shared_prefix` 外的 5 个场景
2. 再重启 vLLM（不带 `--no-enable-prefix-caching`），跑 `shared_prefix`

### 第一类：先准备 manifest，再正式跑

这两个是“开源数据集场景”：

- `longbench`
- `needlebench`

流程是：

1. 先用 `prepare_baseline_manifest.py`
2. 再用 `vllm_baseline_runner.py`

### 第二类：直接跑，跑完自动生成 manifest

这 4 个是“脚本自己合成 prompt 的场景”：

- `short_synth`
- `ultra_long_synth`
- `shared_prefix`
- `mixed_pressure`

流程是：

1. 直接用 `vllm_baseline_runner.py`
2. 跑完后它会自动在结果目录里生成 `sample_manifest.json`
3. 下次如果要完全复用同一批 prompt，再把这份 manifest 喂回去

## 4. 跑测试前你要准备什么

### 4.0 首次运行：下载 LongBench 数据（只需做一次）

编排器现在支持 3 个 LongBench 子集（qasper / qmsum / narrativeqa）。首次运行前，需要把数据拉取到本地：

```bash
# 在服务器上执行（只需一次）
python3 -c "
from datasets import load_dataset
import json, os

local_dir = '/root/autodl-tmp/longbench_local/data'
os.makedirs(local_dir, exist_ok=True)

for subset in ['qasper', 'qmsum', 'narrativeqa']:
    ds = load_dataset('THUDM/LongBench', subset, trust_remote_code=True)
    path = os.path.join(local_dir, f'{subset}.jsonl')
    with open(path, 'w') as f:
        for i in range(len(ds)):
            f.write(json.dumps(ds[i], ensure_ascii=False) + '\n')
    print(f'{subset}: {len(ds)} samples → {path}')
print('done')
"
```

> 如果已经有过 `qasper.jsonl`，这个命令会覆盖更新。如果你已有手动下载的数据且不想覆盖，把 `qasper` 从循环里去掉即可。

### 4.1 服务要先起来

启动 vLLM 服务。**关键：根据你要跑的场景选择正确的启动命令。**

#### 启动方式 A：关前缀缓存（跑除 shared_prefix 外的 5 个场景）

```bash
vllm serve /root/autodl-tmp/Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 49152 \
  --disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching
```

这条命令用于跑：

- `short_synth`
- `longbench`
- `ultra_long_synth`
- `needlebench`
- `mixed_pressure`

#### 启动方式 B：开前缀缓存（只跑 shared_prefix）

```bash
vllm serve /root/autodl-tmp/Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 49152 \
  --disable-hybrid-kv-cache-manager
```

这条命令用于跑：

- `shared_prefix`

#### 推荐执行顺序

1. 用**启动方式 A** 启动 vLLM
2. 跑完 `short_synth`、`longbench`、`ultra_long_synth`、`needlebench`、`mixed_pressure`
3. Ctrl+C 停掉 vLLM
4. 用**启动方式 B** 重新启动 vLLM
5. 跑 `shared_prefix`

> 为什么不能一次启动跑完所有？
>
> `shared_prefix` 场景的目的就是测前缀缓存命中率，关了缓存就没意义了。
> 而其他 5 个场景用 `repeated_text()` 生成合成文本，开着前缀缓存会产生请求内部的"自命中"，latency 会被虚假压低，测出来的不是纯长度压力。

### 4.2 检查服务是不是 ready

```bash
curl http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/metrics | rg "kv_cache_usage_perc|prefix_cache|num_requests"
```

如果这两条命令都正常，再继续下一步。

### 4.3 Python 依赖要有

```bash
python3 -c "import requests; print('ok')"
python3 -c "import datasets; print('ok')"
python3 -c "from transformers import AutoTokenizer; print('ok')"
```

如果缺依赖：

```bash
pip install datasets transformers
```

### 4.4 先进入仓库根目录

后面的命令默认你已经在仓库根目录：

```bash
cd /home/xuzichun/os/xiyoulinux-kvcache-optimizer
```

## 5. 先做一次脚本自检

只要看帮助信息能不能正常出来就行。

```bash
python3 tests/vllm_baseline_runner.py --help
python3 tests/prepare_baseline_manifest.py --help
python3 tests/summarize_baseline_runs.py --help
```

## 6. baseline v2 的总流程

你就按下面这个顺序跑，不要跳。

1. 用**启动方式 A**（关前缀缓存）启动 vLLM 服务
2. 做脚本自检
3. 先准备 `longbench` / `needlebench` 的 manifest
4. 每个场景先跑一个小样本 sanity check（shared_prefix 除外）
5. 跑除 `shared_prefix` 外的五组场景 baseline（`short_synth` / `longbench` / `ultra_long_synth` / `needlebench` / `mixed_pressure`）
6. Ctrl+C 停掉 vLLM，用**启动方式 B**（开前缀缓存）重新启动
7. 跑 `shared_prefix` 场景
8. 再跑固定长度 microbenchmark（用启动方式 B 或 A 均可，microbenchmark 用随机 token 不涉及重复文本）
9. 全部跑完后，汇总成 `baseline_runs_summary.csv`
10. 把关键结果补进测试记录文档
11. 最后再把结果交给 AI 帮你分析和写报告

这个顺序的核心变化是**按前缀缓存策略分两段跑**：先关缓存跑压力场景，再开缓存跑 shared_prefix。

## 7. 第一步：准备开源数据集的 manifest

这一节做两件事：

- 准备 `longbench`（3 个子集：qasper、qmsum、narrativeqa）
- 准备 `needlebench`

### 7.1 LongBench qasper 4K-8K token 桶

```bash
python3 tests/prepare_baseline_manifest.py \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --subset qasper \
  --longbench-local-dir /root/autodl-tmp/longbench_local \
  --num-samples 10 \
  --min-prompt-tokens 4000 \
  --max-prompt-tokens 8000 \
  --max-input-chars 120000 \
  --max-input-tokens 8000 \
  --output-manifest baseline_manifests/longbench_qasper_4k_8k.json
```

### 7.2 LongBench qasper 8K-16K token 桶

```bash
python3 tests/prepare_baseline_manifest.py \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --subset qasper \
  --longbench-local-dir /root/autodl-tmp/longbench_local \
  --num-samples 10 \
  --min-prompt-tokens 8000 \
  --max-prompt-tokens 16000 \
  --max-input-chars 160000 \
  --max-input-tokens 16000 \
  --output-manifest baseline_manifests/longbench_qasper_8k_16k.json
```

### 7.3 LongBench qmsum 4K-8K token 桶

```bash
python3 tests/prepare_baseline_manifest.py \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --subset qmsum \
  --longbench-local-dir /root/autodl-tmp/longbench_local \
  --num-samples 10 \
  --min-prompt-tokens 4000 \
  --max-prompt-tokens 8000 \
  --max-input-chars 120000 \
  --max-input-tokens 8000 \
  --output-manifest baseline_manifests/longbench_qmsum_4k_8k.json
```

### 7.4 LongBench qmsum 8K-16K token 桶

```bash
python3 tests/prepare_baseline_manifest.py \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --subset qmsum \
  --longbench-local-dir /root/autodl-tmp/longbench_local \
  --num-samples 10 \
  --min-prompt-tokens 8000 \
  --max-prompt-tokens 16000 \
  --max-input-chars 160000 \
  --max-input-tokens 16000 \
  --output-manifest baseline_manifests/longbench_qmsum_8k_16k.json
```

### 7.5 LongBench narrativeqa 4K-8K token 桶

```bash
python3 tests/prepare_baseline_manifest.py \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --subset narrativeqa \
  --longbench-local-dir /root/autodl-tmp/longbench_local \
  --num-samples 10 \
  --min-prompt-tokens 4000 \
  --max-prompt-tokens 8000 \
  --max-input-chars 120000 \
  --max-input-tokens 8000 \
  --output-manifest baseline_manifests/longbench_narrativeqa_4k_8k.json
```

### 7.6 LongBench narrativeqa 8K-16K token 桶

```bash
python3 tests/prepare_baseline_manifest.py \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --subset narrativeqa \
  --longbench-local-dir /root/autodl-tmp/longbench_local \
  --num-samples 10 \
  --min-prompt-tokens 8000 \
  --max-prompt-tokens 16000 \
  --max-input-chars 160000 \
  --max-input-tokens 16000 \
  --output-manifest baseline_manifests/longbench_narrativeqa_8k_16k.json
```

### 7.7 NeedleBench 8K-24K token 桶

```bash
python3 tests/prepare_baseline_manifest.py \
  --scenario needlebench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --needle-subset en_haystack_texts \
  --num-samples 10 \
  --min-prompt-tokens 8000 \
  --max-prompt-tokens 24000 \
  --max-input-chars 180000 \
  --max-input-tokens 24000 \
  --output-manifest baseline_manifests/needlebench_en_haystack_8k_24k.json
```

### 7.8 这一步跑完你要看什么

你只需要确认：

- 终端里没有报错
- `baseline_manifests/` 目录下出现了这 7 个 JSON 文件：
  - `longbench_qasper_4k_8k.json`
  - `longbench_qasper_8k_16k.json`
  - `longbench_qmsum_4k_8k.json`
  - `longbench_qmsum_8k_16k.json`
  - `longbench_narrativeqa_4k_8k.json`
  - `longbench_narrativeqa_8k_16k.json`
  - `needlebench_en_haystack_8k_24k.json`

如果 manifest 文件已经生成，这一步就结束。

**这一步不用给 AI 看。**

除非：

- 命令报错
- 筛不出足够样本
- token 桶分布明显不合理

## 8. 第二步：每个场景先跑一次小样本 sanity check

这一步的目标非常简单：

> 先确认 runner 能正常产出新版结果结构。

### 8.1 short_synth sanity check

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario short_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 2 \
  --prompt-chars 20000 \
  --max-input-tokens 1024 \
  --max-tokens 128 \
  --warmup-samples 0 \
  --concurrency 1 \
  --run-group-id sanity_short_synth \
  --repeat-index 0
```

### 8.2 longbench sanity check

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --sample-manifest baseline_manifests/longbench_qasper_4k_8k.json \
  --max-input-tokens 8000 \
  --max-tokens 128 \
  --warmup-samples 0 \
  --concurrency 1 \
  --run-group-id sanity_longbench_4k_8k \
  --repeat-index 0
```

### 8.3 shared_prefix sanity check

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario shared_prefix \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 2 \
  --prefix-chars 120000 \
  --max-input-tokens 8000 \
  --max-tokens 128 \
  --warmup-samples 0 \
  --concurrency 1 \
  --run-group-id sanity_shared_prefix \
  --repeat-index 0
```

### 8.4 sanity check 跑完你要看什么

每次运行结束后，runner 会打印一个结果目录，例如：

```bash
Saved results to: baseline_runs/short_synth_1778000000
```

你进入这个目录，看 3 个文件：

- `sample_manifest.json`
- `results.jsonl`
- `summary.json`

重点检查：

1. `summary.json` 里有没有这些字段
- `prompt_tokens_mean`
- `prompt_tokens_p50`
- `prompt_tokens_p95`
- `prompt_tokens_max`
- `total_tokens_mean`
- `num_requests_waiting_peak`

2. `results.jsonl` 里每条记录有没有这些字段
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

3. synthetic 场景有没有自动生成 `sample_manifest.json`

如果这 3 件事都正常，说明 baseline v2 runner 已经能正式用了。

**sanity check 一般也不用给 AI 看。**

只有下面几种情况才需要发给 AI：

- `summary.json` 缺字段
- `prompt_tokens` 全是 `null`
- synthetic 场景没有生成 `sample_manifest.json`
- 命令直接报错

## 9. 第三步：正式跑五组比赛场景 baseline

下面这部分就是主线，建议顺序和旧版 `方案 B` 保持一致：

1. `short_synth`
2. `longbench`
3. `ultra_long_synth`
4. `needlebench`
5. `shared_prefix`
6. `mixed_pressure`

## 9.1 short_synth

### 1K token

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario short_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 10 \
  --prompt-chars 20000 \
  --max-input-tokens 1024 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 1 \
  --run-group-id short_synth_1k_c1 \
  --repeat-index 0
```

### 2K token

把上面的：

- `--max-input-tokens 1024`

改成：

- `--max-input-tokens 2048`

## 9.2 longbench

### 4K-8K token，concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --sample-manifest baseline_manifests/longbench_qasper_4k_8k.json \
  --max-input-tokens 8000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 1 \
  --run-group-id longbench_qasper_4k_8k_c1 \
  --repeat-index 0
```

### 4K-8K token，concurrency=2

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --sample-manifest baseline_manifests/longbench_qasper_4k_8k.json \
  --max-input-tokens 8000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 2 \
  --run-group-id longbench_qasper_4k_8k_c2 \
  --repeat-index 0
```

### 8K-16K token，concurrency=1/2

### 8K-16K token，concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --sample-manifest baseline_manifests/longbench_qasper_8k_16k.json \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 1 \
  --run-group-id longbench_qasper_8k_16k_c1 \
  --repeat-index 0
```

### 8K-16K token，concurrency=2

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario longbench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --sample-manifest baseline_manifests/longbench_qasper_8k_16k.json \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 2 \
  --run-group-id longbench_qasper_8k_16k_c2 \
  --repeat-index 0
```

## 9.3 ultra_long_synth

### 16K token，concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario ultra_long_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 5 \
  --ultra-prompt-chars 160000 \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id ultra_long_16k_c1 \
  --repeat-index 0
```

### 8K token，concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario ultra_long_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 5 \
  --ultra-prompt-chars 160000 \
  --max-input-tokens 8000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id ultra_long_8k_c1 \
  --repeat-index 0
```

### 24K token，concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario ultra_long_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 5 \
  --ultra-prompt-chars 240000 \
  --max-input-tokens 24000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id ultra_long_24k_c1 \
  --repeat-index 0
```

### 32K token，concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario ultra_long_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 5 \
  --ultra-prompt-chars 320000 \
  --max-input-tokens 32000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id ultra_long_32k_c1 \
  --repeat-index 0
```

### 48K token，concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario ultra_long_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 5 \
  --ultra-prompt-chars 480000 \
  --max-input-tokens 48000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id ultra_long_48k_c1 \
  --repeat-index 0
```

> **注意**：48K 需要 `--max-model-len` ≥ 49152（编排器已默认设为此值）。如果 GPU 显存不足，这步可能会 OOM——此时跳过 48K，用 32K 及以下数据即可。

### 16K token，concurrency=2

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario ultra_long_synth \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 5 \
  --ultra-prompt-chars 160000 \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 2 \
  --run-group-id ultra_long_16k_c2 \
  --repeat-index 0
```

## 9.4 needlebench

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario needlebench \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --sample-manifest baseline_manifests/needlebench_en_haystack_8k_24k.json \
  --max-input-tokens 24000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id needlebench_8k_24k_c1 \
  --repeat-index 0
```

## 9.5 shared_prefix

> **重要**：shared_prefix 场景会在正式测试前自动插入一个缓存预热步骤（`shared_prefix_cache_warmup`）。
> 这个步骤发送一条 shared_prefix 请求，让 vLLM 的 prefix cache / MiniFlex connector 把共享前缀写入缓存。
> 这保证了后续的 4K/8K/16K 测试能真正测到缓存命中的效果，而不是第一个请求全 miss。
> 预热请求的结果会保存到 `baseline_runs/` 目录，但汇总时可根据 `run_group_id=shared_prefix_cache_warmup` 过滤掉。

### 8K token

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario shared_prefix \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 10 \
  --prefix-chars 120000 \
  --max-input-tokens 8000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id shared_prefix_8k \
  --repeat-index 0
```

### 4K token

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario shared_prefix \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 10 \
  --prefix-chars 120000 \
  --max-input-tokens 4000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id shared_prefix_4k \
  --repeat-index 0
```

### 16K token

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario shared_prefix \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 10 \
  --prefix-chars 240000 \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 1 \
  --concurrency 1 \
  --run-group-id shared_prefix_16k \
  --repeat-index 0
```

## 9.6 mixed_pressure

### concurrency=4

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario mixed_pressure \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 20 \
  --mixed-short-chars 20000 \
  --mixed-medium-chars 80000 \
  --mixed-long-chars 160000 \
  --prefix-chars 120000 \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 4 \
  --run-group-id mixed_pressure_c4 \
  --repeat-index 0
```

### concurrency=1

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario mixed_pressure \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 20 \
  --mixed-short-chars 20000 \
  --mixed-medium-chars 80000 \
  --mixed-long-chars 160000 \
  --prefix-chars 120000 \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 1 \
  --run-group-id mixed_pressure_c1 \
  --repeat-index 0
```

### concurrency=2

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario mixed_pressure \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 20 \
  --mixed-short-chars 20000 \
  --mixed-medium-chars 80000 \
  --mixed-long-chars 160000 \
  --prefix-chars 120000 \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 2 \
  --run-group-id mixed_pressure_c2 \
  --repeat-index 0
```

### concurrency=8

```bash
python3 tests/vllm_baseline_runner.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario mixed_pressure \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-samples 20 \
  --mixed-short-chars 20000 \
  --mixed-medium-chars 80000 \
  --mixed-long-chars 160000 \
  --prefix-chars 120000 \
  --max-input-tokens 16000 \
  --max-tokens 128 \
  --warmup-samples 2 \
  --concurrency 8 \
  --run-group-id mixed_pressure_c8 \
  --repeat-index 0
```

## 10. 第四步：怎么做重复试验

正式 baseline 建议每个配置跑 3 次。

例如同一个配置：

- 第一次：`--repeat-index 0`
- 第二次：`--repeat-index 1`
- 第三次：`--repeat-index 2`

注意：

- `run_group_id` 要保持一样
- `sample-manifest` 对数据集场景要保持一样
- 不要一边重复一边换 prompt

### 最重要的简化规则

为了避免执行时太麻烦，直接记这一条就够了：

- `longbench` / `needlebench`：重复试验时必须继续带同一个 `--sample-manifest`
- `short_synth` / `ultra_long_synth` / `shared_prefix` / `mixed_pressure`：重复试验时**不需要**手动回填上一次 run 目录里的 `sample_manifest.json`

也就是说：

- 开源数据集场景：固定 manifest 是必须的
- synthetic 场景：只要参数不变，直接重复跑 3 次就行，通常不用额外加 `--sample-manifest`

### synthetic 场景怎么复用第一次生成的 prompt

这一段是“可选做法”，不是必须做法。

只有当你特别希望 synthetic 场景也严格复用第一次生成的 prompt 时，才需要这样做。

假设你第一次跑 `short_synth`，生成目录是：

- `baseline_runs/short_synth_1778000000/`

那第二次如果要严格复用同一批 prompt，可以加上：

```bash
--sample-manifest baseline_runs/short_synth_1778000000/sample_manifest.json
```

这对下面这些场景都适用：

- `short_synth`
- `ultra_long_synth`
- `shared_prefix`
- `mixed_pressure`

但再次强调：

- 这一步对 synthetic 场景不是默认必做
- 默认推荐做法仍然是：保持参数一致，只改 `--repeat-index`

## 11. 第五步：跑固定长度 microbenchmark

这一步保留旧版 `方案 B` 的结构，目的不是替代五组 baseline，而是补一条更标准的长度参考线。

### 11.1 2K 输入参考线

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model qwen3-8b \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-prompts 20 \
  --random-input-len 2048 \
  --random-output-len 128 \
  --ignore-eos \
  --temperature 0 \
  --request-rate 2
```

### 11.2 8K 输入参考线

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model qwen3-8b \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-prompts 20 \
  --random-input-len 8192 \
  --random-output-len 128 \
  --ignore-eos \
  --temperature 0 \
  --request-rate 1
```

### 11.3 16K 输入参考线

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model qwen3-8b \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --num-prompts 10 \
  --random-input-len 16384 \
  --random-output-len 128 \
  --ignore-eos \
  --temperature 0 \
  --request-rate 0.5
```

### 11.4 这一步跑完你要自己记录什么

从终端输出里手动记下：

- `Successful requests`
- `Failed requests`
- `Request throughput`
- `Output token throughput`
- `Mean TTFT`
- `P99 TTFT`
- `Mean TPOT`
- `P99 TPOT`

这一步通常不用马上发给 AI，先记到你自己的测试记录表里。

## 12. 第六步：全部跑完后统一汇总

所有场景都跑完之后，执行：

```bash
python3 tests/summarize_baseline_runs.py \
  --runs-dir baseline_runs \
  --output-csv baseline_runs_summary.csv
```

跑完后你会得到：

- `baseline_runs_summary.csv`

这张表就是后面统一分析和写报告最重要的输入之一。

## 13. 第七步：把结果补进测试记录文档

这一步是刻意模仿旧版 `baseline测试记录.md` 的做法。

建议你每完成一个正式场景，就补一小段记录。  
最少记录这些：

- 场景名
- 样本数
- 并发度
- prompt token 范围或均值
- success_rate
- mean / p95 latency
- kv_cache_usage_max
- num_requests_waiting_peak
- gpu_memory_used_mb_peak
- 结果目录

microbenchmark 则记录：

- 输入长度
- 输出长度
- 请求数
- req/s
- output tok/s
- Mean TTFT
- P99 TTFT
- Mean TPOT

如果你们还没有新的文档，可以直接新建一份 `baseline_v2测试记录.md`，沿用旧版记录风格。

## 14. 跑完结果以后你要做什么

每次单个 run 跑完，不用立刻交给 AI。

### 12.1 单个 run 跑完时，你自己先做这 3 件事

1. 看 `summary.json`
- 成功率是不是 `1.0`
- `prompt_tokens_mean` 有没有正常值
- `num_requests_waiting_peak` 有没有值

2. 看 `results.jsonl`
- 每条请求是否都有 `prompt_tokens`
- 是否有失败请求

3. 记下 run 目录
- 后面要重复试验、复用 manifest、写报告都会用到

### 12.2 什么情况下需要立刻发给 AI

只有下面这些情况，才建议你中途发给 AI：

- 命令报错，不知道怎么修
- `prompt_tokens` 是空的
- `summary.json` 缺关键字段
- 某个场景成功率不是 `100%`
- 你发现某组结果明显异常，比如：
  - token 很长，但 `kv_cache_usage_max` 非常低
  - `num_requests_waiting_peak` 一直是空，但你怀疑已经排队
  - 同一配置 3 次重复差异特别大

### 12.3 什么情况下不用每次都发给 AI

正常情况下：

- 每个单独 run
- 每个单独 repeat
- 每次 sanity check

都不需要发给 AI。

你先本地检查一遍即可。

### 12.4 什么时候最适合把结果给 AI

最推荐的时机有两个：

#### 时机 A：一个场景全部跑完后

比如：

- `longbench` 的 4 个配置都跑完了
- `mixed_pressure` 的几个并发档位都跑完了

这时候你可以把：

- 这一组的 `summary.json`
- 或者从 `baseline_runs_summary.csv` 里截出这几行

发给 AI，让 AI 帮你看：

- 结果是否合理
- 有没有异常值
- 是否需要补跑

#### 时机 B：全部 baseline 跑完后

这是最重要的一次。

这时候你把这些东西给 AI：

- `baseline_runs_summary.csv`
- 你整理好的测试记录
- 如果需要，再附上主报告草稿

让 AI 帮你做：

- 总结 baseline 结论
- 找出异常点
- 判断哪些结果能进主报告
- 帮你写最终测试报告

## 15. 最后交给 AI 的标准输入是什么

如果你要让 AI 帮你分析 baseline，最推荐发这 3 类东西：

### 第一优先

- `baseline_runs_summary.csv`

### 第二优先

- 某几个关键 run 的 `summary.json`

例如：

- `longbench`
- `ultra_long_synth`
- `shared_prefix`
- `mixed_pressure`

### 第三优先

- 你自己的简短说明

例如：

- 哪些是正式采用结果
- 哪些是 sanity check
- 哪些 run 你怀疑异常

## 16. 一键自动化执行（推荐方式）

从 v3 版本开始，所有步骤已集成到一键编排器中。你不再需要手动复制粘贴任何命令。

### 16.1 在 AutoDL 上第一次跑

```bash
# 1. 进入仓库
cd ~/xiyoulinux-kvcache-optimizer

# 2. 拉取最新代码
git pull

# 3. 确保依赖装好
pip install datasets transformers requests

# 4. 先预览会跑什么（不真跑）
bash run_baseline_v2.sh --dry-run

# 5. 快速验证（每配置只跑 1 次，跳过 sanity check，约 30 分钟）
bash run_baseline_v2.sh --repeat-count 1 --skip-sanity

# 6. 正式跑 baseline（每配置跑 3 次，约 2 小时）
bash run_baseline_v2.sh
```

### 16.2 常用命令

```bash
# 只跑 baseline
bash run_baseline_v2.sh --modes baseline

# 只跑 FlexKV（需要先装好 miniflex）
bash run_baseline_v2.sh --modes miniflex

# 两者都跑（先 baseline，再 miniflex，约 4 小时）
bash run_baseline_v2.sh --modes baseline,miniflex

# 只跑某几个场景
bash run_baseline_v2.sh --scenarios short_synth,shared_prefix

# 断点续跑（中间断了重新跑，自动跳过已完成的）
bash run_baseline_v2.sh --resume

# 跑完发钉钉通知
WEBHOOK_URL="https://oapi.dingtalk.com/robot/send?access_token=xxx" \
  bash run_baseline_v2.sh
```

### 16.3 脚本自动做了什么

```
Phase prepare    → 准备 longbench + needlebench 的 token 分桶 manifest
                  （不依赖 vLLM，先跑）

Phase A          → 自动杀掉旧 vLLM → 启动 vLLM（关前缀缓存）
                   → 等待 /v1/models 就绪
                   → 跑 sanity check（可选）
                   → 跑 short_synth / longbench / ultra_long / needlebench / mixed_pressure
                   → 自动杀掉 vLLM

Phase B          → 启动 vLLM（开前缀缓存）
                   → 等待就绪 → 跑 shared_prefix
                   → 留 vLLM 运行

Microbenchmark   → 跑 vllm bench serve 的 2K / 8K / 16K
                   → 自动杀掉 vLLM

汇总              → 自动执行 summarize_baseline_runs.py
                   → 生成 baseline_runs_summary.csv
```

### 16.4 你只需要做的事

1. 终端输入 `bash run_baseline_v2.sh`
2. 去干别的（约 2 小时）
3. 回来看 `baseline_runs_summary.csv`
4. 把 CSV 发给我分析

### 16.5 查看/恢复进度

```bash
# 查看进度文件
cat .baseline_v2_progress.json

# 删掉进度文件重新开始
rm .baseline_v2_progress.json

# 查看某个 run 的结果
cat baseline_runs/baseline_short_synth_1k_c1__r0_short_synth_*/summary.json | python3 -m json.tool
```

## 17. baseline 和 Miniflex 两种模式的区别

同一个编排脚本、同一套测试场景、同一批 prompt，**唯一的区别是 vLLM 的启动方式**。

### 17.1 两种模式逐项对比

| | baseline 模式 | miniflex 模式 |
|---|---|---|
| **命令** | `bash run_baseline_v2.sh` | `bash run_baseline_v2.sh --modes miniflex` |
| **vLLM 启动** | `vllm serve` 直接启动 | `bash run_vllm_miniflex.sh`（加载 Miniflex connector） |
| **Phase A 前缀缓存** | `--no-enable-prefix-caching` | Miniflex shell 脚本内部自带 `--no-enable-prefix-caching`，缓存命中全走 connector |
| **Phase B 前缀缓存** | 不加 `--no-enable-prefix-caching`（vLLM 原生开） | 同上（Miniflex 永远关原生缓存） |
| **--disable-hybrid-kv-cache-manager** | ✅ 已对齐（baseline 也传） | ✅ shell 脚本自带 |
| **--max-model-len** | ✅ 已对齐：49152（编排器统一传） | ✅ 通过 `MINIFLEX_MAX_MODEL_LEN` 环境变量传入 |
| **--enforce-eager** | baseline 不加（CUDA graph 开着） | miniflex 默认加（可通过 `MINIFLEX_ENFORCE_EAGER=0` 关闭） |
| **--gpu-memory-utilization** | vLLM 默认 0.90 | 0.55（硬编码，设计意图） |
| **runner 的 --mode** | `--mode baseline` | `--mode miniflex` |
| **结果目录名** | `baseline_short_synth_1k_c1__r0_...` | `miniflex_short_synth_1k_c1__r0_...` |
| **汇总 CSV mode 列** | `baseline` | `miniflex` |
| **缓存预热** | Phase B 开头的 `shared_prefix_cache_warmup` 步骤 | 同上 |
| **Prompt** | 同一批 manifest（3 个 LongBench 子集 + NeedleBench） | 同一批 manifest（完全一样） |
| **Telemetry 采集** | Prometheus + GPU + CPU | 同上 + `/tmp/miniflex_metrics.json`（Miniflex 专属埋点） |

### 17.2 对比的意义

跑完两轮后，你能直接回答：

- 同一个场景、同一批 prompt，Miniflex 的 TTFT / latency 比 baseline 快多少
- KV cache 命中后 prefill 省了多少时间
- Miniflex 的 CPU/SSD 搬运开销具体有多大
- GPU 显存是否比 baseline 更空闲（因为 KV 被迁走了）

### 17.3 前提条件

- miniflex 已安装（`cd miniflex && pip install --no-build-isolation .`）
- 系统依赖已装（`sudo apt install liburing-dev`）
- （可选）SSD 缓存目录已设置（`MINIFLEX_ENABLE_SSD=1 MINIFLEX_SSD_CACHE_DIR=/data/miniflex_cache`）

### 17.4 怎么跑

```bash
# 只跑 baseline（默认）
bash run_baseline_v2.sh

# 只跑 Miniflex
bash run_baseline_v2.sh --modes miniflex

# 对比模式：先跑 baseline，再重启 vLLM 加载 Miniflex 再跑一轮
bash run_baseline_v2.sh --modes baseline,miniflex

# Miniflex 模式关掉 --enforce-eager（测试 CUDA graph 兼容性）
MINIFLEX_ENFORCE_EAGER=0 bash run_baseline_v2.sh --modes miniflex
```

`--modes baseline,miniflex` 的执行流程：

1. 启动原始 vLLM（关缓存）→ 跑 Phase A 全部场景
2. 杀掉 vLLM → 启动原始 vLLM（开缓存）→ 跑 Phase B（shared_prefix）+ microbenchmark
3. 汇总 baseline 结果
4. **重新来**：启动 Miniflex → 跑 Phase A → Phase B → microbenchmark
5. 最终汇总 CSV，包含两组数据的 mode 列

Miniflex 启动时编排器自动传 `MINIFLEX_MAX_MODEL_LEN=32768`（可通过 `--miniflex-max-model-len` 调整），确保 32K 测试不会被 shell 脚本里写死的 2048 掐断。

## 18. 结果文件结构

```
baseline_runs_summary.csv          ← 最终汇总表，一行一个 run

baseline_runs/
  baseline_short_synth_1k_c1__r0_short_synth_1717678900/
    sample_manifest.json           ← 可复用的 prompt
    config.json                    ← 完整参数快照
    results.jsonl                  ← 每条请求的 TTFT/TPOT/ITL/Latency/tokens
    summary.json                   ← 该次 run 的统计汇总
    telemetry.jsonl                ← Prometheus + GPU + CPU 时间序列
    telemetry_before.json          ← 测试前快照
    telemetry_after.json           ← 测试后快照
  ...
```

## 19. 一句话原则

你不需要每次都手动敲 67 条命令。

```bash
bash run_baseline_v2.sh    # 跑完去干别的，回来拿结果
```

拿到 CSV 后再找我分析。

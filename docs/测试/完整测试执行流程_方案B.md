# 完整测试执行流程（方案 B）

## 1. 文档目的

本文档用于给当前阶段的测试工作提供一套完整、可执行、可复现的操作流程。  
这里采用的是**方案 B**：

- 保留“原始 vLLM 纯性能参考线”作为补充
- 但为了保证记录格式统一，重新按统一流程补跑
- 所有结果尽量沉淀为结构化文件，便于后续汇总和写报告

本文档适用于当前云服务器环境、当前模型和当前测试脚本。

## 2. 当前测试目标

本轮测试目标不是验证最终版 KVCache/offload 优化，而是建立一套完整的原始 vLLM 基线数据，覆盖以下内容：

1. 五组比赛场景 baseline
2. 外部可观测 baseline
3. 补充性的纯性能参考线
4. 统一格式的结果输出

最终应形成两类结果：

- `vllm bench serve` 输出的固定长度性能参考线
- `/root/autodl-tmp/1.py` 输出的结构化场景测试结果

## 3. 当前测试框架

当前测试框架由四部分组成：

### 3.1 推理服务层

- `vllm serve`
- 提供 `/v1/completions`
- 提供 `/metrics`

### 3.2 benchmark 层

- `vllm bench serve`
- 用于固定输入长度的性能参考线

### 3.3 场景测试与监控采集层

- 云端实际执行脚本：`/root/autodl-tmp/1.py`
- 用于跑五组测试场景
- 自动采集 `/metrics`、GPU、CPU 资源指标

说明：

- `vllm bench serve` 负责官方 benchmark 输出
- 自定义脚本负责五组 baseline 场景与额外可观测指标采集
- 两者不是二选一，而是分工不同

### 3.4 结果汇总层

- 云端实际执行脚本：`/root/autodl-tmp/summarize_baseline_runs.py`
- 用于把多组 `summary.json` 汇总成 CSV

## 4. 本轮测试前提

开始测试前，确认以下条件成立：

### 4.1 模型与服务

- 模型已在云服务器本地下载完成
- 当前模型路径正确
- `vllm serve` 可以稳定启动
- `/v1/models` 可访问
- `/metrics` 可访问

### 4.2 推荐固定参数

建议本轮测试固定以下基础条件：

- 模型：`Qwen3-8B`
- 服务名：`qwen3-8b`
- tokenizer 路径与模型路径一致
- `vLLM` 版本固定
- 推理参数不要在测试过程中随意改动

### 4.3 Python 依赖

如果需要 LongBench 场景，建议确认：

```bash
python3 -c "import datasets; print('ok')"
```

如果未安装，可执行：

```bash
pip install datasets
```

## 5. 测试前准备

## 5.1 启动原始 vLLM 服务

参考当前你已跑通的方式：

```bash
vllm serve /root/autodl-tmp/Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

### 5.2 检查服务是否 ready

```bash
curl http://127.0.0.1:8000/v1/models
```

```bash
curl -s http://127.0.0.1:8000/metrics | rg "kv_cache_usage_perc|prefix_cache|num_requests"
```

如果这两步都正常，再继续下面的测试。

## 6. 本轮完整测试顺序

方案 B 建议按以下顺序执行：

1. 五组外部可观测 baseline
2. 原始 vLLM 纯性能参考线
3. 汇总结果
4. 整理测试记录表

这样做的好处是：

- 先完成比赛场景主线
- 再补一条更干净的长度 microbenchmark 参考线
- 最后统一整理，避免中间来回打断

## 7. 第一步：五组比赛场景 baseline

这一部分统一使用云端脚本 `/root/autodl-tmp/1.py`。

这一阶段的结果会自动保存：

- 请求结果
- `/metrics`
- GPU 指标
- CPU 内存指标
- summary

## 7.1 短上下文基础组

目标：

- 验证基础功能
- 验证监控链路
- 观察轻负载开销

执行命令：

```bash
python3 /root/autodl-tmp/1.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario short_synth \
  --num-samples 10 \
  --prompt-chars 4000 \
  --max-tokens 128
```

## 7.2 中长上下文主测试组

目标：

- 验证真实长上下文场景 baseline
- 观察延迟、资源与 KV cache 使用变化

优先做法：

先在实例中准备本地 LongBench 数据。推荐方式：

```bash
export HF_ENDPOINT=https://hf-mirror.com
wget https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data.zip
mkdir -p /root/autodl-tmp/longbench_local
unzip -o data.zip -d /root/autodl-tmp/longbench_local
```

然后使用本地 JSONL 路径运行：

```bash
python3 /root/autodl-tmp/1.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario longbench \
  --subset qasper \
  --longbench-local-dir /root/autodl-tmp/longbench_local \
  --num-samples 10 \
  --max-tokens 128
```

说明：

- `qasper` 作为当前脚本里的默认长文 QA 子集入口
- 如果后续需要，也可以换成其他更合适的 LongBench 长文 QA / 多文档 QA 子集
- 当前更推荐本地 JSONL 方式，因为它更稳定，不依赖运行时网络
- 如果本地数据未准备好，才退回 Hugging Face 在线加载方式

## 7.3 超长上下文极限组

目标：

- 验证超长输入下的资源压力
- 观察更长上下文下的延迟退化
- 优先拿到真正能形成超长上下文压力的结果

优先执行命令：

```bash
python3 /root/autodl-tmp/1.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario needlebench \
  --needle-subset multi_needle_reasoning_needle \
  --num-samples 5 \
  --max-tokens 128
```

说明：

- 当前命令先给出一个可直接运行的 NeedleBench 子集入口
- 如果你后面确认了更合适的 retrieval 风格子集，可以直接替换 `--needle-subset`

补充说明：

- NeedleBench 的价值在于“真实开源超长数据接入”
- 但如果抽到的样本 prompt 很短，就不足以作为超长上下文极限组主证据
- 因此当前阶段建议：
  - NeedleBench 用于验证真实开源数据链路
  - `ultra_long_synth` 用于稳定构造超长上下文压力

当前阶段的实际决策逻辑是：

- 先优先尝试 `NeedleBench`
- 如果当前选择的子集未形成真正长 prompt，则不强行把它当主结果
- 改由 `ultra_long_synth` 生成稳定超长输入，作为当前阶段超长上下文极限组主结果

如果 `NeedleBench` 下载或运行不稳定，再使用合成超长文本兜底：

```bash
python3 /root/autodl-tmp/1.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario ultra_long_synth \
  --num-samples 5 \
  --ultra-prompt-chars 22000 \
  --max-tokens 128
```

如果服务压力过大，可先把：

- `--num-samples` 降到 `3`
- 或把 `--ultra-prompt-chars` 调到 `18000`

当前阶段实际建议：

- 如果 NeedleBench 样本过短，只将其记为“接入验证成功”
- 以 `ultra_long_synth` 结果作为超长上下文极限组主结果

## 7.4 共享前缀命中组

目标：

- 观察 prefix 查询与命中
- 验证共享前缀场景下缓存行为可被观测

执行命令：

```bash
python3 /root/autodl-tmp/1.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario shared_prefix \
  --num-samples 10 \
  --prefix-chars 12000 \
  --max-tokens 128
```

## 7.5 混合压力组

目标：

- 观察不同长度请求混跑下的稳定性
- 观察资源竞争与负载压力

执行命令：

```bash
python3 /root/autodl-tmp/1.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-8b \
  --scenario mixed_pressure \
  --num-samples 20 \
  --concurrency 4 \
  --mixed-short-chars 2000 \
  --mixed-medium-chars 8000 \
  --mixed-long-chars 16000 \
  --prefix-chars 12000 \
  --max-tokens 128
```

如果压力太大，可先把：

- `--concurrency` 降到 `2`
- `--num-samples` 降到 `12`

## 8. 第二步：补充性的纯性能参考线

这一部分使用 `vllm bench serve`，目的在于获得一条最标准、最干净的固定长度 microbenchmark 参考线。

注意：

- 这一步使用随机生成输入
- 它的作用是观察长度增长带来的纯性能退化
- 它不是五组比赛场景 baseline 的主体
- 它只是一条补充参考线，用来帮助解释性能趋势
### 8.1 2K 输入参考线

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

### 8.2 8K 输入参考线

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

### 8.3 16K 输入参考线

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

### 8.4 这一阶段需要手动记录的指标

每一组至少记录：

- Successful requests
- Failed requests
- Request throughput
- Output token throughput
- Mean TTFT
- P99 TTFT
- Mean TPOT
- P99 TPOT

建议直接复制到你的记录表里。

## 9. 第三步：统一汇总测试结果

跑完上面的场景后，执行：

```bash
python3 /root/autodl-tmp/summarize_baseline_runs.py \
  --runs-dir /root/autodl-tmp/baseline_runs \
  --output-csv /root/autodl-tmp/baseline_runs_summary.csv
```

输出文件：

- `/root/autodl-tmp/baseline_runs_summary.csv`

这个 CSV 会汇总：

- 场景名
- 样本数
- 成功率
- 平均延迟
- P50 / P95 延迟
- completion tok/s
- `kv_cache_usage_max`
- `prefix_queries_delta`
- `prefix_hits_delta`
- GPU 显存峰值
- CPU 内存峰值
- 并发度

## 10. 每轮测试的输出文件说明

每个场景都会在 `baseline_runs/<run_id>/` 下生成：

- `config.json`
  - 本轮参数配置
- `results.jsonl`
  - 每条请求结果
- `telemetry_before.json`
  - 测试前快照
- `telemetry_after.json`
  - 测试后快照
- `telemetry.jsonl`
  - 测试过程中的轮询数据
- `summary.json`
  - 场景级汇总结果

## 11. 建议记录表结构

最终建议整理成两张表。

### 11.1 五组场景 baseline 表

| 场景 | 样本数 | 并发度 | 成功率 | 平均延迟 | P95 延迟 | completion tok/s | kv_cache_usage 峰值 | prefix_queries 增量 | prefix_hits 增量 | GPU 显存峰值 | CPU 内存峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|

这张表填 `baseline_runs_summary.csv` 里的结果。

说明：

- 这张表是当前 baseline 汇报的主体
- 五组里如果某个真实开源数据组暂时不稳定，可用文档说明“当前先用合成兜底”

### 11.2 固定长度 microbenchmark 表

| 模型 | 输入长度 | 输出长度 | 请求数 | RPS | 成功率 | req/s | output tok/s | Mean TTFT | P99 TTFT | Mean TPOT |
|---|---|---|---|---|---|---|---|---|---|---|

这张表填 `vllm bench serve` 的 2K / 8K / 16K 结果。

说明：

- 这张表是补充参考表
- 它帮助你说明“随着长度增长，原始系统纯性能如何退化”
- 但它不替代五组比赛场景 baseline

## 12. 失败与异常处理建议

如果某条命令失败，优先按下面顺序排查：

### 12.1 服务未准备好

先检查：

```bash
curl http://127.0.0.1:8000/v1/models
```

### 12.2 `/metrics` 无法访问

先检查：

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

### 12.3 LongBench 无法运行

原因通常是：

- 未安装 `datasets`
- 网络下载失败

这时先跳过 `longbench`，优先完成另外 4 组。

### 12.4 超长组压力过大

调整：

- `--num-samples`
- `--ultra-prompt-chars`
- `--concurrency`

不要一上来把服务压死。

## 13. 当前阶段的完成标准

如果本轮测试完成了以下内容，就可以认为“原始 vLLM 基线测试”已经比较完整：

1. 五组场景测试至少完成 4 组
2. 2K / 8K / 16K 固定长度参考线已完成
3. `baseline_runs_summary.csv` 已生成
4. 结果已经整理成两张表
5. 可以写出一段总结：

- 原始 vLLM 在长上下文下如何退化
- 共享前缀场景是否可观测
- 资源压力和 KV cache 使用趋势如何

## 14. 结论

方案 B 的核心不是推翻昨天的数据，而是把它升级成一套完整、统一、结构化的基线测试流程。

具体来说：

- 先用 `vllm_baseline_runner.py` 完成五组比赛场景 baseline
- 再用 `vllm bench serve` 补一条固定长度纯性能参考线
- 用 `summarize_baseline_runs.py` 汇总成表

这样你后续就可以持续按同一套流程重复测试，而不需要每次重新组织思路。

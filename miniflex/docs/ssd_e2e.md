# 真实 SSD 命中验证

`bench_ssd_e2e.py` 验证真实 vLLM 请求的完整
`CPU 淘汰 -> SSD 命中 -> DISK2H（CPU 回填）-> H2D（GPU 恢复）` 路径。

它不根据 TTFT 阈值猜测缓存命中，而是同时校验 SSD/CPU hit block、
matched token 和 miss 指标。请求使用 greedy 解码和 vLLM 的
`return_token_ids`，冷重算、SSD hit 与随后 CPU hit 的生成 token ID 也必须
完全一致。脚本同时输出三条路径的 TTFT 原始样本、p50 与 p95。

## 启动服务

先选择一个支持 O_DIRECT 且空间足够的本地 NVMe 目录。下面的容量关系适合
RTX 5090、Qwen3-8B、约 7–8k token 的四条独立 prompt：

```bash
MINIFLEX_ENABLE_SSD=1 \
MINIFLEX_NUM_CPU_BLOCKS=512 \
MINIFLEX_NUM_SSD_BLOCKS=2048 \
MINIFLEX_SSD_CACHE_DIR=/path/on/nvme/miniflex-cache \
MINIFLEX_USE_DIRECT_IO=1 \
MINIFLEX_EVICTION_POLICY=lru \
MINIFLEX_MAX_MODEL_LEN=16384 \
MODEL=Qwen/Qwen3-8B \
bash run_vllm_miniflex.sh
```

服务必须使用 MiniFlex connector，并保持 `--no-enable-prefix-caching`，避免
vLLM 原生 GPU prefix cache 掩盖外部缓存路径。

对于 Qwen3-8B 的默认 KV 结构，2048 个 SSD block 大约会预分配 4.5 GiB，
运行前应检查目标目录的可用空间。

## 运行验证

该测试已纳入 `demo.sh` 的第五幕。要在指定 NVMe 目录上一次运行完整演示与
SSD E2E，可以直接执行：

```bash
SSD_E2E_CACHE_DIR=/path/on/nvme/miniflex-cache \
SSD_E2E_USE_DIRECT_IO=1 \
PAUSE=0 bash demo.sh
```

如果只需要单独运行 SSD 验证，可以使用下面的 benchmark 命令。

另开终端执行：

```bash
cd miniflex
PYTHONPATH=pysrc python bench_ssd_e2e.py \
  --model qwen3-8b \
  --cpu-blocks 512 \
  --ssd-blocks 2048 \
  --body-repeat 250 \
  --num-prefixes 4 \
  --rounds 3 \
  --require-full-ssd-hit \
  --out /tmp/miniflex_ssd_e2e.json
```

`--cpu-blocks` 和 `--ssd-blocks` 是对服务配置的声明，必须和启动服务时的
环境变量一致。验证固定使用 `--max-tokens 1`，并假设测试期间没有其他请求
并发修改 MiniFlex 指标。脚本会通过 vLLM 的 `/tokenize` 接口取得真实 token
数，并在请求开始前检查：

- 单条 prompt 放得进 CPU pool，因为当前 `DISK2H` 需要 CPU staging；
- 全部 prompt 的工作集大于 CPU pool，能够触发 CPU 淘汰；
- 除目标外的压力 prompt 足以将目标完整淘汰，而不只是部分淘汰；
- 冷启动工作集放得进 SSD pool。

测试过程为：

1. 对所有独立 prompt 做冷计算，要求每条请求实际完成 H2DISK；
2. 依次请求压力 prompt，使目标 prompt 被 CPU LRU 完整淘汰；
3. 请求目标 prompt，要求全部可复用 block 来自 SSD，matched token 数完整；
4. 立即再次请求目标 prompt，要求全部 block 来自 CPU，证明 CPU 晋升成功；
5. 比较三条路径的非空生成 token ID，并保存机器可读的原始结果。

等待后台 H2DISK 完成时，脚本会发送普通 CPU-hit 请求，让项目原有的
`query_finished_tasks() -> try_wait() -> _wait_impl()` 路径处理 graph-complete。
它不要求 KVTaskEngine 新增公开 `poll()` 接口，也不改变生产调度逻辑。

默认指标文件为 `/tmp/miniflex_metrics.json`，因此 benchmark 应与 vLLM
服务运行在同一台机器。读取代码会重试短暂的 JSON 非原子写窗口，但测试期间
不要删除或重置该指标文件。

## 如何解释性能结果

SSD hit 比冷重算慢不会令功能验证失败。脚本会将其记录为
`recompute_faster_than_ssd`，这是一个有效的 crossover 观测，而不是传输错误。

是否值得从 SSD 恢复主要取决于：

- prompt 长度和模型 prefill 计算量；
- 每 token KV 大小；
- SSD 实际有效带宽及是否被其他任务争用；
- 当前串行 `DISK2H -> H2D` 的总搬运成本。

短 prompt 可能重算更快，长 prompt 通常更容易摊薄固定调度成本。可以改变
`--body-repeat` 重复测试多个上下文长度，寻找当前硬件上的 crossover。除了
单请求 TTFT，还应在后续性能实验中观察并发吞吐，因为 SSD 恢复即使 TTFT
接近重算，也可能减少 GPU prefill 计算和排队。

## 纯逻辑测试

不启动 vLLM 时，可以验证 benchmark 的 token/SSE 解析、block 对齐边界、
完整淘汰条件、部分命中拒绝、指标增量和统计逻辑：

```bash
cd miniflex
python test/bench_ssd_e2e_test.py
```


## 本机验证记录

2026-07-31 在以下环境完成了一次真实服务验证：

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU 8 GiB；
- 模型：Qwen1.5-0.5B-Chat；
- 软件：vLLM 0.21.0、torch 2.11.0+cu130；
- 缓存：CPU 256 blocks、SSD 1024 blocks、O_DIRECT；
- 工作集：4 条约 3.1k token 的独立 prompt，共 780 blocks；
- 测量：3 轮纯 SSD hit 及其后的 CPU hit，并比较生成 token ID。

结果：

| 路径 | TTFT p50 | TTFT p95 |
|---|---:|---:|
| 冷重算 | 153.3 ms | 166.2 ms |
| SSD hit | 120.7 ms | 120.8 ms |
| SSD 晋升后的 CPU hit | 50.4 ms | 51.0 ms |

三轮目标请求均精确记录 `SSD=195 blocks、CPU=0 blocks、miss=0` 和
`matched=3120 tokens`；紧接着的请求均为
`CPU=195 blocks、SSD=0 blocks、miss=0`。冷重算、三轮 SSD hit 和三轮 CPU hit
返回的非 EOS 生成签名均为 token ID `576`、文本 ` The`。在这个约 3.1k token
的点上，SSD 相对冷重算为 1.27x。

这组数据用于证明真实 SSD 生命周期和测试方法可行，不代表 Qwen3-8B/RTX 5090 的
最终比赛成绩。正式报告应在目标比赛机器上重新执行，并保留脚本输出的 JSON 原始数据。

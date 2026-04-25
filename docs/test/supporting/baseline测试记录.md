# baseline 测试记录

## 1. 记录目的

本文档用于持续记录当前原始 vLLM baseline 测试中的关键结果，便于后续：
- 与后续 KVCache/offload 版本做对比
- 回溯测试环境与关键参数

本记录当前包含两部分：

- 2026-04-24 的固定长度 microbenchmark 结果
- 2026-04-25 的服务启动与环境验证结果

## 2. 当前测试对象

- 模型：`Qwen3-8B`
- 模型路径：`/root/autodl-tmp/Qwen/Qwen3-8B`
- 服务名：`qwen3-8b`
- 推理框架：`vLLM 0.19.1`
- 推理设备：当前云服务器本地 GPU

说明：

- 当前服务是“本机部署、本机推理”的 vLLM OpenAI 兼容服务
- 不是外部第三方 API 调用

## 3. 2026-04-24 固定长度 microbenchmark 结果

来源：

- `vllm bench serve`
- `random` workload
- 固定输出长度 `128`

### 3.1 结果表

| 模型 | 输入长度 | 输出长度 | 请求数 | RPS | 成功率 | req/s | output tok/s | Mean TTFT | P99 TTFT | Mean TPOT |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 2048 | 128 | 20 | 2.0 | 100% | 1.73 | 220.95 | 291.88 ms | 621.65 ms | 15.66 ms |
| qwen3-8b | 8192 | 128 | 20 | 1.0 | 100% | 0.88 | 112.53 | 1178.67 ms | 2934.98 ms | 33.56 ms |
| qwen3-8b | 16384 | 128 | 10 | 0.5 | 100% | 0.41 | 52.38 | 3105.55 ms | 5877.08 ms | 48.97 ms |

### 3.2 当前结论

随着输入长度从 `2K` 增长到 `16K`：

- 请求吞吐从 `1.73 req/s` 下降到 `0.41 req/s`
- 输出 token 吞吐从 `220.95 tok/s` 下降到 `52.38 tok/s`
- Mean TTFT 从 `291.88 ms` 上升到 `3105.55 ms`
- Mean TPOT 从 `15.66 ms` 上升到 `48.97 ms`

这说明：

- 长上下文会显著拉低原始 vLLM 的吞吐
- Prefill/首 token 延迟随上下文长度增长明显恶化
- 这组数据可作为后续优化对照的固定长度性能参考线

说明：

- 该组结果属于补充性的 fixed-length microbenchmark
- 它不是五组比赛场景 baseline 的主体，但非常适合做趋势参考线

## 4. 2026-04-25 服务启动与环境验证结果

来源：

- `python vllm_model.py`
- `vllm serve /root/autodl-tmp/Qwen/Qwen3-8B ...`

### 4.1 本地 Python API 验证

验证结果：

- `python vllm_model.py` 成功运行
- 模型加载成功
- 成功生成回答

说明：

- `from vllm import LLM` 路径已验证可用
- 本地模型目录与 tokenizer 路径均有效

### 4.2 vLLM 服务启动关键数据

来源为 2026-04-25 20:11 左右的 `vllm serve` 启动日志。

关键数据如下：

| 项目 | 值 |
|---|---|
| 模型架构 | `Qwen3ForCausalLM` |
| 最大上下文长度 | `40960` |
| 模型加载显存占用 | `15.27 GiB` |
| 模型加载耗时 | `3.86 s` |
| torch.compile 总耗时 | `4.10 s` |
| Engine 初始化总耗时 | `10.74 s` |
| Available KV cache memory | `12.09 GiB` |
| GPU KV cache size | `88,064 tokens` |
| 40,960 token/request 时最大并发 | `2.15x` |

### 4.3 服务状态

服务已成功启动，关键路由已就绪，包括：

- `/metrics`
- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`

这说明：

- 原始 vLLM OpenAI 兼容服务已经 ready
- 可以开始运行五组 baseline 测试
- 可以开始采集 `/metrics` 与场景级 summary

## 5. 当前阶段总结

截至目前，已经确认：

1. 固定长度性能参考线已拿到
2. 模型本地加载正常
3. vLLM 服务已成功启动
4. `/metrics` 路由已开启
5. 当前环境可以继续执行五组 baseline 测试

## 6. 待补充数据

接下来需要继续补齐以下结果，并追加到本文档：

- 中长上下文主测试组
- 超长上下文极限组
- 共享前缀命中组
- 混合压力组
- `baseline_runs_summary.csv` 汇总结果

## 7. 2026-04-25 五组比赛场景 baseline 结果

### 7.1 短上下文基础组

运行方式：

- 场景：`short_synth`
- 样本数：`10`
- 并发度：`1`
- 平均 prompt 长度：约 `4068 chars`
- 输出长度：`128 tokens`

结果摘要：

| 指标 | 值 |
|---|---|
| success_count | `10` |
| failed_count | `0` |
| success_rate | `1.0` |
| mean_latency_sec | `1.4041` |
| p50_latency_sec | `1.3945` |
| p95_latency_sec | `1.4482` |
| max_latency_sec | `1.4914` |
| mean_completion_tokens | `128` |
| completion_tokens_total | `1280` |
| end_to_end_completion_toks_per_sec | `91.16` |
| kv_cache_usage_start | `0.0` |
| kv_cache_usage_end | `0.009449` |
| kv_cache_usage_max | `0.010176` |
| gpu_memory_used_mb_peak | `29876.0 MB` |
| gpu_util_pct_peak | `100.0%` |
| cpu_memory_used_mb_peak | `54085.32 MB` |
| cpu_memory_used_pct_peak | `7.0%` |
| loadavg_1m_peak | `13.025` |
| telemetry_samples | `13` |

当前结论：

- 短上下文基础组已稳定跑通
- 成功率为 `100%`
- 平均延迟约 `1.40 s`
- KV cache 占用很低，峰值约 `1%`
- 该组适合作为基础服务可用性与轻负载参考

结果目录：

- `baseline_runs/short_synth_1777119588`

### 7.2 中长上下文主测试组

运行方式：

- 场景：`longbench`
- 子集：`qasper`
- 数据来源：本地 LongBench JSONL
- 样本数：`10`
- 并发度：`1`
- prompt 长度范围：约 `10873` 到 `24000 chars`
- 输出长度：`128 tokens`

结果摘要：

| 指标 | 值 |
|---|---|
| success_count | `10` |
| failed_count | `0` |
| success_rate | `1.0` |
| mean_latency_sec | `1.7424` |
| p50_latency_sec | `1.7726` |
| p95_latency_sec | `1.8579` |
| max_latency_sec | `1.8758` |
| mean_completion_tokens | `128` |
| completion_tokens_total | `1280` |
| end_to_end_completion_toks_per_sec | `73.46` |
| kv_cache_usage_start | `0.0` |
| kv_cache_usage_end | `0.042886` |
| kv_cache_usage_max | `0.058514` |
| gpu_memory_used_mb_peak | `29876.0 MB` |
| gpu_util_pct_peak | `100.0%` |
| cpu_memory_used_mb_peak | `54579.93 MB` |
| cpu_memory_used_pct_peak | `7.06%` |
| loadavg_1m_peak | `13.072` |
| telemetry_samples | `16` |

当前结论：

- LongBench `qasper` 子集已稳定跑通
- 成功率为 `100%`
- 平均延迟约 `1.74 s`
- 相比短上下文基础组，KV cache 占用明显上升，峰值约 `5.85%`
- 该组可作为常规真实长上下文场景 baseline

结果目录：

- `baseline_runs/longbench_1777121802`

### 7.3 超长上下文极限组（主结果）

运行方式：

- 场景：`ultra_long_synth`
- 样本数：`5`
- 并发度：`1`
- prompt 长度：约 `22155 chars`
- 输出长度：`128 tokens`

结果摘要：

| 指标 | 值 |
|---|---|
| success_count | `5` |
| failed_count | `0` |
| success_rate | `1.0` |
| mean_latency_sec | `1.7237` |
| p50_latency_sec | `1.7232` |
| p95_latency_sec | `1.7289` |
| max_latency_sec | `1.7301` |
| mean_completion_tokens | `128` |
| completion_tokens_total | `640` |
| end_to_end_completion_toks_per_sec | `74.26` |
| kv_cache_usage_start | `0.0` |
| kv_cache_usage_end | `0.046702` |
| kv_cache_usage_max | `0.047429` |
| gpu_memory_used_mb_peak | `29876.0 MB` |
| gpu_util_pct_peak | `100.0%` |
| cpu_memory_used_mb_peak | `54093.75 MB` |
| cpu_memory_used_pct_peak | `7.0%` |
| loadavg_1m_peak | `15.086` |
| telemetry_samples | `8` |

当前结论：

- 超长上下文合成组已稳定跑通
- 成功率为 `100%`
- prompt 长度约 `22K chars`
- `kv_cache_usage_max` 约为 `4.74%`，明显高于短上下文组，也接近真实长上下文组的量级
- 该组可以作为当前阶段“超长上下文极限组”的主结果

结果目录：

- `baseline_runs/ultra_long_synth_1777122360`

### 7.4 开源超长数据补充结果（NeedleBench / en_haystack_texts）

运行方式：

- 场景：`needlebench`
- 子集：`en_haystack_texts`
- 样本数：`5`
- 并发度：`1`
- prompt 长度范围：约 `4680` 到 `24000 chars`
- 输出长度：`128 tokens`

结果摘要：

| 指标 | 值 |
|---|---|
| success_count | `5` |
| failed_count | `0` |
| success_rate | `1.0` |
| mean_latency_sec | `1.6348` |
| p50_latency_sec | `1.5439` |
| p95_latency_sec | `1.8889` |
| max_latency_sec | `1.8942` |
| mean_completion_tokens | `128` |
| completion_tokens_total | `640` |
| end_to_end_completion_toks_per_sec | `78.30` |
| kv_cache_usage_start | `0.0` |
| kv_cache_usage_end | `0.062693` |
| kv_cache_usage_max | `0.062693` |
| prefix_queries_delta | `15655.0` |
| prefix_hits_delta | `32.0` |
| gpu_memory_used_mb_peak | `29876.0 MB` |
| gpu_util_pct_peak | `100.0%` |
| cpu_memory_used_mb_peak | `54747.57 MB` |
| cpu_memory_used_pct_peak | `7.09%` |
| loadavg_1m_peak | `9.831` |
| telemetry_samples | `8` |

当前结论：

- `NeedleBench / en_haystack_texts` 已成功形成真实开源长文本输入
- 本组 `kv_cache_usage_max` 达到 `0.06269`
- 因此本组可以作为“开源超长长文本补充结果”
- 但当前阶段仍建议把 `ultra_long_synth` 作为超长上下文极限组主结果，因为其长度和压力更稳定可控

长度分布说明：

- 本组样本长度不完全均匀，并不单纯是谁“写错了”
- 主要由两部分共同造成：
  1. 开源数据本身长度分布不均
     - `en_haystack_texts` 不同样本的原始文本长度天然不同
  2. 脚本设置了 `--max-input-chars 24000`
     - 过长样本会被截断到 `24000 chars`
     - 较短样本则保留原始长度

因此：

- 出现一部分样本是 `24000 chars`
- 另一部分只有 `4680`、`6189`、`10381 chars`

这是“数据分布 + 截断策略”共同作用的结果，不是单一脚本 bug。

结果目录：

- `baseline_runs/needlebench_1777126408`

### 7.5 共享前缀命中组

运行方式：

- 场景：`shared_prefix`
- 样本数：`10`
- 并发度：`1`
- 共享前缀长度参数：`--prefix-chars 12000`
- prompt 长度范围：约 `12158` 到 `12171 chars`
- 输出长度：`128 tokens`

结果摘要：

| 指标 | 值 |
|---|---|
| success_count | `10` |
| failed_count | `0` |
| success_rate | `1.0` |
| mean_latency_sec | `1.3868` |
| p50_latency_sec | `1.3697` |
| p95_latency_sec | `1.4671` |
| max_latency_sec | `1.5434` |
| mean_completion_tokens | `128` |
| completion_tokens_total | `1280` |
| end_to_end_completion_toks_per_sec | `92.30` |
| kv_cache_usage_start | `0.0` |
| kv_cache_usage_end | `0.024896` |
| kv_cache_usage_max | `0.025259` |
| gpu_memory_used_mb_peak | `29876.0 MB` |
| gpu_util_pct_peak | `100.0%` |
| cpu_memory_used_mb_peak | `54104.99 MB` |
| cpu_memory_used_pct_peak | `7.0%` |
| loadavg_1m_peak | `18.63` |
| telemetry_samples | `13` |

当前结论：

- 共享前缀组已稳定跑通
- 成功率为 `100%`
- 该组已经构造出约 `12K chars` 的共享前缀输入
- `prefix_queries_delta = 21002.0`
- `prefix_hits_delta = 20960.0`
- 说明共享前缀场景下 prefix cache 查询与命中均已实际发生
- 命中数与查询数非常接近，说明当前场景中前缀复用效果明显

结果目录：

- `baseline_runs/shared_prefix_1777124420`

### 7.6 混合压力组

运行方式：

- 场景：`mixed_pressure`
- 样本数：`20`
- 并发度：`4`
- 短样本参数：`--mixed-short-chars 2000`
- 中样本参数：`--mixed-medium-chars 8000`
- 长样本参数：`--mixed-long-chars 16000`
- 共享前缀参数：`--prefix-chars 12000`
- prompt 长度范围：约 `2069` 到 `16098 chars`
- 输出长度：`128 tokens`

结果摘要（以补跑后的有效版本为准）：

| 指标 | 值 |
|---|---|
| success_count | `20` |
| failed_count | `0` |
| success_rate | `1.0` |
| mean_latency_sec | `1.5227` |
| p50_latency_sec | `1.5221` |
| p95_latency_sec | `1.5741` |
| max_latency_sec | `1.5761` |
| mean_completion_tokens | `128` |
| completion_tokens_total | `2560` |
| end_to_end_completion_toks_per_sec | `84.06` |
| kv_cache_usage_start | `0.0` |
| kv_cache_usage_end | `0.080683` |
| kv_cache_usage_max | `0.083772` |
| prefix_queries_delta | `34311.0` |
| prefix_hits_delta | `34208.0` |
| gpu_memory_used_mb_peak | `29876.0 MB` |
| gpu_util_pct_peak | `100.0%` |
| cpu_memory_used_mb_peak | `54708.55 MB` |
| cpu_memory_used_pct_peak | `7.08%` |
| loadavg_1m_peak | `9.964` |
| telemetry_samples | `7` |

当前结论：

- 混合压力组已稳定跑通
- 成功率为 `100%`
- 并发度 `4` 下，系统仍能稳定完成不同长度请求混跑
- `kv_cache_usage_max` 达到 `8.38%`，是当前已完成组里最高的一组
- `prefix_queries_delta = 34311.0`
- `prefix_hits_delta = 34208.0`
- 说明混合压力组中包含的 shared prefix bucket 也已经观测到明显的 prefix 查询与命中
- 该组可以作为当前阶段资源竞争与混合负载稳定性的主结果

结果目录：

- `baseline_runs/mixed_pressure_1777127138`

## 8. 当前阶段总总结

截至当前，原始 vLLM 的五组 baseline 主体已经完成，并已生成统一汇总表：

- 汇总文件：`baseline_runs_summary.csv`

### 8.1 整体完成情况

本轮已完成的主体场景包括：

- 短上下文基础组：`short_synth`
- 中长上下文主测试组：`longbench`
- 超长上下文极限组主结果：`ultra_long_synth`
- 共享前缀命中组：`shared_prefix`
- 混合压力组：`mixed_pressure`

开源补充完成的场景：

- `needlebench / en_haystack_texts`

### 8.2 当前阶段整体结论

1. 原始 vLLM 在五组 baseline 场景下均可稳定运行

- 当前所有已完成主体场景成功率均为 `100%`
- 说明当前服务部署、请求执行和外部采集链路总体稳定

2. 长上下文和混合负载会显著提高 KV cache 占用

- 短上下文基础组 `kv_cache_usage_max` 约为 `0.0102`
- 共享前缀组 `kv_cache_usage_max` 约为 `0.0253`
- 超长上下文合成组 `kv_cache_usage_max` 约为 `0.0474`
- LongBench 真实长文本组 `kv_cache_usage_max` 约为 `0.0585`
- 混合压力组 `kv_cache_usage_max` 约为 `0.0836`

这说明：

- 随着场景复杂度和负载混合程度增加，KV cache 压力明显上升
- 其中混合压力组对缓存占用的拉升最明显

3. 中长上下文与超长上下文场景已形成可对比的基线

- LongBench `qasper` 子集已经提供真实长文本基线
- `ultra_long_synth` 提供稳定可控的超长上下文主结果

因此当前阶段已经能够同时回答：

- 真实长文本场景下系统表现如何
- 超长上下文压力下系统表现如何

4. 共享前缀场景已构造成功，且 prefix 指标现已验证有效

- `shared_prefix` 组已稳定跑通
- 修正脚本后，当前已得到：
  - `prefix_queries_delta = 21002.0`
  - `prefix_hits_delta = 20960.0`

这说明：

- 场景构造本身没有问题
- prefix cache 查询与命中确实已经发生
- 共享前缀组可以作为当前阶段缓存复用效果的有效证据

5. 开源超长长文本补充结果已补齐

- `NeedleBench / en_haystack_texts`
  - 已形成真实开源长文本输入
  - 可作为开源超长长文本补充结果

因此当前阶段的结果组合为：

- `ultra_long_synth` 作为超长上下文极限组主结果
- `NeedleBench / en_haystack_texts` 作为开源超长补充结果

### 8.3 对后续工作的意义

本轮结果已经足以支撑后续两项工作：

1. 作为原始 vLLM 对照组

- 后续 KVCache/offload 版本可直接与当前五组结果对比

2. 作为比赛阶段汇报材料

- 当前已经可以较完整地说明：
  - baseline 如何设计
  - 五组场景如何覆盖比赛目标
  - 长上下文和混合负载如何提升缓存压力
  - 当前系统哪些环节已经稳定，哪些指标还需进一步完善

## 9. 最终采用结果总表

下表用于明确当前阶段最终保留哪些结果，哪些结果作为主结果，哪些结果仅作为补充。

| 分组 | 最终采用结果 | 关键说明 | success_rate | mean_latency_sec | kv_cache_usage_max | prefix_queries_delta | prefix_hits_delta |
|---|---|---|---|---|---|---|---|
| 短上下文基础组 | `short_synth_1777119588` | 基础功能与轻负载参考 | 1.0 | 1.4041 | 0.010176 | - | - |
| 中长上下文主测试组 | `longbench_1777121802` | LongBench `qasper`，本地 JSONL | 1.0 | 1.7424 | 0.058514 | - | - |
| 超长上下文极限组 | `ultra_long_synth_1777122360` | 当前阶段主结果，稳定构造约 `22K chars` 超长输入 | 1.0 | 1.7237 | 0.047429 | - | - |
| 共享前缀命中组 | `shared_prefix_1777124420` | 修正脚本后有效结果 | 1.0 | 1.3681 | 0.025259 | 21002.0 | 20960.0 |
| 混合压力组 | `mixed_pressure_1777127138` | 修正脚本后有效结果，并发 `4` | 1.0 | 1.5227 | 0.083772 | 34311.0 | 34208.0 |
| 开源超长补充结果 | `needlebench_1777126408` | `NeedleBench / en_haystack_texts`，开源超长长文本补充 | 1.0 | 1.6348 | 0.062693 | 15655.0 | 32.0 |

### 9.1 当前最终口径

当前阶段建议采用以下口径进行汇报：

- 五组 baseline 主体已经完成
- 所有最终采用结果成功率均为 `100%`
- `mixed_pressure` 组的 `kv_cache_usage_max` 最高，达到 `0.083772`
- `shared_prefix` 组已经验证 prefix cache 查询与命中均确实发生
- `ultra_long_synth` 作为当前阶段超长上下文极限组主结果
- `NeedleBench / en_haystack_texts` 作为开源超长长文本补充结果

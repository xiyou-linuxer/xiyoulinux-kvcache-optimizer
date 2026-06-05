# MiniFlexKV 验证报告

记录 MiniFlexKV vLLM connector 的正确性 / 性能验证结论与复现方法。

## 测试环境
- vLLM 0.21.0，单卡 NVIDIA RTX 4060 Laptop（8GB），模型 Qwen1.5-0.5B-Chat。
- 启动参数：`--disable-hybrid-kv-cache-manager`，验证缓存命中时另加 `--no-enable-prefix-caching`
  以隔离 vLLM 自带的显存前缀缓存，只测 MiniFlex 自身贡献。

## 1. 正确性 ✅

| 检查项 | 结论 |
|---|---|
| 单元测试 | `test/vllm_v1_adapter_test.py` **13/13 通过**（含真实 GPU↔CPU 传输逐字节校验） |
| KV 传输 | PUT 把 GPU block 存到 CPU、GET 读回，`torch.equal` **逐字节相等** |
| 真机命中 | 长 prompt 复发后 `GET_MATCH matched>0`；vLLM 指标 `External prefix cache hit rate` 由 0% 升到 **30.2%**，与命中 token 数吻合 |
| 输出一致性 | 命中后输出与冷算**不逐字相同**，但这是**良性浮点差异**——原生 vLLM 用自己的前缀缓存时冷/热也不一致（chunked-prefill 的浮点累加顺序不同）。单测已证明 KV 字节本身无误 |

### 行为说明：异步 commit "慢一拍"
PUT 的最终 commit 只在引擎"有请求在 step"时推进。因此**空闲后立刻重复同一请求**，
其 GET 可能早于上一个 PUT 的 commit 而 miss；**连续流量下会自愈**（中间的请求会
把引擎 step 起来、把之前的 PUT 落地）。这是异步设计的固有取舍，不是 bug。
低频场景若要稳定命中，可用 `MINIFLEX_SYNC_GET=1`（牺牲性能，见 usage.md）。

## 2. 性能

收益 = 省下的 prefill 重算 − KV 搬运成本。两者随规模增长速度不同，因此**收益随
模型变大、上下文变长而放大**。

### 实测（0.5B / 8GB 笔记本，对缓存最不利）
冷 = 全新 prompt 全量重算；热 = 预热到确认命中后从 CPU 加载。TTFT 中位数：

| 缓存前缀长度 | 冷 | 热 | 加速 |
|---|---|---|---|
| ~300 token | 30.4 ms | 31.6 ms | 持平（搬运 ≈ 重算） |
| ~416 token | 30.4 ms | 25.5 ms | **1.19x（省 16%）** |

这台机器的 TTFT 被固定开销（HTTP/调度/采样/首 token）主导，真正的 prefill 计算只占
几 ms，所以即便缓存把 prefill 砍到 0，上限也就 ~15–20%。**16% 是最差值，不是上限。**

### 理论外推（7B + 8K 上下文，A100 + PCIe Gen4）
按 FLOPs 与 KV 带宽估算（数量级）：

| KV 结构 | 重算 prefill | MiniFlex 加载 | 加速 |
|---|---|---|---|
| MHA（Llama-2-7B，KV 4GB） | ~1070 ms | ~220 ms | **~5x** |
| GQA-8（Mistral-7B，1GB） | ~1070 ms | ~70 ms | ~15x |
| GQA-4（Qwen2-7B，0.5GB） | ~1070 ms | ~45 ms | ~24x |

模型越大、上下文越长，prefill 计算量随 `模型规模×上下文` 涨，而 KV 搬运只随
`上下文×KV大小` 涨且是带宽瓶颈——所以收益从 16% 一路放大到数倍。

## 3. 稳定性（并发压测）✅

90s 持续压测，32 并发，`--max-num-seqs 8` + `--no-enable-prefix-caching`
（制造调度压力、让复用全走外部缓存）：

| 指标 | 结果 |
|---|---|
| 完成请求 | 14020（**0 失败 / 0 超时**） |
| 延迟 | p50 213ms / p95 289ms / max 417ms（无长尾） |
| 显存 | start 5501 → end 5522 MiB（**无泄漏**） |
| 外部缓存命中率 | ~76–78%（确认压力确实打在缓存路径上） |

重点：高并发 + 高命中率（大量请求走异步 GET）下**没有出现请求挂死**——
"只剩等待远程 KV、无其他可调度 token" 的边界场景被 sentinel stats 机制正确处理。

## 4. 复现方法

启动（debug 模式可看 PUT/GET 匹配日志）：
```bash
MINIFLEX_DEBUG=1 bash run_vllm_miniflex.sh
```

冷/热 TTFT 基准（`--body-repeat` 越大上下文越长、收益越明显）：
```bash
PYTHONPATH=pysrc python bench_ttft.py --body-repeat 14 --runs 6
```
> 基准必须把"热"prompt **预热到确认命中**再计时（脚本内已做 + 插 pump 请求推进引擎），
> 否则量到的全是 miss，会误判成"没用"。debug 日志里热请求应出现 `GET_MATCH matched>0`。

单元测试：
```bash
PYTHONPATH=pysrc python test/vllm_v1_adapter_test.py
```

## 5. 已知限制 / 未覆盖
- `Request` 字段假设（`all_token_ids` / `num_tokens` / `scheduled_cached_reqs.req_ids` /
  preemption 回退分支）只在 vLLM 0.21 上验证过，跨版本偏脆。
- `MINIFLEX_SYNC_GET` 已修正 `finished_recving` 上报（带单测），但仅在单元测试下验证，
  未做真机长跑。
- 单卡 90s 压测无挂死/泄漏，但更长时间（数小时）的长跑稳定性未测。

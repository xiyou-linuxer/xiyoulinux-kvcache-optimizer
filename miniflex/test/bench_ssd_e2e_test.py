"""bench_ssd_e2e 的纯逻辑测试，不需要启动 vLLM 或访问 GPU/SSD。"""

from pathlib import Path
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_ssd_e2e as bench


def assert_raises(exc_type, fn):
  try:
    fn()
  except exc_type:
    return
  raise AssertionError(f"expected {exc_type.__name__}")


def test_parse_token_count_supports_vllm_response_variants():
  assert bench.parse_token_count({"count": 3}) == 3
  assert bench.parse_token_count({"tokens": [11, 12, 13, 14]}) == 4
  assert_raises(
    bench.VerificationError,
    lambda: bench.parse_token_count({"token_strs": ["a"]}),
  )


def test_parse_stream_event_collects_text_and_token_ids():
  assert bench.parse_stream_event(b"event: ping") is None
  assert bench.parse_stream_event(b"data: [DONE]") is None
  assert bench.parse_stream_event(
    b'data: {"choices": ['
    b'{"text": "hello", "token_ids": [7]}, '
    b'{"text": " world", "token_ids": [8]}]}'
  ) == ("hello world", (7, 8))
  assert bench.parse_stream_event(
    b'data: {"usage": {"completion_tokens": 1}}'
  ) == ("", ())
  assert_raises(
    bench.VerificationError,
    lambda: bench.parse_stream_event(b"data: not-json"),
  )


def test_metric_delta_handles_missing_counters_and_rejects_reset():
  assert bench.metric_delta({}, {}, "counter") == 0
  assert bench.metric_delta({"counter": 2}, {"counter": 5}, "counter") == 3
  assert_raises(
    bench.VerificationError,
    lambda: bench.metric_delta({"counter": 5}, {"counter": 1}, "counter"),
  )


def test_missing_metrics_file_is_an_empty_baseline():
  with tempfile.TemporaryDirectory() as directory:
    missing = Path(directory) / "metrics.json"
    assert bench.read_metrics(missing, timeout=0) == {}


def test_cached_and_put_block_calculation():
  assert bench.cached_prompt_blocks(0, 16) == 0
  assert bench.cached_prompt_blocks(15, 16) == 0
  assert bench.cached_prompt_blocks(16, 16) == 1
  assert bench.cached_prompt_blocks(17, 16) == 1
  assert bench.cached_prompt_blocks(32, 16) == 2
  assert bench.maximum_put_blocks(31, 1, 16) == 1
  assert bench.maximum_put_blocks(32, 1, 16) == 2

def test_warmup_must_not_create_cacheable_blocks():
  bench.validate_uncacheable_warmup(15, 1, 16)
  assert_raises(
    bench.VerificationError,
    lambda: bench.validate_uncacheable_warmup(16, 1, 16),
  )


def test_capacity_accepts_single_fit_cpu_overflow_and_ssd_fit():
  result = bench.validate_capacity(
    token_counts=[3000, 3001, 3002, 3003],
    block_size=16,
    max_tokens=1,
    cpu_blocks=256,
    ssd_blocks=1024,
  )
  assert max(result["estimated_put_blocks"]) <= 256
  assert result["estimated_working_set_blocks"] > 256
  assert result["estimated_working_set_blocks"] <= 1024


def test_capacity_rejects_non_evicting_working_set():
  assert_raises(
    bench.VerificationError,
    lambda: bench.validate_capacity(
      token_counts=[100, 100, 100],
      block_size=16,
      max_tokens=1,
      cpu_blocks=64,
      ssd_blocks=128,
    ),
  )


def test_capacity_rejects_pressure_that_cannot_fully_evict_target():
  assert_raises(
    bench.VerificationError,
    lambda: bench.validate_capacity(
      token_counts=[1600, 800, 800],
      block_size=16,
      max_tokens=1,
      cpu_blocks=128,
      ssd_blocks=256,
    ),
  )


def test_capacity_rejects_non_single_token_generation():
  assert_raises(
    bench.VerificationError,
    lambda: bench.validate_capacity(
      token_counts=[3000, 3001, 3002, 3003],
      block_size=16,
      max_tokens=2,
      cpu_blocks=256,
      ssd_blocks=1024,
    ),
  )


def test_capacity_rejects_prefix_larger_than_cpu_staging():
  assert_raises(
    bench.VerificationError,
    lambda: bench.validate_capacity(
      token_counts=[5000, 5000, 5000],
      block_size=16,
      max_tokens=1,
      cpu_blocks=256,
      ssd_blocks=1024,
    ),
  )


def test_capacity_rejects_working_set_larger_than_ssd():
  assert_raises(
    bench.VerificationError,
    lambda: bench.validate_capacity(
      token_counts=[3000, 3000, 3000, 3000],
      block_size=16,
      max_tokens=1,
      cpu_blocks=256,
      ssd_blocks=512,
    ),
  )


def test_verify_complete_hit_rejects_partial_or_wrong_tier():
  full_ssd = {
    bench.CPU_HIT_BLOCKS: 0,
    bench.SSD_HIT_BLOCKS: 2,
    bench.GET_MATCHED_TOKENS: 32,
    bench.GET_MISS_BLOCKS: 0,
  }
  bench.verify_complete_hit(full_ssd, 2, 16, required_tier="ssd")

  partial = dict(full_ssd)
  partial[bench.SSD_HIT_BLOCKS] = 1
  partial[bench.GET_MATCHED_TOKENS] = 16
  assert_raises(
    bench.VerificationError,
    lambda: bench.verify_complete_hit(partial, 2, 16, required_tier="ssd"),
  )

  wrong_matched_tokens = dict(full_ssd)
  wrong_matched_tokens[bench.GET_MATCHED_TOKENS] = 16
  assert_raises(
    bench.VerificationError,
    lambda: bench.verify_complete_hit(
      wrong_matched_tokens, 2, 16, required_tier="ssd"
    ),
  )

  unexpected_miss = dict(full_ssd)
  unexpected_miss[bench.GET_MISS_BLOCKS] = 1
  assert_raises(
    bench.VerificationError,
    lambda: bench.verify_complete_hit(
      unexpected_miss, 2, 16, required_tier="ssd"
    ),
  )

  mixed = dict(full_ssd)
  mixed[bench.CPU_HIT_BLOCKS] = 1
  mixed[bench.SSD_HIT_BLOCKS] = 1
  bench.verify_complete_hit(mixed, 2, 16)
  assert_raises(
    bench.VerificationError,
    lambda: bench.verify_complete_hit(mixed, 2, 16, required_tier="ssd"),
  )

  full_cpu = dict(full_ssd)
  full_cpu[bench.CPU_HIT_BLOCKS] = 2
  full_cpu[bench.SSD_HIT_BLOCKS] = 0
  bench.verify_complete_hit(full_cpu, 2, 16, required_tier="cpu")


def test_only_first_round_requires_a_pure_ssd_hit():
  assert bench.required_ssd_tier(True, 0) == "ssd"
  assert bench.required_ssd_tier(True, 1) is None
  assert bench.required_ssd_tier(True, 2) is None
  assert bench.required_ssd_tier(False, 0) is None


def test_timing_summary_reports_median_and_p95():
  summary = bench.summarize_timings([
    bench.RequestTiming(ttft_ms=10, latency_ms=20),
    bench.RequestTiming(ttft_ms=30, latency_ms=40),
    bench.RequestTiming(ttft_ms=20, latency_ms=30),
  ])
  assert summary["count"] == 3
  assert summary["ttft_p50_ms"] == 20
  assert summary["ttft_p95_ms"] == 29
  assert summary["latency_p50_ms"] == 30


TEST_CASES = [
  ("解析 vLLM tokenize 响应", test_parse_token_count_supports_vllm_response_variants),
  ("解析 OpenAI SSE 输出", test_parse_stream_event_collects_text_and_token_ids),
  ("指标增量与 reset 检测", test_metric_delta_handles_missing_counters_and_rejects_reset),
  ("缓存 block 数计算", test_cached_and_put_block_calculation),
  ("warmup 不产生可缓存 block", test_warmup_must_not_create_cacheable_blocks),
  ("缺失指标文件作为全零基线", test_missing_metrics_file_is_an_empty_baseline),
  ("接受正确的 CPU/SSD 容量关系", test_capacity_accepts_single_fit_cpu_overflow_and_ssd_fit),
  ("拒绝不会触发 CPU 淘汰的工作集", test_capacity_rejects_non_evicting_working_set),
  ("拒绝不能完整淘汰目标的压力集", test_capacity_rejects_pressure_that_cannot_fully_evict_target),
  ("拒绝多 token 生成", test_capacity_rejects_non_single_token_generation),
  ("拒绝单条前缀大于 CPU staging", test_capacity_rejects_prefix_larger_than_cpu_staging),
  ("拒绝超过 SSD 容量的工作集", test_capacity_rejects_working_set_larger_than_ssd),
  ("拒绝部分命中和错误层级", test_verify_complete_hit_rejects_partial_or_wrong_tier),
  ("仅首轮要求纯 SSD 命中", test_only_first_round_requires_a_pure_ssd_hit),
  ("TTFT 统计", test_timing_summary_reports_median_and_p95),
]


def run_all_tests():
  print("开始运行 bench_ssd_e2e 纯逻辑测试")
  for index, (name, test_fn) in enumerate(TEST_CASES, start=1):
    print(f"[{index}/{len(TEST_CASES)}] 开始：{name}")
    test_fn()
    print(f"[{index}/{len(TEST_CASES)}] 通过：{name}")
  print(f"bench_ssd_e2e 测试完成：通过 {len(TEST_CASES)}/{len(TEST_CASES)}")


if __name__ == "__main__":
  run_all_tests()

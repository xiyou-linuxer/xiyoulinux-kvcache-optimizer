#!/usr/bin/env python3
"""真实 vLLM 服务上的 MiniFlex SSD 命中与 CPU 回填验证。

这个脚本验证的是完整请求路径，而不是单独的文件读写：

1. 用多个互不共享首 block 的长 prompt 冷启动缓存；
2. 工作集大于 CPU cache，迫使最早的目标 prompt 从 CPU 淘汰；
3. 再次请求目标 prompt，要求 ``miniflex_get_hit_ssd_blocks`` 增长；
4. 紧接着第三次请求目标 prompt，要求 CPU hit 增长，证明 DISK2H
   已经把数据晋升回 CPU cache；
5. 要求三条路径的 greedy 生成 token ID 完全一致，验证回填 KV 的内容正确性。

同时报告冷重算、SSD hit 和随后 CPU hit 的 TTFT。SSD hit 比冷重算慢
不是功能失败：短上下文或低带宽 SSD 本来就可能落在负收益区间。

服务必须显式开启 SSD，并关闭 vLLM 原生 prefix caching。推荐启动参数见
``docs/ssd_e2e.md``。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


NO_PROXY = {"http": None, "https": None}
BODY_UNIT = (
    "artificial intelligence has transformed many industries over the past decade "
    "including healthcare finance transportation education manufacturing agriculture "
    "and entertainment by enabling machines to learn patterns from large amounts of data "
)

SSD_HIT_BLOCKS = "miniflex_get_hit_ssd_blocks"
CPU_HIT_BLOCKS = "miniflex_get_hit_cpu_blocks"
GET_MISS_BLOCKS = "miniflex_get_miss_blocks"
GET_MATCHED_TOKENS = "miniflex_get_matched_tokens"
PUT_SSD_BLOCKS = "miniflex_put_h2disk_blocks"
PUT_SSD_COMPLETED_BLOCKS = "miniflex_put_h2disk_completed_blocks"


class VerificationError(RuntimeError):
    """Raised when the observed request path does not match the test contract."""


@dataclass(frozen=True)
class RequestTiming:
    ttft_ms: float
    latency_ms: float
    output_text: str = ""
    output_token_ids: tuple[int, ...] = ()


def _counter(snapshot: dict[str, Any], name: str) -> float:
    value = snapshot.get(name, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def metric_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    name: str,
) -> float:
    """Return a monotonic counter delta and reject an unexpected reset."""
    delta = _counter(after, name) - _counter(before, name)
    if delta < 0:
        raise VerificationError(
            f"metric {name!r} decreased; the metrics registry was probably reset "
            "while the benchmark was running"
        )
    return delta


def parse_token_count(payload: dict[str, Any]) -> int:
    """Accept the vLLM tokenization response formats used across releases."""
    count = payload.get("count")
    if isinstance(count, int) and count >= 0:
        return count
    tokens = payload.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    raise VerificationError(
        f"/tokenize response has neither an integer count nor a token list: {payload}"
    )


def parse_stream_event(raw: bytes) -> tuple[str, tuple[int, ...]] | None:
    """Extract generated text and token IDs from one completion SSE event."""
    if not raw.startswith(b"data: "):
        return None
    data = raw[len(b"data: "):]
    if data.strip() == b"[DONE]":
        return None
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerificationError(f"invalid completion SSE event: {raw!r}") from exc
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return "", ()
    text_parts = []
    token_ids = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        text = choice.get("text")
        if isinstance(text, str):
            text_parts.append(text)
        choice_token_ids = choice.get("token_ids")
        if isinstance(choice_token_ids, list):
            token_ids.extend(
                token_id for token_id in choice_token_ids
                if isinstance(token_id, int)
            )
    return "".join(text_parts), tuple(token_ids)


def tokenize_prompt(
    url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> int:
    """Ask the serving tokenizer for the exact prompt length."""
    endpoint = f"{url.rstrip('/')}/tokenize"
    last_error: Exception | None = None
    # Older vLLM accepts only prompt; newer deployments may require a model.
    for body in ({"prompt": prompt}, {"model": model, "prompt": prompt}):
        try:
            response = requests.post(
                endpoint,
                json=body,
                proxies=NO_PROXY,
                timeout=timeout,
            )
            response.raise_for_status()
            return parse_token_count(response.json())
        except (requests.RequestException, ValueError, VerificationError) as exc:
            last_error = exc
    raise VerificationError(f"failed to tokenize prompt via {endpoint}: {last_error}")


def request_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> RequestTiming:
    """Run one completion and consume the entire SSE stream to avoid aborting PUT."""
    started = time.perf_counter()
    response = requests.post(
        f"{url.rstrip('/')}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "min_tokens": 1,
            "stream": True,
            "return_token_ids": True,
        },
        stream=True,
        proxies=NO_PROXY,
        timeout=timeout,
    )
    response.raise_for_status()
    ttft: float | None = None
    output_parts: list[str] = []
    output_token_ids: list[int] = []
    try:
        for raw in response.iter_lines():
            event = parse_stream_event(raw)
            if event is None:
                continue
            if ttft is None:
                ttft = time.perf_counter() - started
            text, token_ids = event
            output_parts.append(text)
            output_token_ids.extend(token_ids)
    finally:
        response.close()
    finished = time.perf_counter()
    if ttft is None:
        ttft = finished - started
    return RequestTiming(
        ttft_ms=ttft * 1000,
        latency_ms=(finished - started) * 1000,
        output_text="".join(output_parts),
        output_token_ids=tuple(output_token_ids),
    )


def _read_metrics_once(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def read_metrics(path: Path, timeout: float = 5.0) -> dict[str, Any]:
    """Read metrics, treating a not-yet-created file as an all-zero baseline."""
    if not path.exists():
        return {}

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = _read_metrics_once(path)
        if snapshot is not None:
            return snapshot
        time.sleep(0.05)
    return {}


def wait_for_counter_delta(
    path: Path,
    before: dict[str, Any],
    name: str,
    minimum: float,
    timeout: float,
) -> dict[str, Any]:
    """Wait until a request-visible counter proves that a path was planned."""
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snapshot = _read_metrics_once(path)
        if snapshot is not None:
            latest = snapshot
            if metric_delta(before, snapshot, name) >= minimum:
                return snapshot
        time.sleep(0.05)
    observed = metric_delta(before, latest, name) if latest else 0
    raise VerificationError(
        f"timed out waiting for {name} delta >= {minimum}; observed {observed}. "
        "Confirm that the service uses MiniFlex, SSD is enabled, and the metrics "
        f"file {path} belongs to this service."
    )


def cached_prompt_blocks(token_count: int, block_size: int) -> int:
    """Complete prompt blocks reusable after one generated token is PUT."""
    if token_count <= 0:
        return 0
    return token_count // block_size


def maximum_put_blocks(
    token_count: int,
    max_tokens: int,
    block_size: int,
) -> int:
    """Conservative block count occupied after prompt plus generated tokens."""
    return max(0, (token_count + max_tokens - 1) // block_size)

def validate_uncacheable_warmup(
    token_count: int,
    max_tokens: int,
    block_size: int,
) -> None:
    """Require a warmup request that cannot create a cache PUT."""
    put_blocks = maximum_put_blocks(token_count, max_tokens, block_size)
    if put_blocks != 0:
        raise VerificationError(
            "warmup prompt unexpectedly produces cacheable blocks: "
            f"tokens={token_count}, max_tokens={max_tokens}, block_size={block_size}"
        )


def validate_capacity(
    token_counts: list[int],
    block_size: int,
    max_tokens: int,
    cpu_blocks: int,
    ssd_blocks: int,
) -> dict[str, Any]:
    """Validate the capacity relationship required to force an SSD hit."""
    if len(token_counts) < 3:
        raise VerificationError("at least three independent prefixes are required")
    if max_tokens != 1:
        raise VerificationError(
            "this verification requires max_tokens=1 so PUT occupancy and "
            "greedy output comparisons are exact"
        )
    if block_size <= 0 or cpu_blocks <= 0 or ssd_blocks <= 0:
        raise VerificationError("block and cache capacities must be positive")

    reusable = [cached_prompt_blocks(count, block_size) for count in token_counts]
    occupied = [
        maximum_put_blocks(count, max_tokens, block_size)
        for count in token_counts
    ]
    if min(reusable) <= 0:
        raise VerificationError("every prompt must contain at least one reusable block")
    if max(occupied) > cpu_blocks:
        raise VerificationError(
            "one prefix does not fit in the CPU staging pool: "
            f"requires up to {max(occupied)} blocks, configured {cpu_blocks}"
        )
    if sum(occupied) <= cpu_blocks:
        raise VerificationError(
            "the complete working set still fits in CPU cache: "
            f"requires about {sum(occupied)} blocks, configured {cpu_blocks}"
        )
    if sum(occupied[1:]) < cpu_blocks:
        raise VerificationError(
            "the pressure prefixes cannot fully evict the target from CPU cache: "
            f"provide at least {cpu_blocks} non-target blocks, got {sum(occupied[1:])}"
        )
    if sum(occupied) > ssd_blocks:
        raise VerificationError(
            "the cold working set does not fit in SSD cache: "
            f"requires about {sum(occupied)} blocks, configured {ssd_blocks}"
        )
    return {
        "token_counts": token_counts,
        "reusable_blocks": reusable,
        "estimated_put_blocks": occupied,
        "estimated_working_set_blocks": sum(occupied),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_timings(samples: list[RequestTiming]) -> dict[str, float]:
    ttft = [sample.ttft_ms for sample in samples]
    latency = [sample.latency_ms for sample in samples]
    return {
        "count": len(samples),
        "ttft_p50_ms": round(statistics.median(ttft), 3),
        "ttft_p95_ms": round(percentile(ttft, 0.95), 3),
        "latency_p50_ms": round(statistics.median(latency), 3),
    }


def make_prompts(num_prefixes: int, body_repeat: int) -> list[str]:
    run_id = uuid.uuid4().hex
    body = BODY_UNIT * body_repeat
    prompts = []
    for index in range(num_prefixes):
        # Put enough unique text first so different prompts do not share block 0.
        marker = uuid.uuid4().hex
        prompts.append(
            f"{marker} {marker} {marker} MiniFlex SSD E2E {run_id} "
            f"prefix {index}. {body}\nSummarize this passage."
        )
    return prompts


def _request_and_wait(
    args: argparse.Namespace,
    prompt: str,
    metric_name: str,
    minimum_delta: float = 1,
) -> tuple[RequestTiming, dict[str, Any], dict[str, Any]]:
    before = read_metrics(args.metrics_file)
    timing = request_completion(
        args.url,
        args.model,
        prompt,
        args.max_tokens,
        args.request_timeout,
    )
    after = wait_for_counter_delta(
        args.metrics_file,
        before,
        metric_name,
        minimum_delta,
        args.metrics_timeout,
    )
    return timing, before, after


def wait_for_ssd_commit(
    args: argparse.Namespace,
    prompt: str,
    before: dict[str, Any],
    minimum_blocks: float,
) -> dict[str, Any]:
    """Use ordinary GET/wait cycles until H2DISK marks its SSD node ready."""
    deadline = time.monotonic() + args.metrics_timeout
    latest = read_metrics(args.metrics_file)
    while time.monotonic() < deadline:
        if metric_delta(before, latest, PUT_SSD_COMPLETED_BLOCKS) >= minimum_blocks:
            return latest
        # Reusing the just-cached prompt creates a normal CPU GET task. Its
        # existing wait_impl path consumes the earlier PUT graph-complete.
        request_completion(
            args.url,
            args.model,
            prompt,
            args.max_tokens,
            args.request_timeout,
        )
        latest = read_metrics(args.metrics_file)
        if metric_delta(before, latest, PUT_SSD_COMPLETED_BLOCKS) >= minimum_blocks:
            return latest
        time.sleep(args.settle_interval)
    observed = metric_delta(before, latest, PUT_SSD_COMPLETED_BLOCKS)
    raise VerificationError(
        "timed out waiting for H2DISK completion: "
        f"expected {minimum_blocks} ready SSD blocks, observed {observed}"
    )


def _deltas(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float]:
    return {
        key: metric_delta(before, after, key)
        for key in (
            SSD_HIT_BLOCKS,
            CPU_HIT_BLOCKS,
            GET_MISS_BLOCKS,
            GET_MATCHED_TOKENS,
            PUT_SSD_BLOCKS,
            PUT_SSD_COMPLETED_BLOCKS,
        )
    }


def verify_complete_hit(
    deltas: dict[str, float],
    expected_blocks: int,
    block_size: int,
    required_tier: str | None = None,
) -> None:
    """Reject partial hits, misses, and an unexpected cache tier."""
    cpu_blocks = deltas[CPU_HIT_BLOCKS]
    ssd_blocks = deltas[SSD_HIT_BLOCKS]
    matched_tokens = deltas[GET_MATCHED_TOKENS]
    miss_blocks = deltas[GET_MISS_BLOCKS]
    expected_tokens = expected_blocks * block_size

    if cpu_blocks + ssd_blocks != expected_blocks:
        raise VerificationError(
            "request did not hit every reusable block: "
            f"expected {expected_blocks}, observed CPU={cpu_blocks:.0f}, "
            f"SSD={ssd_blocks:.0f}"
        )
    if matched_tokens != expected_tokens:
        raise VerificationError(
            "matched-token count proves a partial hit: "
            f"expected {expected_tokens}, observed {matched_tokens:.0f}"
        )
    if miss_blocks != 0:
        raise VerificationError(
            f"request unexpectedly recorded {miss_blocks:.0f} miss blocks"
        )
    if required_tier == "ssd" and (cpu_blocks != 0 or ssd_blocks != expected_blocks):
        raise VerificationError(
            "target was not a full SSD hit: "
            f"CPU={cpu_blocks:.0f}, SSD={ssd_blocks:.0f}, "
            f"expected={expected_blocks}"
        )
    if required_tier == "cpu" and (ssd_blocks != 0 or cpu_blocks != expected_blocks):
        raise VerificationError(
            "SSD restore did not become a full CPU hit: "
            f"CPU={cpu_blocks:.0f}, SSD={ssd_blocks:.0f}, "
            f"expected={expected_blocks}"
        )


def required_ssd_tier(require_full_ssd_hit: bool, round_index: int) -> str | None:
    """Require one pure SSD sample without assuming whole-prefix LRU eviction forever."""
    if require_full_ssd_hit and round_index == 0:
        return "ssd"
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Keep model warmup below one KV block so it cannot alter cache metrics or
    # consume cache capacity. A missing metrics file is a valid all-zero baseline;
    # the first cold request below must materialize it and advance H2DISK.
    warmup_prompt = "Hi"
    warmup_token_count = tokenize_prompt(
        args.url,
        args.model,
        warmup_prompt,
        args.request_timeout,
    )
    validate_uncacheable_warmup(warmup_token_count, 1, args.block_size)
    request_completion(
        args.url,
        args.model,
        warmup_prompt,
        1,
        args.request_timeout,
    )
    prompts = make_prompts(args.num_prefixes, args.body_repeat)
    token_counts = [
        tokenize_prompt(args.url, args.model, prompt, args.request_timeout)
        for prompt in prompts
    ]
    capacity = validate_capacity(
        token_counts,
        args.block_size,
        args.max_tokens,
        args.cpu_blocks,
        args.ssd_blocks,
    )
    target = prompts[0]
    expected_target_blocks = capacity["reusable_blocks"][0]

    print(
        ">>> capacity: "
        f"prompt_tokens={token_counts}, "
        f"working_set≈{capacity['estimated_working_set_blocks']} blocks, "
        f"CPU={args.cpu_blocks}, SSD={args.ssd_blocks}"
    )
    print(">>> cold fill: every request must plan H2DISK")
    cold_timings: list[RequestTiming] = []
    cold_metrics: list[dict[str, float]] = []
    for index, prompt in enumerate(prompts):
        timing, before, after = _request_and_wait(
            args,
            prompt,
            PUT_SSD_BLOCKS,
        )
        deltas = _deltas(before, after)
        completed = wait_for_ssd_commit(
            args,
            prompt,
            before,
            deltas[PUT_SSD_BLOCKS],
        )
        deltas[PUT_SSD_COMPLETED_BLOCKS] = metric_delta(
            before,
            completed,
            PUT_SSD_COMPLETED_BLOCKS,
        )
        cold_timings.append(timing)
        cold_metrics.append(deltas)
        print(
            f"  cold[{index}] TTFT={timing.ttft_ms:.1f}ms "
            f"H2DISK={deltas[PUT_SSD_BLOCKS]:.0f}, "
            f"completed={deltas[PUT_SSD_COMPLETED_BLOCKS]:.0f} blocks"
        )

    target_output = cold_timings[0].output_text
    target_token_ids = cold_timings[0].output_token_ids
    if not target_token_ids:
        raise VerificationError(
            "vLLM returned no generated token IDs; correctness comparison "
            "requires completion return_token_ids support"
        )

    ssd_timings: list[RequestTiming] = []
    cpu_timings: list[RequestTiming] = []
    ssd_metrics: list[dict[str, float]] = []
    cpu_metrics: list[dict[str, float]] = []

    for round_index in range(args.rounds):
        # Each filler becomes most recently used and pushes the target out of the
        # deliberately small CPU pool. Their own hit tier is irrelevant here.
        for filler in prompts[1:]:
            request_completion(
                args.url,
                args.model,
                filler,
                args.max_tokens,
                args.request_timeout,
            )

        ssd_timing, before_ssd, after_ssd = _request_and_wait(
            args,
            target,
            SSD_HIT_BLOCKS,
        )
        ssd_delta = _deltas(before_ssd, after_ssd)
        verify_complete_hit(
            ssd_delta,
            expected_target_blocks,
            args.block_size,
            required_tier=required_ssd_tier(
                args.require_full_ssd_hit,
                round_index,
            ),
        )
        if (
            ssd_timing.output_token_ids != target_token_ids
            or ssd_timing.output_text != target_output
        ):
            raise VerificationError(
                "SSD-restored KV produced a different greedy output: "
                f"cold_ids={target_token_ids!r}, "
                f"SSD_ids={ssd_timing.output_token_ids!r}, "
                f"cold_text={target_output!r}, SSD_text={ssd_timing.output_text!r}"
            )

        # DISK2H marks the CPU node ready before H2D completes. The immediately
        # following request must therefore be served by CPU rather than SSD.
        cpu_timing, before_cpu, after_cpu = _request_and_wait(
            args,
            target,
            CPU_HIT_BLOCKS,
        )
        cpu_delta = _deltas(before_cpu, after_cpu)
        verify_complete_hit(
            cpu_delta,
            expected_target_blocks,
            args.block_size,
            required_tier="cpu",
        )
        if (
            cpu_timing.output_token_ids != target_token_ids
            or cpu_timing.output_text != target_output
        ):
            raise VerificationError(
                "CPU-promoted KV produced a different greedy output: "
                f"cold_ids={target_token_ids!r}, "
                f"CPU_ids={cpu_timing.output_token_ids!r}, "
                f"cold_text={target_output!r}, CPU_text={cpu_timing.output_text!r}"
            )

        ssd_timings.append(ssd_timing)
        cpu_timings.append(cpu_timing)
        ssd_metrics.append(ssd_delta)
        cpu_metrics.append(cpu_delta)
        print(
            f"  round[{round_index}] "
            f"SSD TTFT={ssd_timing.ttft_ms:.1f}ms "
            f"(SSD={ssd_delta[SSD_HIT_BLOCKS]:.0f}, "
            f"CPU={ssd_delta[CPU_HIT_BLOCKS]:.0f}) -> "
            f"CPU TTFT={cpu_timing.ttft_ms:.1f}ms "
            f"(CPU={cpu_delta[CPU_HIT_BLOCKS]:.0f})"
        )

    cold_summary = summarize_timings(cold_timings)
    ssd_summary = summarize_timings(ssd_timings)
    cpu_summary = summarize_timings(cpu_timings)
    cold_p50 = cold_summary["ttft_p50_ms"]
    ssd_p50 = ssd_summary["ttft_p50_ms"]
    cpu_p50 = cpu_summary["ttft_p50_ms"]
    cold_over_ssd = cold_p50 / ssd_p50 if ssd_p50 else 0
    cold_over_cpu = cold_p50 / cpu_p50 if cpu_p50 else 0
    performance = (
        "ssd_faster_than_recompute"
        if ssd_p50 < cold_p50
        else "recompute_faster_than_ssd"
    )

    print("")
    print("=" * 72)
    print(
        f"cold p50={cold_p50:.1f}ms, SSD p50={ssd_p50:.1f}ms, "
        f"CPU p50={cpu_p50:.1f}ms"
    )
    print(
        f"cold/SSD={cold_over_ssd:.2f}x, "
        f"cold/CPU={cold_over_cpu:.2f}x, verdict={performance}"
    )
    if performance == "recompute_faster_than_ssd":
        print(
            "NOTE: SSD path is correct but slower at this context length. "
            "Repeat with longer prefixes to locate the crossover point."
        )
    print("PASS: observed SSD hit and the following CPU-cache promotion.")
    print("=" * 72)

    return {
        "schema_version": 1,
        "timestamp": time.time(),
        "service": {"url": args.url, "model": args.model},
        "configuration_claim": {
            "block_size": args.block_size,
            "cpu_blocks": args.cpu_blocks,
            "ssd_blocks": args.ssd_blocks,
            "num_prefixes": args.num_prefixes,
            "body_repeat": args.body_repeat,
            "rounds": args.rounds,
            "require_full_ssd_hit": args.require_full_ssd_hit,
        },
        "capacity": capacity,
        "timings": {
            "cold": cold_summary,
            "ssd_hit": ssd_summary,
            "cpu_hit_after_ssd": cpu_summary,
            "cold_over_ssd": round(cold_over_ssd, 4),
            "cold_over_cpu": round(cold_over_cpu, 4),
            "verdict": performance,
        },
        "samples": {
            "cold": [asdict(item) for item in cold_timings],
            "ssd_hit": [asdict(item) for item in ssd_timings],
            "cpu_hit_after_ssd": [asdict(item) for item in cpu_timings],
        },
        "metric_deltas": {
            "cold": cold_metrics,
            "ssd_hit": ssd_metrics,
            "cpu_hit_after_ssd": cpu_metrics,
        },
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a real vLLM CPU-eviction -> SSD-hit -> CPU-promotion path."
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--metrics-file", type=Path, default=Path("/tmp/miniflex_metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/miniflex_ssd_e2e.json"))
    parser.add_argument("--num-prefixes", type=int, default=4)
    parser.add_argument("--body-repeat", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--cpu-blocks",
        type=int,
        default=256,
        help="must match MINIFLEX_NUM_CPU_BLOCKS used by the service",
    )
    parser.add_argument(
        "--ssd-blocks",
        type=int,
        default=1024,
        help="must match MINIFLEX_NUM_SSD_BLOCKS used by the service",
    )
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--metrics-timeout", type=float, default=30)
    parser.add_argument(
        "--settle-interval",
        type=float,
        default=0.1,
        help="delay between requests that drive background H2DISK completion",
    )
    parser.add_argument(
        "--require-full-ssd-hit",
        action="store_true",
        help=(
            "require the first measured recovery to be a pure SSD hit; later "
            "rounds may retain a small CPU-resident prefix"
        ),
    )
    args = parser.parse_args()
    if args.num_prefixes < 3:
        parser.error("--num-prefixes must be at least 3")
    if args.body_repeat <= 0 or args.rounds <= 0:
        parser.error("--body-repeat and --rounds must be positive")
    if args.max_tokens != 1:
        parser.error("--max-tokens must be exactly 1 for deterministic validation")
    if args.settle_interval < 0:
        parser.error("--settle-interval must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f">>> machine-readable result: {args.out}")


if __name__ == "__main__":
    main()

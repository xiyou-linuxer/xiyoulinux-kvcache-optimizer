#!/usr/bin/env python3
"""
vLLM baseline runner — 发送请求、采集遥测、输出 summary。

兼容 run_baseline_v2.py 编排器：接受 --tokenizer / --max-input-tokens /
--sample-manifest / --warmup-samples / --run-group-id / --repeat-index / --mode
等参数。支持流式请求采集 TTFT / TPOT / ITL。
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

# 确保 tests 包可导入（编排器从 repo root 运行，也支持直接 python tests/runner.py）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# ---- 从 common.py 导入标准 prompt 构造 & tokenizer 工具 ----
from tests.common import (
    count_tokens,
    load_tokenizer,
    needlebench_prompt_from_sample as _common_needlebench_prompt,
    prompt_from_longbench as _common_prompt_from_longbench,
    truncate_by_tokens,
)

METRIC_FAMILIES = [
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:time_to_first_token_seconds",
    "vllm:time_per_output_token_seconds",
]

# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM baseline workload runner with streaming telemetry."
    )
    parser.add_argument("--base-url",
                        default="http://127.0.0.1:8000",
                        help="Base URL of the running vLLM OpenAI-compatible server.")
    parser.add_argument("--model",
                        default="qwen3-8b",
                        help="Served model name exposed by the vLLM server.")
    parser.add_argument("--scenario",
                        choices=[
                            "short_synth",
                            "shared_prefix",
                            "longbench",
                            "needlebench",
                            "ultra_long_synth",
                            "mixed_pressure",
                        ],
                        required=True,
                        help="Workload scenario to run.")
    parser.add_argument("--num-samples",
                        type=int,
                        default=10,
                        help="Number of requests to send.")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=128,
                        help="Generation length.")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Sampling seed.")
    parser.add_argument("--poll-interval",
                        type=float,
                        default=1.0,
                        help="Seconds between telemetry snapshots.")
    parser.add_argument("--output-dir",
                        default="baseline_runs",
                        help="Directory to store run outputs.")
    parser.add_argument("--request-timeout",
                        type=int,
                        default=600,
                        help="Request timeout in seconds.")
    parser.add_argument("--concurrency",
                        type=int,
                        default=1,
                        help="Maximum number of concurrent requests.")
    parser.add_argument("--max-input-chars",
                        type=int,
                        default=24000,
                        help="Hard truncate prompt text by characters.")
    parser.add_argument("--max-input-tokens",
                        type=int,
                        default=None,
                        help="Truncate prompt by token count (overrides "
                             "char truncation when both are set).")
    parser.add_argument("--subset",
                        default="qasper",
                        help="LongBench subset name when scenario=longbench.")
    parser.add_argument("--longbench-local-dir",
                        default=None,
                        help="Local LongBench directory containing data/<subset>.jsonl.")
    parser.add_argument("--needle-subset",
                        default="en_haystack_texts",
                        help="NeedleBench subset name when scenario=needlebench.")
    parser.add_argument("--split",
                        default="test",
                        help="Dataset split when scenario=longbench.")
    parser.add_argument("--prefix-chars",
                        type=int,
                        default=12000,
                        help="Shared prefix length for shared_prefix scenario.")
    parser.add_argument("--prompt-chars",
                        type=int,
                        default=6000,
                        help="Synthetic prompt length for short_synth scenario.")
    parser.add_argument("--ultra-prompt-chars",
                        type=int,
                        default=22000,
                        help="Synthetic prompt length for ultra_long_synth.")
    parser.add_argument("--mixed-short-chars",
                        type=int,
                        default=2000,
                        help="Short prompt length for mixed_pressure.")
    parser.add_argument("--mixed-medium-chars",
                        type=int,
                        default=8000,
                        help="Medium prompt length for mixed_pressure.")
    parser.add_argument("--mixed-long-chars",
                        type=int,
                        default=16000,
                        help="Long prompt length for mixed_pressure.")
    # ---- 编排器兼容参数 ----
    parser.add_argument("--tokenizer",
                        default=None,
                        help="Tokenizer path for token-level truncation / stats.")
    parser.add_argument("--sample-manifest",
                        default=None,
                        help="Pre-built prompt manifest JSON (bypasses build_*).")
    parser.add_argument("--warmup-samples",
                        type=int,
                        default=0,
                        help="Number of warmup requests before the actual run.")
    parser.add_argument("--run-group-id",
                        default="",
                        help="Run group ID assigned by orchestrator.")
    parser.add_argument("--repeat-index",
                        type=int,
                        default=0,
                        help="Repeat index assigned by orchestrator.")
    parser.add_argument("--mode",
                        default="baseline",
                        help="Mode label (baseline / miniflex).")
    parser.add_argument("--retry",
                        type=int,
                        default=0,
                        help="Number of retries for failed requests.")
    return parser.parse_args()


# ============================================================
# 遥测采集（保持不变）
# ============================================================
def fetch_prometheus_metrics(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()

    values: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+eE0-9.]+)$",
                         line)
        if not match:
            continue
        family = match.group(1)
        if family not in METRIC_FAMILIES:
            continue
        values[family] = values.get(family, 0.0) + float(match.group(3))
    return values


def fetch_gpu_stats() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command,
                                check=True,
                                capture_output=True,
                                text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "utilization_gpu_pct": float(parts[4]),
        })
    return gpus


def fetch_cpu_memory_stats() -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fin:
            for line in fin:
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                value = rest.strip().split()[0]
                meminfo[key] = int(value)
    except OSError:
        return {}

    total_kb = meminfo.get("MemTotal")
    available_kb = meminfo.get("MemAvailable")
    if total_kb is None or available_kb is None:
        return {}

    used_kb = total_kb - available_kb
    stats: dict[str, Any] = {
        "memory_total_mb": round(total_kb / 1024, 2),
        "memory_available_mb": round(available_kb / 1024, 2),
        "memory_used_mb": round(used_kb / 1024, 2),
        "memory_used_pct": round((used_kb / total_kb) * 100, 2),
    }

    try:
        load1, load5, load15 = os.getloadavg()
        stats["loadavg_1m"] = round(load1, 3)
        stats["loadavg_5m"] = round(load5, 3)
        stats["loadavg_15m"] = round(load15, 3)
    except OSError:
        pass

    return stats


def fetch_telemetry_snapshot(base_url: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "ts": time.time(),
        "metrics": {},
        "gpus": [],
        "cpu": {},
        "errors": [],
    }
    try:
        snapshot["metrics"] = fetch_prometheus_metrics(base_url)
    except Exception as exc:  # noqa: BLE001
        snapshot["errors"].append(f"metrics:{exc}")
    try:
        snapshot["gpus"] = fetch_gpu_stats()
    except Exception as exc:  # noqa: BLE001
        snapshot["errors"].append(f"gpu:{exc}")
    try:
        snapshot["cpu"] = fetch_cpu_memory_stats()
    except Exception as exc:  # noqa: BLE001
        snapshot["errors"].append(f"cpu:{exc}")
    return snapshot


# ============================================================
# Prompt 构造
# ============================================================
def repeated_text(target_chars: int, topic: str) -> str:
    chunk = (
        f"{topic} This paragraph is repeated to create a stable long context for "
        "baseline testing. It should be semantically boring but structurally "
        "consistent so that changes in latency and cache metrics mainly come from "
        "context length and not from prompt diversity.\n"
    )
    pieces: list[str] = []
    while sum(len(piece) for piece in pieces) < target_chars:
        pieces.append(chunk)
    return "".join(pieces)[:target_chars]


def _apply_truncation(prompt: str, max_input_chars: int,
                      max_input_tokens: int | None,
                      tok: Any) -> str:
    """字符级 + token 级双重截断。token 级在字符级之后生效。"""
    prompt = prompt[:max_input_chars]
    if max_input_tokens is not None and tok is not None:
        prompt = truncate_by_tokens(prompt, max_input_tokens, tok)
    return prompt


def prompt_from_longbench(sample: dict[str, Any],
                          max_input_chars: int,
                          max_input_tokens: int | None = None,
                          tok: Any = None) -> str:
    raw = _common_prompt_from_longbench(sample, max_input_chars)
    if max_input_tokens is not None and tok is not None:
        raw = truncate_by_tokens(raw, max_input_tokens, tok)
    return raw


def build_short_synth_prompts(num_samples: int,
                              prompt_chars: int,
                              max_input_chars: int,
                              max_input_tokens: int | None = None,
                              tok: Any = None) -> list[dict[str, Any]]:
    prompts = []
    for idx in range(num_samples):
        topic = f"Sample {idx}: LLM KV cache baseline test."
        raw = (
            "Summarize the following text in 5 concise bullet points.\n\n"
            f"{repeated_text(prompt_chars, topic)}\n\nSummary:"
        )
        prompts.append({
            "sample_id": f"short_synth_{idx}",
            "prompt": _apply_truncation(raw, max_input_chars, max_input_tokens, tok),
            "meta": {"group": "short_synth"},
        })
    return prompts


def build_shared_prefix_prompts(num_samples: int,
                                prefix_chars: int,
                                max_input_chars: int,
                                max_input_tokens: int | None = None,
                                tok: Any = None) -> list[dict[str, Any]]:
    raw_prefix = (
        "You are assisting with a long document QA benchmark.\n\n"
        "Shared Reference Document:\n"
        f"{repeated_text(prefix_chars, 'Shared prefix document for cache reuse.')}\n\n"
    )
    suffixes = [
        "Question: What are the three main themes discussed in the document?\nAnswer:",
        "Question: Identify two repeated operational risks mentioned in the document.\nAnswer:",
        "Question: Summarize the document as if reporting to a project manager.\nAnswer:",
        "Question: Which parts look like background information rather than action items?\nAnswer:",
        "Question: Extract the sections most related to performance bottlenecks.\nAnswer:",
    ]

    prompts = []
    for idx in range(num_samples):
        suffix = suffixes[idx % len(suffixes)]

        # ---- 后缀保留式截断：先算 suffix 占多少 token，再截 prefix ----
        if max_input_tokens is not None and tok is not None:
            suffix_tokens = count_tokens(suffix, tok)
            prefix_max_tokens = max(0, max_input_tokens - suffix_tokens)
            truncated_prefix = truncate_by_tokens(raw_prefix, prefix_max_tokens, tok)
            prompt = f"{truncated_prefix}{suffix}"
        else:
            suffix_len = len(suffix)
            prefix_max_chars = max(0, max_input_chars - suffix_len)
            prompt = raw_prefix[:prefix_max_chars] + suffix

        prompts.append({
            "sample_id": f"shared_prefix_{idx}",
            "prompt": prompt[:max_input_chars],  # 兜底：字符级上限
            "meta": {
                "group": "shared_prefix",
                "shared_prefix_chars": min(len(raw_prefix), max_input_chars),
            },
        })
    return prompts


def build_ultra_long_synth_prompts(num_samples: int,
                                   ultra_prompt_chars: int,
                                   max_input_chars: int,
                                   max_input_tokens: int | None = None,
                                   tok: Any = None) -> list[dict[str, Any]]:
    prompts = []
    for idx in range(num_samples):
        topic = f"Ultra sample {idx}: long-context KV cache stress test."
        raw = (
            "Read the following long context and provide a concise structured "
            "summary of recurring themes, operational risks, and repeated facts.\n\n"
            f"{repeated_text(ultra_prompt_chars, topic)}\n\nStructured Summary:"
        )
        prompts.append({
            "sample_id": f"ultra_long_synth_{idx}",
            "prompt": _apply_truncation(raw, max_input_chars, max_input_tokens, tok),
            "meta": {"group": "ultra_long_synth"},
        })
    return prompts


def build_mixed_pressure_prompts(num_samples: int,
                                 short_chars: int,
                                 medium_chars: int,
                                 long_chars: int,
                                 prefix_chars: int,
                                 max_input_chars: int,
                                 max_input_tokens: int | None = None,
                                 tok: Any = None) -> list[dict[str, Any]]:
    prompts = []
    shared_prefix = (
        "You are assisting with a mixed-pressure benchmark.\n\n"
        "Shared Reference:\n"
        f"{repeated_text(prefix_chars, 'Mixed pressure shared prefix reference.')}\n\n"
    )
    shared_suffixes = [
        "Question: Summarize the most repeated statements.\nAnswer:",
        "Question: Identify the operational concerns in the reference.\nAnswer:",
        "Question: Extract the parts related to latency and throughput.\nAnswer:",
    ]

    for idx in range(num_samples):
        bucket = idx % 4
        if bucket == 0:
            raw = (
                "Provide three key points from the following short context.\n\n"
                f"{repeated_text(short_chars, f'Mixed short {idx}.')}\n\nAnswer:"
            )
            bucket_name = "mixed_short"
        elif bucket == 1:
            raw = (
                "Summarize the following medium-length engineering note.\n\n"
                f"{repeated_text(medium_chars, f'Mixed medium {idx}.')}\n\nAnswer:"
            )
            bucket_name = "mixed_medium"
        elif bucket == 2:
            raw = (
                "Analyze the following long technical context and list the main "
                "performance bottlenecks.\n\n"
                f"{repeated_text(long_chars, f'Mixed long {idx}.')}\n\nAnswer:"
            )
            bucket_name = "mixed_long"
        else:
            raw = f"{shared_prefix}{shared_suffixes[idx % len(shared_suffixes)]}"
            bucket_name = "mixed_shared_prefix"

        prompts.append({
            "sample_id": f"mixed_pressure_{idx}",
            "prompt": _apply_truncation(raw, max_input_chars, max_input_tokens, tok),
            "meta": {
                "group": "mixed_pressure",
                "bucket": bucket_name,
            },
        })
    return prompts


def build_longbench_prompts(args: argparse.Namespace,
                            tok: Any = None) -> list[dict[str, Any]]:
    records = None

    if args.longbench_local_dir:
        local_dir = Path(args.longbench_local_dir)
        data_file = local_dir / "data" / f"{args.subset}.jsonl"
        if not data_file.exists():
            raise SystemExit(f"LongBench local file not found: {data_file}")

        records = []
        with data_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        if load_dataset is None:
            raise SystemExit(
                "Missing dependency: datasets. Install with `pip install datasets`.")
        dataset = load_dataset("THUDM/LongBench",
                               args.subset,
                               split=args.split,
                               trust_remote_code=True)
        records = [dataset[idx] for idx in range(len(dataset))]

    random.shuffle(records)
    selected = records[:args.num_samples]

    prompts = []
    for ds_index, sample in enumerate(selected):
        prompts.append({
            "sample_id": str(sample.get("_id", ds_index)),
            "prompt": prompt_from_longbench(sample, args.max_input_chars,
                                            args.max_input_tokens, tok),
            "meta": {
                "group": "longbench",
                "subset": args.subset,
                "dataset_index": ds_index,
                "answers": sample.get("answers", []),
                "source": "local_jsonl" if args.longbench_local_dir else "hf_dataset",
            },
        })
    return prompts


def pick_first_value(sample: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]
    return None


def normalize_answer_field(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_needlebench_prompt(sample: dict[str, Any],
                             max_input_chars: int,
                             max_input_tokens: int | None = None,
                             tok: Any = None) -> tuple[str, dict[str, Any]]:
    """使用 common.py 的 prompt 构造 + 可选的 token 级截断。"""
    raw, meta = _common_needlebench_prompt(sample, max_input_chars)
    if max_input_tokens is not None and tok is not None:
        raw = truncate_by_tokens(raw, max_input_tokens, tok)
    return raw, meta


def build_needlebench_prompts(args: argparse.Namespace,
                              tok: Any = None) -> list[dict[str, Any]]:
    if load_dataset is None:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`.")

    dataset = load_dataset("opencompass/NeedleBench",
                           args.needle_subset,
                           split=args.split)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[:args.num_samples]

    prompts = []
    for ds_index in selected:
        sample = dataset[ds_index]
        raw, extra_meta = build_needlebench_prompt(
            sample, args.max_input_chars, args.max_input_tokens, tok)
        prompts.append({
            "sample_id": str(sample.get("_id", ds_index)),
            "prompt": raw,
            "meta": {
                "group": "needlebench",
                "subset": args.needle_subset,
                "dataset_index": ds_index,
                **extra_meta,
            },
        })
    return prompts


# ============================================================
# 流式请求（核心改动）
# ============================================================
def call_completion(base_url: str,
                    model: str,
                    prompt: str,
                    max_tokens: int,
                    temperature: float,
                    request_timeout: int,
                    retry: int = 0) -> dict[str, Any]:
    """流式 POST /v1/completions，采集 TTFT / TPOT / ITL。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    last_exc = None
    for attempt in range(retry + 1):
        try:
            t_start = time.perf_counter()
            resp = requests.post(f"{base_url}/v1/completions",
                                 json=payload,
                                 stream=True,
                                 timeout=request_timeout,
                                 proxies={"http": None, "https": None})
            if resp.status_code != 200:
                last_exc = RuntimeError(
                    f"HTTP {resp.status_code}: {resp.text[:500]}")
                if attempt < retry:
                    time.sleep(1)
                    continue
                return {
                    "status_code": resp.status_code,
                    "latency_sec": time.perf_counter() - t_start,
                    "ttft_sec": None,
                    "tpot_sec": None,
                    "itl_values": [],
                    "success": False,
                    "prompt_chars": len(prompt),
                    "completion_tokens": 0,
                    "response_text": "",
                    "error": resp.text[:1000],
                }

            ttft = None
            completion_text = ""
            itl_values: list[float] = []
            t_last_token = t_start
            token_count = 0

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                text = choices[0].get("text", "")
                finish = choices[0].get("finish_reason")

                t_now = time.perf_counter()
                if ttft is None:
                    ttft = t_now - t_start
                else:
                    itl_values.append(t_now - t_last_token)

                t_last_token = t_now
                completion_text += text
                token_count += 1

                if finish is not None:
                    break

            latency = time.perf_counter() - t_start

            # TPOT = 首个token之后平均每token时间
            tpot = (statistics.mean(itl_values) if itl_values
                    else (latency - ttft if ttft and token_count > 1 else None))

            return {
                "status_code": 200,
                "latency_sec": latency,
                "ttft_sec": ttft,
                "tpot_sec": tpot,
                "itl_values": itl_values,
                "itl_mean_sec": statistics.mean(itl_values) if itl_values else None,
                "itl_p50_sec": (float(sorted(itl_values)[len(itl_values)//2])
                                if itl_values else None),
                "itl_p95_sec": (float(sorted(itl_values)[int(len(itl_values)*0.95)])
                                if len(itl_values) >= 2 else None),
                "success": True,
                "prompt_chars": len(prompt),
                "completion_tokens": token_count,
                "response_text": completion_text,
                "error": None,
            }

        except Exception as exc:
            last_exc = exc
            if attempt < retry:
                time.sleep(1)

    return {
        "status_code": 0,
        "latency_sec": 0,
        "ttft_sec": None,
        "tpot_sec": None,
        "itl_values": [],
        "success": False,
        "prompt_chars": len(prompt),
        "completion_tokens": 0,
        "response_text": "",
        "error": str(last_exc),
    }


# ============================================================
# TelemetryPoller（保持不变）
# ============================================================
class TelemetryPoller:

    def __init__(self, base_url: str, output_path: Path,
                 interval_sec: float) -> None:
        self.base_url = base_url
        self.output_path = output_path
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        with self.output_path.open("w", encoding="utf-8") as fout:
            while not self._stop.is_set():
                snapshot = fetch_telemetry_snapshot(self.base_url)
                fout.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                fout.flush()
                self._stop.wait(self.interval_sec)


# ============================================================
# load_prompts（支持 manifest）
# ============================================================
def load_prompts(args: argparse.Namespace, tok: Any = None) -> list[dict[str, Any]]:
    """加载 prompt 列表。如果指定了 --sample-manifest 则从 JSON 加载。"""

    # ---- manifest 路径：直接加载 ----
    if args.sample_manifest:
        manifest_path = Path(args.sample_manifest)
        if not manifest_path.exists():
            raise SystemExit(f"Manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # 兼容两种格式：prepare 产出 dict（prompts 在 "prompts" key 里），runner 产出 list
        if isinstance(raw, dict):
            prompts = raw.get("prompts", [])
        else:
            prompts = raw

        if not isinstance(prompts, list):
            raise SystemExit(f"Manifest prompts must be a list, got {type(prompts).__name__}")

        # 确保与 manifest 的 sample 数一致（编排器可能不传 --num-samples）
        if len(prompts) != args.num_samples:
            print(f"[load_prompts] manifest has {len(prompts)} samples, "
                  f"adjusting --num-samples from {args.num_samples}")
            args.num_samples = len(prompts)

        # 对 manifest 的 prompt 也做截断
        if args.max_input_tokens is not None and tok is not None:
            for item in prompts:
                item["prompt"] = _apply_truncation(
                    item["prompt"], args.max_input_chars,
                    args.max_input_tokens, tok)
        return prompts

    # ---- 非 manifest：生成 ----
    if args.scenario == "short_synth":
        return build_short_synth_prompts(
            args.num_samples, args.prompt_chars, args.max_input_chars,
            args.max_input_tokens, tok)
    if args.scenario == "shared_prefix":
        return build_shared_prefix_prompts(
            args.num_samples, args.prefix_chars, args.max_input_chars,
            args.max_input_tokens, tok)
    if args.scenario == "needlebench":
        return build_needlebench_prompts(args, tok)
    if args.scenario == "ultra_long_synth":
        return build_ultra_long_synth_prompts(
            args.num_samples, args.ultra_prompt_chars, args.max_input_chars,
            args.max_input_tokens, tok)
    if args.scenario == "mixed_pressure":
        return build_mixed_pressure_prompts(
            args.num_samples, args.mixed_short_chars, args.mixed_medium_chars,
            args.mixed_long_chars, args.prefix_chars, args.max_input_chars,
            args.max_input_tokens, tok)
    return build_longbench_prompts(args, tok)


# ============================================================
# 统计工具
# ============================================================
def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_telemetry(telemetry_path: Path) -> dict[str, Any]:
    snapshots = []
    with telemetry_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))

    if not snapshots:
        return {}

    kv_values = [
        snap.get("metrics", {}).get("vllm:kv_cache_usage_perc")
        for snap in snapshots
        if "vllm:kv_cache_usage_perc" in snap.get("metrics", {})
    ]

    def metric_series(*names: str) -> list[float]:
        values = []
        for snap in snapshots:
            metrics = snap.get("metrics", {})
            for name in names:
                if name in metrics:
                    values.append(metrics[name])
                    break
        return values

    prefix_queries = metric_series("vllm:prefix_cache_queries_total",
                                   "vllm:prefix_cache_queries")
    prefix_hits = metric_series("vllm:prefix_cache_hits_total",
                                "vllm:prefix_cache_hits")

    # 新增：从 vLLM metrics 提取 TTFT / TPOT / waiting
    vllm_ttft = metric_series("vllm:time_to_first_token_seconds")
    vllm_tpot = metric_series("vllm:time_per_output_token_seconds")
    waiting_series = metric_series("vllm:num_requests_waiting")

    gpu_used_values = []
    gpu_util_values = []
    cpu_used_values = []
    cpu_used_pct_values = []
    load1_values = []
    for snap in snapshots:
        for gpu in snap.get("gpus", []):
            gpu_used_values.append(gpu.get("memory_used_mb"))
            gpu_util_values.append(gpu.get("utilization_gpu_pct"))
        cpu = snap.get("cpu", {})
        if "memory_used_mb" in cpu:
            cpu_used_values.append(cpu["memory_used_mb"])
        if "memory_used_pct" in cpu:
            cpu_used_pct_values.append(cpu["memory_used_pct"])
        if "loadavg_1m" in cpu:
            load1_values.append(cpu["loadavg_1m"])

    return {
        "kv_cache_usage_start": kv_values[0] if kv_values else None,
        "kv_cache_usage_end": kv_values[-1] if kv_values else None,
        "kv_cache_usage_max": max(kv_values) if kv_values else None,
        "prefix_queries_delta":
        (prefix_queries[-1] - prefix_queries[0]) if len(prefix_queries) >= 2 else None,
        "prefix_hits_delta":
        (prefix_hits[-1] - prefix_hits[0]) if len(prefix_hits) >= 2 else None,
        "gpu_memory_used_mb_peak":
        max(gpu_used_values) if gpu_used_values else None,
        "gpu_util_pct_peak":
        max(gpu_util_values) if gpu_util_values else None,
        "cpu_memory_used_mb_peak":
        max(cpu_used_values) if cpu_used_values else None,
        "cpu_memory_used_pct_peak":
        max(cpu_used_pct_values) if cpu_used_pct_values else None,
        "loadavg_1m_peak":
        max(load1_values) if load1_values else None,
        "num_requests_waiting_peak":
        max(waiting_series) if waiting_series else None,
        "vllm_ttft_sec_mean":
        statistics.mean(vllm_ttft) if vllm_ttft else None,
        "vllm_tpot_sec_mean":
        statistics.mean(vllm_tpot) if vllm_tpot else None,
        "telemetry_samples": len(snapshots),
    }


# ============================================================
# 请求执行器
# ============================================================
def run_requests(base_url: str, model: str, prompts: list[dict[str, Any]],
                 max_tokens: int, temperature: float, request_timeout: int,
                 concurrency: int, retry: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [None] * len(prompts)  # type: ignore[list-item]

    def worker(index: int, item: dict[str, Any]) -> dict[str, Any]:
        result = call_completion(
            base_url=base_url,
            model=model,
            prompt=item["prompt"],
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout=request_timeout,
            retry=retry,
        )
        record = {
            "sample_id": item["sample_id"],
            "meta": item["meta"],
            "prompt_chars": result["prompt_chars"],
            "latency_sec": result["latency_sec"],
            "ttft_sec": result["ttft_sec"],
            "tpot_sec": result["tpot_sec"],
            "itl_mean_sec": result.get("itl_mean_sec"),
            "itl_p50_sec": result.get("itl_p50_sec"),
            "itl_p95_sec": result.get("itl_p95_sec"),
            "success": result["success"],
            "status_code": result["status_code"],
            "completion_tokens": result["completion_tokens"],
            "response_text": result["response_text"],
            "error": result["error"],
            "_index": index,
        }
        return record

    max_workers = max(1, concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, idx, item): idx
            for idx, item in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            record = future.result()
            records[record["_index"]] = record
            ttft_str = f"ttft={record['ttft_sec']:.3f}s" if record["ttft_sec"] else ""
            print(
                f"[{record['_index'] + 1}/{len(prompts)}] "
                f"{record['meta'].get('group')} sample={record['sample_id']} "
                f"success={record['success']} latency={record['latency_sec']:.3f}s "
                f"{ttft_str} prompt_chars={record['prompt_chars']}"
            )

    for record in records:
        del record["_index"]
    return records


# ============================================================
# main
# ============================================================
def main() -> None:
    args = parse_args()
    random.seed(args.seed + args.repeat_index)

    # ---- 当 --max-input-tokens 生效时，自动放大 max_input_chars 防止字符级过早截断 ----
    if args.max_input_tokens is not None:
        args.max_input_chars = max(args.max_input_chars, args.max_input_tokens * 6)

    # ---- tokenizer ----
    tok = None
    if args.tokenizer:
        try:
            tok = load_tokenizer(args.tokenizer)
        except Exception as exc:
            print(f"[WARN] tokenizer load failed: {exc}")

    # ---- run_id 格式：兼容 summarize_baseline_runs.py ----
    ts = int(time.time())
    run_id = f"{args.mode}_{args.run_group_id}__r{args.repeat_index}_{args.scenario}_{ts}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args, tok)

    # ---- 写入 config + manifest ----
    config = vars(args).copy()
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_out = output_dir / "sample_manifest.json"
    manifest_out.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] {manifest_out} ({len(prompts)} samples)")

    # ---- 预热 ----
    if args.warmup_samples > 0:
        warmup_prompts = prompts[-args.warmup_samples:]
        print(f"[warmup] {len(warmup_prompts)} requests...")
        run_requests(
            base_url=args.base_url, model=args.model,
            prompts=warmup_prompts, max_tokens=args.max_tokens,
            temperature=args.temperature, request_timeout=args.request_timeout,
            concurrency=args.concurrency, retry=args.retry,
        )
        print("[warmup] done")

    # ---- 遥测 ----
    before_snapshot = fetch_telemetry_snapshot(args.base_url)
    (output_dir / "telemetry_before.json").write_text(
        json.dumps(before_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    telemetry_path = output_dir / "telemetry.jsonl"
    poller = TelemetryPoller(args.base_url, telemetry_path, args.poll_interval)
    poller.start()

    try:
        records = run_requests(
            base_url=args.base_url, model=args.model, prompts=prompts,
            max_tokens=args.max_tokens, temperature=args.temperature,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency, retry=args.retry,
        )
    finally:
        poller.stop()

    # ---- 请求级别统计 ----
    latencies = [r["latency_sec"] for r in records if r["success"]]
    ttfts = [r["ttft_sec"] for r in records if r["success"] and r["ttft_sec"] is not None]
    tpots = [r["tpot_sec"] for r in records if r["success"] and r["tpot_sec"] is not None]
    itl_means = [r["itl_mean_sec"] for r in records if r["success"] and r.get("itl_mean_sec")]
    completion_tokens = [r["completion_tokens"] for r in records if r["success"]]
    success_count = sum(1 for r in records if r["success"])
    failed_errors = [r["error"] for r in records if not r["success"] and r.get("error")]

    # ---- token 统计 ----
    prompt_token_counts = []
    if tok is not None:
        for item in prompts:
            prompt_token_counts.append(count_tokens(item["prompt"], tok))

    # ---- 写 results.jsonl ----
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    after_snapshot = fetch_telemetry_snapshot(args.base_url)
    (output_dir / "telemetry_after.json").write_text(
        json.dumps(after_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    telemetry_summary = summarize_telemetry(telemetry_path)
    total_latency = sum(latencies)
    total_completion_tokens = sum(completion_tokens)

    # ---- Miniflex metrics（如果存在） ----
    miniflex_metrics = None
    miniflex_metrics_path = Path("/tmp/miniflex_metrics.json")
    if miniflex_metrics_path.exists():
        try:
            miniflex_metrics = json.loads(miniflex_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # ---- summary.json ----
    summary = {
        "run_group_id": args.run_group_id,
        "repeat_index": args.repeat_index,
        "mode": args.mode,
        "scenario": args.scenario,
        "model": args.model,
        "concurrency": args.concurrency,
        "num_samples": len(prompts),
        "warmup_samples": args.warmup_samples,
        "success_count": success_count,
        "failed_count": len(prompts) - success_count,
        "success_rate": success_count / len(prompts) if prompts else 0.0,
        "failed_errors": failed_errors,
        # latency
        "mean_latency_sec": statistics.mean(latencies) if latencies else None,
        "p50_latency_sec": percentile(latencies, 0.50),
        "p95_latency_sec": percentile(latencies, 0.95),
        "p99_latency_sec": percentile(latencies, 0.99),
        "max_latency_sec": max(latencies) if latencies else None,
        # TTFT
        "mean_ttft_sec": statistics.mean(ttfts) if ttfts else None,
        "p50_ttft_sec": percentile(ttfts, 0.50) if ttfts else None,
        "p95_ttft_sec": percentile(ttfts, 0.95) if ttfts else None,
        "p99_ttft_sec": percentile(ttfts, 0.99) if ttfts else None,
        # TPOT
        "mean_tpot_sec": statistics.mean(tpots) if tpots else None,
        "p50_tpot_sec": percentile(tpots, 0.50) if tpots else None,
        "p95_tpot_sec": percentile(tpots, 0.95) if tpots else None,
        # ITL
        "mean_itl_sec": statistics.mean(itl_means) if itl_means else None,
        "p50_itl_sec": percentile(itl_means, 0.50) if itl_means else None,
        "p95_itl_sec": percentile(itl_means, 0.95) if itl_means else None,
        "p99_itl_sec": percentile(itl_means, 0.99) if itl_means else None,
        # token
        "prompt_tokens_mean": (statistics.mean(prompt_token_counts)
                               if prompt_token_counts else None),
        "prompt_tokens_p50": (percentile([float(x) for x in prompt_token_counts], 0.50)
                              if prompt_token_counts else None),
        "prompt_tokens_p95": (percentile([float(x) for x in prompt_token_counts], 0.95)
                              if prompt_token_counts else None),
        "prompt_tokens_max": (max(prompt_token_counts)
                              if prompt_token_counts else None),
        "completion_tokens_mean": (statistics.mean(completion_tokens)
                                   if completion_tokens else None),
        "completion_tokens_total": total_completion_tokens,
        "end_to_end_toks_per_sec":
        (total_completion_tokens / total_latency)
        if total_latency > 0 and total_completion_tokens > 0 else None,
        # miniflex
        "miniflex_metrics": miniflex_metrics,
    }
    summary.update(telemetry_summary)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()

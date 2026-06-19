#!/usr/bin/env python3
"""
summarize_baseline_runs.py — 汇总所有 run 的 summary.json 到 CSV

用法:
  python3 tests/summarize_baseline_runs.py \
    --runs-dir baseline_runs --output-csv baseline_runs_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate summary.json files from baseline runs.")
    p.add_argument("--runs-dir", default="baseline_runs")
    p.add_argument("--output-csv", default="baseline_runs_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows: list[dict] = []

    for sp in sorted(runs_dir.glob("*/summary.json")):
        run_dir = sp.parent
        with sp.open(encoding="utf-8") as f:
            s = json.load(f)

        # config
        cp = run_dir / "config.json"
        cfg = {}
        if cp.exists():
            cfg = json.loads(cp.read_text(encoding="utf-8"))

        rows.append({
            "run_dir": str(run_dir),
            "run_group_id": s.get("run_group_id"),
            "repeat_index": s.get("repeat_index"),
            "scenario": s.get("scenario"),
            "model": s.get("model"),
            "mode": s.get("mode", "baseline"),
            "concurrency": s.get("concurrency"),
            "num_samples": s.get("num_samples"),
            "warmup_samples": s.get("warmup_samples"),
            "success_rate": s.get("success_rate"),
            "success_count": s.get("success_count"),
            "failed_count": s.get("failed_count"),
            "failed_errors": "; ".join(s.get("failed_errors", [])),
            # Latency
            "mean_latency_sec": s.get("mean_latency_sec"),
            "p50_latency_sec": s.get("p50_latency_sec"),
            "p95_latency_sec": s.get("p95_latency_sec"),
            "p99_latency_sec": s.get("p99_latency_sec"),
            "max_latency_sec": s.get("max_latency_sec"),
            # TTFT
            "mean_ttft_sec": s.get("mean_ttft_sec"),
            "p50_ttft_sec": s.get("p50_ttft_sec"),
            "p95_ttft_sec": s.get("p95_ttft_sec"),
            "p99_ttft_sec": s.get("p99_ttft_sec"),
            # TPOT
            "mean_tpot_sec": s.get("mean_tpot_sec"),
            "p50_tpot_sec": s.get("p50_tpot_sec"),
            "p95_tpot_sec": s.get("p95_tpot_sec"),
            # ITL
            "mean_itl_sec": s.get("mean_itl_sec"),
            "p50_itl_sec": s.get("p50_itl_sec"),
            "p95_itl_sec": s.get("p95_itl_sec"),
            "p99_itl_sec": s.get("p99_itl_sec"),
            # Token
            "prompt_tokens_mean": s.get("prompt_tokens_mean"),
            "prompt_tokens_p50": s.get("prompt_tokens_p50"),
            "prompt_tokens_p95": s.get("prompt_tokens_p95"),
            "prompt_tokens_max": s.get("prompt_tokens_max"),
            "completion_tokens_mean": s.get("completion_tokens_mean"),
            "completion_tokens_total": s.get("completion_tokens_total"),
            "end_to_end_toks_per_sec": s.get("end_to_end_toks_per_sec"),
            # Telemetry
            "kv_cache_usage_max": s.get("kv_cache_usage_max"),
            "prefix_queries_delta": s.get("prefix_queries_delta"),
            "prefix_hits_delta": s.get("prefix_hits_delta"),
            "vllm_ttft_sec_mean": s.get("vllm_ttft_sec_mean"),
            "vllm_tpot_sec_mean": s.get("vllm_tpot_sec_mean"),
            "gpu_memory_used_mb_peak": s.get("gpu_memory_used_mb_peak"),
            "gpu_util_pct_peak": s.get("gpu_util_pct_peak"),
            "cpu_memory_used_mb_peak": s.get("cpu_memory_used_mb_peak"),
            "cpu_memory_used_pct_peak": s.get("cpu_memory_used_pct_peak"),
            "loadavg_1m_peak": s.get("loadavg_1m_peak"),
            # Config
            "config_subset": cfg.get("subset"),
            "config_prompt_chars": cfg.get("prompt_chars"),
            "config_prefix_chars": cfg.get("prefix_chars"),
            "config_max_tokens": cfg.get("max_tokens"),
            "config_max_input_tokens": cfg.get("max_input_tokens"),
            "config_max_input_chars": cfg.get("max_input_chars"),
            # Miniflex
            "miniflex_has_metrics": bool(s.get("miniflex_metrics")),
        })

    if not rows:
        raise SystemExit(f"No summary.json found under {runs_dir}")

    # P2 fix: 所有 row 的 keys 取并集，避免旧 run 缺字段导致新字段被丢弃
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

    scenarios = sorted(set(r["scenario"] for r in rows))
    modes = sorted(set(r["mode"] for r in rows))
    print(f"Wrote {len(rows)} rows ({len(all_keys)} columns) → {args.output_csv}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Modes: {modes}")


if __name__ == "__main__":
    main()

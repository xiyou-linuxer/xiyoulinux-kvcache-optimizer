#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate summary.json files from baseline runs.")
    parser.add_argument("--runs-dir",
                        default="baseline_runs",
                        help="Directory containing run subdirectories.")
    parser.add_argument("--output-csv",
                        default="baseline_runs_summary.csv",
                        help="CSV file to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows: list[dict[str, object]] = []

    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        run_dir = summary_path.parent
        with summary_path.open("r", encoding="utf-8") as fin:
            summary = json.load(fin)

        config_path = run_dir / "config.json"
        config = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fin:
                config = json.load(fin)

        rows.append({
            "run_dir": str(run_dir),
            "scenario": summary.get("scenario"),
            "model": summary.get("model"),
            "concurrency": summary.get("concurrency"),
            "num_samples": summary.get("num_samples"),
            "success_rate": summary.get("success_rate"),
            "mean_latency_sec": summary.get("mean_latency_sec"),
            "p50_latency_sec": summary.get("p50_latency_sec"),
            "p95_latency_sec": summary.get("p95_latency_sec"),
            "completion_toks_per_sec":
            summary.get("end_to_end_completion_toks_per_sec"),
            "kv_cache_usage_max": summary.get("kv_cache_usage_max"),
            "prefix_queries_delta": summary.get("prefix_queries_delta"),
            "prefix_hits_delta": summary.get("prefix_hits_delta"),
            "gpu_memory_used_mb_peak": summary.get("gpu_memory_used_mb_peak"),
            "cpu_memory_used_mb_peak": summary.get("cpu_memory_used_mb_peak"),
            "cpu_memory_used_pct_peak":
            summary.get("cpu_memory_used_pct_peak"),
            "loadavg_1m_peak": summary.get("loadavg_1m_peak"),
            "subset": config.get("subset"),
            "prompt_chars": config.get("prompt_chars"),
            "prefix_chars": config.get("prefix_chars"),
            "max_tokens": config.get("max_tokens"),
        })

    if not rows:
        raise SystemExit(f"No summary.json files found under {runs_dir}")

    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

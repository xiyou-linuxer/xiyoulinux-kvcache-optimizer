#!/usr/bin/env python3
"""
prepare_baseline_manifest.py — 为 longbench / needlebench 预筛选 token 级别的 prompt

用法:
  python3 tests/prepare_baseline_manifest.py \
    --scenario longbench --tokenizer /root/autodl-tmp/Qwen/Qwen3-8B \
    --subset qasper --longbench-local-dir /root/autodl-tmp/longbench_local \
    --num-samples 10 --min-prompt-tokens 4000 --max-prompt-tokens 8000 \
    --max-input-chars 120000 --max-input-tokens 8000 \
    --output-manifest baseline_manifests/longbench_qasper_4k_8k.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# 确保 tests 包可导入（编排器从 repo root 运行，也支持直接 python tests/prepare_baseline_manifest.py）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.common import (
    count_tokens,
    load_tokenizer,
    needlebench_prompt_from_sample,
    prompt_from_longbench,
    truncate_by_tokens,
)

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare token-bucketed prompt manifest for baseline runs.")
    p.add_argument("--scenario", required=True,
                   choices=["longbench", "needlebench"])
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-prompt-tokens", type=int, default=0)
    p.add_argument("--max-prompt-tokens", type=int, default=None)
    p.add_argument("--max-input-chars", type=int, default=24000)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--subset", default="qasper")
    p.add_argument("--longbench-local-dir", default=None)
    p.add_argument("--needle-subset", default="en_haystack_texts")
    p.add_argument("--split", default="test")
    return p.parse_args()


def build_longbench_candidates(args: argparse.Namespace, tok: Any) -> list[dict]:
    if args.longbench_local_dir:
        path = Path(args.longbench_local_dir) / "data" / f"{args.subset}.jsonl"
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        with path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        if hf_load_dataset is None:
            raise SystemExit("pip install datasets")
        ds = hf_load_dataset("THUDM/LongBench", args.subset, split=args.split,
                             trust_remote_code=True)
        records = [ds[i] for i in range(len(ds))]

    prompts = []
    for i, sample in enumerate(records):
        raw = prompt_from_longbench(sample, args.max_input_chars)
        prompt = truncate_by_tokens(raw, args.max_input_tokens, tok)
        n = count_tokens(prompt, tok)
        prompts.append({"sample_id": str(sample.get("_id", i)),
                         "prompt": prompt,
                         "meta": {"group": "longbench", "subset": args.subset,
                                   "dataset_index": i,
                                   "answers": sample.get("answers", []),
                                   "source": "local_jsonl" if args.longbench_local_dir else "hf"},
                         "prompt_chars": len(prompt),
                         "prompt_tokens": n})
    return prompts


def build_needlebench_candidates(args: argparse.Namespace, tok: Any) -> list[dict]:
    if hf_load_dataset is None:
        raise SystemExit("pip install datasets")
    ds = hf_load_dataset("opencompass/NeedleBench", args.needle_subset,
                         split=args.split)
    prompts = []
    for i in range(len(ds)):
        sample = ds[i]
        raw, extra_meta = needlebench_prompt_from_sample(sample, args.max_input_chars)
        prompt = truncate_by_tokens(raw, args.max_input_tokens, tok)
        n = count_tokens(prompt, tok)
        prompts.append({"sample_id": str(sample.get("_id", i)),
                         "prompt": prompt,
                         "meta": {"group": "needlebench",
                                   "subset": args.needle_subset,
                                   "dataset_index": i, **extra_meta},
                         "prompt_chars": len(prompt),
                         "prompt_tokens": n})
    return prompts


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tok = load_tokenizer(args.tokenizer)

    if args.scenario == "longbench":
        candidates = build_longbench_candidates(args, tok)
    else:
        candidates = build_needlebench_candidates(args, tok)

    # 按 token 桶筛选
    in_bucket = []
    for item in candidates:
        n = item.get("prompt_tokens")
        if n is None:
            continue
        if n < args.min_prompt_tokens:
            continue
        if args.max_prompt_tokens is not None and n > args.max_prompt_tokens:
            continue
        in_bucket.append(item)

    print(f"  Bucket [{args.min_prompt_tokens}, {args.max_prompt_tokens}]: "
          f"{len(in_bucket)} candidates, need {args.num_samples}")

    if len(in_bucket) < args.num_samples:
        raise SystemExit(
            f"Not enough prompts: found {len(in_bucket)}, need {args.num_samples}. "
            f"Try a wider bucket or smaller --num-samples.")

    random.shuffle(in_bucket)
    selected = in_bucket[:args.num_samples]
    token_values = [it["prompt_tokens"] for it in selected]

    manifest = {
        "scenario": args.scenario,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "subset": (args.subset if args.scenario == "longbench"
                   else args.needle_subset),
        "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "num_samples": len(selected),
        "prompt_tokens_min": min(token_values),
        "prompt_tokens_max": max(token_values),
        "prompts": selected,
    }

    out = Path(args.output_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"  Wrote {len(selected)} prompts → {out}")
    print(f"  Tokens: {min(token_values)}–{max(token_values)}")


if __name__ == "__main__":
    main()

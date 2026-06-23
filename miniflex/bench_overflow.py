#!/usr/bin/env python3
"""容量溢出对比:N 个长前缀(总量 >> GPU KV 容量)轮询访问。
APC 驱逐后 miss=重算(慢);MiniFlex 摊到 CPU/SSD 仍命中(快)。
对 APC / MiniFlex 服务各跑一次,比 median/p90 TTFT 与 疑似 miss 占比。
用法: PYTHONPATH=pysrc python bench_overflow.py --url http://localhost:8000 --tag apc --num-prefixes 12
"""
import argparse
import statistics
import time
import uuid

import requests

NO_PROXY = {"http": None, "https": None}
BODY_UNIT = ("artificial intelligence has transformed many industries over the past decade "
    "including healthcare finance transportation education manufacturing agriculture "
    "and entertainment by enabling machines to learn patterns from large amounts of data ")


def one(url, model, prompt, max_tokens=8, timeout=900):
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0, "stream": True},
        stream=True, proxies=NO_PROXY, timeout=timeout)
    r.raise_for_status()
    ttft = None
    for raw in r.iter_lines():
        if raw and raw.startswith(b"data: ") and raw != b"data: [DONE]":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
    r.close()
    if ttft is None:
        ttft = (time.perf_counter() - t0) * 1000
    return ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--num-prefixes", type=int, default=12)
    ap.add_argument("--prefix-br", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    approx = args.prefix_br * 30
    total_k = args.num_prefixes * approx // 1000
    prefixes = [f"PFX-{i}-{uuid.uuid4().hex} " + BODY_UNIT * args.prefix_br
                for i in range(args.num_prefixes)]
    print(f">>> [{args.tag}] {args.num_prefixes} prefixes x ~{approx} tok = ~{total_k}k working set (GPU KV ~68k)")

    print(">>> warmup (2 遍,确保 PUT 全部 commit) ...")
    for _ in range(2):
        for p in prefixes:
            one(args.url, args.model, p + " q: summarize.")
            one(args.url, args.model, "pump " + uuid.uuid4().hex, 4)

    print(">>> measure (round-robin) ...")
    ttfts = []
    for rd in range(args.rounds):
        for i, p in enumerate(prefixes):
            ttfts.append(one(args.url, args.model, p + f" q{rd}: explain point {i}."))

    med = statistics.median(ttfts)
    p90 = sorted(ttfts)[min(len(ttfts) - 1, int(len(ttfts) * 0.9))]
    th = approx * 0.04
    slow = sum(1 for t in ttfts if t > th)
    pct = 100 * slow // len(ttfts)
    print("")
    print(f"[{args.tag}] median={med:.0f}ms  p90={p90:.0f}ms  miss(>{th:.0f}ms): {slow}/{len(ttfts)} ({pct}%)")
    print("TTFT:", [round(t) for t in ttfts])


if __name__ == "__main__":
    main()

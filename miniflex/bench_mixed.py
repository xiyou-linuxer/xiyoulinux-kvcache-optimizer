#!/usr/bin/env python3
"""混合负载: 高频热 prompt(留GPU->APC) + 长尾前缀(溢出->MiniFlex)。分量热/长尾 TTFT。
对 apc / miniflex / both 各跑一次,看 both 是否在两类上都拿到最好。
用法: PYTHONPATH=pysrc python bench_mixed.py --url http://localhost:8000 --tag both
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
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--hot-br", type=int, default=130)
    ap.add_argument("--tail", type=int, default=12)
    ap.add_argument("--tail-br", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    hots = [f"HOT-{i}-{uuid.uuid4().hex} " + BODY_UNIT * args.hot_br for i in range(args.hot)]
    tails = [f"TAIL-{i}-{uuid.uuid4().hex} " + BODY_UNIT * args.tail_br for i in range(args.tail)]
    tail_k = args.tail * args.tail_br * 30 // 1000
    print(f">>> [{args.tag}] hot {args.hot} + tail {args.tail} (tail 工作集 ~{tail_k}k, GPU ~68k)")
    print(">>> warmup ...")
    for p in hots + tails:
        one(args.url, args.model, p + " q: summarize.")
        one(args.url, args.model, "pump " + uuid.uuid4().hex, 4)
    hot_t = []
    tail_t = []
    print(">>> measure ...")
    for rd in range(args.rounds):
        for i, tp in enumerate(tails):
            h = hots[(rd * args.tail + i) % len(hots)]
            hot_t.append(one(args.url, args.model, h + f" hq{rd}-{i}."))
            tail_t.append(one(args.url, args.model, tp + f" tq{rd}-{i}."))
    med = lambda x: statistics.median(x) if x else 0
    th = args.tail_br * 30 * 0.04
    miss = sum(1 for t in tail_t if t > th)
    print("")
    print(f"[{args.tag}] hot_med={med(hot_t):.0f}ms  tail_med={med(tail_t):.0f}ms  overall={med(hot_t+tail_t):.0f}ms")
    print(f"  tail miss(>{th:.0f}ms): {miss}/{len(tail_t)}")
    print(f"  hot:  {[round(x) for x in hot_t]}")
    print(f"  tail: {[round(x) for x in tail_t]}")


if __name__ == "__main__":
    main()

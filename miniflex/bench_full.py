#!/usr/bin/env python3
"""全面基准:baseline vs MiniFlex,4 个合成场景(short / ultra_long / shared_prefix / mixed_pressure)。

两个关键正确性点:
  1. 每条请求**读完整个流**再结束 —— 否则首 token 后关连接 = server abort = MiniFlex 不 PUT。
  2. 热/命中路径先**预热 + pump** —— 让 MiniFlex 异步 PUT commit、随后 GET 能命中。
  (baseline 模式下这两步无害。)

用法(分别对 baseline / apc / miniflex 服务各跑一次,再 compare3.py 对比):
  PYTHONPATH=pysrc python bench_full.py --url http://localhost:8000 --model qwen3-8b --tag miniflex --out /tmp/mf.json
"""
import argparse
import json
import statistics
import threading
import time
import uuid

import requests

NO_PROXY = {"http": None, "https": None}
WARM_ROUNDS = 3

BODY_UNIT = ("artificial intelligence has transformed many industries over the past decade "
    "including healthcare finance transportation education manufacturing agriculture "
    "and entertainment by enabling machines to learn patterns from large amounts of data ")


def gen_body(repeat):
    return BODY_UNIT * repeat


def one_request(url, model, prompt, max_tokens, timeout=900):
    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0, "stream": True},
        stream=True, proxies=NO_PROXY, timeout=timeout,
    )
    resp.raise_for_status()
    ttft = None
    n = 0
    for raw in resp.iter_lines():
        if raw and raw.startswith(b"data: ") and raw != b"data: [DONE]":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            n += 1
    resp.close()
    return (ttft if ttft is not None else (time.perf_counter() - t0) * 1000), \
        (time.perf_counter() - t0) * 1000, n


def warm(url, model, prompt, max_tokens, rounds=WARM_ROUNDS):
    for _ in range(rounds):
        one_request(url, model, prompt, max_tokens)
        one_request(url, model, "pump " + uuid.uuid4().hex, 4)


def measure(url, model, prompt, max_tokens, runs):
    res = [one_request(url, model, prompt, max_tokens) for _ in range(runs)]
    ttfts = [r[0] for r in res]
    lats = [r[1] for r in res]
    toks = sum(r[2] for r in res)
    return {"ttft_ms": round(statistics.median(ttfts), 1),
            "lat_ms": round(statistics.median(lats), 1),
            "tps": round(toks / (sum(lats) / 1000), 2) if sum(lats) else 0.0}


def scen_context_sweep(url, model, name, brs, labels, runs, max_tokens):
    out = {}
    for br, lab in zip(brs, labels):
        prompt = f"FIXED-{name}-{lab} " + gen_body(br)
        warm(url, model, prompt, max_tokens)
        out[lab] = measure(url, model, prompt, max_tokens, runs)
        m = out[lab]
        print(f"  [{name}] {lab}: TTFT={m['ttft_ms']:.0f}ms lat={m['lat_ms']:.0f}ms tps={m['tps']:.1f}")
    return out


def scen_shared_prefix(url, model, runs, max_tokens):
    out = {}
    for br, lab in [(250, "8k"), (500, "16k")]:
        prefix = f"SHAREDPFX-{lab} " + gen_body(br)
        warm(url, model, prefix + " question: summarize the text.", max_tokens)
        res = [one_request(url, model, prefix + f" question {i}: explain point {i}.", max_tokens)
               for i in range(runs)]
        ttfts = [r[0] for r in res]
        lats = [r[1] for r in res]
        toks = sum(r[2] for r in res)
        out[lab] = {"ttft_ms": round(statistics.median(ttfts), 1),
                    "lat_ms": round(statistics.median(lats), 1),
                    "tps": round(toks / (sum(lats) / 1000), 2) if sum(lats) else 0.0}
        print(f"  [shared_prefix] {lab}: TTFT={out[lab]['ttft_ms']:.0f}ms tps={out[lab]['tps']:.1f}")
    return out


def scen_mixed_pressure(url, model, runs, max_tokens):
    out = {}
    pool = [f"MIX-{k} " + gen_body(120) for k in range(8)]
    for p in pool:
        warm(url, model, p, max_tokens, rounds=2)
    for c in (1, 2, 4, 8):
        results = []
        lock = threading.Lock()

        def work(idx):
            r = one_request(url, model, pool[idx % len(pool)], max_tokens)
            with lock:
                results.append(r)

        t0 = time.perf_counter()
        for _ in range(runs):
            threads = [threading.Thread(target=work, args=(j,)) for j in range(c)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        wall = time.perf_counter() - t0
        toks = sum(r[2] for r in results)
        out[f"c{c}"] = {"tps": round(toks / wall, 2), "reqs": len(results)}
        print(f"  [mixed_pressure] c={c}: aggregate tps={out[f'c{c}']['tps']:.1f} ({len(results)} reqs)")
    return out


def run_all(url, model, runs, max_tokens):
    data = {}
    print(">>> short_synth ...")
    data["short_synth"] = scen_context_sweep(url, model, "short", [30, 65], ["1k", "2k"], runs, max_tokens)
    print(">>> ultra_long ...")
    data["ultra_long"] = scen_context_sweep(url, model, "ultra", [250, 500, 750, 1000],
                                             ["8k", "16k", "24k", "32k"], runs, max_tokens)
    print(">>> shared_prefix ...")
    data["shared_prefix"] = scen_shared_prefix(url, model, runs, max_tokens)
    print(">>> mixed_pressure ...")
    data["mixed_pressure"] = scen_mixed_pressure(url, model, runs, max_tokens)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    print(f">>> tag={args.tag} url={args.url} runs={args.runs} max_tokens={args.max_tokens}")
    t0 = time.perf_counter()
    data = run_all(args.url, args.model, args.runs, args.max_tokens)
    out = args.out or f"/tmp/bench_{args.tag}.json"
    json.dump({"tag": args.tag, "data": data}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n>>> 完成 用时 {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""冷/热 TTFT 基准:衡量 MiniFlex 缓存命中能否降低首 token 延迟(prefill 提速)。

方法(关键:必须把"热"prompt 预热到**确认命中**,否则量到的全是 miss):
  - 冷:每次用【全新 prompt】 -> 保证 miss -> 全量重算 prefill。
  - 热:固定一个 prompt，先反复发若干次并在中间插 pump 请求推进引擎，
        让异步 PUT commit 并对 GET 可见，再计时 -> 走 MiniFlex 加载而非重算。
  - 对比两者 TTFT 中位数。

为什么需要 pump:MiniFlex 的 PUT commit 只在引擎"有请求在 step"时推进。
冷->热成对发、中间空闲时，热请求的 GET 会早于上一个 PUT 的 commit -> miss。
插入 pump 请求能逼引擎多走几步，把 PUT 落地。

前提:服务用 --no-enable-prefix-caching 启动(隔离掉 vLLM 自带显存缓存，
只测 MiniFlex 自身贡献)。开 MINIFLEX_DEBUG=1 可在服务日志里核对 GET_MATCH matched>0。

用法:
  PYTHONPATH=pysrc .venv/bin/python bench_ttft.py
  PYTHONPATH=pysrc .venv/bin/python bench_ttft.py --body-repeat 14 --runs 6
"""
import argparse
import statistics
import time
import uuid

import requests

NO_PROXY = {"http": None, "https": None}

BODY_UNIT = (
    "artificial intelligence has transformed many industries over the past decade "
    "including healthcare finance transportation education manufacturing agriculture "
    "and entertainment by enabling machines to learn patterns from large amounts of data "
)


def measure_ttft(url, model, prompt, max_tokens=8, timeout=120):
    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0, "stream": True},
        stream=True, proxies=NO_PROXY, timeout=timeout,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines():
        if raw and raw.startswith(b"data: ") and raw != b"data: [DONE]":
            t = time.perf_counter() - t0
            resp.close()
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen1.5-0.5B-Chat")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--body-repeat", type=int, default=14,
                    help="正文重复几遍,拉长上下文让 prefill 变贵(越大收益越明显)")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    body = BODY_UNIT * args.body_repeat
    mt = lambda p: measure_ttft(args.url, args.model, p, args.max_tokens)

    # 冷:每次全新 prompt(必 miss = 全量重算)
    print(">>> 测冷(全新 prompt,全量重算)...")
    cold = [mt(f"COLD-{uuid.uuid4().hex} {body}") * 1000 for _ in range(args.runs)]

    # 热:固定 prompt 预热到稳定命中(发若干次 + pump 推进引擎让 PUT commit)
    print(">>> 预热热 prompt 到稳定命中 ...")
    hot_prompt = "HOTFIXED " + body
    for _ in range(4):
        mt(hot_prompt)
        mt("pump " + uuid.uuid4().hex)        # pump:逼引擎 step,落地上一次 PUT
    print(">>> 测热(命中,走 MiniFlex 加载)...")
    hot = [mt(hot_prompt) * 1000 for _ in range(args.runs)]

    cm, hm = statistics.median(cold), statistics.median(hot)
    print("\n" + "=" * 50)
    print(f"冷(重算) TTFT 中位数: {cm:6.1f} ms   {[round(x) for x in cold]}")
    print(f"热(命中) TTFT 中位数: {hm:6.1f} ms   {[round(x) for x in hot]}")
    print(f"加速比: {cm/hm:.2f}x   (节省 {cm-hm:.1f} ms / {100*(cm-hm)/cm:.1f}%)")
    print("=" * 50)
    if cm - hm <= 0:
        print("热没快于冷:上下文/模型太小,搬运开销≈省下的重算。把 --body-repeat 调大,"
              "或换更大模型/更长上下文再看。")
    print("提示:开 MINIFLEX_DEBUG=1 启服务,可在日志核对热请求 GET_MATCH matched>0(确认真命中)。")


if __name__ == "__main__":
    main()

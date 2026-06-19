#!/usr/bin/env python3
"""
run_baseline_v2.py — baseline v2 + FlexKV 一键自动化编排器

一条命令自动跑完 baseline 和/或 FlexKV 的全部测试，无需手动复制粘贴。

用法:
  python3 tests/run_baseline_v2.py                          # 跑完整 baseline
  python3 tests/run_baseline_v2.py --modes miniflex          # 只跑 FlexKV
  python3 tests/run_baseline_v2.py --modes baseline,miniflex # 对比两者
  python3 tests/run_baseline_v2.py --dry-run                 # 预览命令
  python3 tests/run_baseline_v2.py --resume                  # 断点续跑
  python3 tests/run_baseline_v2.py --repeat-count 1 --skip-sanity  # 快速验证
  python3 tests/run_baseline_v2.py --scenarios short_synth,shared_prefix
  python3 tests/run_baseline_v2.py --notify-webhook https://your-webhook

环境变量:
  MODEL_PATH     — 模型路径（默认 /root/autodl-tmp/Qwen/Qwen3-8B）
  LONGBENCH_DIR  — LongBench 目录（默认 /root/autodl-tmp/longbench_local）
  REPEAT_COUNT   — 每配置重复次数（默认 3）
  WEBHOOK_URL    — 通知 webhook URL
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
MINIFLEX_DIR = REPO_ROOT / "miniflex"
RUNNER = str(REPO_ROOT / "tests" / "vllm_baseline_runner.py")
PREPARE = str(REPO_ROOT / "tests" / "prepare_baseline_manifest.py")
SUMMARIZE = str(REPO_ROOT / "tests" / "summarize_baseline_runs.py")
MINIFLEX_START = str(MINIFLEX_DIR / "run_vllm_miniflex.sh")

DEFAULT_MODEL = "/root/autodl-tmp/Qwen/Qwen3-8B"
DEFAULT_LONGBENCH = "/root/autodl-tmp/longbench_local"


# ============================================================
# CLI
# ============================================================
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="baseline v2 + FlexKV 一键编排器")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-sanity", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--repeat-count", type=int, default=3)
    p.add_argument("--modes", default="baseline")
    p.add_argument("--scenarios", default=None)
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    p.add_argument("--longbench-dir", default=os.environ.get("LONGBENCH_DIR", DEFAULT_LONGBENCH))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--output-dir", default="baseline_runs")
    p.add_argument("--manifest-dir", default="baseline_manifests")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--notify-webhook", default=None)
    p.add_argument("--vllm-extra-args", default="")
    p.add_argument("--no-summarize", action="store_true")
    # Miniflex 专属
    p.add_argument("--miniflex-max-model-len", type=int, default=40960,
                   help="max-model-len（默认 40960，Qwen3-8B 上限，同时用于 baseline 和 miniflex）")
    return p.parse_args()


# ============================================================
# 命令构造
# ============================================================
def runner_cmd(cli: argparse.Namespace, scenario: str, group_id: str,
               repeat: int, extra: list[str], mode: str) -> list[str]:
    return (["python3", RUNNER,
             "--base-url", cli.base_url, "--model", cli.model,
             "--scenario", scenario,
             "--tokenizer", cli.model_path,
             "--output-dir", cli.output_dir,
             "--run-group-id", group_id,
             "--repeat-index", str(repeat),
             "--mode", mode] + extra)


def prepare_cmd(cli: argparse.Namespace, scenario: str, subset: str,
                n: int, mn: int, mx: int, mxc: int, mxti: int, out: str, *,
                needle_subset: str | None = None) -> list[str]:
    cmd = ["python3", PREPARE,
           "--scenario", scenario,
           "--tokenizer", cli.model_path,
           "--num-samples", str(n),
           "--min-prompt-tokens", str(mn),
           "--max-prompt-tokens", str(mx),
           "--max-input-chars", str(mxc),
           "--max-input-tokens", str(mxti),
           "--output-manifest", str(Path(cli.manifest_dir) / out)]
    if scenario == "longbench":
        cmd += ["--subset", subset, "--longbench-local-dir", cli.longbench_dir]
    else:
        cmd += ["--needle-subset", needle_subset or "en_haystack_texts"]
    return cmd


def bench_cmd(cli: argparse.Namespace, ilen: int, olen: int = 128,
              n: int = 20, rate: str = "2") -> list[str]:
    return ["vllm", "bench", "serve",
            "--backend", "vllm",
            "--base-url", cli.base_url,
            "--endpoint", "/v1/completions",
            "--model", cli.model,
            "--tokenizer", cli.model_path,
            "--num-prompts", str(n),
            "--random-input-len", str(ilen),
            "--random-output-len", str(olen),
            "--ignore-eos", "--temperature", "0",
            "--request-rate", rate]


# ============================================================
# 步骤定义（参数与执行方案文档一致）
# ============================================================
def build_pipeline(cli: argparse.Namespace, mode: str) -> list[dict]:
    steps = []
    rc = cli.repeat_count
    mdir = str(Path(cli.manifest_dir))

    def mid(s: str) -> str:
        return f"{mode}_{s}"

    def add_runner(phase: str, scenario: str, gid: str,
                   extra: list[str], desc: str) -> None:
        # 自动追加所有 run 共同的参数
        extra = list(extra)  # 不修改调用者传入的列表
        extra += ["--retry", "3"]
        # 用 group_id hash 做 base seed，保证不同 group 的 prompt shuffle 不同
        base_seed = abs(hash(gid)) % 100000
        for r in range(rc):
            seed_val = base_seed + r * 100
            cmd_extra = extra + ["--seed", str(seed_val)]
            steps.append(dict(id=mid(f"{gid}_r{r}"), type="runner",
                              phase=phase, scenario=scenario,
                              group=gid, repeat=r, mode=mode,
                              description=f"[{mode}] {desc} [{r + 1}/{rc}]",
                              cmd=runner_cmd(cli, scenario, gid, r, cmd_extra, mode)))

    def add_prepare(s: str, sub: str, n: int, mn: int, mx: int,
                    mxc: int, mxti: int, out: str,
                    needle: str | None = None) -> None:
        steps.append(dict(id=mid(f"prepare_{out.replace('.json','')}"),
                          type="prepare", phase="prepare", scenario=s,
                          mode=mode,
                          description=f"[{mode}] prepare {out}",
                          cmd=prepare_cmd(cli, s, sub, n, mn, mx, mxc, mxti, out,
                                          needle_subset=needle)))

    def add_bench(desc: str, ilen: int, olen: int = 128,
                  n: int = 20, rate: str = "2") -> None:
        steps.append(dict(id=mid(f"bench_{ilen}"), type="vllm_bench",
                          phase="microbenchmark", mode=mode,
                          description=f"[{mode}] {desc}",
                          cmd=bench_cmd(cli, ilen, olen, n, rate)))

    # ===== Phase prepare: 不依赖 vLLM，先跑 =====
    # LongBench qasper
    add_prepare("longbench", "qasper", 10, 4000, 8000, 120000, 8000,
                "longbench_qasper_4k_8k.json")
    add_prepare("longbench", "qasper", 10, 8000, 16000, 160000, 16000,
                "longbench_qasper_8k_16k.json")
    # LongBench qmsum
    add_prepare("longbench", "qmsum", 10, 4000, 8000, 120000, 8000,
                "longbench_qmsum_4k_8k.json")
    add_prepare("longbench", "qmsum", 10, 8000, 16000, 160000, 16000,
                "longbench_qmsum_8k_16k.json")
    # LongBench narrativeqa
    add_prepare("longbench", "narrativeqa", 10, 4000, 8000, 120000, 8000,
                "longbench_narrativeqa_4k_8k.json")
    add_prepare("longbench", "narrativeqa", 10, 8000, 16000, 160000, 16000,
                "longbench_narrativeqa_8k_16k.json")
    # NeedleBench (10 samples)
    add_prepare("needlebench", "en_haystack_texts", 5, 8000, 24000,
                180000, 24000, "needlebench_en_haystack_8k_24k.json",
                needle="en_haystack_texts")

    # ===== Phase A: 关前缀缓存 =====
    A = "mode_a"

    if not cli.skip_sanity:
        add_runner(A, "short_synth", "sanity_short_synth", [
            "--num-samples", "2", "--prompt-chars", "20000",
            "--max-input-tokens", "1024", "--max-tokens", "128",
            "--warmup-samples", "0", "--concurrency", "1",
        ], "sanity short_synth")
        add_runner(A, "longbench", "sanity_longbench_qasper_4k_8k", [
            "--sample-manifest", f"{mdir}/longbench_qasper_4k_8k.json",
            "--max-input-tokens", "8000", "--max-tokens", "128",
            "--warmup-samples", "0", "--concurrency", "1",
        ], "sanity longbench qasper 4K-8K")

    # 9.1 short_synth
    add_runner(A, "short_synth", "short_synth_1k_c1", [
        "--num-samples", "10", "--prompt-chars", "20000",
        "--max-input-tokens", "1024", "--max-tokens", "128",
        "--warmup-samples", "2", "--concurrency", "1",
    ], "short_synth 1K c1")
    add_runner(A, "short_synth", "short_synth_2k_c1", [
        "--num-samples", "10", "--prompt-chars", "20000",
        "--max-input-tokens", "2048", "--max-tokens", "128",
        "--warmup-samples", "2", "--concurrency", "1",
    ], "short_synth 2K c1")

    # 9.2 longbench — qasper
    lb_qasper_48 = f"{mdir}/longbench_qasper_4k_8k.json"
    lb_qasper_816 = f"{mdir}/longbench_qasper_8k_16k.json"
    for subset_name, lb48, lb816 in [
        ("qasper", lb_qasper_48, lb_qasper_816),
        ("qmsum", f"{mdir}/longbench_qmsum_4k_8k.json", f"{mdir}/longbench_qmsum_8k_16k.json"),
        ("narrativeqa", f"{mdir}/longbench_narrativeqa_4k_8k.json", f"{mdir}/longbench_narrativeqa_8k_16k.json"),
    ]:
        add_runner(A, "longbench", f"longbench_{subset_name}_4k_8k_c1", [
            "--sample-manifest", lb48,
            "--max-input-tokens", "8000", "--max-tokens", "128",
            "--warmup-samples", "2", "--concurrency", "1",
        ], f"longbench {subset_name} 4-8K c1")
        add_runner(A, "longbench", f"longbench_{subset_name}_4k_8k_c2", [
            "--sample-manifest", lb48,
            "--max-input-tokens", "8000", "--max-tokens", "128",
            "--warmup-samples", "2", "--concurrency", "2",
        ], f"longbench {subset_name} 4-8K c2")
        add_runner(A, "longbench", f"longbench_{subset_name}_8k_16k_c1", [
            "--sample-manifest", lb816,
            "--max-input-tokens", "16000", "--max-tokens", "128",
            "--warmup-samples", "2", "--concurrency", "1",
        ], f"longbench {subset_name} 8-16K c1")
        add_runner(A, "longbench", f"longbench_{subset_name}_8k_16k_c2", [
            "--sample-manifest", lb816,
            "--max-input-tokens", "16000", "--max-tokens", "128",
            "--warmup-samples", "2", "--concurrency", "2",
        ], f"longbench {subset_name} 8-16K c2")

    # 9.3 ultra_long_synth
    add_runner(A, "ultra_long_synth", "ultra_long_8k_c1", [
        "--num-samples", "5", "--ultra-prompt-chars", "80000",
        "--max-input-tokens", "8000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "ultra_long 8K")
    add_runner(A, "ultra_long_synth", "ultra_long_16k_c1", [
        "--num-samples", "5", "--ultra-prompt-chars", "160000",
        "--max-input-tokens", "16000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "ultra_long 16K")
    add_runner(A, "ultra_long_synth", "ultra_long_24k_c1", [
        "--num-samples", "5", "--ultra-prompt-chars", "240000",
        "--max-input-tokens", "24000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "ultra_long 24K")
    add_runner(A, "ultra_long_synth", "ultra_long_32k_c1", [
        "--num-samples", "5", "--ultra-prompt-chars", "320000",
        "--max-input-tokens", "32000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "ultra_long 32K")
    add_runner(A, "ultra_long_synth", "ultra_long_16k_c2", [
        "--num-samples", "5", "--ultra-prompt-chars", "160000",
        "--max-input-tokens", "16000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "2",
    ], "ultra_long 16K c2")

    # 9.4 needlebench (10 samples)
    nb_mf = f"{mdir}/needlebench_en_haystack_8k_24k.json"
    add_runner(A, "needlebench", "needlebench_8k_24k_c1", [
        "--sample-manifest", nb_mf,
        "--max-input-tokens", "24000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "needlebench 8-24K c1")

    # 9.6 mixed_pressure
    for c in [1, 2, 4, 8]:
        add_runner(A, "mixed_pressure", f"mixed_pressure_c{c}", [
            "--num-samples", "20",
            "--mixed-short-chars", "20000",
            "--mixed-medium-chars", "80000",
            "--mixed-long-chars", "160000",
            "--prefix-chars", "120000",
            "--max-input-tokens", "16000", "--max-tokens", "128",
            "--warmup-samples", "2", "--concurrency", str(c),
        ], f"mixed_pressure c{c}")

    # ===== Phase B: 开前缀缓存 — shared_prefix =====
    B = "mode_b"

    # 缓存预热：先发一个 shared_prefix 请求把 prefix 写入缓存（不计入正式结果）
    add_runner(B, "shared_prefix", "shared_prefix_cache_warmup", [
        "--num-samples", "1", "--prefix-chars", "120000",
        "--max-input-tokens", "8000", "--max-tokens", "128",
        "--warmup-samples", "0", "--concurrency", "1",
    ], "shared_prefix cache warmup")

    if not cli.skip_sanity:
        add_runner(B, "shared_prefix", "sanity_shared_prefix", [
            "--num-samples", "2", "--prefix-chars", "120000",
            "--max-input-tokens", "8000", "--max-tokens", "128",
            "--warmup-samples", "0", "--concurrency", "1",
        ], "sanity shared_prefix")

    add_runner(B, "shared_prefix", "shared_prefix_4k", [
        "--num-samples", "10", "--prefix-chars", "120000",
        "--max-input-tokens", "4000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "shared_prefix 4K")
    add_runner(B, "shared_prefix", "shared_prefix_8k", [
        "--num-samples", "10", "--prefix-chars", "120000",
        "--max-input-tokens", "8000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "shared_prefix 8K")
    add_runner(B, "shared_prefix", "shared_prefix_16k", [
        "--num-samples", "10", "--prefix-chars", "240000",
        "--max-input-tokens", "16000", "--max-tokens", "128",
        "--warmup-samples", "1", "--concurrency", "1",
    ], "shared_prefix 16K")

    # ===== Microbenchmark =====
    add_bench("micro 2K", 2048, 128, 20, "2")
    add_bench("micro 8K", 8192, 128, 20, "1")
    add_bench("micro 16K", 16384, 128, 10, "0.5")

    return steps


# ============================================================
# 进度追踪
# ============================================================
class Progress:
    def __init__(self, path: Path):
        self.path = path
        self.done: set[str] = set()
        self.failed: dict[str, str] = {}
        if path.exists():
            try:
                d = json.loads(path.read_text())
                self.done = set(d.get("done", []))
                self.failed = d.get("failed", {})
            except (json.JSONDecodeError, KeyError):
                pass

    def ok(self, sid: str) -> None:
        self.done.add(sid); self.failed.pop(sid, None); self._save()

    def fail(self, sid: str, why: str) -> None:
        self.failed[sid] = why; self._save()

    def is_done(self, sid: str) -> bool:
        return sid in self.done

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"done": sorted(self.done), "failed": dict(self.failed)},
            ensure_ascii=False, indent=2))


# ============================================================
# vLLM 生命周期
# ============================================================
def kill_vllm(port: int) -> None:
    try:
        r = subprocess.run(["lsof", "-ti", f":{port}"],
                           capture_output=True, text=True)
        if r.stdout.strip():
            for pid in r.stdout.strip().split():
                subprocess.run(["kill", "-15", pid], capture_output=True)
            time.sleep(2)
            for pid in r.stdout.strip().split():
                subprocess.run(["kill", "-9", pid], capture_output=True)
            print(f"  [kill] port {port}: done")
            time.sleep(2)
    except FileNotFoundError:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "vllm serve"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"], capture_output=True)
        time.sleep(2)
    except Exception:
        pass


def start_vllm_baseline(cli: argparse.Namespace, prefix_caching: bool) -> subprocess.Popen:
    cmd = ["vllm", "serve", cli.model_path,
           "--served-model-name", cli.model,
           "--tokenizer", cli.model_path,
           "--trust-remote-code",
           "--host", "0.0.0.0", "--port", str(cli.port),
           "--max-model-len", str(cli.miniflex_max_model_len),
           "--disable-hybrid-kv-cache-manager"]
    if not prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if cli.vllm_extra_args:
        cmd.extend(cli.vllm_extra_args.split())
    tag = "开" if prefix_caching else "关"
    print(f"  [start baseline] 前缀缓存={tag}")
    print(f"  [start] {' '.join(cmd)}")
    log = (REPO_ROOT / "vllm_startup.log").open("a")  # ← P2: 追加不覆盖
    log.write(f"\n--- [{datetime.now()}] baseline prefix_cache={prefix_caching} ---\n")
    log.flush()
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def start_vllm_miniflex(cli: argparse.Namespace) -> subprocess.Popen:
    env = os.environ.copy()
    env["MODEL"] = cli.model_path
    env["PORT"] = str(cli.port)
    env["PYTHONPATH"] = "pysrc"
    env["MINIFLEX_MAX_MODEL_LEN"] = str(cli.miniflex_max_model_len)

    print(f"  [start miniflex] MODEL={cli.model_path} PORT={cli.port} "
          f"MAX_MODEL_LEN={cli.miniflex_max_model_len}")
    log = (REPO_ROOT / "vllm_startup.log").open("a")
    log.write(f"\n--- [{datetime.now()}] miniflex ---\n")
    log.flush()
    return subprocess.Popen(["bash", MINIFLEX_START],
                            cwd=str(MINIFLEX_DIR),
                            stdout=log, stderr=subprocess.STDOUT,
                            env=env)


def wait_ready(base_url: str, timeout: int = 180) -> bool:
    print(f"  [wait] 等待 vLLM 就绪...", end="", flush=True)
    dl = time.time() + timeout
    while time.time() < dl:
        try:
            if urllib.request.urlopen(f"{base_url}/v1/models", timeout=5).status == 200:
                print(" OK")
                return True
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print(" TIMEOUT!")
    return False


# ============================================================
# Manifest 存在性检查（P1）
# ============================================================
def check_manifests(cli: argparse.Namespace, steps: list[dict]) -> None:
    """在 vLLM 启动前检查所有引用的 manifest 文件是否存在。"""
    needed = set()
    for s in steps:
        for i, arg in enumerate(s["cmd"]):
            if arg == "--sample-manifest" and i + 1 < len(s["cmd"]):
                needed.add(s["cmd"][i + 1])
    missing = [p for p in needed if not Path(p).exists()]
    if missing:
        print(f"\n  [ERROR] Missing manifest files:")
        for p in missing:
            print(f"    - {p}")
        print(f"  Hint: run prepare_baseline_manifest.py first, or check --manifest-dir")
        raise SystemExit(1)
    if needed:
        print(f"  [check] {len(needed)} manifest files OK")


# ============================================================
# 执行
# ============================================================
def run_one_streaming(cmd: list[str], sid: str, desc: str,
                      to: int = 900) -> tuple[bool, str]:
    """P0 fix: 实时流式输出，不再缓冲。"""
    print(f"\n{'='*65}")
    print(f"  [{sid}] {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*65}")

    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    try:
        # 实时打印每一行
        for line in proc.stdout:
            print(f"  | {line}", end="")
    except Exception:
        pass
    try:
        proc.wait(timeout=to)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False, f"Timeout {to}s"

    return (True, "OK") if proc.returncode == 0 else (False, f"rc={proc.returncode}")


def run_phase(steps: list[dict], prog: Progress | None) -> tuple[int, int]:
    ok = fail = 0
    for i, s in enumerate(steps):
        if prog and prog.is_done(s["id"]):
            print(f"\n  [{i + 1}/{len(steps)}] SKIP: {s['description']}")
            ok += 1
            continue
        success, reason = run_one_streaming(s["cmd"], s["id"], s["description"])
        if success:
            ok += 1
            if prog: prog.ok(s["id"])
        else:
            fail += 1
            if prog: prog.fail(s["id"], reason)
            print(f"\n  [FAIL] {s['description']}: {reason}")
    return ok, fail


def summarize(cli: argparse.Namespace) -> None:
    cmd = ["python3", SUMMARIZE,
           "--runs-dir", cli.output_dir,
           "--output-csv", f"{cli.output_dir}_summary.csv"]
    print(f"\n  [summarize] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT))


def notify(url: str, msg: str) -> None:
    try:
        payload = json.dumps({"msgtype": "text",
                              "text": {"content": msg}}).encode()
        urllib.request.urlopen(urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}), timeout=10)
    except Exception as e:
        print(f"  [notify] fail: {e}")


# ============================================================
# 单 mode 执行
# ============================================================
def run_mode(cli: argparse.Namespace, mode: str, prog: Progress | None) -> tuple[int, int]:
    steps = build_pipeline(cli, mode)

    if cli.scenarios:
        wanted = set(cli.scenarios.split(","))
        steps = [s for s in steps
                 if s.get("scenario", "") in wanted
                 or any(w in s.get("group", "") for w in wanted)]
        if not steps:
            print(f"  [{mode}] 无匹配场景")
            return 0, 0

    # 按 phase 分组
    phases: dict[str, list[dict]] = {}
    for s in steps:
        phases.setdefault(s["phase"], []).append(s)

    ok_total = fail_total = 0

    # === Phase prepare: 不依赖 vLLM ===
    if "prepare" in phases:
        print(f"\n{'#'*65}\n  [{mode}] PHASE PREPARE\n{'#'*65}")
        ok, fail = run_phase(phases["prepare"], prog)
        ok_total += ok; fail_total += fail

    # Manifest 存在性检查（vLLM 启动前）
    check_manifests(cli, steps)

    # === Phase A: 关前缀缓存 ===
    if "mode_a" in phases:
        print(f"\n{'#'*65}\n  [{mode}] PHASE A — 关闭前缀缓存\n{'#'*65}")
        kill_vllm(cli.port)
        if mode == "baseline":
            start_vllm_baseline(cli, prefix_caching=False)
        else:
            start_vllm_miniflex(cli)
        if not wait_ready(cli.base_url):
            print(f"  [{mode}] [FATAL] vLLM 启动失败")
            return ok_total, fail_total + len(phases["mode_a"])
        ok, fail = run_phase(phases["mode_a"], prog)
        ok_total += ok; fail_total += fail
        kill_vllm(cli.port)

    # === Phase B: 开前缀缓存 — shared_prefix ===
    if "mode_b" in phases:
        print(f"\n{'#'*65}\n  [{mode}] PHASE B — 开启前缀缓存\n{'#'*65}")
        kill_vllm(cli.port)
        if mode == "baseline":
            start_vllm_baseline(cli, prefix_caching=True)
        else:
            # Miniflex 模式下也用 miniflex 启动（run_vllm_miniflex.sh 内部
            # 有 --no-enable-prefix-caching，外部缓存命中由 connector 实现）
            start_vllm_miniflex(cli)
        if not wait_ready(cli.base_url):
            print(f"  [{mode}] [FATAL] vLLM 启动失败")
            return ok_total, fail_total + len(phases["mode_b"])
        ok, fail = run_phase(phases["mode_b"], prog)
        ok_total += ok; fail_total += fail

    # === Microbenchmark ===
    if "microbenchmark" in phases:
        print(f"\n{'#'*65}\n  [{mode}] PHASE — Microbenchmark\n{'#'*65}")
        if not wait_ready(cli.base_url, timeout=10):
            kill_vllm(cli.port)
            if mode == "baseline":
                start_vllm_baseline(cli, prefix_caching=True)
            else:
                start_vllm_miniflex(cli)
            wait_ready(cli.base_url)
        ok, fail = run_phase(phases["microbenchmark"], prog)
        ok_total += ok; fail_total += fail

    kill_vllm(cli.port)
    return ok_total, fail_total


# ============================================================
# 主入口
# ============================================================
def main() -> None:
    cli = parse_cli()
    modes = [m.strip() for m in cli.modes.replace("-", ",").split(",") if m.strip()]

    # dry-run
    if cli.dry_run:
        print(f"\n  DRY RUN — modes: {modes}\n")
        for mode in modes:
            steps = build_pipeline(cli, mode)
            if cli.scenarios:
                wanted = set(cli.scenarios.split(","))
                steps = [s for s in steps
                         if s.get("scenario", "") in wanted
                         or any(w in s.get("group", "") for w in wanted)]
            print(f"  [{mode}] {len(steps)} steps:")
            for i, s in enumerate(steps):
                print(f"    [{i + 1:03d}] {s['description']}")
                print(f"          {' '.join(s['cmd'])}\n")
        return

    # resume
    prog_path = REPO_ROOT / ".baseline_v2_progress.json"
    prog = Progress(prog_path) if cli.resume else None
    if prog:
        print(f"  Resume: {len(prog.done)} done, {len(prog.failed)} failed")

    t0 = time.time()
    grand_ok = grand_fail = 0

    print(f"\n{'#'*70}")
    print(f"  baseline v2 + FlexKV 自动化编排器")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模式: {modes}  |  每配置重复: {cli.repeat_count}x")
    print(f"  模型: {cli.model_path}")
    print(f"{'#'*70}")

    for mode in modes:
        print(f"\n{'#'*70}\n  MODE: {mode}\n{'#'*70}")
        ok, fail = run_mode(cli, mode, prog)
        grand_ok += ok; grand_fail += fail
        print(f"\n  [{mode}] 完成: {ok} OK / {fail} FAIL")

    if not cli.no_summarize:
        print(f"\n{'#'*70}\n  汇总 CSV...\n{'#'*70}")
        summarize(cli)

    elapsed = f"{int((time.time() - t0) // 60)}m {int((time.time() - t0) % 60)}s"
    msg = (f"baseline v2 完成\n"
           f"modes: {modes} | OK: {grand_ok} | FAIL: {grand_fail}\n"
           f"耗时: {elapsed}\nCSV: {cli.output_dir}_summary.csv")
    print(f"\n{'#'*70}\n  {msg.replace(chr(10), chr(10) + '  ')}\n{'#'*70}")

    if cli.notify_webhook:
        notify(cli.notify_webhook, msg)

    if grand_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

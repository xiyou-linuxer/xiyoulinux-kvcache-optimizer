#!/usr/bin/env python3
"""N 路对比: python compare3.py base.json apc.json mf.json (倍率相对第一个)"""
import json
import sys


def load(p):
    d = json.load(open(p))
    return d.get("tag", p), d["data"]


def fmt(v, nd):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def rat(base, v, hb):
    if not isinstance(base, (int, float)) or not isinstance(v, (int, float)) or base == 0 or v == 0:
        return ""
    r = (v / base) if hb else (base / v)
    return f"({r:.2f}x)"


def col(metric, scen, lab, datas, nd):
    base = datas[0].get(scen, {}).get(lab, {}).get(metric)
    out = []
    for i, d in enumerate(datas):
        v = d.get(scen, {}).get(lab, {}).get(metric)
        out.append(f"{fmt(v, nd)}{'' if i == 0 else rat(base, v, metric == 'tps')}")
    return out


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("用法: python compare3.py base.json apc.json mf.json")
        return
    runs = [load(p) for p in paths]
    tags = [t for t, _ in runs]
    datas = [d for _, d in runs]
    W = 18
    print(f"\n===== 对比({' / '.join(tags)});倍率相对 [{tags[0]}] =====")
    for scen in ("short_synth", "ultra_long", "shared_prefix"):
        print(f"\n# {scen}")
        for metric, nd, note in [("ttft_ms", 0, "TTFT ms 越低越好"), ("tps", 1, "tps 越高越好")]:
            print("  " + note)
            print("  " + "档位".ljust(6) + "".join(t.rjust(W) for t in tags))
            for lab in datas[0].get(scen, {}):
                print("  " + lab.ljust(6) + "".join(c.rjust(W) for c in col(metric, scen, lab, datas, nd)))
    print("\n# mixed_pressure (聚合 tps 越高越好)")
    print("  " + "并发".ljust(6) + "".join(t.rjust(W) for t in tags))
    for lab in datas[0].get("mixed_pressure", {}):
        print("  " + lab.ljust(6) + "".join(c.rjust(W) for c in col("tps", "mixed_pressure", lab, datas, 1)))
    print("=" * 60)


if __name__ == "__main__":
    main()

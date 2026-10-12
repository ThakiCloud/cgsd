#!/usr/bin/env python3
"""Paired K comparison — CI overlap is not a test of difference (v1.11 D73).

The three K runs share selection seeds and training seeds, so the difference can be taken
replicate by replicate instead of comparing two marginal CIs:

    d_rep = E_{K=a}(sel, seed) - E_{K=b}(sel, seed)

which cancels the shared selection/training noise. Bootstrap is hierarchical (selection seed,
then training seed), matching the main analysis. No new training runs.

    python3 experiments/scripts/ksens_paired.py
"""
import json, math, random, statistics as st
from pathlib import Path

# docs/experiments in the working tree, results/ in the public release — and this file sits at a
# different depth in each (experiments/scripts/ vs scripts/), so walk up rather than assume one.
_D = next((a / c for a in Path(__file__).resolve().parents
           for c in ("docs/experiments", "results") if (a / c).is_dir()), None)
if _D is None:
    raise SystemExit("⛔ 결과 디렉터리를 찾을 수 없다 (docs/experiments 또는 results)")
BASE = str(_D / "2026-09-18-cgsd-pilot-en-ksens-k{}-b59900.ckpt.jsonl")
UNIT = 3914


def load(k):
    y = {}
    for line in open(BASE.format(k)):
        r = json.loads(line)
        y.setdefault((r["sel"], r["seed"]), {})[r["arm"]] = r["y"]
    out = {}
    for key, a in y.items():
        if len(a) == 5:
            out[key] = {"E": a["d10"] - a["d00"], "S": a["d01"] - a["d00"],
                        "I": a["d11"] - a["d10"] - a["d01"] + a["d00"],
                        "delta_G": a["d11"] - a["d00"]}
    return out


def hier_boot(pairs, rng, n=4000):
    by = {}
    for (s_, ts), v in pairs.items():
        by.setdefault(s_, []).append(v)
    sels = list(by)
    ms = []
    for _ in range(n):
        vals = []
        for s_ in (rng.choice(sels) for _ in sels):
            vals += [rng.choice(by[s_]) for _ in by[s_]]
        ms.append(sum(vals) / len(vals))
    ms.sort()
    at = lambda q: ms[min(len(ms) - 1, int(q / 100 * len(ms)))]
    return at(2.5), at(97.5)


def main():
    runs = {k: load(k) for k in (4, 8, 16)}
    rng = random.Random(23)
    print(f"공통 예산 59,900 · selection 3 × training 3 · 1건 = 1/{UNIT}")
    for a, b in ((8, 4), (16, 8), (16, 4)):
        keys = sorted(set(runs[a]) & set(runs[b]))
        print(f"\n[K={a} − K={b}]  paired replicates = {len(keys)}")
        for c in ("E", "S", "I", "delta_G"):
            d = {k: runs[a][k][c] - runs[b][k][c] for k in keys}
            m = st.mean(d.values()); lo, hi = hier_boot(d, rng)
            pos = sum(x > 0 for x in d.values()); neg = sum(x < 0 for x in d.values())
            zero = " (0 포함 — 차이 근거 없음)" if lo <= 0 <= hi else " ⚠ 0 제외 — 차이 있음"
            print(f"  {c:<8} mean={m * UNIT:+6.1f}건 CI95[{lo * UNIT:+6.1f},{hi * UNIT:+6.1f}]"
                  f" 부호 +{pos}/-{neg}{zero}")


if __name__ == "__main__":
    main()

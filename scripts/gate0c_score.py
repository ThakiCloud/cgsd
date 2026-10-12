#!/usr/bin/env python3
"""Score the TRAINED Gate 0C by known truth only (v1.6 D39).

In the trained layer the magnitude of a planted effect is unknown. What IS known exactly:

  zero-truth components   e_only: S* = 0 (gate support == source support, so D01 ~ D00)
                                  I* = 0 (and D11 ~ D10)
                          s_only: E* = 0 (no corruption, eligibility random inside a support)
                                  I* = 0
  sign                    E* > 0 in e_only, e_and_s, e_s_i
  invariance              E*(e_only) == E*(e_and_s): D10 is all-clean in both (eligible slots are
                          filled clean-first and 0.25, 0.479 < clean rate 0.55), D00 is identical

Bias/RMSE are reported ONLY for zero-truth components. No bias is invented for nonzero ones.

    python3 experiments/scripts/gate0c_score.py docs/experiments/2026-09-16-cgsd-gate0c-en.ckpt.jsonl --unit 3914
"""
import argparse, json, math, random

ZERO = {"e_only": ("S", "I"), "s_only": ("E", "I")}
# v1.7 D49: with n=12 and a discrete item-count outcome, primary inference is the paired bootstrap
# CI plus the sign balance. The t statistic is reported but is supplementary — a single rep can
# drag the mean (2026-09-17: two reps at -11/-15건 made a -4.1건 "leak" that vanished at n=12).


def boot_ci(v, rng, n=4000, lo=2.5, hi=97.5):
    ms = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    at = lambda q: ms[min(len(ms) - 1, int(q / 100 * len(ms)))]
    return at(lo), at(hi)


def summarize(v, rng):
    n = len(v); m = sum(v) / n
    med = sorted(v)[n // 2] if n % 2 else (sorted(v)[n // 2 - 1] + sorted(v)[n // 2]) / 2
    se = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n) if n > 1 else float("nan")
    lo, hi = boot_ci(v, rng) if n > 1 else (float("nan"),) * 2
    pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v)
    return {"n": n, "mean": m, "median": med, "ci95": [lo, hi], "se": se,
            "t": (m / se) if se == se and se else float("nan"), "sign_pos": pos, "sign_neg": neg}
POS_E = ("e_only", "e_and_s", "e_s_i")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--unit", type=int, required=True)
    ap.add_argument("--budget", type=int, default=None)
    a = ap.parse_args()
    u = 1 / a.unit
    rng = random.Random(0)
    recs = [json.loads(l) for l in open(a.ckpt)]
    if a.budget:
        recs = [r for r in recs if r["budget"] == a.budget]
    by = {}
    for r in recs:
        by.setdefault(r["cond"], {})[r["rep"]] = r
    out = {"unit": a.unit, "zero_truth": {}, "sign": {}, "invariance": {}}
    print(f"budget(s)={sorted({r['budget'] for r in recs})}  (1건 = {u:.6f})")
    for cond, comps in ZERO.items():
        for k in comps:
            v = [r[k] for r in by.get(cond, {}).values()]
            if not v:
                continue
            g = summarize(v, rng) | {"rmse": math.sqrt(sum(x * x for x in v) / len(v))}
            out["zero_truth"][f"{cond}.{k}"] = g
            print(f"  zero-truth {cond:<7} {k}: n={g['n']} mean={g['mean'] / u:+.1f}건 "
                  f"med={g['median'] / u:+.1f} CI95[{g['ci95'][0] / u:+.1f},{g['ci95'][1] / u:+.1f}]"
                  f"{' ✅0포함' if g['ci95'][0] <= 0 <= g['ci95'][1] else ' ⛔0제외'} "
                  f"부호 +{g['sign_pos']}/-{g['sign_neg']}  rmse={g['rmse'] / u:.1f}건  (t={g['t']:+.2f})")
    for cond in POS_E:
        v = [r["E"] for r in by.get(cond, {}).values()]
        if v:
            out["sign"][cond] = {"n": len(v), "positive": sum(x > 0 for x in v)}
            print(f"  sign E>0   {cond:<7} {sum(x > 0 for x in v)}/{len(v)}")
    a1, a2 = by.get("e_only", {}), by.get("e_and_s", {})
    reps = sorted(set(a1) & set(a2))
    if reps:
        g = summarize([a2[i]["E"] - a1[i]["E"] for i in reps], rng)
        out["invariance"]["E(e_and_s)-E(e_only)"] = g
        print(f"  invariance E(e_and_s)−E(e_only): n={g['n']} mean={g['mean'] / u:+.1f}건 "
              f"CI95[{g['ci95'][0] / u:+.1f},{g['ci95'][1] / u:+.1f}]"
              f"{' ✅0포함' if g['ci95'][0] <= 0 <= g['ci95'][1] else ' ⛔0제외'}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()

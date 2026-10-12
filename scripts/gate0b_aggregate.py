#!/usr/bin/env python3
"""Gate 0B aggregate — delta with its Monte Carlo uncertainty (v1.6 D42).

Reads every checkpoint for one placebo run (single-process or sharded), dedups by rep, and reports
per component: delta = q95(|null|), a distribution-free 95% CI for that quantile from order
statistics (binomial), a bootstrap SE, the null median, and an exact sign test.

    python3 experiments/scripts/gate0b_aggregate.py docs/experiments/2026-09-17-cgsd-gate0b-en-b55300.json \
        --unit 3914 --write
"""
import argparse, json, math, random
from pathlib import Path


def calibration_signature(locale, K, budget, n_items, seeds, gate, pass_rate,
                          endpoint="leaf_top1_full"):
    """Everything a delta must share with the run it judges (v1.12 D78).

    ⛔ The GATE and its realized pass rate are part of this (v1.16). The null replaces
    eligibility with a random mask OF THE SAME RATE, so a gate that passes 60% of the pool has a
    different null than one that passes 48% — and a second gate family cannot borrow the first
    one's band. We found this by trying to add one.

    ⛔ Locale alone is not enough. delta measured at one exposure does not transfer to another —
    we measured E growing and per-run noise shrinking with budget — and K changes the support
    geometry, hence the null variance. The constructor file is hashed too: change how arms are
    built and the old null no longer describes them.
    """
    import hashlib, json
    ctor = (Path(__file__).resolve().parents[2] / "cgsd" / "constructor" / "build_arms.py")
    payload = {"locale": locale, "K": K, "budget": budget, "n_items": n_items,
               "seeds": seeds, "gate": gate, "pass_rate": round(float(pass_rate), 4),
               "endpoint": endpoint,
               "constructor_sha": hashlib.sha256(ctor.read_bytes()).hexdigest()[:12]}
    payload["signature"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return payload


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def q_ci(xs, q=0.95, level=0.95):
    """Order-statistic CI for the q-quantile: smallest (l,u) with P(l <= rank <= u) >= level."""
    n, s = len(xs), sorted(xs)
    a = (1 - level) / 2
    lo = max(0, next((k for k in range(n) if binom_cdf(k, n, q) >= a), 0) - 1)
    hi = next((k for k in range(n) if binom_cdf(k, n, q) >= 1 - a), n - 1)
    return s[lo], s[min(hi, n - 1)], hi < n - 1       # False = upper end is not identified


def pctl(xs, q):
    s = sorted(xs); i = q * (len(s) - 1); f = math.floor(i)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (i - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--unit", type=int, required=True, help="holdout size (1 item = 1/unit)")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None,
                    help="training seeds per arm — required if the checkpoints predate v1.13")
    ap.add_argument("--k", type=int, default=None,
                    help="support count — required if the checkpoints predate v1.13")
    ap.add_argument("--locale", default=None, choices=("en", "jp"),
                    help="required when the filename does not name exactly one locale")
    ap.add_argument("--gate", default=None,
                    help="gate identity — required if the checkpoints predate v1.16")
    ap.add_argument("--pass-rate", type=float, default=None,
                    help="realized eligibility rate — required if not in the checkpoints")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    recs = {}
    for f in sorted(out.parent.glob(out.stem + ".*ckpt.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if a.budget is None or r["budget"] == a.budget:
                recs[r["rep"]] = r
    budgets = {r["budget"] for r in recs.values()}
    assert len(budgets) == 1, f"mixed budgets {budgets} — pass --budget"
    n = len(recs); u = 1 / a.unit
    bud = budgets.pop()
    # ⛔ EVERY signature input is observed-or-explicit; missing -> reject (v1.14 D98).
    # Never missing -> default. A default stamped seeds=5 onto a null generated with 3
    # (v1.13 D93), and the same shape lurked in K and in the locale sniff below.
    first = recs[next(iter(recs))]

    def observed_or_explicit(field, flag, flagname):
        v = first.get(field)
        if v is None:
            v = flag
        if v is None:
            raise SystemExit(
                f"⛔ 체크포인트에 `{field}` 가 없다. {flagname} 로 명시하라 — 기본값으로 채우면 "
                f"서명이 측정하지 않은 사실을 주장하게 된다 (v1.14 D98).")
        return v

    K = observed_or_explicit("K", a.k, "--k")
    seeds = observed_or_explicit("seeds", a.seeds, "--seeds")
    gate = observed_or_explicit("gate", a.gate, "--gate")
    rate = observed_or_explicit("pass_rate", a.pass_rate, "--pass-rate")
    hits = [x for x in ("en", "jp") if f"-{x}-" in out.name or out.name.endswith(f"-{x}.json")]
    if a.locale:
        loc = a.locale
    elif len(hits) == 1:
        loc = hits[0]
    else:
        raise SystemExit(
            f"⛔ 파일명 {out.name} 에서 로케일을 확정할 수 없다(후보 {hits}). --locale 로 명시하라 "
            "— 예전 코드는 여기서 조용히 'en' 으로 떨어졌다 (v1.14 D98).")
    res = {"budget": bud, "reps": n, "unit": a.unit,
           "calibration": calibration_signature(loc, K, bud, a.unit, seeds, gate, rate),
           "components": {}}
    print("signature:", res["calibration"]["signature"], res["calibration"])
    rng = random.Random(0)
    print(f"reps={n}  budget={res['budget']:,}  (1건 = {u:.6f})")
    for c in "ESI":
        v = [recs[k][c] for k in sorted(recs)]; av = [abs(x) for x in v]
        d = pctl(av, 0.95); lo, hi, ident = q_ci(av)
        boot = [pctl([rng.choice(av) for _ in av], 0.95) for _ in range(2000)]
        bm = sum(boot) / len(boot); se = math.sqrt(sum((b - bm) ** 2 for b in boot) / (len(boot) - 1))
        pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v); k = pos + neg
        p = min(1.0, 2 * binom_cdf(min(pos, neg), k, 0.5)) if k else 1.0
        med = sorted(v)[n // 2] if n % 2 else (sorted(v)[n // 2 - 1] + sorted(v)[n // 2]) / 2
        exc = {i: sum(x >= i / a.unit - 1e-12 for x in av) / n for i in (1, 2, 3, 5)}
        res["components"][c] = {"exceedance": exc, "delta": d, "delta_items": d / u, "delta_ci95": [lo, hi],
                                "delta_ci95_upper_identified": ident, "delta_boot_se": se,
                                "null_median": med, "sign": [pos, neg, n - k], "sign_p": p,
                                "n_upper_tail": sum(x >= d for x in av)}
        print(f"  {c}: delta={d:.5f} ({d / u:.1f}건)  CI95=[{lo / u:.1f}, {hi / u:.1f}]건"
              f"{'' if ident else ' (상한 미식별)'}  boot SE={se / u:.2f}건  tail n={res['components'][c]['n_upper_tail']}"
              f"  null median={med / u:+.1f}건  sign +{pos}/-{neg} p={p:.3f}\n"
              + "      P(|null| >= k건): " + "  ".join(f"{k}건 {v:.2f}" for k, v in exc.items()))
    if a.write:
        p = out.with_name(out.stem + "-calibrated.json")
        p.write_text(json.dumps(res, indent=2), encoding="utf-8"); print(f"wrote {p}")


if __name__ == "__main__":
    main()

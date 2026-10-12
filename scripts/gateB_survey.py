#!/usr/bin/env python3
"""Gate B — model-based confidence-threshold gate on the SAME pool (feasibility survey).

Gate A (the one in the paper) is a verifier: pass/fail from two deterministic checks.
Gate B thresholds the mapper's own continuous confidence. Same data, same learner, same
support partition -- only the eligibility rule changes. That is what makes it a second gate
FAMILY rather than another locale.

⛔ Support is unchanged: k-means on the pre-gate query embedding. `map_confidence` is a gate
output and may define eligibility, never support.

⛔ Feasibility first. This script commits no training -- it reports M, G_min and the extinction
mass at each threshold so we can pick operating points that are inside F_Pi before spending
compute.

    python3 experiments/scripts/gateB_survey.py --locale en
"""
import argparse, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cgsd" / "constructor"))
import build_arms as BA      # noqa: E402  (read-only; never modified)
import feasibility as FE     # noqa: E402
import extinction as EX      # noqa: E402

TOL = BA.BUDGET_TOL
M_SAFE = 0.08


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="en", choices=("en", "jp"))
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    BA.K = a.k

    import json
    fn, _, _ = BA.DS[a.locale]
    conf = {}
    for line in (BA.SRC / fn).open():
        r = json.loads(line)
        if r.get("map_confidence") is not None:
            conf[r["example_id"]] = float(r["map_confidence"])

    pool, _ = BA.load_pool(a.locale)
    T = sum(r["tokens"] for r in pool)
    have = [conf[r["example_id"]] for r in pool if r["example_id"] in conf]
    print(f"{a.locale}: N={len(pool):,} T={T:,}  confidence 있는 행 {len(have):,}")
    print("  분포:", {f"{v:.2f}": c for v, c in sorted(Counter(round(x, 2) for x in have).items())})

    base_rate = sum(r["passed"] for r in pool) / len(pool)
    print(f"  Gate A pass rate = {base_rate:.3f}\n")
    print(f"{'thr':>6}{'pass':>8}{'min_e':>8}{'B/T@M_safe':>12}{'B':>10}{'G_min':>9}{'ext':>7}  판정")
    for thr in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for r in pool:
            r["passed"] = conf.get(r["example_id"], 0.0) >= thr
        rate = sum(r["passed"] for r in pool) / len(pool)
        A, Z, st = EX.survivor_split(pool)
        if not A:
            print(f"{thr:>6.2f}{rate:>8.3f}  전 support 소멸"); continue
        min_e = min(st["e"].values())
        cap = min_e - M_SAFE
        B = int(cap * T) if cap > 0 else 0
        if B <= 0:
            print(f"{thr:>6.2f}{rate:>8.3f}{min_e:>8.3f}{'—':>12}{'—':>10}{'—':>9}"
                  f"{st['ext_tok']:>7.1%}  ⛔ interior 없음")
            continue
        plan = FE.four_arm_shares(pool)
        G = min(FE.granularity(sh, src, B, TOL)[0] for _, sh, src in plan)
        ok = "✅ 구성 가능" if G >= 0 else "⛔ 조립 불가"
        print(f"{thr:>6.2f}{rate:>8.3f}{min_e:>8.3f}{cap:>12.3f}{B:>10,}{G:>+9.0f}"
              f"{st['ext_tok']:>7.1%}  {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

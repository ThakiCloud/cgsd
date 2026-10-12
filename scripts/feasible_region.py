#!/usr/bin/env python3
"""F_Pi over a budget grid, at several support partitions (v1.15 D4).

⛔ Parameterised by the PARTITION, not by K — two partitions with the same K differ in
granularity. Reports the ESCI retrieval pool at K = 4 / 8 / 16 because that is where we
actually control the partition (ADF supports are fixed pre-gate tags).

⚠️ No claim that "K up => G down" in general. This measures what OUR partitions did.

    nice -n 19 python3 experiments/scripts/feasible_region.py --locale en
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cgsd" / "constructor"))
import build_arms as BA      # noqa: E402  (imported, never modified — delta calibration in flight)
import feasibility as FE     # noqa: E402

TOL = BA.BUDGET_TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="en", choices=("en", "jp"))
    ap.add_argument("--ks", default="4,8,16")
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--json", default=None, help="dump the region for figure generation")
    a = ap.parse_args()

    dump = {}
    print(f"locale={a.locale}  tol={TOL:.1%}")
    for k in [int(x) for x in a.ks.split(",")]:
        BA.K = k
        pool, _ = BA.load_pool(a.locale)
        T = sum(r["tokens"] for r in pool)
        grid = [int(T * f) for f in
                [0.05 + i * (0.55 - 0.05) / (a.points - 1) for i in range(a.points)]]
        res = FE.feasible_region(pool, grid, TOL)
        feas = [B for B, _, _, ok in res if ok]
        span = (f"{min(feas) / T:.2f}–{max(feas) / T:.2f}" if feas else "∅")
        print(f"\nK={k:<3} N={len(pool):,} T={T:,}  F_Pi(B/T) = {span}  "
              f"({len(feas)}/{len(res)} 격자점)")
        print(f"  {'B/T':>6}{'B':>10}{'M':>9}{'min G':>10}  판정")
        for B, M, G, ok in res:
            print(f"  {B / T:>6.2f}{B:>10,}{M:>+9.3f}{G:>+10.0f}  "
                  + ("✅" if ok else ("⛔ M<0" if M < 0 else "⛔ G<0")))
        dump[k] = {"T": T, "N": len(pool),
                   "rows": [{"B": B, "B_over_T": B / T, "M": M, "G": G, "ok": ok}
                            for B, M, G, ok in res]}
    if a.json:
        import json
        Path(a.json).write_text(json.dumps(dump, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

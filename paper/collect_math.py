#!/usr/bin/env python3
"""Math feasibility facts for the paper — computed, never transcribed.

    python3 paper/collect_math.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cgsd" / "constructor"))
# experiments/scripts in the working tree, scripts/ in the public release.
for _c in ("experiments/scripts", "scripts"):
    if (ROOT / _c).is_dir():
        sys.path.insert(0, str(ROOT / _c))
import build_arms as BA          # noqa: E402
import extinction as EX          # noqa: E402
import feasibility as FE         # noqa: E402
from math_extinction_build import load, M_SAFE       # noqa: E402
from granularity_condition import arm_shares         # noqa: E402


def main():
    pool = load()
    A, Z, st = EX.survivor_split(pool)
    B = int((st["min_e_A"] - M_SAFE) * st["T_A"])
    out = {"N": st["N"], "T": st["T"], "T_A": st["T_A"], "M_safe": M_SAFE,
           "min_e_A": st["min_e_A"], "interior_B": B, "tol": BA.BUDGET_TOL,
           "ext_token_mass": st["ext_tok"], "ext_row_mass": st["ext_row"],
           "supports": {c: {"rows": st["rows"][c], "tokens": st["tok"][c], "e": st["e"][c],
                            "extinct": c in Z} for c in sorted(st["tok"])},
           "four_arm_infeasible_at": []}
    for frac in (0.001, 0.01, 0.05, 0.10, 0.30, 0.54):
        try:
            BA.build(pool, max(int(frac * st["T"]), 1), seed=0)
            out["four_arm_infeasible_at"].append([frac, False])
        except BA.InfeasibleBudget:
            out["four_arm_infeasible_at"].append([frac, True])
    out["G"] = {}
    for name, sh, src in arm_shares(pool)[1]:
        G, Q, q, exact = FE.granularity(sh, src, B, BA.BUDGET_TOL)
        out["G"][name] = {"G": G, "Q": Q, "tolB": BA.BUDGET_TOL * B, "exact": exact,
                          "q_by_support": q}
    # ⛔ G is not monotone in B, so a verdict from ONE budget is a verdict about that budget.
    # The paper concluded "not realizable under this finite-pool design" from B alone; an
    # exhaustive scan found 54 budgets under the same margin where every arm assembles
    # (2026-09-19). Generated, never transcribed.
    cap_all = int(st["min_e_A"] * st["T_A"])          # F_Pi bound: M_A >= 0
    cap_safe = int((st["min_e_A"] - M_SAFE) * st["T_A"])   # the pre-registered power margin
    shares = arm_shares(pool)[1]
    ok_all, ok_safe = [], []
    for b in range(1, cap_all + 1):
        if all(FE.granularity(sh, src, b, BA.BUDGET_TOL)[0] >= 0 for _, sh, src in shares):
            ok_all.append(b)
            if b <= cap_safe:
                ok_safe.append(b)

    def bands(xs):
        out, s0 = [], (xs[0] if xs else None)
        for a, b in zip(xs, xs[1:] + [None]):
            if b != (a or 0) + 1:
                out.append([s0, a]); s0 = b
        return out

    # ⛔ Attempt EVERY certified budget, not a sample of three. Reporting successes from a
    # truncated attempt next to the full G count puts two denominators in one sentence — the
    # defect an external review caught in the SQL table on 2026-09-20.
    built, why = {}, {}
    for b in sorted({B, *ok_safe}):
        try:
            EX.build_extinction(pool, b, seed=0, strict=True)
            built[b] = "built"
        except BA.InfeasibleBudget as e:
            k = ("granularity" if "GRANULARITY" in str(e)
                 else "packer" if "PACKER" in str(e) else "frontier")
            built[b] = k
            why[k] = why.get(k, 0) + 1
    out["budget_scan"] = {
        "cap_mass": cap_all, "cap_margin": cap_safe,
        "n_all": len(ok_all), "n_margin": len(ok_safe),
        "bands_margin": bands(ok_safe), "bands_all": bands(ok_all)[:8],
        "BT_margin": [min(ok_safe) / st["T"], max(ok_safe) / st["T"]] if ok_safe else None,
        "n_attempted": len(built), "n_built": sum(v == "built" for v in built.values()),
        "build_failures": why,
        "build_at": {str(k): v for k, v in built.items()},
    }
    p = Path(__file__).resolve().parent / "results-math.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {p}")
    print(f"  N={out['N']} |Z|={sum(1 for v in out['supports'].values() if v['extinct'])} "
          f"ext_tok={out['ext_token_mass']:.3f} ext_row={out['ext_row_mass']:.3f} B={B:,}")
    print("  G: " + " ".join(f"{k}={v['G']:+.0f}" for k, v in out["G"].items()))
    print(f"  4팔 모든 예산 실패: {all(x[1] for x in out['four_arm_infeasible_at'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

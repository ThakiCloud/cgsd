#!/usr/bin/env python3
"""Granularity floor — how many rows a pool needs before matched exposure is achievable.

⛔ A FOURTH boundary, distinct from the three we already have:
  resolution   holdout too coarse to resolve the effect      (jp, v1.4)
  frontier     M < 0 at this budget — lower B and it is fine (v1.6 D40)
  extinction   e_c = 0 — no budget helps                     (v1.13 D83)
  granularity  rows are coarse relative to B, so no arm can land within the matching
               tolerance AT ANY BUDGET — the arms are never matched, so nothing is estimable

The mechanism: arms are packed per support, and each support leaves a residual smaller than
its smallest remaining row. Summed over |C| supports that residual is roughly |C| * t_bar / 2,
so matching to `tol` needs  B >= |C| * t_bar / (2 * tol)  — and B is itself capped by the
interior rule. Below that, "budget-matched arms" is not an approximation, it is unavailable.

⛔ That bound is an AVERAGE statement and not the verdict. The formal condition is
`G(B) = tau*B - Q(B) >= 0` in constructor/feasibility.py (v1.14 D96), which solves the
attainable-sum lattice exactly — use `granularity_condition.py` to decide, this script
only to SIZE. And never quote its row counts as a method threshold (v1.14 D99).

This script measures the floor by RESAMPLING the real pool to larger N (with replacement).
⚠️ The resampled pool is a SIZING CALCULATION, never a result — no outcome is computed on it.

    python3 experiments/scripts/granularity_floor.py
"""
import random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cgsd" / "constructor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_arms as BA      # noqa: E402
import extinction as EX      # noqa: E402
from math_extinction_build import load, M_SAFE   # noqa: E402
from domain_operating_point import load as load_pool   # noqa: E402


def grow(pool, n, seed=0):
    """Resample to n rows, preserving each support's share and length distribution."""
    rng = random.Random(seed)
    by = {}
    for r in pool:
        by.setdefault(r["support"], []).append(r)
    out, N = [], len(pool)
    for c, rs in by.items():
        for i in range(max(1, round(n * len(rs) / N))):
            s = dict(rng.choice(rs))
            s["example_id"] = f"{c}-{i}"
            out.append(s)
    return out


def passes(pool, m_safe=M_SAFE, seed=0):
    A, Z, st = EX.survivor_split(pool)
    B = int((st["min_e_A"] - m_safe) * st["T_A"])
    if B <= 0:
        return None, B
    try:
        EX.build_extinction(pool, B, seed=seed)
        return True, B
    except BA.InfeasibleBudget:
        return False, B


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "math-1"
    pool = load() if name == "math-1" else load_pool(name)
    A, Z, st = EX.survivor_split(pool)
    tbar = st["T"] / st["N"]
    need_B = len(st["tok"]) * tbar / (2 * BA.BUDGET_TOL)
    print(f"{name}: N={st['N']}  t̄={tbar:.1f} 토큰/행  |C|={len(st['tok'])}  "
          f"interior B={int((st['min_e_A'] - M_SAFE) * st['T_A']):,}")
    B_int = (st["min_e_A"] - M_SAFE) * st["T_A"]
    ratio = (f"{need_B / B_int:.1f}배가 필요하다" if B_int > 0
             else "interior 예산 자체가 없다 — 배수는 정의되지 않는다")
    print(f"경험칙 B ≥ |C|·t̄/(2·tol) = {need_B:,.0f} 토큰  →  현재 예산의 {ratio}\n")

    print(f"{'N(행)':>7}{'B':>8}  3 seed 결과")
    first = None
    for n in (100, 200, 400, 800, 1600, 3200, 6400):
        g = grow(pool, n)
        res = [passes(g, seed=s)[0] for s in range(3)]
        _, B = passes(g)
        ok = all(res)
        print(f"{len(g):>7}{B:>8,}  " + " ".join("✅" if r else "⛔" for r in res)
              + ("   ← 처음으로 3/3 통과" if ok and first is None else ""))
        if ok and first is None:
            first = len(g)
    print(f"\n⇒ 매칭(±0.5%)이 가능해지는 최소 규모 ≈ **{first}행** "
          f"(현재 {st['N']}행의 {first / st['N']:.0f}배)" if first else "\n⇒ 6,400행까지도 실패")
    print("⚠️ 이건 '분해를 추정할 수 있는' 규모가 아니라 '팔을 맞출 수 있는' 하한이다. "
          "검정력은 그 위에서 따로 따져야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""G(B) on the real pools — does an in-tolerance assembly EXIST? (v1.14)

⛔ Separates two things the earlier sizing script conflated:
     G(B) <  0   no assembly exists      -> pool property, fix = data
     G(B) >= 0   one exists but the greedy packer missed it -> algorithm, fix = packer

    python3 experiments/scripts/granularity_condition.py [pool]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cgsd" / "constructor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_arms as BA          # noqa: E402
import extinction as EX          # noqa: E402
import feasibility as FE         # noqa: E402
from math_extinction_build import load, M_SAFE      # noqa: E402
from domain_operating_point import load as load_pool  # noqa: E402

TOL = BA.BUDGET_TOL


def arm_shares(pool):
    A, Z, st = EX.survivor_split(pool)
    passed = [r for r in pool if r["passed"]]
    PT = sum(r["tokens"] for r in passed)
    sup = sorted(st["tok"])
    src = {c: [r for r in pool if r["support"] == c] for c in sup}
    gate = {c: [r for r in passed if r["support"] == c] for c in sup}
    p_all = {c: st["tok"][c] / st["T"] for c in sup}
    p_A = {c: st["tok"][c] / st["T_A"] for c in A}
    p_G = {c: sum(r["tokens"] for r in gate[c]) / PT for c in sup}
    return st, (("d00_all", p_all, src), ("d00_A", p_A, src), ("d10_A", p_A, gate),
                ("d01_A", {c: p_G[c] for c in A}, src), ("d11", p_G, gate))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "math-1"
    pool = load() if name == "math-1" else load_pool(name)
    st, plan = arm_shares(pool)
    B_int = int((st["min_e_A"] - M_SAFE) * st["T_A"])
    print(f"{name}: N={st['N']} T={st['T']:,} T_A={st['T_A']:,} min_e_A={st['min_e_A']:.3f} "
          f"interior B={B_int:,}  tol={TOL:.1%}")
    print(f"\n{'B':>7} {'tol*B':>7} | " + " ".join(f"{n:>10}" for n, _, _ in plan) + "   판정")
    for B in [b for b in (300, 600, 900, 1200, 1500, 1800, B_int, 2400, 3000) if b > 0]:
        cells, worst = [], None
        for _, sh, src in plan:
            G, Q, q, exact = FE.granularity(sh, src, B, TOL)
            cells.append(f"{G:>+10.0f}" + ("" if exact else "~"))
            worst = G if worst is None else min(worst, G)
        mark = "✅ 조립 존재" if worst >= 0 else "⛔ 조립 불가"
        star = "  ← interior 상한" if B == B_int else ""
        print(f"{B:>7} {TOL * B:>7.1f} | " + " ".join(cells) + f"   {mark}{star}")

    print("\n[판별] interior 예산에서 G 의 부호가 결론을 가른다")
    G, Q, q, exact = FE.granularity(plan[0][1], plan[0][2], B_int, TOL)
    print(f"  d00_all @ B={B_int:,}: Q={Q:.0f} 토큰, 허용 {TOL * B_int:.1f} → G={G:+.0f}"
          + ("" if exact else "  (근사)"))
    print("  support 별 q_c: " + ", ".join(f"{c}={v:.0f}" for c, v in sorted(q.items())))
    if G < 0:
        print(f"  ⇒ ⛔ **조립 자체가 불가능하다.** 어떤 패커도 못 만든다 — 풀을 키워야 한다.")
    else:
        print(f"  ⇒ ⚠️ **조립은 존재한다.** 2.56% 미달은 greedy packer 의 한계다 — 패커를 고쳐야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

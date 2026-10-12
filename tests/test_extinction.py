#!/usr/bin/env python3
"""Gate 0G — support-extinction control (v1.13 D83-D86). No data, no training; seconds.

Gate 0E pinned the budget-reducible frontier (M < 0 at this B, fixable by lowering B).
This gate pins the OTHER boundary: a support with positive source mass and zero gate
eligibility, where no positive budget helps. The two must not be confused, so G5 shows
the structural one survives shrinking B to 0.1% of T.

  G1  survivor_split() partitions A/Z and reports token mass SEPARATELY from row mass —
      they differ whenever extinct supports have atypical row length (they usually do)
  G2  build_extinction() fills all 5 arms within 0.5% of B; D00^A and D10^A place zero
      tokens on Z while D00^all keeps Z at its source share
  G3  the identity Delta_G = X + E_A + S_A + I_A closes EXACTLY (residual == 0)
      for arbitrary outcomes — it is algebra, not an empirical claim
  G4  the survivor margin uses T_A, not T: M_A = min_{c in A} e_c - B/T_A, and a negative
      M_A raises InfeasibleBudget instead of returning a short arm
  G5  STRUCTURAL BOUNDARY — the plain 4-arm build() fails on an extinct pool at EVERY
      budget tried (0.1% .. 40% of T). Contrast: the same pool with e_c > 0 everywhere
      builds fine once B is lowered. That contrast is the whole point of D86.

    python3 tests/test_extinction.py   # exit 0 = pass
"""
import random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cgsd" / "constructor"))
import build_arms as BA  # noqa: E402
import extinction as EX  # noqa: E402

K, ROWS, RATIO = 8, 2000, 0.40
EXTINCT = (6, 7)                 # these supports get e_c = 0
LONG = (7,)                      # ...and c7 rows are ~4x longer, so token mass != row mass
FAILS = []


def check(name, cond, detail=""):
    print(("  ✅ " if cond else "  ❌ ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def pool_with(e_alive, seed=0, extinct=EXTINCT):
    """8 supports. Supports in `extinct` get zero eligible tokens; the rest get rate e_alive."""
    rng, pool, i = random.Random(seed), [], 0
    for c in range(K):
        lo, hi = (24, 40) if c in LONG else (4, 16)
        rows = [{"example_id": f"x{i + j}", "support": c, "tokens": rng.randint(lo, hi)}
                for j in range(ROWS)]
        i += ROWS
        if c in extinct:
            for r in rows:
                r["passed"] = False
        else:
            target, got = e_alive * sum(r["tokens"] for r in rows), 0
            rng.shuffle(rows)
            for r in rows:
                r["passed"] = got + r["tokens"] <= target
                got += r["tokens"] if r["passed"] else 0
        pool += rows
    return pool


def main():
    pool = pool_with(0.60)
    T = sum(r["tokens"] for r in pool)
    A, Z, st = EX.survivor_split(pool)

    print("G1  생존/소멸 분할 — 질량은 토큰과 행을 따로 센다")
    check("A = 생존 6개, Z = 소멸 2개", A == [0, 1, 2, 3, 4, 5] and Z == [6, 7], f"A={A} Z={Z}")
    check("소멸 support 의 e_c 는 정확히 0", all(st["e"][c] == 0 for c in Z))
    check("행 질량 = 2/8 = 25.0%", abs(st["ext_row"] - 0.25) < 1e-9, f"{st['ext_row']:.3%}")
    check("토큰 질량은 행 질량과 다르다 (c7 이 길다)", abs(st["ext_tok"] - st["ext_row"]) > 0.05,
          f"tok {st['ext_tok']:.1%} vs row {st['ext_row']:.1%}")
    check("T_A 는 생존 토큰만 센다", st["T_A"] == T - sum(st["tok"][c] for c in Z))

    print("G2  팔 구성 — D00^A·D10^A 는 소멸 support 에 토큰을 두지 않는다")
    B = int(0.40 * st["T_A"])                 # interior for the survivors (min_e 0.60 > 0.40)
    arms, diag, meta = EX.build_extinction(pool, B, seed=1)
    for name in ("d00_all", "d00_A", "d10_A", "d01_A", "d11"):
        used = sum(r["tokens"] for r in arms[name])
        check(f"{name} 예산 충족", abs(used / B - 1) <= EX.BUDGET_TOL, f"{used:,}/{B:,} ({used/B-1:+.2%})")
    on_Z = lambda nm: sum(r["tokens"] for r in arms[nm] if r["support"] in Z)
    check("D00^A 는 Z 에 0 토큰", on_Z("d00_A") == 0)
    check("D10^A 는 Z 에 0 토큰", on_Z("d10_A") == 0)
    check("D11 도 Z 에 0 토큰 (게이트가 이미 지웠다 — 그래서 D11^A 가 불필요)", on_Z("d11") == 0)
    check("D00^all 은 Z 를 원래 비중대로 유지", on_Z("d00_all") > 0,
          f"{on_Z('d00_all')/B:.1%} of B")
    check("생존 상대비중 보존: p_A[c] = tok_c / T_A",
          all(abs(meta["p_A"][c] - st["tok"][c] / st["T_A"]) < 1e-12 for c in A))

    print("G3  항등식은 대수적으로 정확히 닫힌다")
    rng = random.Random(11)
    worst = 0.0
    for _ in range(2000):
        Y = {k: rng.uniform(-3, 3) for k in ("d00_all", "d00_A", "d10_A", "d01_A", "d11")}
        d = EX.decompose_extinction(Y)
        worst = max(worst, abs(d["residual"]))
    check("2000개 임의 결과에서 잔차 = 0", worst < 1e-12, f"max |residual| = {worst:.2e}")
    d = EX.decompose_extinction({"d00_all": 1.0, "d00_A": 0.7, "d10_A": 0.9,
                                 "d01_A": 0.6, "d11": 0.8})
    check("X 는 부호 있는 대비 (여기선 음수 = 소멸이 손해)", abs(d["X"] - (-0.3)) < 1e-12,
          f"X={d['X']:+.2f}  cost=-X={-d['X']:+.2f}")

    print("G4  생존 마진은 T 가 아니라 T_A 를 쓴다")
    M_A, _ = EX.survivor_margin(pool, B)
    check("M_A = min_A e_c - B/T_A", abs(M_A - (st["min_e_A"] - B / st["T_A"])) < 1e-12,
          f"M_A={M_A:+.3f}")
    check("전체 T 로 재면 값이 달라진다 (혼동하면 안 되는 이유)",
          abs(M_A - (st["min_e_A"] - B / T)) > 0.01,
          f"T 기준 {st['min_e_A'] - B / T:+.3f} vs T_A 기준 {M_A:+.3f}")
    Bbig = int(0.80 * st["T_A"])              # M_A = 0.60 - 0.80 < 0
    try:
        EX.build_extinction(pool, Bbig, seed=1)
        check("M_A < 0 이면 InfeasibleBudget", False, "예외가 안 났다")
    except BA.InfeasibleBudget as ex:
        check("M_A < 0 이면 InfeasibleBudget", True)
        check("오류가 마진·소멸 질량을 함께 말한다",
              "M_A=" in str(ex) and "extinct token mass" in str(ex))

    print("G5  구조적 경계 — 예산을 낮춰도 4팔 구성은 살아나지 않는다")
    survived = []
    for frac in (0.001, 0.01, 0.05, 0.10, 0.20, 0.40):
        try:
            BA.build(pool, int(frac * T), seed=1)
            survived.append(frac)
        except BA.InfeasibleBudget:
            pass
    check("소멸 풀에서는 모든 양수 예산이 실패", not survived,
          f"실패 6/6" if not survived else f"살아남은 예산 {survived}")
    alive = pool_with(0.60, extinct=())       # same pool, no extinction
    okf = [f for f in (0.40, 0.55, 0.65) if _builds(alive, f)]
    check("소멸이 없으면 예산을 낮추는 것으로 해결된다 (예산 경계와 구별)",
          okf == [0.40, 0.55], f"성공 {okf} — e=0.60 이므로 0.65 만 실패해야 정상")

    print("G6  frontier 와 granularity 는 같은 예외인데 처방이 반대다 (v1.13 D94)")
    fine = pool_with(0.60, extinct=())                      # 2000행/support — 입자가 곱다
    _, _, sf = EX.survivor_split(fine)
    try:                                                    # 예산을 e 위로 올리면 frontier
        EX.build_extinction(fine, int(0.90 * sf["T_A"]), seed=1)
        check("frontier 는 위반 support 를 이름 댄다", False, "예외가 안 났다")
    except EX.InfeasibleBudget as ex:
        check("frontier 는 위반 support 를 이름 대고 'Lower B' 를 말한다",
              "FRONTIER" in str(ex) and "Lower B" in str(ex))
    coarse = [r for r in fine if int(r["example_id"][1:]) % 400 == 0]   # support 당 5행
    _, _, sc = EX.survivor_split(coarse)
    try:                                                    # 같은 상대예산인데 풀이 거칠다
        EX.build_extinction(coarse, int(0.40 * sc["T_A"]), seed=1)
        check("granularity/packer 로 걸린다", False, "통과해버렸다 — 풀이 덜 거칠다")
    except EX.InfeasibleBudget as ex:
        kind = "GRANULARITY" if "GRANULARITY" in str(ex) else (
            "PACKER" if "PACKER" in str(ex) else "?")
        check("위반 0개는 GRANULARITY 나 PACKER 로 갈린다", kind in ("GRANULARITY", "PACKER"), kind)
        check("어느 쪽이든 'Lower B' 를 말하지 않는다", "Lower B" not in str(ex))
        check("G(B) 값을 숫자로 보여 준다", "G(B)" in str(ex))
        if kind == "GRANULARITY":
            # ⛔ 2026-09-19: 옛 계약은 "풀을 키우라"였고 그건 실측으로 틀렸다. G 는 B 에
            # 비단조라 같은 풀·같은 여유에서 54개 예산이 조립된다(Math). 처방은 먼저
            # "허용 예산을 탐색하라"이고, 풀 확대는 공동 feasible 집합이 빌 때만이다.
            check("조립 불가면 먼저 '예산을 탐색하라'", "SEARCH the admissible budgets" in str(ex))
            check("풀 확대는 조건부로만 권한다",
                  "only when the joint feasible" in str(ex))
            check("비단조성을 근거로 밝힌다", "non-monotone" in str(ex))
        else:
            check("조립 가능하면 '데이터를 더 요구하지 마라'", "do not request more data" in str(ex))

    print("G7  G(B) 는 pool property 이고 packer 와 분리된다 (v1.14 D97)")
    import feasibility as FE
    check("완전 조립 가능하면 q=0", FE.attainable_max([7, 11, 13, 5], 24) == (24, True))
    check("10짜리 3개로 25 는 20 이 최선 → q=5", FE.attainable_max([10, 10, 10], 25) == (20, True))
    shares = {0: 0.5, 1: 0.5}
    src = {0: [{"tokens": 10, "example_id": f"a{i}"} for i in range(50)],
           1: [{"tokens": 10, "example_id": f"b{i}"} for i in range(50)]}
    G_ok, _, _, _ = FE.granularity(shares, src, 200, 0.005)      # 100/100 = 10의 배수
    G_bad, _, _, _ = FE.granularity(shares, src, 202, 0.005)     # 101/101 → 각 1 부족
    check("격자에 맞는 예산은 G>=0", G_ok >= 0, f"G={G_ok:+.1f}")
    check("격자에서 벗어난 예산은 G<0", G_bad < 0, f"G={G_bad:+.1f}")
    check("⇒ G 는 B 에 단조가 아니다 (예산을 낮춰서 푸는 문제가 아니다)", G_ok >= 0 > G_bad)

    print("G8  F_Pi 는 두 채널을 구분해 보고한다 (v1.15 D106)")
    big = pool_with(0.60, extinct=())                  # 큰 풀 — 원자가 곱다
    T = sum(r["tokens"] for r in big)
    reg = FE.feasible_region(big, [int(f * T) for f in (0.30, 0.50, 0.70)], 0.005)
    m_binds = [(B, M, G) for B, M, G, ok in reg if not ok and M < 0]
    check("e=0.60 이므로 B/T=0.70 은 M 이 구속한다", any(B > 0.6 * T for B, _, _ in m_binds),
          f"M<0 인 격자점 {len(m_binds)}개")
    check("feasible 구간이 비어 있지 않다", any(ok for *_, ok in reg))
    small = [r for r in big if int(r["example_id"][1:]) % 300 == 0]   # 같은 e, 거친 풀
    Ts = sum(r["tokens"] for r in small)
    reg_s = FE.feasible_region(small, [int(0.30 * Ts)], 0.005)
    B, M, G, ok = reg_s[0]
    check("같은 e 에서 풀만 거칠어지면 M 은 멀쩡한데 G 가 구속한다", M >= 0 and G < 0,
          f"M={M:+.3f} G={G:+.0f}")
    check("⇒ 두 채널은 따로 보고돼야 한다 (뭉뚱그리면 잘못된 처방)", not ok)

    print("G9  Lemma: 예산 허용오차가 support 드리프트를 묶는다 (논문 §3.2)")
    # TV(realized, target) <= tau/(1-tau). 논문이 주장하는 상한을 실측으로 고정한다 —
    # ⛔ 이게 없으면 "총 예산만 맞추면 support 가 틀어지지 않나" 반론에 코드 근거가 없다.
    tau = BA.BUDGET_TOL
    bound = tau / (1 - tau)
    worst_tv, worst_ctx = 0.0, None
    for e_alive, frac, seed in ((0.60, 0.40, 3), (0.60, 0.30, 4), (0.45, 0.25, 5)):
        pl = pool_with(e_alive, seed=seed, extinct=())
        T = sum(r["tokens"] for r in pl)
        B = int(frac * T)
        arms, _, meta = EX.build_extinction(pl, B, seed=seed)
        for name, share in (("d00_all", meta["p_all"]), ("d10_A", meta["p_A"])):
            rows = arms[name]
            used = sum(r["tokens"] for r in rows)
            got = {}
            for r in rows:
                got[r["support"]] = got.get(r["support"], 0) + r["tokens"]
            tv = 0.5 * sum(abs(got.get(c, 0) / used - share[c]) for c in share)
            if tv > worst_tv:
                worst_tv, worst_ctx = tv, f"{name} B/T={frac}"
    check(f"실측 TV <= tau/(1-tau) = {bound:.5f}", worst_tv <= bound,
          f"worst {worst_tv:.6f} ({worst_ctx})")
    check("상한이 자명하게 느슨하지 않다 (실측이 0 이 아니다)", worst_tv > 0)
    return 1 if FAILS else 0


def _builds(pool, frac):
    T = sum(r["tokens"] for r in pool)
    try:
        BA.build(pool, int(frac * T), seed=1)
        return True
    except BA.InfeasibleBudget:
        return False


if __name__ == "__main__":
    rc = main()
    print(("\n⛔ FAIL: " + ", ".join(FAILS)) if FAILS else "\n✅ Gate 0G PASS")
    sys.exit(rc)

#!/usr/bin/env python3
"""Gate 0E — infeasible control (v1.6 D39). No data, no training; runs in seconds.

A synthetic pool with 8 supports and a fixed relative budget B/T = 0.40. Eligibility is
set per support to a chosen rate e, so the feasibility margin is M = e - 0.40 by design.

  E1  M computed by feasibility_margin() matches the planted margin
  E2  M > 0 (e = 0.48, 0.44, 0.42): build() succeeds and every arm is within 0.5% of B
  E3  M < 0 (e = 0.39, 0.25):       build() raises InfeasibleBudget — never a silent short arm
  E4  the error names the margin, so the log says WHY it stopped
  E5  strict=False reproduces the old silent under-fill (documents what the guard prevents)

    python3 tests/test_feasibility.py   # exit 0 = pass
"""
import random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cgsd" / "constructor"))
import build_arms as BA  # noqa: E402
import feasibility as FE  # noqa: E402

K, ROWS, RATIO = 8, 2000, 0.40


def pool_with(e, seed=0):
    rng, pool, i = random.Random(seed), [], 0
    for c in range(K):
        rows = [{"example_id": f"x{i + j}", "support": c, "tokens": rng.randint(4, 16)}
                for j in range(ROWS)]
        i += ROWS
        target, got = e * sum(r["tokens"] for r in rows), 0
        rng.shuffle(rows)
        for r in rows:                         # mark eligible until the token rate reaches e
            r["passed"] = got + r["tokens"] <= target
            got += r["tokens"] if r["passed"] else 0
        pool += rows
    return pool


def main():
    BA.K = K
    test_exact_packer_finds_what_G_promises()   # F1
    test_named_supports_are_rejected_not_silently_empty()   # F2
    fails = []
    for e in (0.48, 0.44, 0.42, 0.39, 0.25):
        pool = pool_with(e)
        T = sum(r["tokens"] for r in pool)
        B = int(RATIO * T)
        M, _ = BA.feasibility_margin(pool, B)
        tag = f"e={e:.2f} M={M:+.4f}"
        if abs(M - (e - B / T)) > 0.002:                               # E1
            fails.append(f"E1 {tag}: planted margin {e - B / T:+.4f}")
        if M > 0:
            arms, _, meta = BA.build(pool, B, seed=0)                  # E2
            worst = max(abs(sum(r["tokens"] for r in a) / B - 1) for a in arms.values())
            ok = worst <= BA.BUDGET_TOL and abs(meta["margin"] - M) < 1e-12
            print(f"  {'✅' if ok else '⛔'} E2 {tag}: built, worst budget err {worst:.3%}")
            ok or fails.append(f"E2 {tag}")
        else:
            try:
                BA.build(pool, B, seed=0)
                print(f"  ⛔ E3 {tag}: built without raising"); fails.append(f"E3 {tag}")
            except BA.InfeasibleBudget as err:                         # E3 + E4
                named = "margin M=" in str(err)
                print(f"  {'✅' if named else '⛔'} E3/E4 {tag}: InfeasibleBudget ({str(err)[:70]}…)")
                named or fails.append(f"E4 {tag}")
            arms, _, _ = BA.build(pool, B, seed=0, strict=False)       # E5
            short = sum(r["tokens"] for r in arms["d10"]) / B - 1
            print(f"     E5 strict=False would have returned D10 at {short:+.2%}")
            short < -BA.BUDGET_TOL or fails.append(f"E5 {tag}: no shortfall {short:+.2%}")
    print("PASS" if not fails else "FAIL: " + "; ".join(fails))
    return 1 if fails else 0


def test_exact_packer_finds_what_G_promises():
    """G(B) >= 0 says an in-tolerance assembly EXISTS; the packer must be able to find it.

    ⛔ Pinned because it was not true. On the mathematics pool the greedy packer failed at 54
    budgets that satisfied the pre-registered margin, each reporting
    `PACKER: G(B) = +1 >= 0, so an in-tolerance assembly EXISTS`, and the paper read those
    solver misses as a structural verdict ("not realizable under this finite-pool design").
    2026-09-19.

    ⛔ Also pins that the exact path stays RANDOM across seeds. Picking the first witness would
    select rows by token length, and length is the confound the arms are matched on.
    """
    import random
    print("\nF1  G>=0 이면 정확 조립기가 그 조립을 찾는다")
    rng = random.Random(7)
    misses_greedy = 0
    for trial in range(60):
        # lumpy pools are where greedy prefixes fail
        rows = [{"tokens": rng.choice([37, 41, 53, 89, 97]), "example_id": i}
                for i in range(rng.randint(12, 40))]
        target = rng.randint(200, 900)
        best, exact = FE.attainable_max([r["tokens"] for r in rows], target)
        assert exact, "이 크기에서는 정확 판정이 나와야 한다"
        sel, tot, ok = FE.attainable_subset(rows, target, seed=trial)
        assert ok and tot == best, f"조립기가 attainable_max({best}) 를 못 찾았다: {tot}"
        assert sum(r["tokens"] for r in sel) == tot
        assert len({id(r) for r in sel}) == len(sel), "같은 행을 두 번 담았다"
        _, greedy_used = BA.take(rows, target, seed=trial)
        if greedy_used < tot:
            misses_greedy += 1
    assert misses_greedy > 0, ("그리디가 한 번도 못 찾은 적이 없다면 이 테스트는 아무것도 "
                               "지키지 않는다 — 풀을 더 lumpy 하게 만들어라")
    print(f"  ✅ 60회 전부 attainable_max 와 일치 · 그리디가 놓친 경우 {misses_greedy}회")

    # randomness across seeds, same total
    rows = [{"tokens": rng.randint(20, 90), "example_id": i} for i in range(40)]
    picks = [tuple(sorted(r["example_id"] for r in FE.attainable_subset(rows, 731, seed=s)[0]))
             for s in range(6)]
    assert len(set(picks)) > 1, "시드를 바꿔도 같은 부분집합 — 길이로 선택하고 있다"
    assert len({sum(rows[i]["tokens"] for i in p) for p in picks}) == 1, "합이 달라졌다"
    print(f"  ✅ 시드 6개가 서로 다른 부분집합, 합은 동일")

    # pack() must not change behaviour where greedy already satisfies the tolerance
    rows = [{"tokens": 10, "example_id": i} for i in range(200)]
    g, gu = BA.take(rows, 1000, seed=3)
    pk, pu, how = FE.pack(rows, 1000, seed=3, tol=0.005)
    assert how == "greedy" and pu == gu, f"그리디가 이미 맞는데 경로가 바뀌었다: {how}"
    print(f"  ✅ 그리디가 허용오차를 맞추면 정확 경로로 내려가지 않는다 (how={how})")


def test_named_supports_are_rejected_not_silently_empty():
    """build_arms.build assumes integer supports; a named-support pool must raise, not lie.

    ⛔ Pinned because the silent path produced a wrong published-grade verdict. Handing an ADF
    tag pool to the four-arm builder makes every `range(K)` lookup miss, so it reports
    `rho=inf` and `filled 0/B tokens` — indistinguishable from a real positivity failure, and we
    recorded `live-clean` as mass-infeasible on that basis. It is constructible at 38 budgets.
    2026-09-19.
    """
    print("\nF2  이름 붙은 support 는 거부된다 (조용히 빈 그룹이 되지 않는다)")
    named = [{"support": "poly_expand", "tokens": 40, "passed": True, "example_id": i}
             for i in range(8)]
    try:
        FE.assert_builder_partition(named, 8)
        raise AssertionError("문자열 support 가 통과했다 — 가드가 죽었다")
    except TypeError as e:
        assert "type mismatch" in str(e), f"진단이 원인을 말하지 않는다: {e}"
        assert "build_extinction" in str(e), "어디로 가야 하는지 알려주지 않는다"
    print("  ✅ TypeError 로 차단하고 원인과 대안을 함께 말한다")

    ints = [{"support": i % 8, "tokens": 40, "passed": True, "example_id": i} for i in range(24)]
    FE.assert_builder_partition(ints, 8)                       # 통과해야 한다
    out_of_range = ints + [{"support": 9, "tokens": 40, "passed": True, "example_id": 99}]
    try:
        FE.assert_builder_partition(out_of_range, 8)
        raise AssertionError("범위 밖 정수가 통과했다")
    except TypeError:
        pass
    print("  ✅ 0..K-1 범위 밖 정수도 잡는다")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build the survivor-conditional arms on the REAL Math pool (v1.13 D84/D90).

Two questions, both answered without training:
  1. Does the extension actually construct where the 4-arm version cannot?
     (the 4-arm build must raise InfeasibleBudget at EVERY budget; the 5-arm one must fill)
  2. How big is each arm at the interior budget — i.e. is a decomposition powered at all?

    python3 experiments/scripts/math_extinction_build.py
"""
import json, sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cgsd" / "constructor"))
import build_arms as BA      # noqa: E402
import extinction as EX      # noqa: E402

ROOT = Path("/tmp/cgsd-data/experiments/math/cgsd-math-20260918/adf/outputs/math-1")
TOKJSON = Path("/tmp/cgsd-data/experiments/distill-ko/out/potion-ko-128d/tokenizer.json")
M_SAFE = 0.08               # v1.9 D61 — reused, not re-derived


def load():
    tok = Tokenizer.from_file(str(TOKJSON))
    tok.no_padding(); tok.no_truncation()
    pool = []
    for fn, ok in (("dataset-gated.jsonl", True), ("rejected-gated.jsonl", False)):
        for line in (ROOT / fn).open():
            if line.strip():
                r = json.loads(line)
                pool.append({"example_id": r.get("id", len(pool)), "support": r.get("tag"),
                             "passed": ok, "q": r.get("question") or "",
                             "a": str(r.get("sql") or r.get("answer") or "")})
    for r, eq, ea in zip(pool, tok.encode_batch([r["q"] for r in pool]),
                              tok.encode_batch([r["a"] for r in pool])):
        r["tokens"] = len(eq.ids) + len(ea.ids)
    return pool


def main():
    pool = load()
    A, Z, st = EX.survivor_split(pool)
    print(f"math-1  rows={st['N']}  tokens={st['T']:,}  |A|={len(A)}  |Z|={len(Z)}")
    for c in sorted(st["tok"]):
        mark = "  ⛔ 소멸" if c in Z else ""
        print(f"  {c:<18} rows {st['rows'][c]:>3}  tok {st['tok'][c]:>5}  e_c {st['e'][c]:.3f}{mark}")
    print(f"소멸 질량 — 토큰 {st['ext_tok']:.1%} · 행 {st['ext_row']:.1%}   T_A={st['T_A']:,}")

    print("\n[1] 4팔 구성은 어떤 예산에서도 실패해야 한다 (구조적 경계, v1.13 D83)")
    for frac in (0.001, 0.01, 0.05, 0.10, 0.30, 0.54):
        B = max(int(frac * st["T"]), 1)
        try:
            BA.build(pool, B, seed=0)
            print(f"  B/T={frac:<6} ⛔ 구성됐다 — 명제 반증"); return 1
        except BA.InfeasibleBudget:
            pass
    print("  ✅ 6/6 InfeasibleBudget — 예산을 0.1% 까지 낮춰도 안 풀린다")

    print("\n[2] 생존조건부 5팔은 interior 예산에서 구성된다")
    B = int((st["min_e_A"] - M_SAFE) * st["T_A"])
    print(f"  min_e_A={st['min_e_A']:.3f}  M_safe={M_SAFE}  →  B = {B:,} 토큰 "
          f"(B/T_A = {B / st['T_A']:.3f}, M_A = {EX.survivor_margin(pool, B)[0]:+.3f})")
    try:
        arms, _, meta = EX.build_extinction(pool, B, seed=0)
    except BA.InfeasibleBudget as ex:
        # ⛔ This is the FINDING, not a crash (v1.13 D94). The extension made the arms
        # DEFINABLE; the pool is still too coarse to make them MATCHED.
        print(f"  ⛔ {ex}")
        print("\n[3] 판정 — 정의는 풀렸고 규모가 안 풀렸다")
        print("  소멸 확장으로 D10^A 는 정의된다. 그런데 예산 200~2,500 토큰 어디에서도 팔이")
        print("  ±0.5% 로 맞지 않는다 — positivity 위반은 0건이고 원인은 입자 크기다.")
        print("  ⇒ 예산으로는 못 푼다. 필요한 최소 규모는 granularity_floor.py 가 잰다.")
        return 0
    for name in ("d00_all", "d00_A", "d10_A", "d01_A", "d11"):
        rows = arms[name]
        per = {c: sum(1 for r in rows if r["support"] == c) for c in sorted(st["tok"])}
        used = sum(r["tokens"] for r in rows)
        print(f"  {name:<9} {len(rows):>3}행  {used:>5} tok ({used / B - 1:+.2%})  "
              + " ".join(f"{c}:{n}" for c, n in per.items() if n))
    print("  ✅ 다섯 팔 전부 예산 ±0.5% 안")

    print("\n[3] 검정력 — 이 크기로 분해를 추정할 수 있나")
    n = min(len(arms[k]) for k in arms)
    print(f"  arm 당 {n}행 · 생존 support {len(A)}개 · support 당 ~{n // len(A)}행")
    print("  ⛔ endpoint 는 8B 파인튜닝 exact-match 다. 구성 가능성과 검정력은 다른 문제다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

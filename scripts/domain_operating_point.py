import json, sys
from collections import defaultdict
from pathlib import Path
from tokenizers import Tokenizer
sys.path.insert(0, "cgsd/constructor")
import build_arms as BA
import extinction as EX

ROOT = Path("/tmp/cgsd-data/experiments/math/cgsd-math-20260918/adf/outputs")
tok = Tokenizer.from_file("/tmp/cgsd-data/experiments/distill-ko/out/potion-ko-128d/tokenizer.json")
tok.no_padding(); tok.no_truncation()
M_SAFE = 0.08

def load(name):
    pool = []
    for fn, ok in (("dataset-gated.jsonl", True), ("rejected-gated.jsonl", False)):
        p = ROOT / name / fn
        if not p.exists(): continue
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                pool.append({"example_id": r.get("id", len(pool)), "support": r.get("tag"),
                             "passed": ok, "_q": r.get("question") or r.get("prompt") or "",
                             "_a": str(r.get("sql") or r.get("answer") or "")})
    for r, e1, e2 in zip(pool, tok.encode_batch([r["_q"] for r in pool]),
                              tok.encode_batch([r["_a"] for r in pool])):
        r["tokens"] = len(e1.ids) + len(e2.ids)
    return pool

print(f"{'pool':<20}{'T':>9}{'min_e':>8}{'B/T':>8}{'B':>9}  판정")
for name in ("power-wide", "ab-weak8b", "gen-27b-2026-08-20", "live-clean", "live-scale", "math-1"):
    pool = load(name)
    if not pool: continue
    A, Z, st = EX.survivor_split(pool)
    T = st["T"]
    if Z:
        print(f"{name:<20}{T:>9,}{'—':>8}{'—':>8}{'—':>9}  ⛔ 소멸 {len(Z)}개, "
              f"token mass {st['ext_tok']:.1%} — 4팔 구성 불가 (생존조건부 필요)")
        cap = st["min_e_A"] - M_SAFE
        print(f"{'':<20}{'':>9}{st['min_e_A']:>8.3f}{max(cap,0):>8.3f}"
              f"{int(max(cap,0)*st['T_A']):>9,}  ↳ 생존조건부 A: T_A={st['T_A']:,}")
        continue
    cap = st["min_e_A"] - M_SAFE
    v = "✅ interior" if cap > 0 else "⛔ M_safe=0.08 불가 (frontier 아래)"
    print(f"{name:<20}{T:>9,}{st['min_e_A']:>8.3f}{max(cap,0):>8.3f}{int(max(cap,0)*T):>9,}  {v}")

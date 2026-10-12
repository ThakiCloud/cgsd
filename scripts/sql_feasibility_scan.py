"""SQL pools, re-adjudicated with the partition the pools actually have.

⛔ BA.build iterates `range(K)` — integer KMeans supports from the ESCI loader. ADF pools carry
string tag supports, so every lookup misses and it reports rho=inf / 0 tokens filled. That is a
type mismatch, not a feasibility verdict. The extension builder handles string supports.
"""
import json, sys, io, contextlib
from pathlib import Path
from tokenizers import Tokenizer
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/"cgsd"/"constructor")); sys.path.insert(0, str(R/"experiments"/"scripts"))
import build_arms as BA, extinction as EX, feasibility as FE
with contextlib.redirect_stdout(io.StringIO()):
    from granularity_condition import arm_shares
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

out = {}
print(f"{'pool':<22}{'N':>5}{'sup':>5}{'Z':>3}{'T':>8}{'min_e':>7}"
      f"{'Bmax':>7}{'G>=0':>6}{'build OK':>9}  판정")
for name in ("power-wide", "ab-weak8b", "gen-27b-2026-08-20", "live-clean", "live-scale"):
    with contextlib.redirect_stdout(io.StringIO()):
        pool = load(name); A, Z, st = EX.survivor_split(pool); shares = arm_shares(pool)[1]
    T, T_A, min_e = st["T"], st["T_A"], st["min_e_A"]
    cap = int(min_e * T_A)                       # M_A >= 0
    cap_safe = int(max(min_e - M_SAFE, 0) * T_A)  # pre-registered margin
    step = max(1, cap // 2500)
    okG = []
    for B in range(1, cap + 1, step):
        with contextlib.redirect_stdout(io.StringIO()):
            if all(FE.granularity(sh, src, B, BA.BUDGET_TOL)[0] >= 0 for _, sh, src in shares):
                okG.append(B)
    # ⛔ Attempt EVERY budget that passed the assembly condition. Attempting a truncated prefix
    # and reporting the successes next to the full G count puts two different denominators in
    # the same sentence — which is exactly how "86 vs 38" became unexplainable (2026-09-20).
    built, why = [], {}
    for B in okG:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                EX.build_extinction(pool, B, seed=0, strict=True); built.append(B)
            except Exception as e:
                t = str(e)
                k = ("packer" if "PACKER" in t else "granularity" if "GRANULARITY" in t
                     else "frontier" if "FRONTIER" in t else type(e).__name__)
                why[k] = why.get(k, 0) + 1
    v = ("✅ 구성 가능" if built else ("⚠️ G>=0 있으나 build 실패" if okG else "⛔ F_Pi 비어있음"))
    extra = f"  B/T {min(built)/T:.3f}…{max(built)/T:.3f}" if built else ""
    print(f"{name:<22}{st['N']:>5}{len(st['tok']):>5}{len(Z):>3}{T:>8,}{min_e:>7.3f}"
          f"{cap:>7,}{len(okG):>6}{len(built):>9}  {v}{extra}")
    out[name] = {"N": st["N"], "supports": len(st["tok"]), "extinct": len(Z), "T": T,
                 "min_e_A": min_e, "cap_mass": cap, "cap_margin": cap_safe,
                 # every count below shares ONE denominator: the budgets actually examined
                 "grid_step": step, "n_scanned": len(range(1, cap + 1, step)),
                 "n_G_ok": len(okG), "n_attempted": len(okG), "n_built": len(built),
                 "build_failures": why,
                 "BT_built": [min(built)/T, max(built)/T] if built else None}
Path(R/"paper"/"results-sql.json").write_text(json.dumps(out, indent=2, default=float))
print(f"\nwrote paper/results-sql.json")

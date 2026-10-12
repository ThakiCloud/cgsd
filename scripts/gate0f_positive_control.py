#!/usr/bin/env python3
"""Gate 0F — controlled positive controls for S and I (v1.7 D47).

Gate 0C proved E: detection (12/12, +20.7건) AND exclusion (s_only E = +0.0건). It did NOT prove
that S and I can be DETECTED, because in ESCI the support effect is only 1~3 items — the same size
as the resolution. That is a property of the data, not of the estimator, so the positive control
has to come from a controlled DGP where the truth is large and known by construction.

⛔ Realism is NOT the goal here. The goal is an identifiability unit test: plant S* = -20 items (or
I* != 0), and check the decomposition returns it. No learner, no GPU — the outcome is a closed-form
function of arm composition, and the SAME constructor (`build_arms.build`) makes the arms.

DGP. K supports, evaluation partitioned evenly over supports (w_c = 1/K). For an arm with token
share s_c and clean-token fraction q_c inside support c:

    acc_c = A_c * (1 - exp(-s_c / tau)) * (q0_c + (1 - q0_c) * q_c)
    Y     = sum_c w_c * acc_c   + eps,      eps ~ N(0, sigma^2)

  * exposure sensitivity   -> starving a support's ELIGIBILITY moves p_gate, so D01/D11 lose
                             exposure there: that is S.
  * quality sensitivity    -> q0_c < 1 makes clean rows worth more inside support c: that is E.
  * heterogeneous q0_c     -> the value of eligibility differs per support, so the gate-induced
                             support mix changes what E is worth: that is I.

Scenarios
  s_positive   q0_c = 1 everywhere (quality worthless) + one support starved to 5% eligibility
               => E* = 0, S* << 0, I* = 0
  i_positive   q0_A = 0 (clean matters a lot), q0_B = 1 (clean worthless), eligibility rates differ
               => I* != 0
  e_positive   q0_c = 0.5 uniform, uniform eligibility  => E* > 0, S* = 0, I* = 0  (sanity)

    python3 experiments/scripts/gate0f_positive_control.py --reps 12 --eval 2000
"""
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
assert (REPO / "README.md").exists()
sys.path.insert(0, str(REPO / "cgsd" / "constructor"))
import build_arms as BA                                    # noqa: E402

K, ROWS, TOK, TAU = 4, 4000, 10, 0.20
SEEDS, SIGMA = 5, 0.0015
RATIO = 0.05      # B/T. ⛔ A deep starve needs a SMALL relative budget: feasibility is B/T <= min_c e_c
                  # (D40). The first try used 0.25 with a 5% starve and the constructor refused it
                  # (M = -0.20) — the positive control has to live inside the frontier too.

SCEN = {                                                   # (q0 per support, eligibility per support)
    "s_positive": ([1.0, 1.0, 1.0, 1.0], [0.60, 0.60, 0.10, 0.60]),   # M = 0.10 - 0.05 = +0.05
    "i_positive": ([0.0, 1.0, 0.5, 0.5], [0.60, 0.20, 0.60, 0.60]),   # M = +0.15
    "e_positive": ([0.5, 0.5, 0.5, 0.5], [0.60, 0.60, 0.60, 0.60]),
}
A_C = [1.0] * K
CLEAN = 0.5                                                # population clean rate


def make_pool(q0, elig, rng):
    """Rows with equal tokens; eligible slots are filled with clean rows first (eligibility carries
    quality exactly when q0_c < 1 makes clean rows worth more)."""
    pool = []
    for c in range(K):
        rows = [{"example_id": f"c{c}_{i}", "support": c, "tokens": TOK,
                 "clean": rng.random() < CLEAN} for i in range(ROWS)]
        k = int(round(elig[c] * ROWS))
        clean = [r for r in rows if r["clean"]]; dirty = [r for r in rows if not r["clean"]]
        rng.shuffle(clean); rng.shuffle(dirty)
        for r in rows:
            r["passed"] = False
        for r in (clean + dirty)[:k]:
            r["passed"] = True
        pool += rows
    return pool


def Y(share, q, q0):
    return sum((1 / K) * A_C[c] * (1 - math.exp(-share[c] / TAU)) * (q0[c] + (1 - q0[c]) * q[c])
               for c in range(K))


def composition(rows):
    tok, cl = defaultdict(int), defaultdict(int)
    for r in rows:
        tok[r["support"]] += r["tokens"]; cl[r["support"]] += r["tokens"] * r["clean"]
    T = sum(tok.values())
    return ({c: tok[c] / T for c in range(K)},
            {c: (cl[c] / tok[c]) if tok[c] else 0.0 for c in range(K)})


def truth(pool, q0):
    """Design shares (p_src / p_gate) x population clean fractions -> model-implied E*, S*, I*."""
    T = sum(r["tokens"] for r in pool); PT = sum(r["tokens"] for r in pool if r["passed"])
    ps, pg = defaultdict(float), defaultdict(float)
    ta, ca, te, ce = (defaultdict(int) for _ in range(4))
    for r in pool:
        c, t = r["support"], r["tokens"]
        ps[c] += t / T; ta[c] += t; ca[c] += t * r["clean"]
        if r["passed"]:
            pg[c] += t / PT; te[c] += t; ce[c] += t * r["clean"]
    qa = {c: ca[c] / ta[c] for c in range(K)}
    qe = {c: (ce[c] / te[c]) if te[c] else 0.0 for c in range(K)}
    y = {"d00": Y(ps, qa, q0), "d10": Y(ps, qe, q0), "d01": Y(pg, qa, q0), "d11": Y(pg, qe, q0)}
    return {"E": y["d10"] - y["d00"], "S": y["d01"] - y["d00"],
            "I": y["d11"] - y["d10"] - y["d01"] + y["d00"], "delta_G": y["d11"] - y["d00"]}


def boot_ci(v, rng, n=4000, lo=2.5, hi=97.5):
    ms = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    at = lambda p: ms[min(len(ms) - 1, int(p / 100 * len(ms)))]
    return at(lo), at(hi)


def run(name, reps, unit):
    q0, elig = SCEN[name]
    BA.K = K
    est = {k: [] for k in ("E", "S", "I", "delta_G")}
    tr = []
    for rep in range(reps):
        rng = random.Random(1000 + rep)
        pool = make_pool(q0, elig, rng)
        B = int(RATIO * sum(r["tokens"] for r in pool))
        tr.append(truth(pool, q0))
        arms, _, meta = BA.build(pool, B, seed=rep)         # the real constructor, strict
        base = {n_: Y(*composition(arms[n_]), q0) for n_ in ("d00", "d10", "d01", "d11")}
        nr = random.Random(50000 + rep)
        y = {n_: [base[n_] + nr.gauss(0, SIGMA) for _ in range(SEEDS)] for n_ in base}
        d = {"E": [y["d10"][i] - y["d00"][i] for i in range(SEEDS)],
             "S": [y["d01"][i] - y["d00"][i] for i in range(SEEDS)],
             "I": [y["d11"][i] - y["d10"][i] - y["d01"][i] + y["d00"][i] for i in range(SEEDS)],
             "delta_G": [y["d11"][i] - y["d00"][i] for i in range(SEEDS)]}
        for k, v in d.items():
            est[k].append(sum(v) / SEEDS)
    rng = random.Random(7)
    out = {"scenario": name, "reps": reps, "margin": meta["margin"], "B_over_T": RATIO, "components": {}}
    print(f"== {name}  (M={meta['margin']:+.3f}, B/T={RATIO}, 1건 = 1/{unit})")
    for k in ("E", "S", "I", "delta_G"):
        e = est[k]; t = [x[k] for x in tr]
        err = [a - b for a, b in zip(e, t)]
        n = len(e); m = sum(err) / n
        lo, hi = boot_ci(err, rng)
        med = sorted(err)[n // 2]
        neg = sum(x < 0 for x in e); pos = sum(x > 0 for x in e)
        out["components"][k] = {"truth_mean": sum(t) / n, "est_mean": sum(e) / n, "bias": m,
                                "bias_ci95": [lo, hi], "bias_median": med,
                                "rmse": math.sqrt(sum(x * x for x in err) / n),
                                "sign_pos": pos, "sign_neg": neg, "recovered": lo <= 0 <= hi}
        print(f"   {k:<7} truth={sum(t) / n * unit:+7.1f}건  est={sum(e) / n * unit:+7.1f}건  "
              f"bias={m * unit:+5.1f} CI95[{lo * unit:+.1f},{hi * unit:+.1f}] "
              f"{'✅' if lo <= 0 <= hi else '⛔'}  부호 +{pos}/-{neg}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--eval", type=int, default=2000, help="evaluation size, only for 건 units")
    ap.add_argument("--out", default="docs/experiments/2026-09-18-cgsd-gate0f-positive-control.json")
    a = ap.parse_args()
    res = [run(s, a.reps, a.eval) for s in SCEN]
    Path(a.out).write_text(json.dumps({"K": K, "rows_per_support": ROWS, "tau": TAU, "sigma": SIGMA,
                                       "seeds": SEEDS, "eval": a.eval, "scenarios": res},
                                      indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    ok = all(c["recovered"] for r in res for c in r["components"].values())
    big = {"s_positive": "S", "i_positive": "I", "e_positive": "E"}
    detect = all(abs(r["components"][big[r["scenario"]]]["truth_mean"]) * a.eval >= 10 for r in res)
    print("PASS" if ok and detect else "FAIL"
          + ("" if detect else " — planted effect under 10건, not a detection test"))
    return 0 if (ok and detect) else 1


if __name__ == "__main__":
    sys.exit(main())

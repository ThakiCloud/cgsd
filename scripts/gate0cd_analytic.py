#!/usr/bin/env python3
"""Gate 0C/0D — known-truth recovery with an ANALYTIC outcome (v1.6 D38, D41).

Why an analytic layer: the trained-model Gate 0C knows only the SIGN of the planted effects,
so bias/RMSE/coverage can be scored there only on components whose truth is exactly zero.
Here the outcome is a known function of an arm's composition, so E*, S*, I* are computed
exactly from population quantities, while the estimate goes through the REAL constructor
(`build_arms.build` on the real pool, real supports, real token counts) and the same paired
per-seed contrast. What this validates: constructor + decomposition + interval procedure.
What it does NOT validate: the learner. That stays with the trained 0C.

Outcome of an arm A (token units), with support shares s_c and clean-token fraction q_c:
    Y(A) = 0.20 + a * sum_c w_c * g(s_c) * q_c  + eps_seed,   eps ~ N(0, sigma^2)
    g(x) = 1 - exp(-K x / tau)      concave: starving a support costs more than it returns
    w_c  = source share x fixed demand factor u_c in [0.5, 1.5]
           ⛔ w = source share makes the source mix a stationary point of sum w g(s): starving a
           support then moves Y only to SECOND order and S* is ~0 (smoke: S* = -0.0005). Real
           evaluation demand is not the source mix, so a fixed, seeded demand tilt is used.
Truth uses the arm's DESIGN shares (p_src / p_gate) and the population clean fraction of the
pool it draws from (all rows vs eligible rows); the estimate uses the realized arm.

  0C interior  budget 55,300 (en) — the trained-0C budget, M > 0 in every condition
  0D frontier  budget 95,162 (en, B/T = 0.40) — starve rate e set so M = 0.08, 0.04, 0.02

    python3 experiments/scripts/gate0cd_analytic.py --reps 400
"""
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
assert (REPO / "README.md").exists()
sys.path.insert(0, str(REPO / "cgsd" / "constructor"))
sys.path.insert(0, str(REPO / "experiments" / "scripts"))
sys.path.insert(0, "/tmp/cgsd-repro-205536")
import build_arms as BA                     # noqa: E402
import gate0c_synthetic as G                # noqa: E402

SIGMA, TAU, SEEDS = 0.002, 1.0, 5
T975 = {4: 2.776, 9: 2.262}


def compose(rows, K):
    tok, clean = defaultdict(int), defaultdict(int)
    for r in rows:
        tok[r["support"]] += r["tokens"]
        clean[r["support"]] += 0 if r["_corrupt"] else r["tokens"]
    T = sum(tok.values())
    return {c: tok[c] / T for c in range(K)}, {c: (clean[c] / tok[c]) if tok[c] else 0.0 for c in range(K)}


def outcome(share, q, w, a, K):
    return 0.20 + a * sum(w[c] * (1 - math.exp(-K * share[c] / TAU)) * q[c] for c in range(K))


def truth(pool, w, a, K):
    """E*, S*, I* from design shares and population clean fractions."""
    T = sum(r["tokens"] for r in pool); PT = sum(r["tokens"] for r in pool if r["passed"])
    p_src = defaultdict(float); p_gate = defaultdict(float)
    tok_all, cl_all, tok_el, cl_el = (defaultdict(int) for _ in range(4))
    for r in pool:
        c, t, ok = r["support"], r["tokens"], not r["_corrupt"]
        p_src[c] += t / T; tok_all[c] += t; cl_all[c] += t * ok
        if r["passed"]:
            p_gate[c] += t / PT; tok_el[c] += t; cl_el[c] += t * ok
    q_all = {c: cl_all[c] / tok_all[c] for c in range(K)}
    q_el = {c: (cl_el[c] / tok_el[c]) if tok_el[c] else 0.0 for c in range(K)}
    Y = {"d00": outcome(p_src, q_all, w, a, K), "d10": outcome(p_src, q_el, w, a, K),
         "d01": outcome(p_gate, q_all, w, a, K), "d11": outcome(p_gate, q_el, w, a, K)}
    return {"E": Y["d10"] - Y["d00"], "S": Y["d01"] - Y["d00"],
            "I": Y["d11"] - Y["d10"] - Y["d01"] + Y["d00"]}


def one_rep(pool, B, w, a, K, rng_seed, rep):
    arms, _, _ = BA.build(pool, B, seed=rep)          # strict: raises if M < 0
    Ybar = {n: outcome(*compose(arms[n], K), w, a, K) for n in ("d00", "d10", "d01", "d11")}
    nrng = random.Random(rng_seed)
    Y = {n: [Ybar[n] + nrng.gauss(0, SIGMA) for _ in range(SEEDS)] for n in Ybar}
    d = {"E": [Y["d10"][i] - Y["d00"][i] for i in range(SEEDS)],
         "S": [Y["d01"][i] - Y["d00"][i] for i in range(SEEDS)],
         "I": [Y["d11"][i] - Y["d10"][i] - Y["d01"][i] + Y["d00"][i] for i in range(SEEDS)]}
    out = {}
    for k, v in d.items():
        m = sum(v) / SEEDS
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (SEEDS - 1))
        h = T975[SEEDS - 1] * sd / math.sqrt(SEEDS)
        out[k] = {"mean": m, "median": sorted(v)[SEEDS // 2], "lo": m - h, "hi": m + h}
    return out


def score(recs, trs):
    """Errors are taken PER REP against that rep's own truth (the starved set differs per rep)."""
    res = {}
    for k in ("E", "S", "I"):
        err = [r[k]["mean"] - t[k] for r, t in zip(recs, trs)]
        errm = [r[k]["median"] - t[k] for r, t in zip(recs, trs)]
        n = len(err); b = sum(err) / n
        cov = sum(r[k]["lo"] <= t[k] <= r[k]["hi"] for r, t in zip(recs, trs)) / n
        res[k] = {"truth_mean": sum(t[k] for t in trs) / n,
                  "truth_min": min(t[k] for t in trs), "truth_max": max(t[k] for t in trs),
                  "bias": b, "bias_mcse": math.sqrt(sum((x - b) ** 2 for x in err) / (n - 1) / n),
                  "bias_median_est": sum(errm) / n,
                  "rmse": math.sqrt(sum(x * x for x in err) / n),
                  "coverage95": cov, "coverage_mcse": math.sqrt(cov * (1 - cov) / n)}
    return res


def run(pool, leaves, p_base, K, B, cond, starve, reps, w, a):
    G.STARVE = starve
    recs, trs, M = [], [], []
    for rep in range(reps):
        G.apply_condition(pool, cond, leaves, p_base, random.Random(7000 + rep))
        M.append(BA.feasibility_margin(pool, B)[0])
        trs.append(truth(pool, w, a, K))
        recs.append(one_rep(pool, B, w, a, K, 424242 + rep, rep))
    return {"cond": cond, "budget": B, "starve": starve, "margin_min": min(M),
            "margin_mean": sum(M) / reps, "reps": reps, **score(recs, trs)}


_CTX = None


def _job(item):
    gate, B, cond, st = item
    pool, leaves, p_base, K, reps, w, a = _CTX
    BA.K = K
    return run([dict(r) for r in pool], leaves, p_base, K, B, cond, st, reps, w, a) | {"gate": gate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="en", choices=("en", "jp"))
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    x = ap.parse_args()
    K = BA.K = 8
    pool, _ = BA.load_pool(x.locale)
    for r in pool:
        r["leaf_orig"] = r["leaf"]
    leaves = sorted({r["leaf"] for r in pool})
    p_base = sum(r["passed"] for r in pool) / len(pool)
    T = sum(r["tokens"] for r in pool)
    tok = defaultdict(int)
    for r in pool:
        tok[r["support"]] += r["tokens"]
    urng = random.Random(99)
    w = {c: tok[c] / T * urng.uniform(0.5, 1.5) for c in range(K)}
    zw = sum(w.values()); w = {c: v / zw for c, v in w.items()}
    # scale so the e_only quality effect is ~ +0.01 (the trained-0C magnitude) — illustrative only
    a = 0.01 / (sum(w[c] * (1 - math.exp(-K * w[c] / TAU)) for c in range(K)) * G.CORRUPT)
    interior = {"en": 55300, "jp": 18700}[x.locale]
    primary = G.BUDGET[x.locale]
    print(f"{x.locale}: B/T interior={interior / T:.3f} primary={primary / T:.3f} p_base={p_base:.3f} "
          f"sigma={SIGMA} seeds={SEEDS} reps={x.reps}", flush=True)
    results = []
    plan = [("0C", interior, c, G.STARVE) for c in G.CONDITIONS]
    plan += [("0D", primary, "s_only", primary / T + m) for m in (0.08, 0.04, 0.02)]
    plan += [("0D", primary, "e_s_i", primary / T + m) for m in (0.08, 0.04, 0.02)]
    import multiprocessing as mp
    global _CTX
    _CTX = (pool, leaves, p_base, K, x.reps, w, a)
    with mp.get_context("fork").Pool(min(x.workers, len(plan))) as P:   # fork shares the loaded pool
        for r in P.imap(_job, plan):
            results.append(r)
            print(f"  {r['gate']} {r['cond']:<8} B={r['budget']:>6,} M_min={r['margin_min']:+.3f}  " + "  ".join(
                f"{k}: truth={r[k]['truth_mean']:+.5f} bias={r[k]['bias']:+.6f}±{r[k]['bias_mcse']:.6f} "
                f"rmse={r[k]['rmse']:.5f} cov={r[k]['coverage95']:.3f}" for k in "ESI"), flush=True)
    for m in (-0.01,):                                              # 0E on the real pool
        G.STARVE = primary / T + m
        G.apply_condition(pool, "s_only", leaves, p_base, random.Random(7000))
        try:
            BA.build(pool, primary, seed=0); e0 = "⛔ built"
        except BA.InfeasibleBudget as err:
            e0 = f"✅ InfeasibleBudget ({str(err)[:60]}…)"
        print(f"  0E s_only M={BA.feasibility_margin(pool, primary)[0]:+.3f}: {e0}", flush=True)
        results.append({"gate": "0E", "margin": BA.feasibility_margin(pool, primary)[0], "result": e0})
    p = Path(x.out or f"docs/experiments/2026-09-17-cgsd-gate0cd-analytic-{x.locale}.json")
    p.write_text(json.dumps({"sigma": SIGMA, "tau": TAU, "seeds": SEEDS, "a": a, "results": results},
                            indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

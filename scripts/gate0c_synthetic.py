#!/usr/bin/env python3
"""Gate 0C — synthetic known-effect recovery.

Gate 0B showed the estimator returns null under a null treatment. 0C asks the other half:
when a KNOWN effect is planted, does the decomposition put it in the right component?

Four conditions. Sign is known by construction; magnitude is not, so recovery is judged on
sign + which component clears its delta (from Gate 0B), not on hitting a target value.

  e_only   labels are corrupted uniformly at random across supports; eligibility = "not
           corrupted". D10 therefore trains on clean labels and D00 on a mixture.
           => E* > 0,  S* = 0,  I* = 0
  s_only   no corruption; eligibility probability VARIES BY SUPPORT (a fraction of supports
           is starved). Within a support, eligibility is random.
           => E* = 0,  S* != 0 (starving supports costs coverage),  I* = 0
  e_and_s  both of the above at once.               => E* > 0, S* != 0
  e_s_i    corruption RATE varies by support, so the value of eligibility differs per support.
           => interaction is planted.               => I* != 0

⛔ The corruption target is the LEAF (the training label), never the query — corrupting the
query would also move the support representation, which is defined on pre-gate text.

Run with the LEGACY venv:
    python3 experiments/scripts/gate0c_synthetic.py \
        --locale jp --reps 4 --seeds 5
"""
import argparse, json, random, statistics as st, sys
from pathlib import Path

REPRO = "/tmp/cgsd-repro-205536"
sys.path.insert(0, REPRO)
REPO = Path(__file__).resolve().parents[2]
assert (REPO / "README.md").exists(), f"REPO wrong: {REPO}"
sys.path.insert(0, str(REPO / "cgsd" / "constructor"))

import numpy as np                                    # noqa: E402
import os as _os
import torch as _torch
# v1.6: thread count does not change the metric (top1 identical to the last digit at 1/2/12 threads,
# measured 2026-09-17), and under host contention 1 thread was FASTER than 12 (122s vs 143s). So
# placebo reps run as many single-thread shards instead of one oversubscribed process.
if _os.environ.get("CGSD_TORCH_THREADS"):
    _torch.set_num_threads(int(_os.environ["CGSD_TORCH_THREADS"]))
import build_arms as BA                               # noqa: E402
from _common import BASE, HP, load                    # noqa: E402
from train_arm import train, leaf_top1_full           # noqa: E402

BUDGET = {"jp": 33188, "en": 95162}
# ⚠️ 2026-09-16: the first pass used 0.35/0.35 and the planted effects landed INSIDE the Gate 0B
# deltas — the estimator could not be judged because the intervention was not detectable. Raised
# deliberately. This is power, not threshold tuning: Gate 0C validates the estimator, so the
# planted effect has to clear the null band or the test says nothing either way.
# ⛔ The intervention strength is capped by POSITIVITY, not by taste. Making eligibility
# "eligible AND not corrupted" multiplies the pass rate by (1-c), so c=0.80 left only
# 0.4886*0.20 = 9.8% eligible and D10 could not fill its budget — it then looked bad because
# it was SMALL, not because its labels were clean. Measured: e_only came out E<0, the wrong sign.
# Fix: hold the pass rate at p_base and fill eligibility with CLEAN rows first, inside each
# support. That needs (1-c) >= p_base, i.e. c <= 1 - 0.4886 = 0.511.
CORRUPT = 0.45          # <= 1 - p_base, so eligibility can still be filled entirely with clean rows
STARVE = 0.25           # eligibility probability inside a starved support (low, but fillable)
CONDITIONS = ("e_only", "s_only", "e_and_s", "e_s_i")


def apply_condition(pool, cond, leaves, p_base, rng):
    """Mutate pool rows in place: set r['passed'] (eligibility) and possibly r['leaf'].

    Eligibility is drawn to a FIXED per-support rate so the pass rate never collapses, and in
    the E conditions the eligible slots are filled with clean rows first. That makes eligibility
    informative about label quality without shrinking the D10 pool.
    """
    from collections import defaultdict
    sups = sorted({r["support"] for r in pool})
    starved = set(rng.sample(sups, max(1, len(sups) // 3)))
    plants_e = cond in ("e_only", "e_and_s", "e_s_i")
    plants_s = cond in ("s_only", "e_and_s", "e_s_i")

    for r in pool:
        r["leaf"] = r["leaf_orig"]                     # reset from a clean copy
        cp = CORRUPT
        if cond == "e_s_i":                            # corruption rate differs per support
            cp = min(CORRUPT * 1.6, 0.90) if r["support"] in starved else CORRUPT * 0.4
        r["_corrupt"] = plants_e and rng.random() < cp
        if r["_corrupt"]:
            r["leaf"] = rng.choice(leaves)             # wrong label

    by_sup = defaultdict(list)
    for r in pool:
        by_sup[r["support"]].append(r)
    for c, rs in by_sup.items():
        p = STARVE if (plants_s and c in starved) else p_base
        k = int(round(p * len(rs)))
        clean = [r for r in rs if not r["_corrupt"]]
        dirty = [r for r in rs if r["_corrupt"]]
        rng.shuffle(clean); rng.shuffle(dirty)
        pick = (clean + dirty)[:k] if plants_e else rng.sample(rs, k)
        for r in rs:
            r["passed"] = False
        for r in pick:
            r["passed"] = True
    return starved


def feasible_budget(pool, leaves, p_base, k):
    """Largest B for which EVERY condition keeps all four arms fillable.

    ⛔ 2026-09-17. The first en/jp runs used the primary budget (0.40*T) and the starve pushed
    rho to ~1.6 in the starved supports, so D10 came in 9.5%/6.5% short. E then read negative
    in `s_only`, where it was planted to be zero — the shortfall, not the estimator. The budget
    is therefore re-derived from the POST-intervention pool with the project's own rule
    (v1.3 D20: B = min(0.40*T, 0.95*B_max)), taking the minimum across conditions so all four
    conditions are compared at one identical exposure.
    """
    from collections import defaultdict
    T = sum(r["tokens"] for r in pool)
    worst = []
    for cond in CONDITIONS:
        apply_condition(pool, cond, leaves, p_base, random.Random(7000))
        by = defaultdict(list)
        for r in pool:
            by[r["support"]].append(r)
        worst.append(min(sum(x["tokens"] for x in rs if x["passed"]) /
                         (sum(x["tokens"] for x in rs) / T) for rs in by.values()))
    return int(min(0.40 * T, 0.95 * min(worst)) // 100 * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=("jp", "en"))
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None, help="default: feasible_budget()")
    ap.add_argument("--starve", type=float, default=None,
                    help="eligibility inside starved supports (v1.6 D41 frontier: B/T + M)")
    ap.add_argument("--conds", default=",".join(CONDITIONS))
    ap.add_argument("--tag", default="", help="output suffix, e.g. 0d-m002")
    a = ap.parse_args()
    BA.K = a.k

    S = load(a.locale)
    pool, _ = BA.load_pool(a.locale)
    for r in pool:
        r["leaf_orig"] = r["leaf"]
    leaves = sorted({r["leaf"] for r in pool})
    p_base = sum(r["passed"] for r in pool) / len(pool)
    global STARVE
    if a.starve is not None:
        STARVE = a.starve
    B = a.budget or feasible_budget(pool, leaves, p_base, a.k)
    print(f"{a.locale}: pool={len(pool)} p_base={p_base:.4f} B={B:,} "
          f"(primary {BUDGET[a.locale]:,}) K={a.k} "
          f"corrupt={CORRUPT} starve={STARVE}", flush=True)

    out = {"locale": a.locale, "K": a.k, "budget": B, "reps": a.reps, "seeds": a.seeds,
           "margin_after_last_condition": BA.feasibility_margin(pool, B)[0],
           "corrupt": CORRUPT, "starve": STARVE, "conditions": {}}
    stem = f"docs/experiments/2026-09-16-cgsd-gate0c-{a.locale}" + (f"-{a.tag}" if a.tag else "")
    ckpt = Path(stem + ".ckpt.jsonl")
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            r = json.loads(line)
            if r.get("budget") == B and r.get("starve", 0.25) == STARVE:
                done[(r["cond"], r["rep"])] = r
        print(f"  resume: {len(done)} (cond,rep) from {ckpt}", flush=True)
    for cond in a.conds.split(","):
        comp = {"E": [], "S": [], "I": []}
        for rep in range(a.reps):
            if (cond, rep) in done:
                for k in comp:
                    comp[k].append(done[(cond, rep)][k])
                continue
            rng = random.Random(7000 + rep)
            apply_condition(pool, cond, leaves, p_base, rng)
            arms, _, _ = BA.build(pool, B, seed=rep)
            if {x["example_id"] for x in arms["d00"]} == {x["example_id"] for x in arms["d10"]}:
                print(f"  ⛔ {cond} rep{rep}: D00 == D10 — no intervention. abort."); return 1
            Y = {}
            for name in ("d00", "d10", "d01", "d11"):
                pairs = [(r["query_ko"], r["leaf"]) for r in arms[name]]
                Y[name] = [leaf_top1_full(train(BASE, pairs, seed=s_, quiet=True, **HP),
                                          S["hold"], S["leaves"])
                           for s_ in range(5, 5 + a.seeds)]
            comp["E"].append(st.median([Y["d10"][i] - Y["d00"][i] for i in range(a.seeds)]))
            comp["S"].append(st.median([Y["d01"][i] - Y["d00"][i] for i in range(a.seeds)]))
            comp["I"].append(st.median([Y["d11"][i] - Y["d10"][i] - Y["d01"][i] + Y["d00"][i]
                                        for i in range(a.seeds)]))
            with ckpt.open("a") as f:
                f.write(json.dumps({"cond": cond, "rep": rep, "budget": B, "starve": STARVE,
                                    **{k: comp[k][-1] for k in comp}}) + "\n")
        med = {k: float(st.median(v)) for k, v in comp.items()}
        out["conditions"][cond] = {"per_rep": comp, "median": med}
        print(f"  {cond:<9} E={med['E']:+.5f}  S={med['S']:+.5f}  I={med['I']:+.5f}", flush=True)

    p = Path(stem + ".json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

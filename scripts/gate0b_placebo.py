#!/usr/bin/env python3
"""Gate 0B — placebo gate: recover E = S = I = 0 and calibrate per-component thresholds.

The placebo assigns eligibility at random with the SAME pass probability in every support,
so in expectation p_c^placebo = p_c (no support shift) and eligibility carries no information
(no within-support selection). Ground truth is therefore E* = S* = I* = 0.

What the arms still differ in is STRUCTURE — D10/D11 draw from a pool that is p times smaller,
so they deplete it harder. That is exactly what we want the null to absorb: whatever E/S/I the
constructor + training randomness produce under a known-null treatment becomes the resolution
threshold for the real gate.

    delta_E = Q_0.95(|E_null|)   delta_S = Q_0.95(|S_null|)   delta_I = Q_0.95(|I_null|)

⛔ v1.3 D25: before trusting any of this, confirm the placebo actually created an intervention —
D00 != D10 by sample identity. A placebo whose arms collapse to the same live data contributes
zero bits (legacy c209b2a943).

Run with the LEGACY venv (needs model2vec):
    python3 experiments/scripts/gate0b_placebo.py --locale jp --reps 2 --seeds 3
"""
import argparse, json, random, statistics as st, sys, time
from pathlib import Path

REPRO = "/tmp/cgsd-repro-205536"
sys.path.insert(0, REPRO)
REPO = Path(__file__).resolve().parents[2]          # scripts -> experiments -> repo root
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

BUDGET = {"jp": 33188, "en": 95162}                   # v1.4 D27, K=8 primary


def placebo_flags(n, p, seed):
    rng = random.Random(seed)
    return [rng.random() < p for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=("jp", "en"))
    ap.add_argument("--reps", type=int, default=10, help="placebo draws (the null sample)")
    ap.add_argument("--seeds", type=int, default=10, help="training seeds per arm")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None,
                    help="override B. delta is exposure-dependent, so a 0C run at a reduced "
                         "budget needs its own null measured at THAT budget (2026-09-17).")
    ap.add_argument("--shard", default=None, help="I/N: run only reps with rep %% N == I (v1.6 D42 — "
                    "200-500 reps do not fit one process). Shards write their own checkpoint; "
                    "gate0b_aggregate.py computes delta and its MC uncertainty.")
    ap.add_argument("--gate", default="verifier",
                    help="eligibility rule: verifier (Gate A) or conf:<thr> (Gate B)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    BA.K = a.k

    S = load(a.locale)                                # holdout + leaf space, legacy-identical
    pool, _ = BA.load_pool(a.locale)
    # ⛔ The gate and its realized rate ride into the signature (v1.16): the null is a
    # random mask of the SAME rate, so a band measured at one pass rate does not
    # describe a gate with another.
    import gates as GT
    gate_id, pass_rate = GT.apply_gate(pool, a.locale, a.gate)
    print(f"gate={gate_id} pass_rate={pass_rate:.4f}", flush=True)
    p_real = sum(r["passed"] for r in pool) / len(pool)
    B = a.budget or BUDGET[a.locale]
    print(f"{a.locale}: pool={len(pool)} real_pass_rate={p_real:.4f} B={B:,} K={a.k}", flush=True)

    null = {"E": [], "S": [], "I": []}
    recs = []
    # ⛔ 2026-09-17: results used to be written only at the end, so a host memory kill after
    # 45 min lost every finished rep. Each rep now lands in a checkpoint before the next starts.
    out_path = Path(a.out or f"docs/experiments/2026-09-16-cgsd-gate0b-placebo-{a.locale}.json")
    shard_i, shard_n = (map(int, a.shard.split("/")) if a.shard else (0, 1))
    ckpt = out_path.with_suffix(".ckpt.jsonl" if not a.shard else f".shard{shard_i}of{shard_n}.ckpt.jsonl")
    done = {}
    for f in out_path.parent.glob(out_path.stem + ".*ckpt.jsonl"):    # any shard layout resumes
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("budget") == B:
                done[r["rep"]] = r
    if done:
        print(f"  resume: {len(done)} rep(s) already on disk", flush=True)
    for rep in range(a.reps):
        if rep % shard_n != shard_i:
            continue
        if rep in done:
            r = done[rep]; recs.append(r)
            for c in ("E", "S", "I"):
                null[c].append(r[c])
            continue
        flags = placebo_flags(len(pool), p_real, 9000 + rep)
        for r, f in zip(pool, flags):
            r["passed"] = f                           # placebo eligibility replaces the gate
        arms, _, _ = BA.build(pool, B, seed=rep)
        ids = {n: {x["example_id"] for x in arms[n]} for n in ("d00", "d10")}
        if ids["d00"] == ids["d10"]:                  # v1.3 D25
            print("  ⛔ placebo collapsed: D00 == D10 by identity — zero bits. abort.", flush=True)
            return 1
        Y = {}
        for name in ("d00", "d10", "d01", "d11"):
            pairs = [(r["query_ko"], r["leaf"]) for r in arms[name]]
            Y[name] = []
            for s_ in range(5, 5 + a.seeds):
                t0 = time.time()
                m = train(BASE, pairs, seed=s_, quiet=True, **HP)
                Y[name].append(leaf_top1_full(m, S["hold"], S["leaves"]))
                if rep == 0 and name == "d00" and s_ == 5:
                    print(f"  (first run {time.time()-t0:.1f}s)", flush=True)
        E = [Y["d10"][i] - Y["d00"][i] for i in range(a.seeds)]
        Sv = [Y["d01"][i] - Y["d00"][i] for i in range(a.seeds)]
        I = [Y["d11"][i] - Y["d10"][i] - Y["d01"][i] + Y["d00"][i] for i in range(a.seeds)]
        e, s2, i2 = st.median(E), st.median(Sv), st.median(I)
        null["E"].append(e); null["S"].append(s2); null["I"].append(i2)
        # ⛔ `seeds` and `K` ride along on every line (v1.13 D93). The aggregator builds the
        # calibration signature from these, and a missing field used to be filled by a DEFAULT —
        # stamping seeds=5 on a null actually generated with 3. Provenance travels with the data.
        rec = {"rep": rep, "budget": B, "seeds": a.seeds, "K": a.k,
               "gate": gate_id, "pass_rate": round(pass_rate, 4), "E": e, "S": s2, "I": i2,
               "arm_medians": {k: st.median(v) for k, v in Y.items()}}
        recs.append(rec)
        with ckpt.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  rep {rep}: E={e:+.5f} S={s2:+.5f} I={i2:+.5f}", flush=True)

    if a.shard:
        print(f"shard {a.shard} done — run gate0b_aggregate.py", flush=True)
        return 0
    out = {"locale": a.locale, "K": a.k, "budget": B, "reps": a.reps, "seeds": a.seeds,
           "real_pass_rate": p_real, "records": recs}
    for comp in ("E", "S", "I"):
        v = [abs(x) for x in null[comp]]
        out[f"delta_{comp}"] = float(np.percentile(v, 95)) if len(v) > 1 else float(v[0])
        out[f"median_{comp}"] = float(st.median(null[comp]))
    print(f"\ndelta_E={out['delta_E']:.5f}  delta_S={out['delta_S']:.5f}  delta_I={out['delta_I']:.5f}")
    print(f"median under null: E={out['median_E']:+.5f} S={out['median_S']:+.5f} I={out['median_I']:+.5f}")
    p = out_path
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

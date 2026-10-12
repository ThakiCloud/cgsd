#!/usr/bin/env python3
"""P3/P4 — the real experiment: decompose the ACTUAL ADF gate effect (v1.4 D27, v1.7 D53).

No planted effect here. The pool's `passed` IS the real map/translation gate, so what comes out is
the empirical decomposition of the gate we set out to audit:

    Delta_G = Y11 - Y00 = E + S + I
    E = Y10 - Y00   within-support gate effect (v1.7 D52 — not "quality")
    S = Y01 - Y00   between-support redistribution
    I = Y11 - Y10 - Y01 + Y00
    R_naive         token-budget-only reference (v1.3 D6/D19), reported next to D11

Selection seeds and training seeds are separate axes: each selection seed rebuilds the arms, each
training seed retrains them. Every (selection seed, arm, training seed) lands in the checkpoint
before the next starts, so a kill costs one training run.

    CGSD_TORCH_THREADS=1 python3 experiments/scripts/run_pilot.py --locale en --sel-seeds 3 --train-seeds 3
"""
import argparse, json, math, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
assert (REPO / "README.md").exists()
sys.path.insert(0, "/tmp/cgsd-repro-205536")
sys.path.insert(0, str(REPO / "cgsd" / "constructor"))

import os                                                   # noqa: E402
import torch                                                # noqa: E402
if os.environ.get("CGSD_TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["CGSD_TORCH_THREADS"]))

import numpy as np                                          # noqa: E402
import build_arms as BA                                     # noqa: E402
from _common import BASE, HP, load                          # noqa: E402
from train_arm import train, leaf_top1_full                 # noqa: E402


def top1_items(model, hold, all_leaves):
    """Per-item correctness over the SAME full-leaf candidate space as leaf_top1_full.

    v1.9 D62: a CI built only over selection and training seeds is conditional on this one
    evaluation set. Keeping the per-item vector lets a third bootstrap level resample items —
    PAIRED across arms, since every arm is scored on the same holdout.
    """
    leaves = list(all_leaves)
    idx = {l: i for i, l in enumerate(leaves)}
    L = model.encode(leaves); L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-9)
    Q = model.encode([q for q, _ in hold]); Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    ok = (Q @ L.T).argmax(1) == np.array([idx[l] for _, l in hold])
    return ok.astype(np.uint8)

BUDGET = {"jp": 33188, "en": 95162}                          # v1.4 D27 primary
ARMS = ("r_naive", "d00", "d10", "d01", "d11")


def boot_ci(v, rng, n=4000, lo=2.5, hi=97.5):
    import random
    r = random.Random(rng)
    ms = sorted(sum(r.choice(v) for _ in v) / len(v) for _ in range(n))
    at = lambda q: ms[min(len(ms) - 1, int(q / 100 * len(ms)))]
    return at(lo), at(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="en", choices=("en", "jp"))
    ap.add_argument("--sel-seeds", type=int, default=3)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gate", default="verifier",
                    help='"verifier" (Gate A) or "conf:0.95" (Gate B). ⛔ The gate and its '
                         'realized pass rate ride into the calibration signature, so a pilot '
                         'run under one gate can never be judged against another gate\'s null.')
    ap.add_argument("--save-items", action="store_true",
                    help="also store the per-item correctness bitmap (for the item-level bootstrap)")
    a = ap.parse_args()
    BA.K = a.k
    B = a.budget or BUDGET[a.locale]

    S = load(a.locale)
    pool, _ = BA.load_pool(a.locale)
    # ⛔ Rewrite eligibility BEFORE the arms are built — build_arms reads r["passed"].
    import gates as GT
    gate_id, pass_rate = GT.apply_gate(pool, a.locale, a.gate)
    print(f"gate={gate_id} pass_rate={pass_rate:.4f}", flush=True)
    unit = len(S["hold"])
    stem = f"docs/experiments/2026-09-18-cgsd-pilot-{a.locale}" + (f"-{a.tag}" if a.tag else "")
    ckpt = Path(stem + ".ckpt.jsonl")
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            r = json.loads(line)
            # ⛔ A checkpoint written under a different gate is a different experiment.
            if r["budget"] == B and r["K"] == a.k and r.get("gate", gate_id) == gate_id:
                done[(r["sel"], r["arm"], r["seed"])] = r["y"]
    print(f"{a.locale}: pool={len(pool)} hold={unit} B={B:,} K={a.k} "
          f"margin={BA.feasibility_margin(pool, B)[0]:+.3f} resume={len(done)}", flush=True)

    Y = {}
    for sel in range(a.sel_seeds):
        arms, rho, meta = BA.build(pool, B, seed=sel)        # strict: refuses an unmatched arm
        for arm in ARMS:
            pairs = [(r["query_ko"], r["leaf"]) for r in arms[arm]]
            for ts in range(5, 5 + a.train_seeds):
                key = (sel, arm, ts)
                if key in done:
                    Y[key] = done[key]; continue
                model = train(BASE, pairs, seed=ts, quiet=True, **HP)
                if a.save_items:
                    import base64, zlib
                    ok = top1_items(model, S["hold"], S["leaves"])
                    y = float(ok.mean())
                    items = base64.b64encode(zlib.compress(np.packbits(ok).tobytes())).decode()
                    with Path(stem + ".items.jsonl").open("a") as f:
                        f.write(json.dumps({"sel": sel, "arm": arm, "seed": ts, "budget": B,
                                            "K": a.k, "gate": gate_id,
                                            "pass_rate": round(pass_rate, 4),
                                            "n": int(ok.size), "items": items}) + "\n")
                else:
                    y = leaf_top1_full(model, S["hold"], S["leaves"])
                Y[key] = y
                with ckpt.open("a") as f:
                    # ⛔ Provenance travels with the data (v1.13 D93): a later aggregator must
                    # never have to DEFAULT the gate or its rate onto records it did not measure.
                    f.write(json.dumps({"sel": sel, "arm": arm, "seed": ts, "y": y,
                                        "budget": B, "K": a.k, "gate": gate_id,
                                        "pass_rate": round(pass_rate, 4),
                                        "n_rows": len(pairs)}) + "\n")
                print(f"  sel{sel} {arm:<8} seed{ts} top1={y:.5f}", flush=True)

    comp = {k: [] for k in ("E", "S", "I", "delta_G", "d11_vs_rnaive", "d00_vs_rnaive")}
    for sel in range(a.sel_seeds):
        g = lambda arm, i: Y[(sel, arm, 5 + i)]
        for i in range(a.train_seeds):
            comp["E"].append(g("d10", i) - g("d00", i))
            comp["S"].append(g("d01", i) - g("d00", i))
            comp["I"].append(g("d11", i) - g("d10", i) - g("d01", i) + g("d00", i))
            comp["delta_G"].append(g("d11", i) - g("d00", i))
            comp["d11_vs_rnaive"].append(g("d11", i) - g("r_naive", i))
            comp["d00_vs_rnaive"].append(g("d00", i) - g("r_naive", i))
    out = {"locale": a.locale, "K": a.k, "budget": B, "gate": gate_id,
           "pass_rate": round(pass_rate, 4), "sel_seeds": a.sel_seeds,
           "train_seeds": a.train_seeds, "holdout": unit,
           "arm_means": {arm: st.mean([Y[(s_, arm, 5 + i)] for s_ in range(a.sel_seeds)
                                       for i in range(a.train_seeds)]) for arm in ARMS},
           "components": {}}
    print(f"\n=== {a.locale} pilot (1건 = 1/{unit}) ===")
    for k, v in comp.items():
        lo, hi = boot_ci(v, 11)
        m = st.mean(v); med = st.median(v)
        pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v)
        out["components"][k] = {"n": len(v), "mean": m, "median": med, "ci95": [lo, hi],
                                "sign_pos": pos, "sign_neg": neg}
        print(f"  {k:<14} mean={m * unit:+6.1f}건 med={med * unit:+6.1f} "
              f"CI95[{lo * unit:+.1f},{hi * unit:+.1f}] 부호 +{pos}/-{neg}")
    Path(stem + ".json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {stem}.json")


if __name__ == "__main__":
    main()

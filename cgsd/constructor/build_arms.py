#!/usr/bin/env python3
"""CGSD P2 — 5-arm constructor.

    R_naive   unrestricted selection, NO support control   (token budget only)
    D00       unrestricted, source-preserved support
    D10       gate-eligible, source-preserved support
    D01       unrestricted, gate-induced support
    D11       gate-eligible, gate-induced support

Decomposition uses D00/D10/D01/D11; R_naive is the literature-facing reference
(v1.3 D6/D19). Everything is in TOKEN units — example count is never the budget
(v1.2 D13, v1.3 D20), and the backbone is a model2vec static table so effective
tokens are exactly tokenizer output with no collator or truncation.

Arm construction follows legacy _common.py so the comparison stays a reproduction:
drop holdout ids, require query_ko AND leaf non-empty, drop rows whose query
appears among holdout queries.

    B_l     = min(0.40*T_l, 0.95*B_max_l)      per locale, frozen in v1.3 appendix B
    B_c     = B * p_c        p_c   = tok_c      / T          (source support share)
    B_c^G   = B * p_c^G      p_c^G = pass_tok_c / pass_tok   (gate-induced share)

⛔ Support comes from PRE-GATE text only (query_ja/query_en), clustered at the frozen
K=16. The gate's own outputs (leaf, map_confidence, query_ko) never define support.
"""
import argparse, hashlib, json, random
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tokenizers import Tokenizer

SRC = Path("/tmp/cgsd-data/experiments/esci-ko")
TOKJSON = Path("/tmp/cgsd-data/experiments/distill-ko/out/potion-ko-128d/tokenizer.json")
EMB = Path("/tmp/cgsd-emb")
K = 8                       # frozen — v1.4 D27 (primary). --k only for the preregistered robustness arm.
CLUSTER_SEED = 42
DS = {"jp": ("esci_ko.jsonl", "esci_split.json", "query_ja"),
      "en": ("esci_ko_us.jsonl", "esci_split_en.json", "query_en")}
ARMS = ("r_naive", "d00", "d10", "d01", "d11")


class InfeasibleBudget(RuntimeError):
    """An arm could not reach the token budget — the arms are no longer matched."""


def load_pool(loc):
    fn, splitfn, pre_field = DS[loc]
    rows = [json.loads(l) for l in (SRC / fn).open()]
    hold_ids = {str(x) for x in json.loads((SRC / splitfn).read_text())["holdout"]}
    pair = lambda r: ((r.get("query_ko") or "").strip(), (r.get("leaf") or "").strip())
    ok = lambda r: r.get("map_gate") == "pass" and r.get("tr_gate") == "pass"
    hq = {pair(r)[0] for r in rows
          if str(r["example_id"]) in hold_ids and ok(r) and all(pair(r))}
    keep = [i for i, r in enumerate(rows)
            if str(r["example_id"]) not in hold_ids and all(pair(r))
            and pair(r)[0] not in hq]
    tok = Tokenizer.from_file(str(TOKJSON))
    tok.no_padding(); tok.no_truncation()   # ⛔ tokenizer.json ships padding=BatchLongest;
    #    legacy train_arm.py calls encode() one row at a time, so no padding ever reaches the model.
    qs = [pair(rows[i])[0] for i in keep]
    ls = [pair(rows[i])[1] for i in keep]
    nq = np.array([len(e.ids) for e in tok.encode_batch(qs)])
    nl = np.array([len(e.ids) for e in tok.encode_batch(ls)])
    lab = KMeans(n_clusters=K, random_state=CLUSTER_SEED,
                 n_init=10).fit_predict(np.load(EMB / f"{loc}.npy")[keep])
    pool = [{"example_id": rows[i]["example_id"], "query_ko": qs[j], "leaf": ls[j],
             "tokens": int(nq[j] + nl[j]), "tok_q": int(nq[j]), "tok_l": int(nl[j]),
             "passed": ok(rows[i]), "support": int(lab[j])}
            for j, i in enumerate(keep)]
    return pool, hq


def take(rows, budget, seed):
    """Shuffle deterministically, then take the longest prefix that fits the token budget."""
    rs = list(rows)
    random.Random(seed).shuffle(rs)
    out, used = [], 0
    for r in rs:
        if used + r["tokens"] > budget:
            continue                      # skip, keep scanning — packs the budget tighter
        out.append(r); used += r["tokens"]
        if budget - used < 2:             # 2 tokens is below any row; stop early
            break
    return out, used


BUDGET_TOL = 0.005          # v1.0 §11 — every arm must land within 0.5% of B


def feasibility_margin(pool, B):
    """CGSD feasibility margin  M = min_c (e_c - B/T)   (v1.6 D40).

    e_c = gate-eligible tokens / all tokens in support c, and B/T is the relative budget.
    D10 keeps the SOURCE support share inside the ELIGIBLE pool, so it needs (B/T)*T_c <= E_c
    in every support, i.e. B/T <= min_c e_c.
      M > 0  the matched counterfactual can be constructed
      M = 0  boundary
      M < 0  the requested counterfactual does not exist at this budget — not an estimator failure
    Positivity itself is textbook; what this pins down is the exact bound for a finite gated pool.
    Returns (M, {c: e_c}).
    """
    T = sum(r["tokens"] for r in pool)
    tot, elig = {}, {}
    for r in pool:
        tot[r["support"]] = tot.get(r["support"], 0) + r["tokens"]
        if r["passed"]:
            elig[r["support"]] = elig.get(r["support"], 0) + r["tokens"]
    e = {c: elig.get(c, 0) / tot[c] for c in tot}
    return min(v - B / T for v in e.values()), e


def build(pool, B, seed, strict=True):
    """Build the 5 arms at a matched token budget.

    ⛔ strict=True refuses to return an arm that could not be filled. The whole premise is
    MATCHED token exposure, so an under-filled arm is not a smaller arm — it is a different
    experiment. `take()` silently skips rows that do not fit, so without this guard a
    positivity violation (rho_c > 1) leaves D10 short and the shortfall is read as an effect.
    Measured 2026-09-17 in Gate 0C: a synthetic starve pushed rho to 1.6 in two supports,
    D10 came in 9.5% (en) / 6.5% (jp) short, and E turned negative in the condition where it
    was planted to be ZERO. Nothing in the pipeline noticed.
    """
    T = sum(r["tokens"] for r in pool)
    passed = [r for r in pool if r["passed"]]
    PT = sum(r["tokens"] for r in passed)
    by = lambda rs: {c: [r for r in rs if r["support"] == c] for c in range(K)}
    src, gate = by(pool), by(passed)
    p_src = {c: sum(r["tokens"] for r in src[c]) / T for c in range(K)}
    p_gate = {c: sum(r["tokens"] for r in gate[c]) / PT for c in range(K)}

    arms, diag = {}, {}
    arms["r_naive"], _ = take(pool, B, seed)          # no support control at all
    for name, share, source in (("d00", p_src, src), ("d10", p_src, gate),
                                ("d01", p_gate, src), ("d11", p_gate, gate)):
        rows, rho = [], {}
        for c in range(K):
            bc = B * share[c]
            sel, used = take(source[c], bc, seed * 1000 + c)
            rows += sel
            avail = sum(r["tokens"] for r in gate[c])   # rho is always vs the PASS pool
            rho[c] = (bc / avail) if avail else float("inf")
        arms[name] = rows
        diag[name] = rho
        if strict:
            used = sum(r["tokens"] for r in rows)
            if used < B * (1 - BUDGET_TOL):
                bad = sorted((c, v) for c, v in rho.items() if v > 1)
                raise InfeasibleBudget(
                    f"{name}: filled {used:,}/{B:,} tokens ({used / B - 1:+.2%}, tol "
                    f"{BUDGET_TOL:.1%}) — positivity violated in {len(bad)} support(s): "
                    + ", ".join(f"c{c} rho={v:.2f}" for c, v in bad[:6])
                    + f". Feasibility margin M={feasibility_margin(pool, B)[0]:+.3f}"
                    + f". Lower B to <= 0.95*B_max = {0.95 * B / max(rho.values()):,.0f}.")
    return arms, diag, {"T": T, "pass_tokens": PT, "p_src": p_src, "p_gate": p_gate,
                        "margin": feasibility_margin(pool, B)[0]}


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=tuple(DS))
    ap.add_argument("--budget", type=int, required=True, help="B_l in TOKENS (v1.3 appendix B)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed (not the training seed)")
    ap.add_argument("--k", type=int, default=K,
                    help="support resolution; primary is 8, 16 only for the matched-low-budget robustness pair")
    ap.add_argument("--tag", default="", help="manifest subdir suffix (e.g. robust_low)")
    ap.add_argument("--out", default="cgsd/manifests")
    a = ap.parse_args()
    K = a.k                      # resolution is set once, before any pool is built

    pool, hq = load_pool(a.locale)
    arms, rho, meta = build(pool, a.budget, a.seed)
    d = Path(a.out) / a.locale / (f"k{a.k}" + (f"_{a.tag}" if a.tag else "")) / f"seed_{a.seed}"
    d.mkdir(parents=True, exist_ok=True)

    stats = {"locale": a.locale, "K": K, "selection_seed": a.seed,
             "budget_tokens": a.budget, "cluster_seed": CLUSTER_SEED,
             "tokenizer": str(TOKJSON), "T_tokens": meta["T"],
             "pass_tokens": meta["pass_tokens"], "n_pool": len(pool),
             "n_pass": sum(1 for r in pool if r["passed"]), "arms": {}}
    for name in ARMS:
        rows = arms[name]
        used = sum(r["tokens"] for r in rows)
        hist = {c: sum(r["tokens"] for r in rows if r["support"] == c) for c in range(K)}
        lens = np.array([r["tok_q"] for r in rows])
        e = {"n_rows": len(rows), "tokens": used,
             "budget_err": (used - a.budget) / a.budget,
             "pass_frac_rows": sum(1 for r in rows if r["passed"]) / max(len(rows), 1),
             "support_share": {c: hist[c] / used for c in range(K)},
             "query_len_mean": float(lens.mean()), "query_len_p90": float(np.percentile(lens, 90))}
        if name in rho:
            v = [x for x in rho[name].values() if np.isfinite(x)]
            e |= {"rho_median": float(np.median(v)), "rho_p95": float(np.percentile(v, 95)),
                  "rho_max": float(max(v))}
        stats["arms"][name] = e
        p = d / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        e["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        print(f"  {name:<8} rows={len(rows):>6} tok={used:>9,} "
              f"err={e['budget_err']:+.4%} pass={e['pass_frac_rows']:.3f}", flush=True)
    (d / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {d}", flush=True)


if __name__ == "__main__":
    main()

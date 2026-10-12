#!/usr/bin/env python3
"""CGSD P0 — exact token accounting, then the token-based budget (v1.3 D20).

The legacy backbone is NOT a transformer: train_arm.py loads a model2vec StaticModel
(potion-ko-128d, distilled from jhgan/ko-sroberta-multitask) and updates only the
embedding table with InfoNCE. There is no collator, no max_length, no truncation and
no padding — mean pooling over looked-up token vectors. So "effective tokens actually
processed" is exactly len(tokenizer.encode(text).ids), and compute is LINEAR in tokens
rather than quadratic. Two texts reach the model per row: query_ko and the leaf label.
Negatives are in-batch (no explicit negative docs), so nothing else is encoded.

Arm construction mirrors legacy _common.py exactly: drop holdout ids, require both
query_ko and leaf non-empty, and drop rows whose query appears in the holdout queries.

    B_l = min( 0.40 * T_l , 0.95 * B_max_l )        (token units)
    B_max_l = min_c ( pass_tok_c / p_tok_c )        p_tok_c = tok_c / T_l
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tokenizers import Tokenizer

SRC = Path("/tmp/cgsd-data/experiments/esci-ko")
TOKJSON = Path("/tmp/cgsd-data/experiments/distill-ko/out/potion-ko-128d/tokenizer.json")
EMB = Path("/tmp/cgsd-emb")
OUT = None  # set in main()
import sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # v1.4 D27: primary = 8
SEED = 42
DS = {"jp": ("esci_ko.jsonl", "esci_split.json", "query_ja"),
      "en": ("esci_ko_us.jsonl", "esci_split_en.json", "query_en")}


def main():
    tok = Tokenizer.from_file(str(TOKJSON))
    tok.no_padding(); tok.no_truncation()   # ⛔ tokenizer.json ships padding=BatchLongest;
    #    legacy train_arm.py calls encode() one row at a time, so no padding ever reaches the model.
    out = {"K": K, "tokenizer": str(TOKJSON), "locales": {}}
    for loc, (fn, splitfn, pre_field) in DS.items():
        rows = [json.loads(l) for l in (SRC / fn).open()]
        hold_ids = {str(x) for x in json.loads((SRC / splitfn).read_text())["holdout"]}

        def pair(r):
            return (r.get("query_ko") or "").strip(), (r.get("leaf") or "").strip()

        ok = lambda r: r.get("map_gate") == "pass" and r.get("tr_gate") == "pass"
        hq = {pair(r)[0] for r in rows if str(r["example_id"]) in hold_ids and ok(r)
              and all(pair(r))}
        pool = [r for r in rows if str(r["example_id"]) not in hold_ids
                and all(pair(r)) and pair(r)[0] not in hq]

        qs = [pair(r)[0] for r in pool]
        ls = [pair(r)[1] for r in pool]
        n_q = np.array([len(e.ids) for e in tok.encode_batch(qs)])
        n_l = np.array([len(e.ids) for e in tok.encode_batch(ls)])
        t_i = n_q + n_l                                   # effective tokens per row
        passed = np.array([ok(r) for r in pool])
        T = int(t_i.sum())

        # support = pre-gate query text only (v1.2 D10), cached embeddings, K frozen
        cache = EMB / f"{loc}.npy"
        full = np.load(cache)
        pre_all = [(r.get(pre_field) or "") for r in rows]
        idx = {i for i, r in enumerate(rows)
               if str(r["example_id"]) not in hold_ids and all(pair(r))
               and pair(r)[0] not in hq}
        keep = sorted(idx)
        assert len(keep) == len(pool), (len(keep), len(pool))
        lab = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(full[keep])

        rowsc = []
        for c in range(K):
            m = lab == c
            tok_c = int(t_i[m].sum())
            pass_tok_c = int(t_i[m & passed].sum())
            if tok_c == 0:
                continue
            rowsc.append({"c": c, "n": int(m.sum()), "tok_c": tok_c,
                          "pass_tok_c": pass_tok_c, "p_tok_c": tok_c / T,
                          "pass_tok_rate": pass_tok_c / tok_c})
        zero = [r for r in rowsc if r["pass_tok_c"] == 0]
        binding = min(rowsc, key=lambda r: r["pass_tok_c"] / r["p_tok_c"])
        bmax = 0.0 if zero else binding["pass_tok_c"] / binding["p_tok_c"]
        B = min(0.40 * T, 0.95 * bmax)
        rho = [ (B * r["p_tok_c"]) / r["pass_tok_c"] if r["pass_tok_c"] else float("inf")
                for r in rowsc ]
        out["locales"][loc] = {
            "n_pool": len(pool), "n_pass": int(passed.sum()),
            "T_tokens": T, "pass_tokens": int(t_i[passed].sum()),
            "mean_tok_query": float(n_q.mean()), "mean_tok_leaf": float(n_l.mean()),
            "B_max_tok": bmax, "B_max_over_T": bmax / T,
            "cap_040T": 0.40 * T, "cap_095Bmax": 0.95 * bmax,
            "binding_term": "0.40*T" if 0.40 * T <= 0.95 * bmax else "0.95*B_max",
            "B_tokens": B, "B_over_T": B / T,
            "n_zero_pass_clusters": len(zero),
            "min_pass_tok_rate": min(r["pass_tok_rate"] for r in rowsc),
            "rho_median": float(np.median(rho)), "rho_p95": float(np.percentile(rho, 95)),
            "rho_max": float(max(rho)),
            "clusters": rowsc,
        }
        o = out["locales"][loc]
        print(f"{loc}: pool={o['n_pool']} T={T:,} tok "
              f"(q {o['mean_tok_query']:.1f} + leaf {o['mean_tok_leaf']:.1f} /row)", flush=True)
        print(f"   B_max={bmax:,.0f} ({o['B_max_over_T']:.3f} of T) | "
              f"0.40T={0.40*T:,.0f} 0.95Bmax={0.95*bmax:,.0f} → B={B:,.0f} "
              f"({o['B_over_T']:.3f}, binding={o['binding_term']})", flush=True)
        print(f"   rho median={o['rho_median']:.3f} p95={o['rho_p95']:.3f} "
              f"max={o['rho_max']:.3f}  zero-pass={len(zero)}", flush=True)
    out_p = Path(f"docs/experiments/2026-09-16-cgsd-token-budget-k{K}.json")
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_p}", flush=True)


if __name__ == "__main__":
    main()

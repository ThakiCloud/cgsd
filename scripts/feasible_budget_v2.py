#!/usr/bin/env python3
"""K feasibility, recomputed on the LEGACY TRAINING POOL and in real tokens.

v1 (feasible_budget.py) had two defects, both mine:
  (a) it scored the raw jsonl instead of the pool the runner actually trains on —
      holdout rows, empty-pair rows and holdout-query-leaking rows were all still in;
  (b) tokenizer.json ships padding=BatchLongest, so encode_batch padded every row to
      the batch max and made token counts uniform. legacy train_arm.py calls encode()
      per row, so no padding ever reaches the model.

This recomputes both ways on the correct pool so the D18 rule is applied to real input.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tokenizers import Tokenizer

S = Path("/tmp/cgsd-data/experiments/esci-ko")
EMB = Path("/tmp/cgsd-emb")
OUT = Path("docs/experiments/2026-09-16-cgsd-feasible-budget-v2.json")
KS = [8, 16, 32, 64, 128]
SEED = 42
DS = {"jp": ("esci_ko.jsonl", "esci_split.json"), "en": ("esci_ko_us.jsonl", "esci_split_en.json")}


def main():
    tok = Tokenizer.from_file(str(S.parent / "distill-ko/out/potion-ko-128d/tokenizer.json"))
    tok.no_padding(); tok.no_truncation()
    out = {}
    for loc, (fn, sp) in DS.items():
        rows = [json.loads(l) for l in (S / fn).open()]
        hold = {str(x) for x in json.loads((S / sp).read_text())["holdout"]}
        pair = lambda r: ((r.get("query_ko") or "").strip(), (r.get("leaf") or "").strip())
        ok = lambda r: r.get("map_gate") == "pass" and r.get("tr_gate") == "pass"
        hq = {pair(r)[0] for r in rows if str(r["example_id"]) in hold and ok(r) and all(pair(r))}
        keep = [i for i, r in enumerate(rows) if str(r["example_id"]) not in hold
                and all(pair(r)) and pair(r)[0] not in hq]
        emb = np.load(EMB / f"{loc}.npy")[keep]
        passed = np.array([ok(rows[i]) for i in keep])
        t = np.array([len(tok.encode(pair(rows[i])[0]).ids) + len(tok.encode(pair(rows[i])[1]).ids)
                      for i in keep])
        N, T = len(keep), int(t.sum())
        out[loc] = {"n_pool": N, "T_tokens": T, "per_k": {}}
        for K in KS:
            lab = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(emb)
            cnt = [((lab == c).sum(), int(passed[lab == c].sum()),
                    int(t[lab == c].sum()), int(t[(lab == c) & passed].sum())) for c in range(K)]
            zc = [1 for n, p, _, _ in cnt if n and p == 0]
            zt = [1 for _, _, tc, pt in cnt if tc and pt == 0]
            bc = 0.0 if zc else min(p / (n / N) for n, p, _, _ in cnt if n)
            bt = 0.0 if zt else min(pt / (tc / T) for _, _, tc, pt in cnt if tc)
            out[loc]["per_k"][K] = {"count_Bmax_over_N": bc / N, "token_Bmax_over_T": bt / T,
                                    "n_zero_pass": len(zc)}
            print(f"{loc} K={K:>3}  count {bc/N:.3f}   token {bt/T:.3f}   zero-pass={len(zc)}",
                  flush=True)
    print("\n=== D18 rule: all primary locales need zero-pass 0 AND ratio >= 0.40 ===")
    for K in KS:
        j, e = out["jp"]["per_k"][K], out["en"]["per_k"][K]
        for unit, key in (("count", "count_Bmax_over_N"), ("token", "token_Bmax_over_T")):
            ok_ = (j["n_zero_pass"] == 0 and e["n_zero_pass"] == 0
                   and j[key] >= 0.40 and e[key] >= 0.40)
            print(f"K={K:>3} [{unit}] jp={j[key]:.3f} en={e[key]:.3f} → {'PASS' if ok_ else 'fail'}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

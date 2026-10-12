#!/usr/bin/env python3
"""CGSD — feasible budget per K (survey v2).

v1 asked "is B_c <= pass_c at B = all passed items?" and found ~half the clusters
violating at every K. That is not a property of K: with B = total pass,
B_c <= pass_c reduces to (overall pass rate) <= (cluster pass rate), so by
construction roughly half the clusters fail. The budget was the binding choice,
not the clustering.

So this asks the right question instead: for each K, what is the LARGEST budget
that keeps every cluster feasible?

    B_max = min_c ( pass_c / p_c )        p_c = total_c / N
    B_max = 0  if any cluster has pass_c == 0   (no budget rescues an empty cell)

K selection rule (D10): pick the largest K that is feasible, decided WITHOUT
looking at downstream performance. With B_max in hand the rule becomes a
resolution-vs-budget tradeoff, which is a decision the paper can state openly.

Run:  /tmp/cgsd-venv/bin/python experiments/scripts/feasible_budget.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

DATA_DIR = Path("/tmp/cgsd-data/experiments/esci-ko")
CACHE = Path("/tmp/cgsd-emb")
OUT = Path("docs/experiments/2026-09-16-cgsd-feasible-budget.json")
SEED = 42
K_VALUES = [8, 16, 32, 64, 128]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCALES = {
    "jp": ("esci_ko.jsonl", "query_ja"),
    "en": ("esci_ko_us.jsonl", "query_en"),
}


def load(path, field):
    texts, passed = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r[field])
            passed.append(r.get("map_gate") == "pass" and r.get("tr_gate") == "pass")
    return texts, np.array(passed)


def embed(texts, tag):
    CACHE.mkdir(exist_ok=True)
    c = CACHE / f"{tag}.npy"
    if c.exists():
        return np.load(c)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL_NAME)
    e = m.encode(texts, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
    np.save(c, e)
    return e


def main():
    out = {}
    for loc, (fn, field) in LOCALES.items():
        texts, passed = load(DATA_DIR / fn, field)
        N, P = len(texts), int(passed.sum())
        emb = embed(texts, loc)
        out[loc] = {"total_n": N, "pass_n": P, "pass_rate": P / N, "per_k": {}}
        for K in K_VALUES:
            lab = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(emb)
            rows = []
            for c in range(K):
                m = lab == c
                tot = int(m.sum())
                if tot == 0:
                    continue
                pc = int(passed[m].sum())
                rows.append({"c": c, "total_c": tot, "pass_c": pc,
                             "p_c": tot / N, "pass_rate_c": pc / tot})
            zero = [r for r in rows if r["pass_c"] == 0]
            if zero:
                bmax, binding = 0.0, zero[0]
            else:
                binding = min(rows, key=lambda r: r["pass_c"] / r["p_c"])
                bmax = binding["pass_c"] / binding["p_c"]
            out[loc]["per_k"][K] = {
                "n_clusters": len(rows),
                "n_zero_pass": len(zero),
                "B_max": bmax,
                "B_max_over_N": bmax / N,
                "B_max_over_pass": bmax / P,
                "binding_cluster": binding,
                "min_pass_rate_c": min(r["pass_rate_c"] for r in rows),
                "median_total_c": float(np.median([r["total_c"] for r in rows])),
                "min_total_c": min(r["total_c"] for r in rows),
            }
            print(f"{loc} K={K:>3} zero={len(zero)} B_max={bmax:>8.0f} "
                  f"({bmax/N:.3f} of N, {bmax/P:.3f} of pass) "
                  f"min_pass_rate_c={min(r['pass_rate_c'] for r in rows):.3f}", flush=True)

    # 두 locale 동시 만족: 같은 K 에서 둘 다 B_max > 0 이고, 표본이 쓸 만한가
    print("\n=== joint ===", flush=True)
    for K in K_VALUES:
        j, e = out["jp"]["per_k"][K], out["en"]["per_k"][K]
        ok = j["B_max"] > 0 and e["B_max"] > 0
        print(f"K={K:>3} feasible={ok}  jp B_max={j['B_max']:>7.0f} ({j['B_max_over_N']:.3f})  "
              f"en B_max={e['B_max']:>8.0f} ({e['B_max_over_N']:.3f})", flush=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

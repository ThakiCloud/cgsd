#!/usr/bin/env python3
"""CGSD latent support feasibility survey.

Pre-gate support representation (query text only, pre-map/pre-translation-gate)
embedded with a frozen multilingual sentence encoder, clustered with k-means at
K in {32,64,128}, and checked for D10-quota positivity feasibility per cluster.

Run with the isolated venv:
    /tmp/cgsd-venv/bin/python /tmp/cgsd-latent-survey.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

DATA_DIR = Path("/tmp/cgsd-data/experiments/esci-ko")
OUT_JSON = Path("/tmp/cgsd-latent-survey-results.json")
SEED = 42
K_VALUES = [32, 64, 128]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LOCALES = {
    "jp": {"file": DATA_DIR / "esci_ko.jsonl", "text_field": "query_ja"},
    "en": {"file": DATA_DIR / "esci_ko_us.jsonl", "text_field": "query_en"},
}


def load_rows(path: Path, text_field: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            passed = (r.get("map_gate") == "pass") and (r.get("tr_gate") == "pass")
            rows.append({
                "text": r[text_field],
                "passed": passed,
                "map_gate": r.get("map_gate"),
                "tr_gate": r.get("tr_gate"),
            })
    return rows


def embed_texts(texts, encoder_kind, model=None, vectorizer=None, svd=None):
    if encoder_kind == "st":
        return model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    else:  # tfidf fallback
        X = vectorizer.transform(texts)
        return svd.transform(X)


def cluster_and_report(rows, embeddings, k_values, seed, locale, encoder_kind):
    total_n = len(rows)
    passed_flags = np.array([r["passed"] for r in rows])
    B = int(passed_flags.sum())  # training budget == number of passed items (assumption stated in report)

    results = {}
    for K in k_values:
        km = KMeans(n_clusters=K, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)

        cluster_stats = []
        for c in range(K):
            mask = labels == c
            total_c = int(mask.sum())
            pass_c = int(passed_flags[mask].sum())
            p_c = total_c / total_n if total_n > 0 else 0.0
            B_c = B * p_c
            violation = B_c > pass_c
            rho = (B_c / pass_c) if pass_c > 0 else float("inf")
            cluster_stats.append({
                "cluster": c,
                "total_c": total_c,
                "pass_c": pass_c,
                "p_c": p_c,
                "B_c": B_c,
                "rho": rho,
                "violation": violation,
            })

        totals = np.array([cs["total_c"] for cs in cluster_stats])
        n_zero_pass = sum(1 for cs in cluster_stats if cs["pass_c"] == 0)
        n_violation = sum(1 for cs in cluster_stats if cs["violation"])
        violating_mass = sum(cs["p_c"] for cs in cluster_stats if cs["violation"])
        rhos_finite = [cs["rho"] for cs in cluster_stats if np.isfinite(cs["rho"])]
        n_rho_gt_09 = sum(1 for r in rhos_finite if r > 0.9)

        results[K] = {
            "locale": locale,
            "K": K,
            "total_n": total_n,
            "B": B,
            "n_clusters": K,
            "n_zero_pass_clusters": n_zero_pass,
            "n_violating_clusters": n_violation,
            "violating_mass_p": violating_mass,
            "total_c_min": int(totals.min()),
            "total_c_p10": float(np.percentile(totals, 10)),
            "total_c_median": float(np.percentile(totals, 50)),
            "rho_median": float(np.median(rhos_finite)) if rhos_finite else None,
            "rho_p95": float(np.percentile(rhos_finite, 95)) if rhos_finite else None,
            "n_rho_gt_0.9": n_rho_gt_09,
            "feasible": n_violation == 0,
            "encoder_kind": encoder_kind,
        }
        print(f"[{locale}] K={K}: violating={n_violation}/{K} zero_pass={n_zero_pass} "
              f"rho_median={results[K]['rho_median']:.3f} rho_p95={results[K]['rho_p95']:.3f} "
              f"feasible={results[K]['feasible']}")
    return results


def main():
    encoder_kind = "st"
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer(MODEL_NAME)
        print(f"[encoder] loaded {MODEL_NAME} in {time.time()-t0:.1f}s", file=sys.stderr)
    except Exception as e:
        print(f"[encoder] sentence-transformers failed ({e}); falling back to TF-IDF+SVD", file=sys.stderr)
        encoder_kind = "tfidf"

    all_results = {}
    for locale, cfg in LOCALES.items():
        print(f"\n=== locale={locale} file={cfg['file'].name} ===")
        rows = load_rows(cfg["file"], cfg["text_field"])
        texts = [r["text"] for r in rows]
        n_pass = sum(r["passed"] for r in rows)
        print(f"rows={len(rows)} pass={n_pass} ({n_pass/len(rows)*100:.1f}%)")

        if encoder_kind == "st":
            embeddings = embed_texts(texts, "st", model=model)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000)
            X = vec.fit_transform(texts)
            svd = TruncatedSVD(n_components=256, random_state=SEED)
            embeddings = svd.fit_transform(X)

        print(f"embeddings shape: {embeddings.shape}")
        res = cluster_and_report(rows, embeddings, K_VALUES, SEED, locale, encoder_kind)
        all_results[locale] = res

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()

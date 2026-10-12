#!/usr/bin/env python3
"""P3/P4 analysis — decomposition identity, hierarchical inference, and frontier diagnostics (v1.8).

⛔ The decomposition identity is checked on UNROUNDED per-replicate values:

    r_decomp = Delta_G - (E + S + I)     must be 0 to float tolerance, every replicate

⚠️ The identity is linear, so it survives MEANS but NOT MEDIANS. A table of per-component medians
can look like it violates the identity (2026-09-18: 38 vs 3 + (-5) + 38). Medians are reported as
robustness, and the identity-bearing aggregate is the mean.

Design has two nested axes — selection seed (rebuilds the arms) and training seed (retrains them) —
so 3x3 is NOT 9 iid replicates. Inference is a hierarchical bootstrap (resample selection seeds,
then training seeds within each), and the variance is split into selection vs training components.

Frontier diagnostics, because the real operating point sits at margin +0.027: rho median/p95/max,
the share of supports above 0.9, and how much the D10 sample actually CHANGES across selection
seeds (Jaccard). If D10 consumes nearly the whole pass pool, three selection seeds are not three
independent realizations and must not be described as such.

    python3 experiments/scripts/pilot_analyze.py docs/experiments/2026-09-18-cgsd-pilot-en.ckpt.jsonl
"""
import argparse, base64, json, math, random, statistics as st, sys, zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cgsd" / "constructor"))
ARMS = ("r_naive", "d00", "d10", "d01", "d11")
COMP = ("E", "S", "I", "delta_G", "resid", "d11_vs_rnaive", "d00_vs_rnaive")


def calibration_signature(locale, K, budget, n_items, seeds, gate, pass_rate,
                          endpoint="leaf_top1_full"):
    """Everything a delta must share with the run it judges (v1.12 D78).

    ⛔ Locale alone is not enough. delta measured at one exposure does not transfer to another —
    we measured E growing and per-run noise shrinking with budget — and K changes the support
    geometry, hence the null variance. The constructor file is hashed too: change how arms are
    built and the old null no longer describes them.
    """
    import hashlib, json
    ctor = (Path(__file__).resolve().parents[2] / "cgsd" / "constructor" / "build_arms.py")
    payload = {"locale": locale, "K": K, "budget": budget, "n_items": n_items,
               "seeds": seeds, "gate": gate, "pass_rate": round(float(pass_rate), 4),
               "endpoint": endpoint,
               "constructor_sha": hashlib.sha256(ctor.read_bytes()).hexdigest()[:12]}
    payload["signature"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return payload


def replicate(y):
    """y: arm -> value for ONE (selection seed, training seed) pair."""
    E = y["d10"] - y["d00"]; S = y["d01"] - y["d00"]
    I = y["d11"] - y["d10"] - y["d01"] + y["d00"]; D = y["d11"] - y["d00"]
    return {"E": E, "S": S, "I": I, "delta_G": D, "resid": D - (E + S + I),
            "d11_vs_rnaive": y["d11"] - y["r_naive"], "d00_vs_rnaive": y["d00"] - y["r_naive"]}


def hier_boot(by_sel, comp, rng, n=4000):
    """Resample selection seeds with replacement, then training seeds inside each."""
    sels = list(by_sel)
    out = []
    for _ in range(n):
        vals = []
        for s in (rng.choice(sels) for _ in sels):
            reps = by_sel[s]
            vals += [rng.choice(reps)[comp] for _ in reps]
        out.append(sum(vals) / len(vals))
    out.sort()
    at = lambda q: out[min(len(out) - 1, int(q / 100 * len(out)))]
    return at(2.5), at(97.5)


def var_components(by_sel, comp):
    """One-way random effects: between-selection vs within-selection (training) variance."""
    groups = [[r[comp] for r in v] for v in by_sel.values()]
    k = len(groups); n = st.mean(len(g) for g in groups)
    if k < 2 or min(len(g) for g in groups) < 2:
        return None, None
    gm = [st.mean(g) for g in groups]; grand = st.mean(gm)
    msb = sum(len(g) * (st.mean(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum((x - st.mean(g)) ** 2 for g in groups for x in g) / sum(len(g) - 1 for g in groups)
    return max(0.0, (msb - msw) / n), msw          # sigma2_selection, sigma2_training


def item_level(ckpt, unit, rng, n=2000, trunc=3914):
    """3-level bootstrap: selection seed -> training seed -> PAIRED evaluation items (v1.9 D62).

    Every arm is scored on the same holdout, so items are resampled ONCE per draw and applied to
    all arms; resampling them independently would destroy the pairing and inflate the interval.
    Returns None when the run was made without --save-items.
    """
    p = Path(str(ckpt).replace(".ckpt.jsonl", ".items.jsonl"))
    if not p.exists():
        return None
    import numpy as np
    v = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        bits = np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(r["items"])), dtype=np.uint8))
        # ⛔ np.packbits pads to a byte boundary: 3914 items -> 3920 bits. Those 6 zero bits would
        # enter the bootstrap as always-wrong items shared by every arm (2026-09-18).
        n_items = r.get("n") or trunc
        v.setdefault(r["sel"], {}).setdefault(r["seed"], {})[r["arm"]] = bits[:n_items]
    per = {}                                    # (sel, seed) -> component -> per-item contrast
    for s_, d in v.items():
        for ts, arms in d.items():
            if len(arms) < 5:
                continue
            a = {k: arms[k][:n_items].astype(np.int8) for k in arms}
            per[(s_, ts)] = {"E": a["d10"] - a["d00"], "S": a["d01"] - a["d00"],
                             "I": a["d11"] - a["d10"] - a["d01"] + a["d00"],
                             "delta_G": a["d11"] - a["d00"]}
    sels = sorted({k[0] for k in per})
    out = {}
    draws = {c: [] for c in ("E", "S", "I", "delta_G")}
    for _ in range(n):
        idx = np.random.randint(0, n_items, n_items)          # paired item resample, once per draw
        picked = []
        for s_ in (rng.choice(sels) for _ in sels):
            ts = [k for k in per if k[0] == s_]
            picked += [rng.choice(ts) for _ in ts]
        for c in draws:
            draws[c].append(float(np.mean([per[k][c][idx].mean() for k in picked])))
    for c, d in draws.items():
        d.sort()
        at = lambda q: d[min(len(d) - 1, int(q / 100 * len(d)))]
        out[c] = [at(2.5), at(97.5)]
    return out, n_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--delta", default="docs/experiments/2026-09-17-cgsd-gate0b-en-b55300-calibrated.json")
    ap.add_argument("--gate", default=None, help="gate identity for pre-v1.16 runs")
    ap.add_argument("--pass-rate", type=float, default=None,
                    help="realized eligibility rate for pre-v1.16 runs")
    ap.add_argument("--diag", action="store_true", help="rebuild arms for rho / overlap diagnostics")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.ckpt)]
    meta = recs[0]
    y = {}
    for r in recs:
        y.setdefault(r["sel"], {}).setdefault(r["seed"], {})[r["arm"]] = r["y"]
    by_sel = {s: [replicate(v) for v in sorted(d.values(), key=lambda x: 0) if len(v) == 5]
              for s, d in y.items()}
    by_sel = {s: v for s, v in by_sel.items() if v}
    unit = a.unit or 3914
    u = 1 / unit
    allr = [r for v in by_sel.values() for r in v]
    print(f"budget={meta['budget']:,} K={meta['K']}  selection seeds={len(by_sel)}  "
          f"replicates={len(allr)}  (1건 = {u:.6f})")

    mx = max(abs(r["resid"]) for r in allr)
    print(f"\n[identity] max |Delta_G - (E+S+I)| over replicates = {mx:.3e} "
          f"({'✅ 0' if mx < 1e-12 else '⛔ NONZERO'})")
    med_res = st.median([r["delta_G"] for r in allr]) - sum(
        st.median([r[c] for r in allr]) for c in ("E", "S", "I"))
    print(f"[identity] same check on per-component MEDIANS = {med_res * unit:+.1f}건 "
          f"— medians are not additive, so this is expected to be nonzero (report means)")

    delta = {}
    # ⛔ gate + realized pass rate are part of the signature (v1.16): the null uses a random
    # mask of the SAME RATE, so a different gate family cannot borrow another's band.
    g = meta.get("gate") or a.gate
    pr = meta.get("pass_rate") if meta.get("pass_rate") is not None else a.pass_rate
    if g is None or pr is None:
        raise SystemExit("⛔ 이 실행의 gate/pass_rate 를 알 수 없다. --gate 와 --pass-rate 로 "
                         "명시하라 — 기본값으로 채우면 서명이 거짓을 말한다 (v1.16).")
    mine = calibration_signature(
        "jp" if "-jp" in Path(a.ckpt).name else "en", meta["K"], meta["budget"], unit,
        len({r["seed"] for r in recs}), g, pr)
    # ⛔ v1.11 D77 / v1.12 D78: delta is specific to locale AND K AND budget AND holdout AND
    # constructor. Locale-only checking let an EN delta label a jp run (2026-09-18). (different holdout, different grid). The default path
    # points at the EN calibration, so a jp run would silently be labelled against EN's resolution.
    if Path(a.delta).exists():
        d = json.load(open(a.delta))
        theirs = d.get("calibration", {})
        if theirs.get("signature") == mine["signature"]:
            delta = {k: v["delta"] for k, v in d["components"].items()}
            print(f"[delta] {Path(a.delta).name} · signature {mine['signature']} 일치: " +
                  "  ".join(f"{k} {v * unit:.1f}건" for k, v in delta.items()))
        else:
            diff = [k for k in ("locale", "K", "budget", "n_items", "seeds", "constructor_sha")
                    if theirs.get(k) != mine[k]]
            print(f"\n[delta] ⛔ 라벨 보류 — calibration signature 불일치"
                  f"{' (' + ', '.join(diff) + ' 다름)' if diff else ' (서명 없음/구버전)'}. "
                  f"run={mine['signature']} delta={theirs.get('signature', 'none')}. "
                  f"resolved/unresolved 를 붙이지 않고 CI 로만 읽는다.")
    else:
        print(f"\n[delta] 보정 파일 없음 — CI 로만 읽는다 (run signature {mine['signature']})")

    print("\n[per selection seed] (median over training seeds)")
    for s, v in sorted(by_sel.items()):
        print(f"  sel{s} n={len(v)}: " + "  ".join(
            f"{c}={st.median([r[c] for r in v]) * unit:+6.1f}" for c in
            ("E", "S", "I", "delta_G", "d11_vs_rnaive", "d00_vs_rnaive")))

    rng = random.Random(17)
    print("\n[pooled] mean · hierarchical bootstrap CI95 · sign · resolved vs delta")
    out = {"budget": meta["budget"], "K": meta["K"], "unit": unit,
           "selection_seeds": len(by_sel), "replicates": len(allr),
           "identity_max_abs_resid": mx, "components": {}}
    for c in COMP:
        v = [r[c] for r in allr]
        m = st.mean(v); lo, hi = hier_boot(by_sel, c, rng)
        s2s, s2t = var_components(by_sel, c)
        pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v)
        zero = len(v) - pos - neg
        res = ""
        if c in delta:
            res = " RESOLVED" if abs(m) > delta[c] else " unresolved(<delta)"
        out["components"][c] = {"mean": m, "median": st.median(v), "ci95": [lo, hi],
                                "sign_pos": pos, "sign_neg": neg, "sign_zero": zero,
                                "sigma2_selection": s2s, "sigma2_training": s2t,
                                "resolved": (abs(m) > delta[c]) if c in delta else None}
        sd = ""
        if s2s is not None:
            sd = (f"  sd_sel={math.sqrt(s2s) * unit:5.1f}건 sd_train={math.sqrt(s2t) * unit:5.1f}건"
                  f" ratio={math.sqrt(s2s / s2t) if s2t else float('inf'):.2f}")
        print(f"  {c:<14} mean={m * unit:+6.1f}건 CI95[{lo * unit:+6.1f},{hi * unit:+6.1f}] "
              f"부호 +{pos}/-{neg}/={zero}{res}{sd}")

    il = item_level(a.ckpt, unit, rng, trunc=unit)
    if il:
        ci, n_items = il
        print(f"\n[item-inclusive] 3-level bootstrap (selection > training > paired items, "
              f"{n_items} items) — wider than the conditional CI by construction")
        for c, (lo, hi) in ci.items():
            print(f"  {c:<14} CI95[{lo * unit:+6.1f},{hi * unit:+6.1f}]건")
        out["item_inclusive_ci95"] = ci
    else:
        print("\n[item-inclusive] 건너뜀 — 이 실행에는 --save-items 항목 벡터가 없다 "
              "(현재 CI 는 이 평가셋 3,914 문항에 조건부다)")

    if a.diag:
        import build_arms as BA
        BA.K = meta["K"]
        pool, _ = BA.load_pool("en" if "-en" in a.ckpt else "jp")
        print("\n[frontier diagnostics]")
        d10 = {}
        for s in sorted(by_sel):
            arms, rho, m2 = BA.build(pool, meta["budget"], seed=s)
            for arm in ("d10", "d11"):
                rv = sorted(rho[arm].values())
                print(f"  sel{s} {arm}: rho median={st.median(rv):.3f} p95={rv[int(.95 * (len(rv) - 1))]:.3f} "
                      f"max={max(rv):.3f}  rho>0.9 in {sum(x > 0.9 for x in rv)}/{len(rv)} supports"
                      + (f"  margin={m2['margin']:+.3f}" if arm == "d10" else ""))
            d10[s] = {x["example_id"] for x in arms["d10"]}
        ks = sorted(d10)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a_, b_ = d10[ks[i]], d10[ks[j]]
                print(f"  D10 overlap sel{ks[i]}~sel{ks[j]}: Jaccard={len(a_ & b_) / len(a_ | b_):.3f} "
                      f"(|A∩B|/|A|={len(a_ & b_) / len(a_):.3f})")
        out["diagnostics_printed"] = True
    p = Path(a.ckpt.replace(".ckpt.jsonl", "-analysis.json"))
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

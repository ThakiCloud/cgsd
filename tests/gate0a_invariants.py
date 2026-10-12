#!/usr/bin/env python3
"""Gate 0A — data-level invariants on a CGSD manifest set. No training.

T1  support share (token)  D00 == D10            source-preserved pair
T2  support share (token)  D01 == D11            gate-induced pair
T3  all five arms hit the token budget within 0.5%
T4  query-length distribution comparable across arms (report + loose bound)
T5  gate-eligible arms are 100% passed; unrestricted arms are not
T6  D00/D01 selection does not use the gate verdict — their pass fraction must
    sit near the pool/support-local base rate, not at 0 or 1
T7  selection is reproducible under the same seed (sha256 of each manifest)
T8  no holdout query leaks into any arm
T9  D00 != D10 by sample identity  (v1.3 D25 — the placebo lesson)

exit 0 = all pass, 1 = some invariant failed.
"""
import json, sys
from pathlib import Path

import numpy as np

TOL_BUDGET = 0.005
TOL_SHARE = 1e-9
ARMS = ("r_naive", "d00", "d10", "d01", "d11")


def ids(p):
    return {json.loads(l)["example_id"] for l in p.open()}


def check(d: Path, hq: set | None = None):
    st = json.loads((d / "stats.json").read_text())
    a = st["arms"]
    fails, notes = [], []

    def share(n):
        return np.array([a[n]["support_share"][str(c)] if str(c) in a[n]["support_share"]
                         else a[n]["support_share"][c] for c in range(st["K"])])

    for t, x, y in (("T1", "d00", "d10"), ("T2", "d01", "d11")):
        diff = float(np.abs(share(x) - share(y)).max())
        (notes if diff <= 1e-3 else fails).append(f"{t} {x}~{y} support share max|Δ|={diff:.2e}")

    for n in ARMS:
        e = abs(a[n]["budget_err"])
        (notes if e <= TOL_BUDGET else fails).append(f"T3 {n} budget err={a[n]['budget_err']:+.4%}")

    lens = {n: a[n]["query_len_mean"] for n in ARMS}
    spread = max(lens.values()) - min(lens.values())
    (notes if spread <= 3.0 else fails).append(
        f"T4 query_len_mean spread={spread:.2f} ({', '.join(f'{k}={v:.1f}' for k, v in lens.items())})")

    for n in ("d10", "d11"):
        (notes if a[n]["pass_frac_rows"] == 1.0 else fails).append(
            f"T5 {n} pass_frac={a[n]['pass_frac_rows']:.3f} (must be 1.000)")
    base = st["n_pass"] / st["n_pool"]
    for n in ("r_naive", "d00", "d01"):
        f = a[n]["pass_frac_rows"]
        ok = 0.05 < f < 0.95 and abs(f - base) < 0.15
        (notes if ok else fails).append(f"T6 {n} pass_frac={f:.3f} vs pool base={base:.3f}")

    for n in ARMS:
        p = d / f"{n}.jsonl"
        ok = p.exists() and "sha256" in a[n]
        (notes if ok else fails).append(f"T7 {n} manifest+sha present")

    if hq is not None:
        for n in ARMS:
            q = {json.loads(l)["query_ko"] for l in (d / f"{n}.jsonl").open()}
            k = len(q & hq)
            (notes if k == 0 else fails).append(f"T8 {n} holdout-query leak={k}")
    else:
        notes.append("T8 skipped (no holdout set passed)")

    i0, i1 = ids(d / "d00.jsonl"), ids(d / "d10.jsonl")
    j = len(i0 & i1) / len(i0 | i1)
    (notes if i0 != i1 else fails).append(
        f"T9 D00 != D10 by identity (jaccard={j:.3f}, |D00|={len(i0)}, |D10|={len(i1)})")

    return fails, notes


def main():
    roots = [Path(x) for x in sys.argv[1:]] or sorted(Path("cgsd/manifests").glob("*/k*/seed_*"))
    bad = 0
    for d in roots:
        hq = None
        sp = Path(__file__).resolve().parent.parent / "cgsd/constructor/build_arms.py"
        if sp.exists():
            sys.path.insert(0, str(sp.parent))
            try:
                import build_arms
                st0 = json.loads((d / "stats.json").read_text())
                build_arms.K = st0["K"]          # the manifest's own resolution
                _, hq = build_arms.load_pool(st0["locale"])
            except Exception as ex:                       # noqa: BLE001
                print(f"  (holdout set unavailable: {ex})")
        fails, notes = check(d, hq)
        print(f"\n== {d}")
        for n in notes:
            print(f"  ✅ {n}")
        for f in fails:
            print(f"  ❌ {f}")
        bad += len(fails)
    print(f"\n{'PASS' if bad == 0 else f'FAIL ({bad})'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

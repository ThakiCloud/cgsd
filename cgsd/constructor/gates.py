#!/usr/bin/env python3
"""Gate families — the eligibility rule, kept OUT of build_arms.py on purpose.

Gate A (`verifier`) is the deterministic rule the paper's primary results use: a pair is
eligible when both the mapping check and the translation check pass.
Gate B (`conf:<thr>`) thresholds the mapper's own continuous confidence — a model-based score
gate, a different gate FAMILY on the same pool, and the axis along which a threshold sweep is
defined.

⛔ This lives in its own module because `calibration_signature()` hashes `build_arms.py`;
adding a gate switch there would move that hash and invalidate every stored delta (the trap
recorded in v1.13 D92).

⛔ Support is never touched here. `map_confidence` is a gate output: it may decide eligibility
and may never define the support partition, which stays k-means on pre-gate query text.
"""
import json
from pathlib import Path

import build_arms as BA

VERIFIER = "verifier:map+tr"


def _confidence(locale):
    fn = BA.DS[locale][0]
    out = {}
    for line in (BA.SRC / fn).open():
        r = json.loads(line)
        c = r.get("map_confidence")
        if c is not None:
            out[r["example_id"]] = float(c)
    return out


def apply_gate(pool, locale, spec):
    """Rewrite r['passed'] in place for the named gate. Returns (spec, realized pass rate).

    spec: "verifier"  -> Gate A, leave load_pool's verdict alone
          "conf:0.75" -> Gate B, eligible iff map_confidence >= 0.75
    """
    if spec in ("verifier", VERIFIER):
        rate = sum(r["passed"] for r in pool) / len(pool)
        return VERIFIER, rate
    if not spec.startswith("conf:"):
        raise ValueError(f"unknown gate spec {spec!r}")
    thr = float(spec.split(":", 1)[1])
    conf = _confidence(locale)
    for r in pool:
        r["passed"] = conf.get(r["example_id"], 0.0) >= thr
    rate = sum(r["passed"] for r in pool) / len(pool)
    return f"confidence>={thr:g}", rate

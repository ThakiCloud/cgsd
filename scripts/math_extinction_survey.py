#!/usr/bin/env python3
"""Extinction survey in TOKEN units for the ADF Math/SQL pools (v1.13 D84).

⛔ Row mass is not the quantity CGSD uses. Exposure is matched in tokens, so the extinct mass and
the feasibility bound must both be token-weighted. Reports row mass and token mass side by side.

    python3 experiments/scripts/math_extinction_survey.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path("/tmp/cgsd-data/experiments/math/cgsd-math-20260918/adf/outputs")
TOKJSON = Path("/tmp/cgsd-data/experiments/distill-ko/out/potion-ko-128d/tokenizer.json")
M_SAFE = 0.08                      # v1.9 D61 — reuse, do not invent a new threshold
POOLS = ("math-1", "power-wide", "ab-weak8b", "gen-27b-2026-08-20", "live-clean", "live-scale")


def load(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return []


def main():
    tok = Tokenizer.from_file(str(TOKJSON))
    tok.no_padding(); tok.no_truncation()
    print(f"{'pool':<20}{'rows':>6}{'tags':>5}  {'row_ext':>8}{'tok_ext':>8}  "
          f"{'min_e_row':>10}{'min_e_tok':>10}  {'B/T max (M>=0.08)':>18}")
    for name in POOLS:
        g, f = load(ROOT / name / "dataset-gated.jsonl"), load(ROOT / name / "rejected-gated.jsonl")
        if not g:
            continue
        rows = [(r, True) for r in g] + [(r, False) for r in f]
        texts = [r.get("question") or r.get("prompt") or "" for r, _ in rows]
        lens = [len(e.ids) for e in tok.encode_batch(texts)]
        tot_r, tot_t, ps_r, ps_t = (defaultdict(int) for _ in range(4))
        for (r, ok), n in zip(rows, lens):
            t = r.get("tag")
            tot_r[t] += 1; tot_t[t] += n
            if ok:
                ps_r[t] += 1; ps_t[t] += n
        tags = sorted(tot_r)
        e_row = {t: ps_r[t] / tot_r[t] for t in tags}
        e_tok = {t: (ps_t[t] / tot_t[t]) if tot_t[t] else 0.0 for t in tags}
        ext = [t for t in tags if e_tok[t] == 0]
        R, T = sum(tot_r.values()), sum(tot_t.values())
        row_ext = sum(tot_r[t] for t in ext) / R
        tok_ext = sum(tot_t[t] for t in ext) / T
        mr, mt = min(e_row.values()), min(e_tok.values())
        cap = (mt - M_SAFE) * 100
        note = f"{max(0.0, mt - M_SAFE):.3f}" + ("" if mt - M_SAFE > 0 else "  ⛔ interior 불가")
        print(f"{name:<20}{R:>6}{len(tags):>5}  {row_ext:>7.1%}{tok_ext:>8.1%}  "
              f"{mr:>10.3f}{mt:>10.3f}  {note:>18}")
        if ext:
            print(f"    소멸 태그: " + ", ".join(f"{t}(row {tot_r[t]}, tok {tot_t[t]})" for t in ext))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Collect every number the paper cites into one results.json.

⛔ Figures and tables are GENERATED from this file — no number is ever retyped into LaTeX
(arxiv-submitter: "results.json -> one script per figure -> vector PDF"). When a measurement
changes you rerun this, not the .tex.

    python3 paper/collect_results.py
"""
import json, subprocess, sys
from pathlib import Path

_R = Path(__file__).resolve().parents[1]
# Results live in docs/experiments in the working tree and results/ in the public release.
D = next((_R / c for c in ("docs/experiments", "results") if (_R / c).is_dir()), _R / "results")
OUT = Path(__file__).resolve().parent / "results.json"


def load(name):
    return json.loads((D / name).read_text())


def comp(a, unit, key):
    c = a["components"][key]
    return {"mean_items": c["mean"] * unit,
            "ci95_items": [c["ci95"][0] * unit, c["ci95"][1] * unit],
            "sign": [c["sign_pos"], c["sign_neg"], c.get("sign_zero")],
            "resolved": c.get("resolved"),   # None = label withheld (signature mismatch)
            "sd_sel_items": c["sigma2_selection"] ** .5 * unit,
            "sd_train_items": c["sigma2_training"] ** .5 * unit}


def pilot(name, unit):
    a = load(name)
    out = {"budget": a["budget"], "K": a["K"], "n_items": unit,
           "replicates": a["replicates"], "selection_seeds": a["selection_seeds"],
           "identity_max_abs_resid": a["identity_max_abs_resid"]}
    for k in ("E", "S", "I", "delta_G", "d11_vs_rnaive", "d00_vs_rnaive"):
        if k in a["components"]:
            out[k] = comp(a, unit, k)
    if "item_bootstrap" in a:
        out["item_ci95_items"] = {k: [v[0] * unit, v[1] * unit]
                                  for k, v in a["item_bootstrap"].items()}
    return out


def delta(name):
    a = load(name)
    c = a["calibration"]
    u = a["unit"]
    return {"signature": c["signature"], "budget": c["budget"], "seeds": c["seeds"],
            "K": c["K"], "n_items": c["n_items"], "reps": a["reps"],
            **{f"delta_{k}_items": a["components"][k]["delta"] * u for k in ("E", "S", "I")},
            **{f"delta_{k}_ci_items": [a["components"][k]["delta_ci95"][0] * u,
                                       a["components"][k]["delta_ci95"][1] * u]
               for k in ("E", "S", "I")},
            **{f"sign_p_{k}": a["components"][k]["sign_p"] for k in ("E", "S", "I")}}


def main():
    _root = Path(__file__).resolve().parents[1]
    _git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_root,
                          capture_output=True, text=True)
    r = {"generated_from": f"{D.relative_to(_root)}/*.json",
         "commit": _git.stdout.strip() if _git.returncode == 0 else "(not a git checkout)"}
    r["delta"] = {"en": delta("2026-09-18-cgsd-gate0b-en-b95162-calibrated.json"),
                  "jp": delta("2026-09-18-cgsd-gate0b-jp-b33188-calibrated.json"),
                  # Second gate family. Its own null: a model-confidence gate passes a different
                  # share of the pool, so it cannot borrow the verifier's band (v1.16).
                  "en_gateB": delta("2026-09-19-cgsd-gate0b-en-gateB075-b95162-calibrated.json")}
    r["pilot"] = {"en": pilot("2026-09-18-cgsd-pilot-en-analysis.json", 3914),
                  "jp": pilot("2026-09-18-cgsd-pilot-jp-analysis.json", 1465),
                  "en_items": pilot("2026-09-18-cgsd-pilot-en-items-analysis.json", 3914),
                  "en_gateB": pilot("2026-09-18-cgsd-pilot-en-gateB075-analysis.json", 3914)}
    r["ksens"] = {k: pilot(f"2026-09-18-cgsd-pilot-en-ksens-k{k}-b59900-analysis.json", 3914)
                  for k in (4, 8, 16)}
    r["positive_control"] = load("2026-09-18-cgsd-gate0f-positive-control.json")
    # paired K contrast — no new training, same replicates differenced (v1.11)
    # experiments/scripts in the working tree, scripts/ in the public release.
    _ks = next((_root / c / "ksens_paired.py" for c in ("experiments/scripts", "scripts")
                if (_root / c / "ksens_paired.py").is_file()), None)
    if _ks is None:
        sys.exit("⛔ ksens_paired.py 를 찾을 수 없다 — 빈 값을 저장하면 논문이 인용하는 "
                 "K 민감도 대비가 조용히 사라진다.")
    out = subprocess.run([sys.executable, str(_ks)], capture_output=True, text=True, cwd=_root)
    if out.returncode != 0 or not out.stdout.strip():
        # ⛔ Never store an empty field. This exact silence hid the missing paired contrast in
        # the public tree until the two trees were diffed (2026-09-19).
        sys.exit(f"⛔ ksens_paired.py 실패 (rc={out.returncode})\n{out.stderr[-800:]}")
    r["ksens_paired_raw"] = out.stdout
    OUT.write_text(json.dumps(r, indent=2))
    print(f"wrote {OUT}")
    for loc in ("en", "jp", "en_gateB"):
        d, p = r["delta"][loc], r["pilot"][loc]
        print(f"  {loc}: delta_E={d['delta_E_items']:.1f}건 sig={d['signature'][:8]} | "
              f"E={p['E']['mean_items']:+.1f} "
              f"[{p['E']['ci95_items'][0]:+.1f},{p['E']['ci95_items'][1]:+.1f}] "
              f"{'RESOLVED' if p['E']['resolved'] else 'unresolved'} "
              f"sign={p['E']['sign']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

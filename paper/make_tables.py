#!/usr/bin/env python3
"""Generate booktabs table bodies from the result JSONs — numbers are never retyped.

    python3 paper/make_tables.py
"""
import json, sys
from pathlib import Path

P = Path(__file__).resolve().parent


def ci(c):
    return f"${c['mean_items']:+.1f}$ & $[{c['ci95_items'][0]:+.1f},\\,{c['ci95_items'][1]:+.1f}]$"


def sign(c):
    a, b, z = c["sign"]
    return f"{a}/{b}/{z}" if z is not None else f"{a}/{b}/--"


def evidence(c, d, key):
    """Three states, named rather than starred (v1.17).

    ⛔ No stars. In an ML table a star reads as "statistically significant", and delta is not a
    level-alpha threshold — the paper says so explicitly, so the table must not imply otherwise.

      point + interval : the effect interval is separated from the null band's OWN interval,
                         CI_lower > delta_upper. Conservative: it charges the threshold's
                         estimation error against the effect rather than ignoring it.
      point only       : the pre-specified rule |theta| > delta passes, but the separation above
                         does not hold. Marginal, not a weaker kind of significance.
      --               : the point estimate does not clear delta. NOT a claim of zero.
    """
    if key == "delta_G":
        return "--"
    t = d[f"delta_{key}_items"]
    dhi = d[f"delta_{key}_ci_items"][1]
    lo, hi = c["ci95_items"]
    if lo * hi > 0 and min(abs(lo), abs(hi)) > dhi:
        return "point $+$ interval"
    return "point only" if c.get("resolved") else "--"


def t_verdict(r):
    L = [r"\begin{tabular}{llrrcr@{\,}ll}", r"\toprule",
         r"& & \multicolumn{1}{c}{mean} & \multicolumn{1}{c}{95\% CI} "
         r"& $n_+/n_-/n_0$ & \multicolumn{2}{c}{$\delta$ [CI]} & evidence \\", r"\midrule"]
    # ⛔ Gate B rides in its OWN null. Same locale, same budget, same K — but a different gate
    # passes a different share of the pool, so borrowing the verifier's band would compare a
    # run against a null that does not describe it.
    panels = (("en", "EN"), ("jp", "JP"), ("en_gateB", r"EN\textsubscript{B}"))
    for loc, nm in panels:
        p, d = r["pilot"][loc], r["delta"][loc]
        for j, k in enumerate(("E", "S", "I", "delta_G")):
            c = p[k]
            head = (rf"\multirow{{4}}{{*}}{{{nm}}}" if j == 0 else "")
            nmk = {"E": "$E$", "S": "$S$", "I": "$I$", "delta_G": r"$\Delta_G$"}[k]
            lo, hi = c["ci95_items"]
            if k == "delta_G":
                dv = r"\multicolumn{2}{c}{--}"
            else:
                t = d[f"delta_{k}_items"]
                dlo, dhi = d[f"delta_{k}_ci_items"]
                dv = f"{t:.1f} & [{dlo:.0f},\\,{dhi:.0f}]"
            L.append(f"{head} & {nmk} & ${c['mean_items']:+.1f}$ & "
                     f"$[{lo:+.1f},\\,{hi:+.1f}]$ & {sign(c)} & {dv} & "
                     f"{evidence(c, d, k)} \\\\")
        L.append(r"\bottomrule" if loc == panels[-1][0] else r"\midrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def tex_pct(x):
    """⛔ `%` is a LaTeX comment character — an unescaped one silently eats the rest of the line."""
    return f"{x * 100:.1f}\\%"


def tex_num(n):
    """Thousands separator that survives math mode (a bare comma adds list spacing)."""
    return f"{n:,}".replace(",", "{,}")


def t_math(m):
    L = [r"\begin{tabular}{lrrrl}", r"\toprule",
         r"support & rows & tokens & $e_c$ & status \\", r"\midrule"]
    for c, v in m["supports"].items():
        st = r"\textbf{extinct}" if v["extinct"] else "surviving"
        L.append(f"\\texttt{{{c.replace('_', chr(92) + '_')}}} & {v['rows']} & "
                 f"{v['tokens']} & {v['e']:.3f} & {st} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_math_caption(m):
    """The numbers that used to sit in wide footnote rows — the caption is the right home.

    ⛔ Emitted as a macro, not as raw text: hyperref expands a caption more than once, so an
    \\input inside \\caption dies with "Argument of \\Hy@tempa has an extra }".
    """
    g = m["G"]
    return ("\\newcommand{\\tabmathcaption}{" + f"Extinct supports hold {tex_pct(m['ext_token_mass'])} of tokens "
            f"({tex_pct(m['ext_row_mass'])} of rows). At the survivor-conditional interior "
            f"budget $B={tex_num(m['interior_B'])}$ tokens "
            f"($M_{{\\mathrm{{safe}}}}={m['M_safe']}$, tolerance "
            f"${m['tol'] * 100:.1f}\\%$), the survivor arms assemble with "
            f"$G={g['d10_A']['G']:+.0f}$ while $D_{{00}}^{{\\mathrm{{all}}}}$ has "
            f"$G={g['d00_all']['G']:+.0f}$." + "}")


def main():
    r = json.loads((P / "results.json").read_text())
    m = json.loads((P / "results-math.json").read_text())
    (P / "tab-verdict.tex").write_text(t_verdict(r) + "\n")
    (P / "tab-math.tex").write_text(t_math(m) + "\n")
    (P / "tab-math-caption.tex").write_text(t_math_caption(m) + "\n")
    print("wrote tab-verdict.tex, tab-math.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())

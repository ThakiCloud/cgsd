#!/usr/bin/env python3
"""Generate every figure from results.json / feasible_region.json — never from retyped numbers.

Invariants (arxiv-submitter figure contract):
  vector PDF · drawn at final column width · colour PLUS marker/linestyle · no in-figure title
  · two different quantities never share an axis · axis/tick/legend >= 7pt on the page.

    python3 paper/make_figures.py
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
COL = 3.3                                  # two-column \columnwidth in inches
# ⛔ No sub/superscripts in figure text. mathtext renders a subscript at ~0.7x, so a 7.5pt
# label carrying "$\\Delta_G$" puts a 5.2pt glyph on the page — under the 7pt floor, and
# unfixable by resizing because the ratio is scale-free. Words in the figure, symbols in the
# caption (the caption is where terms get defined anyway).
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,  # never Type 3
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
DARK, ACCENT = "#1a1a1a", "#B03A2E"        # measured vs threshold — 2 channels with linestyle


def fig_decomposition(r):
    """(a) EN (b) JP — component means with CI, against that locale's own delta.

    ⛔ Separate panels: the two locales have different holdout sizes (3,914 vs 1,465), so an
    item count on a shared axis would invite a comparison the estimand does not support.
    ⛔ Delta_G carries no delta and is drawn with its own marker — a null band is calibrated
    per component, and giving the total an "unresolved" marker would assert a test we never ran.
    """
    # ⛔ A dedicated legend row. Placing the legend in figure coordinates collided first with
    # the data and then with the x-axis label; an empty axes cannot collide with either.
    # ⛔ Three data panels, one legend row. The third is a DIFFERENT GATE on the same pool as
    # the first, so it gets its own panel rather than sharing EN's axis — its null band is its
    # own and the two are not meant to be read off one scale.
    fig, allax = plt.subplots(4, 1, figsize=(COL, 6.2),
                              gridspec_kw={"height_ratios": [1, 1, 1, 0.3]})
    axes, legax = allax[:3], allax[3]
    legax.axis("off")
    keys = ["E", "S", "I", "delta_G"]
    labels = ["E\nselection", "S\nredistrib.", "I\ninteraction", "total"]
    for ax, loc, tag in zip(axes, ("en", "jp", "en_gateB"),
                            ("(a) EN verifier", "(b) JP verifier", "(c) EN confidence gate")):
        p, d = r["pilot"][loc], r["delta"][loc]
        y = list(range(len(keys)))[::-1]
        for i, k in zip(y, keys):
            c = p[k]
            lo, hi = c["ci95_items"]
            ax.plot([lo, hi], [i, i], color=DARK, lw=1.1, solid_capstyle="butt")
            # ⛔ Three states, because two decision rules disagree (v1.16): the implemented
            # rule is |mean| > delta, the stricter reading is that the whole interval clears it.
            # JP E passes the first and fails the second; one marker would hide that.
            lo_, hi_ = c["ci95_items"]
            strict = (k != "delta_G" and c.get("resolved") and lo_ * hi_ > 0 and
                      min(abs(lo_), abs(hi_)) > d[f"delta_{k}_ci_items"][1])
            if k == "delta_G":
                mk, mfc = "D", "0.55"
            elif strict:
                mk, mfc = "o", DARK
            elif c.get("resolved") is True:
                mk, mfc = "o", "0.72"
            else:
                mk, mfc = "s", "white"
            ax.plot([c["mean_items"]], [i], marker=mk, ms=4.0, mfc=mfc, mec=DARK,
                    mew=0.9, ls="none")
        for k, i in zip(("E", "S", "I"), y[:3]):
            t = d[f"delta_{k}_items"]
            for x in (t, -t):
                ax.plot([x, x], [i - .32, i + .32], color=ACCENT, lw=1.0, ls=(0, (2.5, 1.5)))
        ax.axvline(0, color="0.6", lw=0.6, ls=":")
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_ylim(-0.7, len(keys) - 0.25)
        ax.set_xlabel("effect (holdout items)")
        ax.set_title(tag, loc="left", fontsize=7.5, pad=2.5)
        ax.spines[["top", "right"]].set_visible(False)
    h = [plt.Line2D([], [], color=DARK, marker="o", ms=4.0, ls="-", lw=1.1),
         plt.Line2D([], [], color=DARK, marker="o", ms=4.0, mfc="0.72", ls="-", lw=1.1),
         plt.Line2D([], [], color=DARK, marker="s", ms=4.0, mfc="white", ls="-", lw=1.1),
         plt.Line2D([], [], color="0.55", marker="D", ms=4.0, mec=DARK, ls="-", lw=1.1),
         plt.Line2D([], [], color=ACCENT, ls=(0, (2.5, 1.5)), lw=1.0)]
    legax.legend(h, ["interval past band", "mean past band only", "neither",
                     "total (no band)", "null band"],
                 loc="center", ncol=2, frameon=False, handlelength=1.8,
                 columnspacing=1.0, borderpad=0.0, labelspacing=0.3)
    fig.tight_layout(pad=0.3, h_pad=1.4)
    fig.savefig(P / "figures" / "decomposition.pdf")
    plt.close(fig)


def fig_region(reg):
    """F_Pi over B/T at three partitions.

    ⛔ M and G are different quantities and never share an axis — the y-axis is the partition
    and feasibility is shown as a span, with the caption naming which condition bound.
    """
    fig, (ax, legax) = plt.subplots(2, 1, figsize=(COL, 2.2),
                                    gridspec_kw={"height_ratios": [1, 0.16]})
    legax.axis("off")
    styles = {"4": ("-", "o"), "8": ((0, (4, 1.6)), "s"), "16": ((0, (1, 1.4)), "^")}
    for i, k in enumerate(("4", "8", "16")):
        rows = reg[k]["rows"]
        ok = [x["B_over_T"] for x in rows if x["ok"]]
        bad = [x["B_over_T"] for x in rows if not x["ok"]]
        ls, mk = styles[k]
        y = len(styles) - 1 - i
        if ok:
            ax.plot([min(ok), max(ok)], [y, y], color=DARK, lw=2.2, ls=ls,
                    solid_capstyle="butt")
        ax.plot(ok, [y] * len(ok), marker=mk, ms=3.4, ls="none", mfc=DARK, mec=DARK)
        ax.plot(bad, [y] * len(bad), marker="x", ms=4.0, ls="none", color=ACCENT, mew=1.0)
    ax.axvline(0.40, color="0.5", lw=0.8, ls=(0, (5, 1.5, 1, 1.5)))
    ax.annotate("primary\noperating point", xy=(0.395, -0.62), fontsize=7.2, ha="right",
                va="center", color="0.35", linespacing=1.1)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["K = 4", "K = 8", "K = 16"])
    ax.set_xlabel(r"relative training budget $B/T$")
    ax.set_ylim(-1.0, 2.5)
    ax.spines[["top", "right"]].set_visible(False)
    h = [plt.Line2D([], [], color=DARK, marker="o", ms=3.4, ls="-", lw=2.2),
         plt.Line2D([], [], color=ACCENT, marker="x", ms=4.0, ls="none", mew=1.0)]
    legax.legend(h, ["feasible", "infeasible"], loc="center", ncol=2,
                 frameon=False, handlelength=1.8, columnspacing=1.4, borderpad=0.0)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(P / "figures" / "feasible-region.pdf")
    plt.close(fig)


def main():
    r = json.loads((P / "results.json").read_text())
    reg = json.loads((P / "feasible_region.json").read_text())
    fig_decomposition(r)
    fig_region(reg)
    for f in sorted((P / "figures").glob("*.pdf")):
        print(f"  {f.name}  {f.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

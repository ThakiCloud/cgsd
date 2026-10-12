# CGSD — Counterfactual Gate–Support Decomposition

Code and recorded results for *Counterfactual Gate–Support Decomposition: Separating Selection from
Redistribution in Training-Data Filtering*.

A data quality gate does two things at once. It changes **which items inside a group survive**,
and it changes **how much mass each group gets**. Reporting one number for the gate averages
those together, and the two can point in opposite directions. CGSD separates them without ever
asserting what the gate's notion of "quality" is:

```
Δ_G = E + S + I

  E = Y10 − Y00     within-support selection   (gate eligibility, support mass held fixed)
  S = Y01 − Y00     between-support redistribution
  I = Y11 − Y10 − Y01 + Y00
```

`E` is **not** a quality effect. It is the effect of gate eligibility with support mass pinned —
whatever the gate's internal criterion happens to be. That distinction is the point of the
method: an audit that has to name the gate's latent variable cannot audit a gate it did not
build.

## What is actually here

| | |
|---|---|
| `cgsd/constructor/` | arm construction, the two feasibility tests, the extinction split |
| `tests/` | the gate harnesses that pin the invariants — **no data required** |
| `scripts/` | calibration, analysis, feasibility surveys |
| `paper/` | LaTeX source and the scripts that generate every figure and table |
| `results/` | the recorded result files those scripts read |

Every number in the paper is produced by running `paper/make_figures.py` and
`paper/make_tables.py` over `results/`. None of it is transcribed by hand, so the reported
values can be recomputed exactly. See [REPRODUCE.md](REPRODUCE.md).

## Start here: the gates run with no corpus

The structural claims do not depend on our data, and the harnesses that pin them are executable
on a laptop in seconds:

```bash
python3 tests/test_feasibility.py   # exact subset-sum assembly, positivity violations
python3 tests/test_extinction.py    # survivor split, the four failure modes, the drift bound
```

`test_extinction.py` is the one worth reading first. It pins that the four ways a construction
can fail carry **opposite prescriptions** — lowering the budget fixes one and cannot touch
another — which is the practical reason the two feasibility conditions are reported separately
rather than as a single "infeasible" verdict.

## Feasibility is a precondition, not a footnote

An arm can only be built if two things hold at the requested budget `B`:

```
mass         M(B) = min_c e_c − B/T        ≥ 0
assembly     G(B) = τB − Σ_c q_c(b_c)      ≥ 0
```

`M` asks whether every support has enough eligible mass; `G` asks whether the available item
sizes can actually *hit* the per-support quotas inside the budget tolerance `τ`. They fail for
different reasons and are fixed by different actions. When a support has `e_c = 0` the estimand
is undefined rather than merely infeasible, and the paper's extinction extension pulls that term
out of the decomposition instead of hiding it inside a component.

## What is not here

The source pools and evaluation splits are not ours to redistribute, so retraining the arms
requires equivalent data of your own. The method itself — estimand, identity, both feasibility
conditions, the extinction proposition and the drift bound — is fully specified in the paper and
can be reimplemented from the text alone.

## License

Code is Apache-2.0 ([LICENSE](LICENSE)). The paper is CC BY 4.0.

## Citation

```bibtex
@misc{han2026cgsd,
  author = {Hyojung Han and Jongmin Kim and Seung-Hun Jeon},
  title  = {Counterfactual Gate--Support Decomposition:
            Separating Selection from Redistribution in Training-Data Filtering},
  year   = {2026},
  eprint = {arXiv:PENDING},          % filled in on posting
  url    = {https://github.com/ThakiCloud/cgsd}
}
```

# Reproducing the paper

Two things are reproducible here, and they need different amounts of work. Be clear which one
you want before you start.

## 1. Every figure, table and number — no data, no GPU

The recorded result files ship with this repository. Everything the paper prints is computed
from them by script; nothing is typed into the LaTeX by hand. To rebuild it all:

```bash
cd paper
python3 collect_results.py     # results/*.json -> results.json
python3 make_figures.py        # -> figures/*.pdf   (vector)
python3 make_tables.py         # -> tab-*.tex       (booktabs bodies + caption macro)
tectonic main.tex
```

`collect_results.py` prints the two locked retrieval verdicts as it runs, which is the fastest
way to confirm your checkout reproduces ours:

```
en: delta_E=14.0건 sig=5511e26f | E=+29.7 [+26.8,+32.9] RESOLVED sign=[30, 0, 0]
jp: delta_E=5.0건  sig=85b27abd | E=+5.6  [+3.9,+7.3]   RESOLVED sign=[30, 0, 0]
```

`sig` is the calibration signature: a hash over locale, partition cardinality, budget, holdout
size, training seeds, gate identity and its realized pass rate, endpoint, and the constructor's
own source hash. A verdict is only attached when the null's signature matches the run's. If your
`sig` differs from the values above, something in that tuple differs — start there rather than
with the estimates.

`collect_math.py` is deliberately **not** in that list. The mathematics case study recomputes the
support structure from the corpus, so that one step needs the source pool; its output
(`paper/results-math.json`) ships, and the commands above read it. Run `collect_math.py` only if
you hold an equivalent pool — it overwrites the shipped file. Requirements:
Python 3.12 and `matplotlib` for the figures; `tectonic` or a TeX Live distribution for the PDF.

## 2. The structural claims — no data at all

These do not depend on our corpus and run in seconds:

```bash
python3 tests/test_feasibility.py
python3 tests/test_extinction.py
```

`test_feasibility.py` checks the exact bitset subset-sum against brute force, that the greedy
fallback is flagged rather than silently substituted, and that a positivity violation raises
instead of quietly returning a short arm.

`test_extinction.py` is the more interesting one. It pins that the survivor decomposition is
exact, that the structural boundary survives shrinking the budget to 0.1% of the pool (so
"just use a smaller budget" cannot rescue an extinct support), that the four failure modes carry
**opposite** prescriptions, and that the measured support drift stays inside the bound the paper
proves from the budget tolerance.

## 3. Retraining the arms — needs your own data

The source pools and evaluation splits are not ours to redistribute. `cgsd/constructor/` is the
complete construction layer, and `cgsd/manifests/` records the arm compositions we used, so the
protocol is fully specified — but the corpus is not included and the reported estimates cannot be
regenerated from scratch without equivalent data.

Point `build_arms.load_pool()` at your own pool, then:

```python
from cgsd.constructor import build_arms, feasibility, extinction

pool = build_arms.load_pool(...)
print(feasibility.mass_margin(pool, B))        # M(B) >= 0 ?
print(feasibility.granularity(...))            # G(B) >= 0 ?
A, Z, stats = extinction.survivor_split(pool)  # any support with e_c = 0 ?
```

Check feasibility before you spend a single GPU-hour. If `M < 0` the budget is too large for the
scarcest support; if `G < 0` at this budget the item sizes cannot hit the quotas — but `G` is a lattice quantity and
is **not monotone in B**, so scan the budget range rather than concluding the pool is unusable
(on our mathematics pool, 54 budgets satisfying the same margin do assemble while the
pre-registered operating point does not); if `Z` is non-empty the four-arm estimand is undefined at any budget and you want the
extinction extension instead.

#!/usr/bin/env python3
"""CGSD feasibility — two INDEPENDENT conditions (v1.14).

    M(B) >= 0   is there enough eligible mass?                     (v1.6 D40)
    G(B) >= 0   can that mass be ASSEMBLED from indivisible
                examples at the requested budget?                  (v1.14, here)

Math-1 passes the first and fails the second, which is why they must be separate objects.

Definitions. For support c the constructor targets b_c = B * p_c. Let A_c be the set of token
sums attainable from c's examples. The constructor never overshoots a per-support target, so

    q_c(b_c) = b_c - max{ a in A_c : a <= b_c }        atomic mismatch (>= 0)
    Q(B)     = sum_c q_c(B * p_c)
    G(B)     = tau * B - Q(B)                          tau = matching tolerance

G(B) < 0 means NO assembly of these examples lands inside tolerance — a property of the pool,
not of any algorithm. G(B) >= 0 means one EXISTS; a greedy packer may still miss it, and that
gap is an algorithm problem with a different fix. ⛔ Never report a packer's failure as a pool
property without computing G first.

`attainable_max` is an exact subset-sum over a bitset of reachable sums (Python big-int shifts,
so the inner loop runs at C speed). It falls back to the greedy packer only when the product
n * target would make that unreasonable, and SAYS SO — an approximate answer never masquerades
as the exact one.
"""
from build_arms import take   # noqa: F401  (greedy fallback = the constructor's own packer)

EXACT_LIMIT = 4_000_000_000   # n * target word-ops budget for the bitset DP


def attainable_max(tokens, target):
    """(best, exact) — largest attainable sum <= target, and whether it is provably optimal."""
    target = int(target)
    if target <= 0:
        return 0, True
    toks = [t for t in tokens if 0 < t <= target]
    if not toks:
        return 0, True
    if len(toks) * target <= EXACT_LIMIT:
        mask = (1 << (target + 1)) - 1
        reach = 1                                   # bit i set  <=>  sum i is attainable
        for t in toks:
            reach |= (reach << t) & mask
            if reach >> target & 1:                 # exact hit — cannot do better
                return target, True
        return reach.bit_length() - 1, True
    rows = [{"tokens": t, "example_id": i} for i, t in enumerate(toks)]
    _, used = take(rows, target, seed=0)
    return used, False


def attainable_subset(rows, target, seed):
    """Rows whose token sum is the largest attainable <= target, chosen at RANDOM among the
    subsets that achieve it.  (rows_out, total, exact)

    ⛔ `attainable_max` proves a sum exists; it does not hand you the subset. The greedy packer
    in build_arms can miss an assembly that G(B) >= 0 proves exists, and that gap was reported as
    a structural verdict: the mathematics pool failed with `PACKER: G(B)=+1 >= 0, so an
    in-tolerance assembly EXISTS` at 54 budgets that satisfy the pre-registered margin, while the
    paper concluded the design was "not realizable" (2026-09-19).

    ⛔ The backtrack is RANDOMIZED on purpose. Taking the lexicographically-first witness would
    select rows by token length, and length is exactly the confound the arm construction matches
    on — a deterministic exact packer would buy budget precision by silently biasing composition.
    Randomizing among witnesses keeps selection-within-support arbitrary, which is what D00/D01
    mean by "unrestricted".
    """
    import random as _r
    target = int(target)
    usable = [r for r in rows if 0 < r["tokens"] <= target]
    if target <= 0 or not usable:
        return [], 0, True
    if len(usable) * target > EXACT_LIMIT:          # too big to prove — say so, do not pretend
        sel, used = take(usable, target, seed)
        return sel, used, False

    mask = (1 << (target + 1)) - 1
    reach = [1]                                     # reach[i] = sums attainable from rows[:i]
    for r in usable:
        prev = reach[-1]
        reach.append(prev | ((prev << r["tokens"]) & mask))
    best = reach[-1].bit_length() - 1

    rng = _r.Random(seed)
    out, need = [], best
    for i in range(len(usable), 0, -1):             # walk items backwards, choose randomly
        if need == 0:
            break
        t = usable[i - 1]["tokens"]
        can_skip = bool(reach[i - 1] >> need & 1)
        can_take = need >= t and bool(reach[i - 1] >> (need - t) & 1)
        if can_take and (not can_skip or rng.random() < 0.5):
            out.append(usable[i - 1]); need -= t
        elif not can_skip:                          # unreachable — the DP is wrong, do not guess
            raise AssertionError(f"subset-sum backtrack lost the witness at need={need}")
    assert sum(r["tokens"] for r in out) == best
    rng.shuffle(out)
    return out, best, True


def pack(rows, budget, seed, tol):
    """Greedy first; fall back to the exact assembler ONLY when greedy misses the tolerance.

    Keeping greedy as the default matters: it is what every recorded calibration was built with,
    and its sha rides in the calibration signature through build_arms. The exact path changes the
    outcome only where the greedy one was already failing.
    """
    sel, used = take(rows, budget, seed)
    if budget - used <= tol * budget:
        return sel, used, "greedy"
    ex, tot, exact = attainable_subset(rows, budget, seed)
    if budget - tot <= tol * budget and tot > used:
        return ex, tot, "exact" if exact else "exact-approx"
    return (ex, tot, "exact-short") if tot > used else (sel, used, "greedy-short")


def atomic_mismatch(pool_rows, target):
    """q_c(target) for one support, plus the exactness flag."""
    best, exact = attainable_max([r["tokens"] for r in pool_rows], target)
    return target - best, exact


def granularity(shares, sources, B, tol):
    """G(B) for ONE arm.

    shares  {c: p_c}    sources {c: [rows]}    tol  matching tolerance (e.g. 0.005)
    Returns (G, Q, {c: q_c}, all_exact).
    """
    q, exact_all = {}, True
    for c, p in shares.items():
        qc, ex = atomic_mismatch(sources[c], B * p)
        q[c] = qc
        exact_all &= ex
    Q = sum(q.values())
    return tol * B - Q, Q, q, exact_all


# ─── feasible region — joint in exposure and support representation (v1.15 D4) ────────────
# ⛔ Write it as a function of the PARTITION, not of K. Two partitions with the same K can have
# very different granularity, so G(K) is not well defined; G(B, Pi; tau) is.
#
#     F_Pi = { B > 0 : M(B, Pi) >= 0  and  G(B, Pi; tau) >= 0 }
#
# M < 0  there is not enough eligible mass.
# G < 0  the mass is there but indivisible examples cannot assemble the budget.
# With extinction, F_Pi can be EMPTY for the original partition at every positive B; that is when
# the survivor partition Pi_A and the X_B extension take over (v1.13).

def relabel(pool, labels):
    """Return a view of `pool` under a different partition Pi. `labels` maps example_id -> support."""
    return [dict(r, support=labels[r["example_id"]]) for r in pool]


def mass_margin(pool, B):
    """M(B, Pi) = min_c e_c - B/T, computed on whatever partition `pool` currently carries."""
    tot, elig = {}, {}
    for r in pool:
        tot[r["support"]] = tot.get(r["support"], 0) + r["tokens"]
        if r["passed"]:
            elig[r["support"]] = elig.get(r["support"], 0) + r["tokens"]
    T = sum(tot.values())
    return min(elig.get(c, 0) / tot[c] - B / T for c in tot)


def four_arm_shares(pool):
    """(shares, sources) for the four decomposition arms under the pool's current partition."""
    passed = [r for r in pool if r["passed"]]
    T = sum(r["tokens"] for r in pool)
    PT = sum(r["tokens"] for r in passed)
    sup = sorted({r["support"] for r in pool})
    src = {c: [r for r in pool if r["support"] == c] for c in sup}
    gate = {c: [r for r in passed if r["support"] == c] for c in sup}
    p_src = {c: sum(r["tokens"] for r in src[c]) / T for c in sup}
    p_gate = {c: (sum(r["tokens"] for r in gate[c]) / PT if PT else 0.0) for c in sup}
    return (("d00", p_src, src), ("d10", p_src, gate),
            ("d01", p_gate, src), ("d11", p_gate, gate))


def feasible_region(pool, B_grid, tol):
    """[(B, M, min_arm_G, feasible)] over B_grid for the pool's current partition."""
    plan = four_arm_shares(pool)
    out = []
    for B in B_grid:
        M = mass_margin(pool, B)
        gs = [granularity(sh, srcs, B, tol)[0] for _, sh, srcs in plan]
        out.append((B, M, min(gs), M >= 0 and min(gs) >= 0))
    return out


def assert_builder_partition(pool, K):
    """Raise unless `pool` carries the integer supports 0..K-1 that build_arms.build assumes.

    ⛔ `build_arms.build` groups rows with `for c in range(K)`, because the ESCI loader assigns
    KMeans labels. Hand it a pool whose supports are strings — every ADF tag pool is — and every
    lookup misses silently: it reports `rho=inf` and `filled 0/B tokens`, which reads exactly like
    a positivity failure. It cost us a wrong verdict (2026-09-19: `live-clean` was recorded as
    mass-infeasible when it is in fact constructible at 38 budgets).

    build_arms.py cannot carry this check itself — its source hash is an input to
    calibration_signature, so editing it invalidates every recorded calibration. The guard lives
    here and callers use it before reaching for the four-arm builder.
    """
    bad = sorted({repr(r["support"]) for r in pool
                  if not isinstance(r["support"], int) or not 0 <= r["support"] < K})
    if bad:
        raise TypeError(
            f"build_arms.build assumes integer supports 0..{K - 1}; this pool has "
            f"{len(bad)} other label(s): {', '.join(bad[:5])}"
            f"{' …' if len(bad) > 5 else ''}. It would return empty groups and report rho=inf, "
            f"which looks like a positivity failure but is a type mismatch. Pools with named "
            f"supports go through extinction.build_extinction instead.")

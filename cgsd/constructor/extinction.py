#!/usr/bin/env python3
"""CGSD support-extinction extension — survivor-conditional arms (v1.13 D83-D87).

⛔ This lives in its OWN module on purpose. `calibration_signature()` hashes
`build_arms.py`, so appending here would have changed that hash and silently
invalidated every delta calibration in flight (caught 2026-09-19: the sha moved
9a34faa6d04f -> ff2c9d3a67df while two 200-rep placebo runs were mid-flight).
The 4-arm constructor is frozen; extinction is a new object built on top of it.

Runs that use these arms carry `extinction_sha` in their own signature.
"""
import hashlib
from pathlib import Path

from build_arms import BUDGET_TOL, InfeasibleBudget, take   # noqa: F401
from feasibility import granularity, pack


def extinction_sha():
    """Provenance for runs built with this module (paired with constructor_sha)."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# A support with e_c = 0 has NO eligible tokens at all, so the source-support-preserving
# gate-eligible arm (D10) is undefined there at every positive budget — this is a different
# object from the budget-reducible frontier M < 0, which a smaller B always fixes.
# ⛔ Masses are TOKEN-weighted. Rows are reported alongside for description only; the
# constructor, the budget split and the feasibility bound all run in tokens (v1.2 D13).

class SupportExtinction(RuntimeError):
    """Some source support has positive mass but zero gate eligibility (v1.13 D83)."""


def _support_seed(seed, c):
    """Deterministic per-support seed that also works for STRING support ids.

    ESCI supports are ints from KMeans, but ADF supports are tag NAMES — `seed * 1000 + c`
    raised TypeError on the real Math pool. ⛔ Never use built-in hash() here: it is salted
    per process for str, so the same arm would differ between runs.
    """
    if isinstance(c, int):
        return seed * 1000 + c
    h = int(hashlib.sha256(str(c).encode()).hexdigest()[:8], 16)
    return seed * 1000 + (h % 1000)


def survivor_split(pool):
    """Partition supports into A = {c : e_c > 0} and Z = {c : e_c = 0}.

    Returns (A, Z, stats) with A/Z sorted, and stats carrying BOTH masses:
      ext_tok  = sum_{c in Z} tok_c / T        <- the one the constructor uses
      ext_row  = sum_{c in Z} n_c   / N        <- descriptive only
    """
    tot, elig, n, nz = {}, {}, {}, {}
    for r in pool:
        c = r["support"]
        tot[c] = tot.get(c, 0) + r["tokens"]
        n[c] = n.get(c, 0) + 1
        if r["passed"]:
            elig[c] = elig.get(c, 0) + r["tokens"]
            nz[c] = nz.get(c, 0) + 1
    e = {c: elig.get(c, 0) / tot[c] for c in tot}
    A = sorted(c for c in tot if e[c] > 0)
    Z = sorted(c for c in tot if e[c] == 0)
    T, N = sum(tot.values()), sum(n.values())
    return A, Z, {"e": e, "tok": tot, "rows": n, "T": T, "N": N, "T_A": sum(tot[c] for c in A),
                  "ext_tok": sum(tot[c] for c in Z) / T if T else 0.0,
                  "ext_row": sum(n[c] for c in Z) / N if N else 0.0,
                  "min_e_A": min((e[c] for c in A), default=0.0)}


def survivor_margin(pool, B):
    """Survivor-conditional feasibility margin  M_A = min_{c in A} e_c - B/T_A  (v1.13 D85).

    D10^A holds the survivors' ORIGINAL relative shares p_c^A = tok_c / T_A, so it needs
    B * tok_c / T_A <= e_c * tok_c in every surviving support, i.e. B/T_A <= min_{c in A} e_c.
    Identical in form to feasibility_margin(), with T replaced by the survivor mass T_A —
    so the interior rule M_safe carries over unchanged and no new criterion is invented.
    """
    A, Z, st = survivor_split(pool)
    if not A:
        raise SupportExtinction("every support is extinct — no survivor-conditional arm exists")
    return st["min_e_A"] - B / st["T_A"], st


def build_extinction(pool, B, seed, strict=True):
    """Arms for the extinction-extended identity (v1.13 D84).

        Delta_G = X_B + E_A + S_A + I_A
        X_B = Y(D00^A; B) - Y(D00^all; B)      support-extinction contrast (SIGNED)
        E_A = Y(D10^A)  - Y(D00^A)
        S_A = Y(D01^A)  - Y(D00^A)
        I_A = Y(D11)    - Y(D10^A) - Y(D01^A) + Y(D00^A)

    The identity is exact by construction: the four terms telescope to Y(D11) - Y(D00^all).
    ⛔ There is no D11^A. The real gate arm already puts zero mass on extinct supports, so
    restricting it to A would be the same set of rows under a different name — and naming it
    ^A would falsely suggest a second, separate counterfactual.

    Both D00 arms are drawn at the SAME budget B from the SAME unrestricted pool; they differ
    only in which supports may contribute and how the freed mass is redistributed (survivors
    keep their original relative proportions). So X is not "the value of the tag" — it carries
    the replacement policy with it. If a cost is wanted, cost = -X.
    """
    A, Z, st = survivor_split(pool)
    T, T_A = st["T"], st["T_A"]
    passed = [r for r in pool if r["passed"]]
    PT = sum(r["tokens"] for r in passed)
    sup = sorted(st["tok"])
    src = {c: [r for r in pool if r["support"] == c] for c in sup}
    gate = {c: [r for r in passed if r["support"] == c] for c in sup}

    p_all = {c: st["tok"][c] / T for c in sup}                    # source, full support
    p_A = {c: st["tok"][c] / T_A for c in A}                      # source, survivors renormalized
    p_G = {c: sum(r["tokens"] for r in gate[c]) / PT for c in sup}  # gate-induced (0 on Z)

    plan = (("d00_all", p_all, src, sup), ("d00_A", p_A, src, A),
            ("d10_A", p_A, gate, A), ("d01_A", p_G, src, A), ("d11", p_G, gate, sup))
    arms, diag = {}, {}
    for name, share, source, cs in plan:
        rows, rho, packer = [], {}, {}
        for c in cs:
            bc = B * share[c]
            # ⛔ Greedy first, exact only on a miss (feasibility.pack). Before this, a support
            # whose items could not be greedily prefixed into bc reported the arm as infeasible
            # even when G(B) >= 0 proved an in-tolerance assembly existed — a solver limit read
            # as a structural verdict (2026-09-19).
            sel, _, how = pack(source[c], bc, _support_seed(seed, c), BUDGET_TOL)
            packer[c] = how
            rows += sel
            avail = sum(r["tokens"] for r in source[c])
            rho[c] = (bc / avail) if avail else float("inf")
        arms[name], diag[name] = rows, rho
        if any(h != "greedy" for h in packer.values()):
            diag.setdefault("_packer", {})[name] = \
                {c: h for c, h in packer.items() if h != "greedy"}
        if strict:
            used = sum(r["tokens"] for r in rows)
            if used < B * (1 - BUDGET_TOL):
                bad = sorted((c, v) for c, v in rho.items() if v > 1)
                head = (f"{name}: filled {used:,}/{B:,} tokens ({used / B - 1:+.2%}, tol "
                        f"{BUDGET_TOL:.1%}) — ")
                # ⛔ Two different boundaries end up here and the fix is OPPOSITE (v1.13 D94).
                # Frontier: some support cannot supply its share -> LOWER B.
                # Granularity: nobody is over-subscribed, the rows are simply coarser than the
                # per-support budgets -> lowering B makes it WORSE; the pool must grow.
                if bad:
                    raise InfeasibleBudget(
                        head + f"FRONTIER: positivity violated in {len(bad)} support(s): "
                        + ", ".join(f"{c} rho={v:.2f}" for c, v in bad[:6])
                        + f". Survivor margin M_A={survivor_margin(pool, B)[0]:+.3f}"
                        + f" (|A|={len(A)}, |Z|={len(Z)}, extinct token mass"
                        f" {st['ext_tok']:.1%}). Lower B.")
                # ⛔ Two more cases hide behind "0 violations" and their fixes differ again
                # (v1.14 D97). G(B) < 0 means no assembly of these examples lands in tolerance —
                # a property of the POOL. G(B) >= 0 means one exists and this packer missed it —
                # a property of the ALGORITHM. Never report the second as the first.
                G, Q, q, exact = granularity(share, source, B, BUDGET_TOL)
                worst = ", ".join(f"{c}={v:.0f}" for c, v in
                                  sorted(q.items(), key=lambda kv: -kv[1])[:4])
                approx = "" if exact else " (approximate — subset-sum fell back to the packer)"
                if G < 0:
                    raise InfeasibleBudget(
                        head + f"GRANULARITY: no support is over-subscribed (0 violations) and "
                        f"G(B) = tol*B - Q = {BUDGET_TOL * B:.1f} - {Q:.0f} = {G:+.0f} < 0{approx}"
                        + (" — NO assembly of these examples lands in tolerance"
                           if exact else
                           " — the assembly bound is APPROXIMATE here (subset-sum fell back to a "
                           "greedy bound), so a negative G does NOT prove impossibility")
                        + f". Worst q_c: {worst}."
                        f" ⛔ G is non-monotone in B — SEARCH the admissible budgets before"
                        f" changing anything. On the mathematics pool 54 budgets under the same"
                        f" margin assemble while the pre-registered operating point does not."
                        f" Grow the pool or coarsen the supports only when the joint feasible"
                        f" set is empty.")
                raise InfeasibleBudget(
                    head + f"PACKER: G(B) = {G:+.0f} >= 0, so an in-tolerance assembly EXISTS but "
                    f"the greedy packer did not find it{approx}. Worst q_c: {worst}. "
                    f"⛔ This is an algorithm gap, not a pool property — do not request more data.")
    return arms, diag, {"A": A, "Z": Z, "T": T, "T_A": T_A, "pass_tokens": PT,
                        "p_all": p_all, "p_A": p_A, "p_gate": p_G,
                        "margin_A": survivor_margin(pool, B)[0],
                        "ext_tok": st["ext_tok"], "ext_row": st["ext_row"], "e": st["e"]}


def decompose_extinction(Y):
    """Map arm outcomes to the four signed terms + the identity residual (v1.13 D84)."""
    X = Y["d00_A"] - Y["d00_all"]
    E = Y["d10_A"] - Y["d00_A"]
    S = Y["d01_A"] - Y["d00_A"]
    I = Y["d11"] - Y["d10_A"] - Y["d01_A"] + Y["d00_A"]
    dG = Y["d11"] - Y["d00_all"]
    return {"X": X, "E_A": E, "S_A": S, "I_A": I, "delta_G": dG,
            "residual": dG - (X + E + S + I)}


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=tuple(DS))
    ap.add_argument("--budget", type=int, required=True, help="B_l in TOKENS (v1.3 appendix B)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed (not the training seed)")
    ap.add_argument("--k", type=int, default=K,
                    help="support resolution; primary is 8, 16 only for the matched-low-budget robustness pair")
    ap.add_argument("--tag", default="", help="manifest subdir suffix (e.g. robust_low)")
    ap.add_argument("--out", default="cgsd/manifests")
    a = ap.parse_args()
    K = a.k                      # resolution is set once, before any pool is built

    pool, hq = load_pool(a.locale)
    arms, rho, meta = build(pool, a.budget, a.seed)
    d = Path(a.out) / a.locale / (f"k{a.k}" + (f"_{a.tag}" if a.tag else "")) / f"seed_{a.seed}"
    d.mkdir(parents=True, exist_ok=True)

    stats = {"locale": a.locale, "K": K, "selection_seed": a.seed,
             "budget_tokens": a.budget, "cluster_seed": CLUSTER_SEED,
             "tokenizer": str(TOKJSON), "T_tokens": meta["T"],
             "pass_tokens": meta["pass_tokens"], "n_pool": len(pool),
             "n_pass": sum(1 for r in pool if r["passed"]), "arms": {}}
    for name in ARMS:
        rows = arms[name]
        used = sum(r["tokens"] for r in rows)
        hist = {c: sum(r["tokens"] for r in rows if r["support"] == c) for c in range(K)}
        lens = np.array([r["tok_q"] for r in rows])
        e = {"n_rows": len(rows), "tokens": used,
             "budget_err": (used - a.budget) / a.budget,
             "pass_frac_rows": sum(1 for r in rows if r["passed"]) / max(len(rows), 1),
             "support_share": {c: hist[c] / used for c in range(K)},
             "query_len_mean": float(lens.mean()), "query_len_p90": float(np.percentile(lens, 90))}
        if name in rho:
            v = [x for x in rho[name].values() if np.isfinite(x)]
            e |= {"rho_median": float(np.median(v)), "rho_p95": float(np.percentile(v, 95)),
                  "rho_max": float(max(v))}
        stats["arms"][name] = e
        p = d / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        e["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        print(f"  {name:<8} rows={len(rows):>6} tok={used:>9,} "
              f"err={e['budget_err']:+.4%} pass={e['pass_frac_rows']:.3f}", flush=True)
    (d / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {d}", flush=True)


if __name__ == "__main__":
    main()

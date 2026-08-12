#!/usr/bin/env python3
"""Lever 2 — BINDING-ENERGY-weighted KL-triage readout (converts the payoff to experimental kcal/mol).

kl_triage.py measures capture@k of hotspot COUNT. Reviewers ask "your readout isn't binding." This
re-expresses the same triage in EXPERIMENTAL binding free energy: among the interface residues with an
Ala-scan measurement, how much of the complex's total binding energy (Σ max(ΔΔG_bind, 0), kcal/mol) does
a fixed budget of k positions capture when ranked by KL+burial vs by burial alone?

Design-time triage framing: a fixed budget of k measured interface positions per complex for expensive
binding-aware optimisation. Energy-capture@k(ranker) = Σ_{top-k by ranker} w_i / Σ_i w_i, w_i = max(ΔΔG,0).
Complex-level bootstrap (seed 20260803). Uses committed CSVs only.

  python3 src/kl_triage_energy.py --out results/kl_triage_energy.csv
"""
import argparse
import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 10000


def zscore(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def energy_capture(sub, score, k, wcol="w"):
    """Fraction of this complex's total binding energy in the top-k positions by `score`."""
    W = sub[wcol].values
    tot = W.sum()
    if tot <= 0:
        return None
    order = np.argsort(-sub[score].values, kind="mergesort")
    return float(W[order[:k]].sum() / tot)


def abs_energy(sub, score, k, wcol="w"):
    order = np.argsort(-sub[score].values, kind="mergesort")
    return float(sub[wcol].values[order[:k]].sum())


def cluster_boot(fn, cids, seed=SEED, nboot=NBOOT):
    rng = np.random.default_rng(seed)
    vals = {c: fn(c) for c in cids}
    keep = [c for c in cids if vals[c] is not None and np.isfinite(vals[c])]
    obs = float(np.mean([vals[c] for c in keep]))
    boots = [np.mean([vals[c] for c in rng.choice(keep, len(keep), True)]) for _ in range(nboot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi), float(np.mean(np.array(boots) > 0)), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default="results/kl_detector_joined.csv")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--out", default="results/kl_triage_energy.csv")
    ap.add_argument("--ddg", default="ddG", choices=["ddG", "ddG_max"])
    a = ap.parse_args()

    d = pd.read_csv(a.joined)
    d = d[d.is_interface == 1].copy()
    d["icode"] = d["icode"].fillna("").astype(str)
    pos = pd.read_csv(a.positions, usecols=["complex_id", "chain", "resnum", "icode",
                                            "ddG", "ddG_max", "has_meas", "n_meas"])
    pos["icode"] = pos["icode"].fillna("").astype(str)
    d = d.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left")

    # keep only interface positions WITH an Ala-scan measurement (isolates ranking from sparsity)
    d = d[(d.has_meas == True) & np.isfinite(d[a.ddg])].copy()
    d["w"] = np.clip(d[a.ddg].values, 0, None)          # binding energy contributed, kcal/mol
    d["kl_z"] = d.groupby("complex_id")["kl"].transform(zscore)
    d["bur_z"] = d.groupby("complex_id")["nbr"].transform(zscore)   # burial = neighbour count
    d["klbur"] = d["kl_z"] + d["bur_z"]

    by = {c: s.reset_index(drop=True) for c, s in d.groupby("complex_id")}
    # complexes with >=3 measured interface positions and positive total binding energy
    cids = [c for c in by if len(by[c]) >= 3 and by[c]["w"].sum() > 0]
    print(f"complexes with >=3 measured interface positions & energy>0: {len(cids)}")
    print(f"total measured interface positions: {sum(len(by[c]) for c in cids)}; "
          f"mean Σ|ΔΔG>0| per complex = {np.mean([by[c]['w'].sum() for c in cids]):.2f} kcal/mol")

    def budget_k(sub, mode):
        return 3 if mode == "k3" else max(1, int(np.ceil(0.25 * len(sub))))

    rows = []
    for mode in ["k3", "k25pct"]:
        # per-ranker energy capture (fraction) and absolute kcal
        for score in ["nbr", "kl", "klbur"]:
            obs, lo, hi, _, n = cluster_boot(
                lambda c, s=score, m=mode: energy_capture(by[c], s, budget_k(by[c], m)), cids)
            akc, _, _, _, _ = cluster_boot(
                lambda c, s=score, m=mode: abs_energy(by[c], s, budget_k(by[c], m)), cids)
            rows.append(dict(metric=f"energy_capture@{mode}", ranker=score, estimate=obs, lo=lo, hi=hi,
                             p_gt0=np.nan, abs_kcal=akc, n_cx=n))
        # random-selection expectation = k/n (fraction of energy on average)
        obs, lo, hi, _, n = cluster_boot(lambda c, m=mode: budget_k(by[c], m) / len(by[c]), cids)
        rows.append(dict(metric=f"energy_capture@{mode}", ranker="random_expected", estimate=obs, lo=lo,
                         hi=hi, p_gt0=np.nan, abs_kcal=np.nan, n_cx=n))
        # paired deltas vs burial
        for a_, b_ in [("kl", "nbr"), ("klbur", "nbr")]:
            obs, lo, hi, p, n = cluster_boot(
                lambda c, aa=a_, bb=b_, m=mode: (energy_capture(by[c], aa, budget_k(by[c], m)) -
                                                 energy_capture(by[c], bb, budget_k(by[c], m))), cids)
            akc, alo, ahi, ap, _ = cluster_boot(
                lambda c, aa=a_, bb=b_, m=mode: (abs_energy(by[c], aa, budget_k(by[c], m)) -
                                                 abs_energy(by[c], bb, budget_k(by[c], m))), cids)
            rows.append(dict(metric=f"DELTA_energy_capture@{mode}", ranker=f"{a_}_minus_{b_}", estimate=obs,
                             lo=lo, hi=hi, p_gt0=p, abs_kcal=akc, abs_kcal_lo=alo, abs_kcal_hi=ahi, n_cx=n))

    out = pd.DataFrame(rows)
    out["seed"] = SEED; out["nboot"] = NBOOT; out["ddg_col"] = a.ddg
    out["command"] = f"python3 src/kl_triage_energy.py --ddg {a.ddg} --out {a.out}"
    out.to_csv(a.out, index=False)

    def g(metric, ranker):
        r = out[(out.metric == metric) & (out.ranker == ranker)]
        return r.iloc[0] if len(r) else None
    for m in ["k3", "k25pct"]:
        rnd = g(f"energy_capture@{m}", "random_expected")
        b = g(f"energy_capture@{m}", "nbr"); k = g(f"energy_capture@{m}", "kl"); kb = g(f"energy_capture@{m}", "klbur")
        print(f"\n[energy_capture@{m}] random={rnd.estimate:.3f}  burial={b.estimate:.3f} ({b.abs_kcal:.1f} kcal)  "
              f"kl={k.estimate:.3f}  kl+burial={kb.estimate:.3f} ({kb.abs_kcal:.1f} kcal)")
        for rk in ["kl_minus_nbr", "klbur_minus_nbr"]:
            dd = g(f"DELTA_energy_capture@{m}", rk)
            star = "*" if (dd.lo > 0 or dd.hi < 0) else " "
            print(f"    Δ {rk:16} {dd.estimate:+.3f} [{dd.lo:+.3f},{dd.hi:+.3f}] P(>0)={dd.p_gt0:.3f}  "
                  f"(+{dd.abs_kcal:+.2f} kcal [{dd.abs_kcal_lo:+.2f},{dd.abs_kcal_hi:+.2f}]) {star}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

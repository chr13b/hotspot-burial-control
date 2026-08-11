#!/usr/bin/env python3
"""
KL-as-triage prototype (EXPLORATORY method probe, not pre-registered as a falsifier).

Question: can the sequence-free KL signal be turned from a DIAGNOSTIC into an actionable design-time
METHOD? Frame it as triage: a designer with a fixed budget to apply expensive binding-aware effort at
k interface positions per complex. Does ranking those positions by KL (or KL+burial) capture more
experimental hotspots than the naive burial ranking?

This is the honest "is there a method here" test. The prior is weak: kl_analysis found KL's top-k
PRECISION not significant over burial (it adds in AUROC +0.048 but the very top is a wash). Capture@k
with the two signals COMBINED, plus the on-narrative niche (positions the model is least confident
about, where log-prob design would fail), are the two places a gain could still live.

KILL (pre-specified before running): if Delta-capture(KL+burial - burial) CI contains zero at BOTH
budgets AND the model-uncertain niche test is null, the triage method does NOT beat burial -> KL stays
a diagnostic, drop the method claim. Reported either way.

Data: results/kl_detector_joined.csv (per interface position: kl, nbr/rsasa burial, logp_native,
is_hot from Ala-scan ddG). Complex-level bootstrap (complexes are the unit), seed 20260803.

  python3 src/kl_triage.py --joined results/kl_detector_joined.csv --out results/kl_triage.csv
"""
import argparse
import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 10000


def zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def capture_at_k(sub, score, k):
    """fraction of this complex's hotspots ranked in the top-k by `score` (descending)."""
    nh = int(sub.is_hot.sum())
    if nh == 0:
        return None
    order = np.argsort(-sub[score].values, kind="mergesort")  # stable
    topk = set(order[:k])
    hits = sum(1 for i, ishot in enumerate(sub.is_hot.values) if ishot == 1 and i in topk)
    return hits / nh


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based AUROC
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().values
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cluster_boot(per_cx_fn, cids, seed=SEED, nboot=NBOOT):
    """per_cx_fn(cid)->scalar or None; returns obs mean, CI, P(>0) over complex resamples."""
    rng = np.random.default_rng(seed)
    vals = {c: per_cx_fn(c) for c in cids}
    keep = [c for c in cids if vals[c] is not None and np.isfinite(vals[c])]
    obs = float(np.mean([vals[c] for c in keep]))
    boots = []
    for _ in range(nboot):
        samp = rng.choice(keep, len(keep), True)
        boots.append(np.mean([vals[c] for c in samp]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi), float(np.mean(np.array(boots) > 0)), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default="results/kl_detector_joined.csv")
    ap.add_argument("--out", default="results/kl_triage.csv")
    a = ap.parse_args()

    d = pd.read_csv(a.joined)
    d = d[d.is_interface == 1].copy() if "is_interface" in d else d
    d["burial_score"] = d["nbr"]  # more neighbours = more buried = higher rank
    # per-complex combined score (z-summed within complex)
    d["kl_z"] = d.groupby("complex_id")["kl"].transform(zscore)
    d["bur_z"] = d.groupby("complex_id")["burial_score"].transform(zscore)
    d["klbur"] = d["kl_z"] + d["bur_z"]

    by = {c: s.reset_index(drop=True) for c, s in d.groupby("complex_id")}
    cids = [c for c in by if by[c].is_hot.sum() >= 1]
    rows = []

    def budget_k(sub, mode):
        return 3 if mode == "k3" else max(1, int(np.ceil(0.25 * len(sub))))

    # ---- Q1: capture@k, each ranker, + paired deltas vs burial ----
    for mode in ["k3", "k25pct"]:
        for score in ["burial_score", "kl", "klbur"]:
            obs, lo, hi, _, n = cluster_boot(
                lambda c, s=score, m=mode: capture_at_k(by[c], s, budget_k(by[c], m)), cids)
            rows.append(dict(metric=f"capture@{mode}", ranker=score, estimate=obs, lo=lo, hi=hi,
                             p_gt0=np.nan, n_cx=n))
        # random baseline (analytic expectation k/n per complex)
        obs, lo, hi, _, n = cluster_boot(
            lambda c, m=mode: budget_k(by[c], m) / len(by[c]), cids)
        rows.append(dict(metric=f"capture@{mode}", ranker="random_expected", estimate=obs, lo=lo, hi=hi,
                         p_gt0=np.nan, n_cx=n))
        # paired deltas
        for a_, b_ in [("kl", "burial_score"), ("klbur", "burial_score")]:
            obs, lo, hi, p, n = cluster_boot(
                lambda c, aa=a_, bb=b_, m=mode: (capture_at_k(by[c], aa, budget_k(by[c], m)) -
                                                 capture_at_k(by[c], bb, budget_k(by[c], m)))
                if capture_at_k(by[c], aa, budget_k(by[c], m)) is not None else None, cids)
            rows.append(dict(metric=f"DELTA_capture@{mode}", ranker=f"{a_}_minus_{b_}", estimate=obs,
                             lo=lo, hi=hi, p_gt0=p, n_cx=n))

    # ---- Q2: AUROC(is_hot) per complex, kl vs burial vs klbur (sanity vs kl_analysis) ----
    for score in ["burial_score", "kl", "klbur"]:
        obs, lo, hi, _, n = cluster_boot(lambda c, s=score: auroc(by[c].is_hot, by[c][s]), cids)
        rows.append(dict(metric="AUROC_is_hot", ranker=score, estimate=obs, lo=lo, hi=hi, p_gt0=np.nan, n_cx=n))
    for a_, b_ in [("kl", "burial_score"), ("klbur", "burial_score")]:
        obs, lo, hi, p, n = cluster_boot(
            lambda c, aa=a_, bb=b_: (auroc(by[c].is_hot, by[c][aa]) - auroc(by[c].is_hot, by[c][bb]))
            if auroc(by[c].is_hot, by[c][aa]) is not None else None, cids)
        rows.append(dict(metric="DELTA_AUROC_is_hot", ranker=f"{a_}_minus_{b_}", estimate=obs, lo=lo, hi=hi,
                         p_gt0=p, n_cx=n))

    # ---- Q3: the niche -- among model-UNCERTAIN positions (logp below complex median), kl vs burial ----
    def hard_auroc(c, score):
        sub = by[c]
        hard = sub[sub.logp_native <= sub.logp_native.median()]
        if hard.is_hot.sum() < 1 or (hard.is_hot == 0).sum() < 1:
            return None
        return auroc(hard.is_hot, hard[score])
    for score in ["burial_score", "kl"]:
        obs, lo, hi, _, n = cluster_boot(lambda c, s=score: hard_auroc(c, s), cids)
        rows.append(dict(metric="AUROC_is_hot_MODEL_UNCERTAIN", ranker=score, estimate=obs, lo=lo, hi=hi,
                         p_gt0=np.nan, n_cx=n))
    obs, lo, hi, p, n = cluster_boot(
        lambda c: (hard_auroc(c, "kl") - hard_auroc(c, "burial_score"))
        if hard_auroc(c, "kl") is not None else None, cids)
    rows.append(dict(metric="DELTA_AUROC_MODEL_UNCERTAIN", ranker="kl_minus_burial", estimate=obs, lo=lo,
                     hi=hi, p_gt0=p, n_cx=n))

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out["nboot"] = NBOOT
    out["command"] = f"python3 src/kl_triage.py --joined {a.joined} --out {a.out}"
    out.to_csv(a.out, index=False)

    def g(metric, ranker):
        r = out[(out.metric == metric) & (out.ranker == ranker)]
        return r.iloc[0] if len(r) else None
    print(f"complexes with >=1 hotspot: {len(cids)}  (bootstrap {NBOOT} reps, seed {SEED})")
    for m in ["capture@k3", "capture@k25pct"]:
        b = g(m, "burial_score"); k = g(m, "kl"); kb = g(m, "klbur"); rnd = g(m, "random_expected")
        print(f"\n[{m}] random={rnd.estimate:.3f}  burial={b.estimate:.3f}  kl={k.estimate:.3f}  kl+burial={kb.estimate:.3f}")
        for rk in [f"kl_minus_burial_score", f"klbur_minus_burial_score"]:
            dd = g(f"DELTA_{m}", rk)
            if dd is not None:
                star = "*" if (dd.lo > 0 or dd.hi < 0) else " "
                print(f"    Δ {rk:24} {dd.estimate:+.3f} [{dd.lo:+.3f},{dd.hi:+.3f}] P(>0)={dd.p_gt0:.3f} {star}")
    da = g("DELTA_AUROC_is_hot", "klbur_minus_burial_score")
    print(f"\n[AUROC is_hot] burial={g('AUROC_is_hot','burial_score').estimate:.3f} "
          f"kl={g('AUROC_is_hot','kl').estimate:.3f} kl+burial={g('AUROC_is_hot','klbur').estimate:.3f}"
          f"   Δ(kl+bur - bur)={da.estimate:+.3f} [{da.lo:+.3f},{da.hi:+.3f}] P(>0)={da.p_gt0:.3f}")
    dn = g("DELTA_AUROC_MODEL_UNCERTAIN", "kl_minus_burial")
    hu_b = g("AUROC_is_hot_MODEL_UNCERTAIN", "burial_score"); hu_k = g("AUROC_is_hot_MODEL_UNCERTAIN", "kl")
    print(f"[NICHE: model-uncertain positions] burial={hu_b.estimate:.3f} kl={hu_k.estimate:.3f}"
          f"   Δ(kl - bur)={dn.estimate:+.3f} [{dn.lo:+.3f},{dn.hi:+.3f}] P(>0)={dn.p_gt0:.3f}")
    print(f"\nEXPLORATORY. Wrote {a.out}")


if __name__ == "__main__":
    main()

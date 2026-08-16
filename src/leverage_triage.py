#!/usr/bin/env python3
"""Does LEVERAGE win the position-level hotspot triage? (the audit's 'single most important' test)

The paper indicts confidence and recommends free geometry (ΔSASA) for interface-hotspot triage, but never
put its own hero quantity — leverage L — into that benchmark. If L is the best zero-shot single-feature
ranker, that is the paper's actionable headline and fixes the 'so what / just a control on someone else's
score' novelty problem. If it only ties, the honest thesis is 'binding knowledge is real but not yet
actionable for ranking'. Either is publishable; silence is not.

Position-level is_hot over SKEMPI interface positions (leverage_skempi_positions.csv). Marginal AUROC +
capture@k, complex-clustered bootstrap, paired complex-clustered test L vs ΔSASA / burial. seed 20260803.
  python3 src/leverage_triage.py --out results/leverage_triage.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if (n1 == 0 or n0 == 0) else (stats.rankdata(s)[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)


def capture_at(df, col, k, rng=None):
    caps = []
    for _, g in df.groupby("complex_id"):
        nh = int(g.is_hot.sum())
        if nh == 0:
            continue
        n = len(g); kk = min(k, n)
        if col == "random":
            idx = rng.permutation(n)[:kk]; caps.append(g.is_hot.to_numpy()[idx].sum()/nh)
        else:
            order = np.argsort(-g[col].to_numpy(), kind="stable")[:kk]
            caps.append(g.is_hot.to_numpy()[order].sum()/nh)
    return np.array(caps)


def cluster_boot(fn, g, rng, n=3000):
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    out = [fn(np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])) for _ in range(n)]
    out = np.array([v for v in out if np.isfinite(v)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/leverage_triage.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    d = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    d = d[d.is_interface == 1].dropna(subset=["is_hot", "L_rms", "L_ala", "drsasa", "burial", "klP", "conf"]).copy()
    g = d.complex_id.to_numpy(); y = d.is_hot.to_numpy().astype(float)
    print(f"[triage] {len(d)} interface positions, {d.complex_id.nunique()} complexes, {int(y.sum())} hotspots")

    # rankers: high score = predicted hotspot. leverage MAGNITUDE (L_rms), destabilising alanine (-L_ala),
    # worst-sub (-L_min via -L_ala proxy), geometry, KL, confidence, random.
    d["negL_ala"] = -d.L_ala
    rankers = [("leverage |L|_rms", "L_rms"), ("leverage -L(->Ala)", "negL_ala"),
               ("ΔSASA", "drsasa"), ("burial", "burial"), ("KL", "klP"),
               ("confidence", "conf"), ("random", "random")]
    rows = []
    aucs = {}
    for name, col in rankers:
        if col == "random":
            s = rng.random(len(d))
        else:
            s = d[col].to_numpy()
        a0 = auc(s, y); aucs[name] = s
        lo, hi, _ = cluster_boot(lambda t: auc(s[t], y[t]), g, rng, n=2000)
        print(f"  AUROC[{name:20s}] = {a0:.4f} [{lo:.4f},{hi:.4f}]")
        rows.append(dict(metric="marginal_AUROC_is_hot", ranker=name, value=round(float(a0), 4),
                         lo=round(lo, 4), hi=round(hi, 4), n=len(d)))

    # paired complex-clustered: leverage |L|_rms MINUS ΔSASA and MINUS burial (the honest comparison)
    for ref_name, ref in [("ΔSASA", "drsasa"), ("burial", "burial")]:
        sL = d["L_rms"].to_numpy(); sR = d[ref].to_numpy()
        diff = lambda t: auc(sL[t], y[t]) - auc(sR[t], y[t])
        dval = auc(sL, y) - auc(sR, y)
        lo, hi, pgt = cluster_boot(diff, g, rng, n=3000)
        print(f"  PAIRED AUROC[|L|_rms − {ref_name}] = {dval:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={pgt:.3f}")
        rows.append(dict(metric=f"paired_AUROC_Lrms_minus_{ref_name}", ranker="|L|_rms", value=round(dval, 4),
                         lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(pgt, 3), n=len(d)))

    # capture@k
    for k in (3, 5):
        base = {}
        for name, col in rankers:
            caps = capture_at(d, col, k, rng); base[name] = caps
            print(f"  capture@{k}[{name:20s}] = {caps.mean():.4f}")
            rows.append(dict(metric=f"capture_at_{k}", ranker=name, value=round(float(caps.mean()), 4),
                             n_complexes=len(caps)))
        # paired L_rms vs ΔSASA capture
        cL = base["leverage |L|_rms"]; cR = base["ΔSASA"]
        m = min(len(cL), len(cR)); dd = cL[:m] - cR[:m]
        bs = [dd[rng.integers(0, m, m)].mean() for _ in range(3000)]
        print(f"  capture@{k} paired [|L|_rms − ΔSASA] = {dd.mean():+.4f} "
              f"[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] P(>0)={np.mean(np.array(bs)>0):.3f}")

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = "position-level hotspot triage incl. leverage (audit's headline test); complex-clustered boot"
    out["command"] = "python3 src/leverage_triage.py"
    out.to_csv(a.out, index=False)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

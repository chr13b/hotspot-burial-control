#!/usr/bin/env python3
"""AB-Bind: the combiner-free CPI test the Big-Idea-1 analysis never ran.

src/abbind_bigidea1.py decided "logP adds nothing beyond geometry+similarity" on a single
dAUROC over a 0.660 cross-validated baseline (n=420, 27 complexes). src/abbind_readout_audit.py
showed that readout's detection floor sits at a within-mutant-identity AUROC of ~0.60, and logP's
is 0.593 — i.e. the test cannot resolve an effect of exactly the observed size.

CPI (Watson & Wright 2019), the estimator src/nugget_cpi.py already uses on SKEMPI, is the
combiner-free alternative: cross-fit on Z+X, then destroy X's Z-conditional information by
permuting X WITHIN strata of the fitted Z-score; the increase in log-loss is X's conditional
predictive impact. Log-loss is a proper scoring rule and is materially more sensitive than a
rank-only dAUROC. Controls Z = burial + ΔSASA + BLOSUM62 + volume (the committed baseline).

  python3 src/abbind_cpi.py --out results/abbind_cpi.csv
"""
import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 20260803
CACHE = "results/abbind_readout_audit_positions.csv"


def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def logloss(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def z(v):
    v = np.asarray(v, float); s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 1e-12 else v * 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/abbind_cpi.csv")
    a = ap.parse_args()
    e = pd.read_csv(CACHE)
    ei = e[e.is_interface == 1].dropna(subset=["logP", "burial", "drsasa"]).reset_index(drop=True)
    ei["destab"] = (ei.ddg >= 1.0).astype(int)
    y = ei.destab.to_numpy(float); g = ei.pdb.to_numpy()
    rng = np.random.default_rng(SEED)
    print(f"interface mutations {len(ei)}  complexes {ei.pdb.nunique()}  destabilising {int(y.sum())}")

    BASE = ["burial", "drsasa", "blosum", "dvol"]
    tests = [("logP | burial+ΔSASA+BLOSUM+vol", BASE, "logP"),
             ("logP_minus_wt | burial+ΔSASA+BLOSUM+vol", BASE, "logP_minus_wt"),
             ("logodds(P/Q) | burial+ΔSASA+BLOSUM+vol", BASE, "logodds"),
             ("ΔSASA | burial+BLOSUM+vol  [positive control]", ["burial", "blosum", "dvol"], "drsasa"),
             ("BLOSUM | burial+ΔSASA+vol  [positive control]", ["burial", "drsasa", "dvol"], "blosum")]
    rows = []
    for name, Zc, Xc in tests:
        d = ei.dropna(subset=[Xc] + Zc)
        yy = d.destab.to_numpy(float); gg = d.pdb.to_numpy()
        Z = np.column_stack([z(d[c]) for c in Zc]); X = z(d[Xc]); XZ = np.column_stack([Z, X])
        eta = np.zeros(len(yy)); bX = np.zeros(len(yy)); sZ = np.zeros(len(yy))
        for tr, te in GroupKFold(min(5, d.pdb.nunique())).split(XZ, yy, gg):
            m = LogisticRegression(max_iter=2000).fit(XZ[tr], yy[tr])
            eta[te] = XZ[te] @ m.coef_[0] + m.intercept_[0]; bX[te] = m.coef_[0][-1]
            mz = LogisticRegression(max_iter=2000).fit(Z[tr], yy[tr])
            sZ[te] = Z[te] @ mz.coef_[0] + mz.intercept_[0]
        lf = logloss(yy, sig(eta))
        bins = pd.qcut(pd.Series(sZ).rank(method="first"), 10, labels=False)
        order = {b: np.where(bins == b)[0] for b in np.unique(bins)}
        lp_acc = np.zeros(len(yy)); Rperm = 60
        for _ in range(Rperm):
            Xp = X.copy()
            for b, idx in order.items():
                Xp[idx] = X[rng.permutation(idx)]
            lp_acc += logloss(yy, sig(eta - bX * (X - Xp)))
        cpi = lp_acc / Rperm - lf
        cids = np.unique(gg); posby = {c: np.where(gg == c)[0] for c in cids}
        stat = float(cpi.mean())
        b = np.array([cpi[np.concatenate([posby[c] for c in rng.choice(cids, len(cids), True)])].mean()
                      for _ in range(3000)])
        lo, hi, p = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(np.mean(b > 0))
        vd = "ADDS (CI>0)" if lo > 0 else "conditionally INDEPENDENT (CI spans 0)"
        print(f"  CPI[{name:46s}] = {stat:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {vd}")
        rows.append(dict(test=name, cpi=round(stat, 5), lo=round(lo, 5), hi=round(hi, 5),
                         p_gt0=round(p, 3), verdict=vd, n=len(d), n_destab=int(yy.sum()),
                         n_complex=len(cids)))
    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = ("CPI on AB-Bind, the estimator src/nugget_cpi.py uses on SKEMPI; the Big-Idea-1 "
                   "demotion used only dAUROC over a 0.660 baseline, whose detection floor "
                   "(src/abbind_readout_audit.py) is above the observed effect size")
    out["command"] = "python3 src/abbind_cpi.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

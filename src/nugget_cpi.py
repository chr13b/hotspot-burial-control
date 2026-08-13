#!/usr/bin/env python3
"""CPI — Conditional Predictive Impact: the combiner-free statement of the nugget.

The z-sum 'confidence HURTS -0.056' was a combiner artifact. CPI (Watson & Wright 2019) tests whether a
feature X adds predictive value for is_hot BEYOND controls Z, with NO chosen combiner: cross-fit a logistic
model on Z+X, then break X's Z-conditional information by permuting X WITHIN geometry strata; the increase
in log-loss (CPI) is X's conditional predictive impact. CPI>0 => X adds; CPI~=0 => conditional independence.

Tests (SKEMPI crystal interface): confidence | full geometry (expect ~0 = the nugget); ΔSASA | burial+nbr
(expect >0); KL | full geometry (expect ~0, cf. R1). Complex-clustered bootstrap, seed 20260803.
  python3 src/nugget_cpi.py --out results/nugget_cpi.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803


def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def logloss(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nugget_cpi.csv")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    R = "results"
    j = pd.read_csv(f"{R}/kl_detector_joined.csv"); j = j[j.is_interface == 1].copy()
    j["icode"] = j.icode.fillna("").astype(str)
    pos = pd.read_csv(f"{R}/p0_positions.csv", usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    pos["icode"] = pos.icode.fillna("").astype(str)
    j = j.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left").rename(columns={"drsasa": "dsasa"})
    j = j.dropna(subset=["dsasa", "kl", "burial", "nbr", "is_hot", "logp_native"]).reset_index(drop=True)
    j["conf"] = j.logp_native
    for c in ["burial", "nbr", "dsasa", "kl", "conf"]:
        j[c + "z"] = (j[c] - j[c].mean()) / j[c].std()
    y = j.is_hot.to_numpy().astype(float); g = j.complex_id.to_numpy()
    rng = np.random.default_rng(SEED)
    tests = [("confidence | burial+nbr+ΔSASA", ["burialz", "nbrz", "dsasaz"], "confz"),
             ("ΔSASA | burial+nbr", ["burialz", "nbrz"], "dsasaz"),
             ("KL | burial+nbr+ΔSASA", ["burialz", "nbrz", "dsasaz"], "klz")]
    rows = []
    for name, Zc, Xc in tests:
        Z = j[Zc].to_numpy(); X = j[Xc].to_numpy(); XZ = np.column_stack([Z, X])
        eta = np.zeros(len(y)); bX = np.zeros(len(y)); sZ = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(XZ, y, g):
            m = LogisticRegression(max_iter=2000).fit(XZ[tr], y[tr])
            eta[te] = XZ[te] @ m.coef_[0] + m.intercept_[0]; bX[te] = m.coef_[0][-1]
            mz = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
            sZ[te] = Z[te] @ mz.coef_[0] + mz.intercept_[0]         # geometry-only score for strata
        p_full = sig(eta); lf = logloss(y, p_full)
        bins = pd.qcut(pd.Series(sZ).rank(method="first"), 25, labels=False)  # geometry strata
        # conditional permutation of X within geometry strata, averaged over R draws
        lp_acc = np.zeros(len(y))
        Rperm = 40
        order = {b: np.where(bins == b)[0] for b in np.unique(bins)}
        for _ in range(Rperm):
            Xp = X.copy()
            for b, idx in order.items():
                Xp[idx] = X[rng.permutation(idx)]
            lp_acc += logloss(y, sig(eta - bX * (X - Xp)))
        cpi = lp_acc / Rperm - lf                                    # per-obs conditional predictive impact
        cids = np.unique(g); posby = {c: np.where(g == c)[0] for c in cids}
        stat = float(cpi.mean())
        b = np.array([cpi[np.concatenate([posby[c] for c in rng.choice(cids, len(cids), True)])].mean()
                      for _ in range(3000)])
        lo, hi, p = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(np.mean(b > 0))
        verdict = "ADDS (CI>0)" if lo > 0 else ("conditionally INDEPENDENT (CI spans 0)" if hi > 0 else "neg")
        print(f"  CPI[{name:30s}] = {stat:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {verdict}")
        rows.append(dict(test=name, cpi=round(stat, 5), lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3),
                         verdict=verdict))
    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = "CPI>0 => feature adds beyond controls; ~0 => conditional independence. confidence|geometry ~0 is the combiner-free nugget"
    out["command"] = "python3 src/nugget_cpi.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

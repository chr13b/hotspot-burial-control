#!/usr/bin/env python3
"""W4 (audit must-fix): the actionable claim should be the COMBINED ranker, not "|L| beats ΔSASA" (whose CI
includes 0). Does adding the mixed derivative to geometry significantly improve hotspot ranking? Cross-fit
logistic (GroupKFold by complex) of is_hot on geometry vs geometry+|L|; AUROC of out-of-fold scores; paired
complex-clustered bootstrap of the difference. From results/leverage_skempi_positions.csv.

  python3 src/w4_combined_ranker.py --out results/w4_combined_ranker.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEED = 20260803

def oof_scores(X, y, g):
    eta = np.zeros(len(y)); nf = int(min(5, len(np.unique(g))))
    for tr, te in GroupKFold(nf).split(X, y, g):
        m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]); eta[te] = X[te] @ m.coef_[0] + m.intercept_[0]
    return eta

def auroc(s, y):
    y = np.asarray(y); r = stats.rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/w4_combined_ranker.csv"); a = ap.parse_args()
    d = pd.read_csv("results/leverage_skempi_positions.csv")
    d = d[(d.is_interface == True) & d.L_rms.notna() & d.L_ala.notna()].copy()   # noqa: E712
    y = d.is_hot.astype(int).to_numpy(); g = d.complex_id.to_numpy()
    def z(c): v = d[c].to_numpy(float); return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    GEO = np.column_stack([z("burial"), z("nbr"), z("drsasa")])
    rankers = {
        "geometry (burial+nbr+dSASA)": GEO,
        "geometry + |L|_rms": np.column_stack([GEO, z("L_rms")]),
        "geometry + (-L_ala)": np.column_stack([GEO, -z("L_ala")]),
    }
    scores = {k: oof_scores(X, y, g) for k, X in rankers.items()}
    base = "geometry (burial+nbr+dSASA)"
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}; rng = np.random.default_rng(SEED)
    print(f"[W4] {len(d)} interface positions, {len(ids)} complexes, {int(y.sum())} hotspots")
    rows = []
    for k, s in scores.items():
        a_pt = auroc(s, y)
        if k == base:
            print(f"  {k:32s} AUROC = {a_pt:.4f}")
            rows.append(dict(ranker=k, auroc=round(a_pt, 4), delta_vs_geom=0.0, lo=np.nan, hi=np.nan, p_gt0=np.nan))
            continue
        db = scores[base]
        boot = []
        for _ in range(3000):
            ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids), True)])
            boot.append(auroc(s[ix], y[ix]) - auroc(db[ix], y[ix]))
        boot = np.array(boot); lo, hi, p = np.percentile(boot, 2.5), np.percentile(boot, 97.5), float(np.mean(boot > 0))
        dpt = a_pt - auroc(db, y)
        print(f"  {k:32s} AUROC = {a_pt:.4f}   Δ vs geometry = {dpt:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}")
        rows.append(dict(ranker=k, auroc=round(a_pt, 4), delta_vs_geom=round(dpt, 4),
                         lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(p, 3)))
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"[wrote] {a.out}")

if __name__ == "__main__":
    main()

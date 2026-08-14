#!/usr/bin/env python3
"""Robustness: per-POSITION AUROC for Big-Idea-1 (closes the pooled-base-rate caveat).

Pooled interface AUROC(P->binds)=0.615 could in principle be inflated by positions of differing base rate.
Here we compute AUROC WITHIN each interface position (ranking that position's own 19 substitutions by the
model's complex-conditioned probability), for positions with >=3 binders and >=3 non-binders, and bootstrap
the mean. seed 20260803.
  python3 src/bennett_perposition.py --out results/bennett_perposition.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/bennett_occlusion_allatom_pairs.csv")
    ap.add_argument("--out", default="results/bennett_perposition.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.pairs); d = d[d.dsasa > 5]
    per = []
    for (des, rn), g in d.groupby(["design", "resnum"]):
        nb = g.binds.sum()
        if nb >= 3 and (len(g) - nb) >= 3:
            v = auc(g.P.values, g.binds.values)
            if np.isfinite(v):
                per.append(v)
    per = np.array(per)
    rng = np.random.default_rng(20260803)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out = pd.DataFrame([dict(metric="per_position_auroc_mean", value=round(float(per.mean()), 4),
                             lo=round(float(lo), 4), hi=round(float(hi), 4),
                             median=round(float(np.median(per)), 4), frac_gt_0p5=round(float((per > 0.5).mean()), 4),
                             n_positions=len(per), seed=20260803,
                             note="per-position AUROC(P->binds) at interface, >=3 of each class; pooled was 0.615",
                             command="python3 src/bennett_perposition.py")])
    out.to_csv(a.out, index=False)
    print(f"per-position AUROC mean {per.mean():.3f} [{lo:.3f},{hi:.3f}] median {np.median(per):.3f} "
          f"frac>0.5 {(per>0.5).mean():.3f} over {len(per)} positions; wrote {a.out}")


if __name__ == "__main__":
    main()

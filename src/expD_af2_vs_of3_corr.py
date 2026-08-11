"""Experiment D: the per-complex AF2-vs-OF3 deficit correlation (most-informative readout, PREREG 5d).

For each complex, the burial-matched deficit is the mean over its committed SECONDARY_B pairs of
d = logp_native(hot) - logp_native(ctl), evaluated on each predictor's backbone. If the SAME
complexes carry the deficit under both AF2-multimer and OpenFold3, that is strong evidence of a
general (independently-reconstructed-backbone) signal; if the two per-complex deficits are disjoint,
the per-predictor deficits are predictor-specific noise.

Reports Spearman + Pearson of (d_af2, d_of3) across the shared complexes, with a complex-level
bootstrap CI (seed 20260803). Also emits the per-complex table incl. the crystal deficit for context.

Usage (ftax env):
  python3 src/expD_af2_vs_of3_corr.py \
    --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
    --af2-positions $SCRATCH/ftax/expD/expD_p0_positions.csv \
    --of3-positions $SCRATCH/ftax/predicted/expA_p0_positions.csv \
    --crystal-positions results/p0_positions.csv \
    --out results/expD_af2_of3_corr.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260803
NBOOT = 10000


def pos_lookup(csv, col="logp_native"):
    df = pd.read_csv(csv, usecols=lambda c: c in ("complex_id", "chain", "resnum", "icode", col))
    df["icode"] = df["icode"].fillna("").astype(str)
    df = df.sort_values("icode").drop_duplicates(subset=["complex_id", "chain", "resnum"], keep="first")
    return {(r.complex_id, r.chain, int(r.resnum)): getattr(r, col) for r in df.itertuples()}


def per_complex_deficit(pairs, lp):
    """complex_id -> mean pair gap d on the given predictor (nan pairs dropped)."""
    rows = []
    for r in pairs.itertuples():
        h = (r.complex_id, r.hot_chain, int(r.hot_resnum))
        c = (r.complex_id, r.ctl_chain, int(r.ctl_resnum))
        rows.append((r.complex_id, lp.get(h, np.nan) - lp.get(c, np.nan)))
    df = pd.DataFrame(rows, columns=["complex_id", "d"])
    df = df[np.isfinite(df["d"])]
    return df.groupby("complex_id")["d"].mean()


def boot_corr(x, y, kind, seed=SEED, nboot=NBOOT):
    rng = np.random.default_rng(seed)
    n = len(x)
    f = (lambda a, b: stats.spearmanr(a, b).correlation) if kind == "spearman" \
        else (lambda a, b: stats.pearsonr(a, b)[0])
    obs = f(x, y)
    bs = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        if len(set(idx)) < 3:
            continue
        v = f(x[idx], y[idx])
        if np.isfinite(v):
            bs.append(v)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(obs), float(lo), float(hi), float(np.mean(np.array(bs) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/p0_dssp_pairs_SECONDARY_B_any_interface.csv")
    ap.add_argument("--af2-positions", required=True)
    ap.add_argument("--of3-positions", required=True)
    ap.add_argument("--crystal-positions", default="results/p0_positions.csv")
    ap.add_argument("--out", default="results/expD_af2_of3_corr.csv")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    pairs = pd.read_csv(a.pairs)
    d_af2 = per_complex_deficit(pairs, pos_lookup(a.af2_positions))
    d_of3 = per_complex_deficit(pairs, pos_lookup(a.of3_positions))
    d_cry = per_complex_deficit(pairs, pos_lookup(a.crystal_positions))

    tab = pd.DataFrame({"d_af2": d_af2, "d_of3": d_of3, "d_crystal": d_cry}).dropna(subset=["d_af2", "d_of3"])
    tab = tab.reset_index().rename(columns={"index": "complex_id"})
    tab.to_csv(a.out.replace(".csv", "_percomplex.csv"), index=False)
    x, y = tab["d_af2"].values, tab["d_of3"].values
    print(f"[corr] shared complexes: {len(tab)}  "
          f"mean d_af2={x.mean():+.3f}  mean d_of3={y.mean():+.3f}  mean d_cry={tab['d_crystal'].mean():+.3f}")

    rows = []
    for kind in ("spearman", "pearson"):
        obs, lo, hi, pgt0 = boot_corr(x, y, kind)
        excl0 = (lo > 0) or (hi < 0)
        print(f"  {kind}: {obs:+.3f} [{lo:+.3f},{hi:+.3f}] P(>0)={pgt0:.3f}  CI-excludes-0={excl0}")
        rows.append(dict(metric=kind, n_complexes=len(tab), estimate=obs, lo95=lo, hi95=hi,
                         p_gt0=pgt0, ci_excludes_0=excl0,
                         mean_d_af2=float(x.mean()), mean_d_of3=float(y.mean()),
                         mean_d_crystal=float(tab["d_crystal"].mean())))
    pd.DataFrame(rows).assign(seed=SEED, command=cmd).to_csv(a.out, index=False)
    print(f"[corr] wrote {a.out} and *_percomplex.csv")


if __name__ == "__main__":
    main()

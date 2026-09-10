#!/usr/bin/env python3
"""Beyond-X ladder in ONE matched metric: partial rank-correlation of L with experimental ΔΔG, controlling in turn
for each competing feature-CLASS (geometry, conservation, one-pass log-odds, fitted physics). Same metric on every
rung, so the bars ARE comparable — complementary to the CPI placebo-floor analysis in §4 (which tells the same
story in its own framework). FoldX-independent except the physics rung (merges the committed Lane-A FoldX ΔΔG).

partial(L, exp | Z) = Pearson of the rank-residuals of L and exp after regressing each on rank(Z) (+intercept).
Sign: L scores favorability, exp scores cost, so a real signal is NEGATIVE; complex-clustered bootstrap 95% CI.

  python3 src/leverage_ladder.py  ->  results/beyond_x_ladder.csv
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr

SEED, NBOOT = 20260803, 5000
RUNGS = [("geometry", ["drsasa", "rsasa_complex", "nbr", "burial"]),
         ("conservation", ["blosum", "dvol", "dhydro"]),
         ("one-pass log-odds", ["logP_mut", "r_mut", "conf"])]        # physics rung added below (needs FoldX merge)


def partial_rank(y, x, Z):
    ry, rx = rankdata(y), rankdata(x)
    RZ = np.column_stack([np.ones(len(ry))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])])
    beta_y, *_ = np.linalg.lstsq(RZ, ry, rcond=None)
    beta_x, *_ = np.linalg.lstsq(RZ, rx, rcond=None)
    return pearsonr(rx - RZ @ beta_x, ry - RZ @ beta_y)[0]


def boot_ci(y, x, Z, cx):
    ucx = np.unique(cx); idxby = {c: np.where(cx == c)[0] for c in ucx}
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(NBOOT):
        pick = rng.choice(ucx, len(ucx), replace=True)
        ii = np.concatenate([idxby[c] for c in pick])
        try:
            out.append(partial_rank(y[ii], x[ii], Z[ii]))
        except Exception:
            pass
    a = np.array(out)
    return np.percentile(a, 2.5), np.percentile(a, 97.5), (a < 0).mean()


def main():
    d = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    d = d[(d.is_interface == 1)].dropna(subset=["ddG", "L"]).reset_index(drop=True)
    rows = []
    for name, cols in RUNGS:
        m = d.dropna(subset=cols).reset_index(drop=True)
        y, x = m.L.to_numpy(float), m.ddG.to_numpy(float)
        Z = m[cols].to_numpy(float)
        pr = partial_rank(y, x, Z)
        lo, hi, plt0 = boot_ci(y, x, Z, m.complex_id.to_numpy())
        rows.append(dict(rung=name, controls="+".join(cols), partial=round(pr, 4),
                         lo=round(lo, 4), hi=round(hi, 4), p_lt0=round(plt0, 4),
                         n=len(m), n_complex=m.complex_id.nunique()))
        print(f"  {name:18s} partial(L,exp|{'+'.join(cols)}) = {pr:+.4f} [{lo:+.4f},{hi:+.4f}] P(<0)={plt0:.3f} n={len(m)}")
    # physics rung — merge committed FoldX Lane-A ΔΔG
    fx = pd.read_csv("results/foldx_ddg_laneA.csv").dropna(subset=["ddg_bind"]).copy()
    fx["icode"] = fx.get("icode", "").fillna("").astype(str)
    dd = d.copy(); dd["icode"] = dd.icode.fillna("").astype(str)
    for df in (fx, dd):
        df["key"] = (df.complex_id.astype(str) + "|" + df.chain.astype(str) + "|" +
                     df.resnum.astype(int).astype(str) + "|" + df.icode + "|" + df.mut.astype(str))
    m = dd.merge(fx[["key", "ddg_bind"]], on="key", how="inner").dropna(subset=["ddg_bind"]).reset_index(drop=True)
    y, x, Z = m.L.to_numpy(float), m.ddG.to_numpy(float), m[["ddg_bind"]].to_numpy(float)
    pr = partial_rank(y, x, Z); lo, hi, plt0 = boot_ci(y, x, Z, m.complex_id.to_numpy())
    rows.append(dict(rung="fitted physics (FoldX)", controls="foldx_ddg_bind", partial=round(pr, 4),
                     lo=round(lo, 4), hi=round(hi, 4), p_lt0=round(plt0, 4),
                     n=len(m), n_complex=m.complex_id.nunique()))
    print(f"  {'fitted physics':18s} partial(L,exp|FoldX) = {pr:+.4f} [{lo:+.4f},{hi:+.4f}] P(<0)={plt0:.3f} n={len(m)}")
    pd.DataFrame(rows).assign(seed=SEED, nboot=NBOOT).to_csv("results/beyond_x_ladder.csv", index=False)
    print("-> results/beyond_x_ladder.csv")


if __name__ == "__main__":
    main()

"""Continuous slope of the burial-matched gap d vs log10(interface-RMSD) — the C2-PRIMARY statistic.

PREREG_expC2 §4/§5. Consumes a `*_gap_perbackbone.csv` (from src/expC_analyze.py: one row per backbone
with columns backbone_id, complex_id, d, partial_T, irmsd, interface_ok). Computes two slopes and keeps
them sharply distinct:

  physical  = slope over interface-FORMED (interface_ok==1) AND iRMSD<=8 A AND partial_T!=0 backbones
              -- the honest, pre-registered dose-response statistic (the deciding physical regime).
  naive     = slope over ALL backbones (incl. the partial_T=0 crystal anchor at iRMSD~0 and the
              dissolved >10 A tail) -- reported ONLY to expose the crystal-vs-dissolved confound; it is
              NOT evidence for a dose-response.

CI + P via complex-level bootstrap (resample complexes, refit). Units: gap per log10 A. The physical
slope's 90% CI is emitted too, for the C2-NULL TOST reading (equivalence to zero at margin +-0.10).

Positive control: run on Exp C's committed results/expC_gap_perbackbone.csv first; Exp C reported the
physical slope flat and the naive slope apparently significant. Whatever the reproduced values are, they
are written out and reported verbatim.

Usage:
  python3 src/expC2_slope_check.py --gap-perbackbone results/expC_gap_perbackbone.csv \
      --label expC --out results/expC_slope_check.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

SEED, NBOOT, IRMSD_MAX = 20260803, 10000, 8.0


def _slope(x, y):
    """OLS slope of y on x; nan if <3 finite points or x has no spread."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def slope_boot(df, seed=SEED, n=NBOOT):
    """Complex-clustered bootstrap of the OLS slope of d ~ log10(irmsd) over rows of df.

    Returns dict: slope (point est on full data), lo95/hi95, lo90/hi90, p_boot (two-sided cluster
    bootstrap), p_ols (pooled linregress, clustering-naive, for reference), n_bb, n_cx.
    """
    df = df[np.isfinite(df["x"]) & np.isfinite(df["d"])].copy()
    n_bb, n_cx = len(df), df["complex_id"].nunique()
    if n_bb < 3 or n_cx < 2:
        return dict(slope=np.nan, lo95=np.nan, hi95=np.nan, lo90=np.nan, hi90=np.nan,
                    p_boot=np.nan, p_ols=np.nan, n_bb=n_bb, n_cx=n_cx)
    est = _slope(df["x"].values, df["d"].values)
    p_ols = float(stats.linregress(df["x"].values, df["d"].values).pvalue)
    cids = df["complex_id"].unique()
    by = {c: df[df["complex_id"] == c] for c in cids}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        s = pd.concat([by[cids[i]] for i in rng.choice(len(cids), len(cids), True)], ignore_index=True)
        boots.append(_slope(s["x"].values, s["d"].values))
    boots = np.array(boots, float)
    fin = np.isfinite(boots)
    b = boots[fin]
    if b.size == 0:
        return dict(slope=est, lo95=np.nan, hi95=np.nan, lo90=np.nan, hi90=np.nan,
                    p_boot=np.nan, p_ols=p_ols, n_bb=n_bb, n_cx=n_cx)
    lo95, hi95 = np.percentile(b, [2.5, 97.5])
    lo90, hi90 = np.percentile(b, [5.0, 95.0])
    p_boot = 2.0 * min((b <= 0).mean(), (b >= 0).mean())
    return dict(slope=est, lo95=float(lo95), hi95=float(hi95), lo90=float(lo90), hi90=float(hi90),
                p_boot=float(min(p_boot, 1.0)), p_ols=p_ols, n_bb=n_bb, n_cx=n_cx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-perbackbone", required=True)
    ap.add_argument("--label", default="expC")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    g = pd.read_csv(a.gap_perbackbone)
    g = g[np.isfinite(g["irmsd"]) & (g["irmsd"] > 0)].copy()
    g["x"] = np.log10(g["irmsd"])

    phys = g[(g["interface_ok"] == 1) & (g["irmsd"] <= IRMSD_MAX) & (g["partial_T"] != 0)]
    naive = g  # all backbones incl. crystal anchor + dissolved tail

    rows = []
    for kind, sub in (("physical", phys), ("naive", naive)):
        r = slope_boot(sub)
        rows.append(dict(label=a.label, kind=kind, irmsd_max=(IRMSD_MAX if kind == "physical" else np.nan),
                         **r))
        print(f"  [{a.label} {kind:8s}] slope={r['slope']:+.4f}  95%CI[{r['lo95']:+.4f},{r['hi95']:+.4f}] "
              f" 90%CI[{r['lo90']:+.4f},{r['hi90']:+.4f}]  p_boot={r['p_boot']:.4f} p_ols={r['p_ols']:.4f}"
              f"  (n_bb={r['n_bb']}, n_cx={r['n_cx']})")
    # C2-NULL TOST readout on the physical slope
    ph = rows[0]
    if np.isfinite(ph["lo90"]) and np.isfinite(ph["hi90"]):
        tost_equiv = (ph["lo90"] > -0.10) and (ph["hi90"] < 0.10)
        print(f"  [C2-NULL TOST @±0.10] physical 90%CI ⊂ (−0.10,+0.10)? {tost_equiv}")
    pd.DataFrame(rows).assign(command=cmd).to_csv(a.out, index=False)
    print(f"[expC_slope_check] wrote {a.out}")


if __name__ == "__main__":
    main()

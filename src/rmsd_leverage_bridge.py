#!/usr/bin/env python3
"""Independent LOCAL recomputation of the dose-law bridge: does per-complex leverage RETENTION fall with
per-complex interface Cα-RMSD? Both inputs are committed CSVs (predicted_backbone_rmsd.csv interface RMSDs +
deficit_vs_leverage_percomplex.csv retention = mean|L|rms(predicted)/mean|L|rms(crystal)), so this reproduces
the Sherlock bridge number WITHOUT the raw predicted PDBs — an independent cross-check of the Sherlock
computation and a CSV trace for the number. Caveat carried from the source: retention is a MAGNITUDE ratio, so
it understates the discrimination loss the pooled CPI (84%/69% of crystal) shows.

  python3 src/rmsd_leverage_bridge.py --out results/rmsd_leverage_bridge.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
SEED = 20260803


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/rmsd_leverage_bridge.csv"); a = ap.parse_args()
    r = pd.read_csv("results/predicted_backbone_rmsd.csv")
    d = pd.read_csv("results/deficit_vs_leverage_percomplex.csv")
    m = r.merge(d, on="complex_id", how="inner")
    rng = np.random.default_rng(SEED); rows = []

    def boot(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; n = len(x)
        rho = stats.spearmanr(x, y).correlation
        b = [stats.spearmanr(x[i], y[i]).correlation for i in (rng.integers(0, n, n) for _ in range(5000))]
        b = np.array([v for v in b if np.isfinite(v)])
        return rho, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), n

    print(f"[bridge] {len(m)} complexes merged from the two committed CSVs")
    # The four tests document WHY #7 (deficit x leverage) is left out of the main text: interface RMSD drives
    # leverage loss STRONGLY but the confidence deficit only weakly/not at all -> the two readouts are loosely
    # coupled (a partial dissociation), not merely underpowered.
    for rc, yc, lab in [("of3_ca_rmsd_iface", "leverage_metric", "OF3: RMSD->leverage retention"),
                        ("af2_ca_rmsd_iface", "retention_af2", "AF2: RMSD->leverage retention"),
                        ("of3_ca_rmsd_iface", "d_of3", "OF3: RMSD->confidence deficit"),
                        ("af2_ca_rmsd_iface", "d_af2", "AF2: RMSD->confidence deficit")]:
        rho, lo, hi, n = boot(m[rc], m[yc])
        print(f"  Spearman(interface Cα-RMSD, retention) {lab}: {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}")
        rows.append(dict(predictor=lab, x=rc, y=yc, spearman=round(rho, 4), lo=round(lo, 4), hi=round(hi, 4), n=n))
    pd.DataFrame(rows).assign(seed=SEED, source="local recompute from committed CSVs (no PDBs)",
                              command="python3 src/rmsd_leverage_bridge.py").to_csv(a.out, index=False)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

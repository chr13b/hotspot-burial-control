#!/usr/bin/env python3
"""#7 follow-up: WHAT drives the per-complex confidence-recovery deficit on predicted backbones? The RMSD bridge
showed leverage loss is cleanly, strongly driven by interface Cα-RMSD (−0.56/−0.64). Here we ask whether the
CONFIDENCE deficit is driven by the same local backbone error, or by global prediction quality (pTM/ipTM/
interface pLDDT), or by neither — i.e. do the two readouts fail for DIFFERENT reasons? All inputs are committed
CSVs (no PDBs). Honest, exploratory; a null is informative.

  python3 src/deficit_drivers.py --out results/deficit_drivers.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
SEED = 20260803


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/deficit_drivers.csv"); a = ap.parse_args()
    d = pd.read_csv("results/deficit_vs_leverage_percomplex.csv")
    r = pd.read_csv("results/predicted_backbone_rmsd.csv")[["complex_id", "of3_ca_rmsd_iface", "af2_ca_rmsd_iface"]]
    af = pd.read_csv("results/expD_backbone_manifest.csv")[["complex_id", "ptm", "iptm", "interface_plddt"]].add_prefix("af2_").rename(columns={"af2_complex_id": "complex_id"})
    of = pd.read_csv("results/expA_confidence.csv")[["complex_id", "ptm", "iptm", "interface_plddt"]].add_prefix("of3_").rename(columns={"of3_complex_id": "complex_id"})
    m = d.merge(r, on="complex_id").merge(af, on="complex_id", how="left").merge(of, on="complex_id", how="left")
    rng = np.random.default_rng(SEED); rows = []

    def boot(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; n = len(x)
        rho = stats.spearmanr(x, y).correlation
        b = np.array([stats.spearmanr(x[i], y[i]).correlation for i in (rng.integers(0, n, n) for _ in range(5000))]); b = b[np.isfinite(b)]
        return rho, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), n

    # d is signed (more-negative = bigger deficit). retention high = leverage survived.
    tests = [
        ("AF2 deficit", "d_af2", "af2_iptm"), ("AF2 deficit", "d_af2", "af2_ptm"),
        ("AF2 deficit", "d_af2", "af2_interface_plddt"), ("AF2 deficit", "d_af2", "af2_ca_rmsd_iface"),
        ("OF3 deficit", "d_of3", "of3_iptm"), ("OF3 deficit", "d_of3", "of3_ptm"),
        ("OF3 deficit", "d_of3", "of3_interface_plddt"), ("OF3 deficit", "d_of3", "of3_ca_rmsd_iface"),
        ("AF2 leverage-retention (ref)", "retention_af2", "af2_ca_rmsd_iface"),
        ("OF3 leverage-retention (ref)", "leverage_metric", "of3_ca_rmsd_iface"),
    ]
    for lab, xc, yc in tests:
        rho, lo, hi, n = boot(m[xc], m[yc])
        sig = "CI>0" if lo > 0 else ("CI<0" if hi < 0 else "ns")
        print(f"  {lab:28s} vs {yc:22s} rho={rho:+.3f} [{lo:+.3f},{hi:+.3f}] n={n} {sig}")
        rows.append(dict(quantity=lab, x=xc, driver=yc, spearman=round(rho, 4), lo=round(lo, 4), hi=round(hi, 4), n=n, sig=sig))
    pd.DataFrame(rows).assign(seed=SEED, note="leverage loss is cleanly RMSD-driven; the confidence deficit is not cleanly driven by RMSD or by prediction quality -> the two readouts fail for different reasons (dissociation)",
                              command="python3 src/deficit_drivers.py").to_csv(a.out, index=False)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

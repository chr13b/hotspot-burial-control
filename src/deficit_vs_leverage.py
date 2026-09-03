#!/usr/bin/env python3
"""#7 (exploratory): does the per-complex CONFIDENCE-recovery deficit on predicted backbones (§6)
coincide with per-complex LEVERAGE stress? i.e. do the two readouts of the decomposition move together
across complexes, or are they separable? Honest-null is a valid outcome.

Confidence deficit: results/expD_af2_of3_corr_percomplex.csv (d_of3, d_af2, d_crystal; more-negative =
larger burial-matched hotspot-recovery deficit). Leverage: per-complex mean |L|_rms over interface
positions from the R2 predicted-backbone frames (leverage_predicted_{crystal,of3,af2}_positions.csv),
POSITION-MATCHED to the crystal so `retention = predicted/crystal` is a clean per-complex kept-fraction.

Matched per predictor: Spearman(d_of3, retention_of3) and Spearman(d_af2, retention_af2), complex-
bootstrap 95% CI. Also the raw predicted mean |L|_rms as a proxy x.

  python3 src/deficit_vs_leverage.py --out results/deficit_vs_leverage_percomplex.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260803
KEY = ["complex_id", "chain", "resnum", "icode"]


def load_positions(src):
    d = pd.read_csv(f"results/leverage_predicted_{src}_positions.csv", low_memory=False)
    d["icode"] = d.icode.fillna("").astype(str)
    d = d[d.is_interface == 1][KEY + ["L_rms"]].rename(columns={"L_rms": f"Lrms_{src}"})
    return d


def percomplex_leverage():
    """Position-matched per-complex mean |L|_rms for crystal/of3/af2, and retention ratios."""
    cr, o3, a2 = load_positions("crystal"), load_positions("of3"), load_positions("af2")
    m = cr.merge(o3, on=KEY, how="inner").merge(a2, on=KEY, how="inner")   # positions present in ALL 3
    g = m.groupby("complex_id").agg(
        Lrms_crystal=("Lrms_crystal", "mean"), Lrms_of3=("Lrms_of3", "mean"),
        Lrms_af2=("Lrms_af2", "mean"), n_pos=("Lrms_crystal", "size")).reset_index()
    g["retention_of3"] = g.Lrms_of3 / g.Lrms_crystal
    g["retention_af2"] = g.Lrms_af2 / g.Lrms_crystal
    return g


def spearman_boot(x, y, rng, nboot=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = stats.spearmanr(x, y).correlation
    n = len(x)
    b = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        r = stats.spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(r):
            b.append(r)
    b = np.array(b)
    return rho, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), \
        float(np.mean(b > 0)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/deficit_vs_leverage_percomplex.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    dfc = pd.read_csv("results/expD_af2_of3_corr_percomplex.csv")
    lev = percomplex_leverage()
    m = dfc.merge(lev, on="complex_id", how="inner")
    print(f"[deficit-vs-leverage] {len(m)} shared complexes "
          f"(deficit table {len(dfc)} ∩ leverage {len(lev)})")

    # positive control: crystal deficit ~0 (median), predicted deficits clearly negative
    print(f"  [+control] d_crystal median={m.d_crystal.median():+.4f} mean={m.d_crystal.mean():+.4f} "
          f"(should be ~0) | d_of3 mean={m.d_of3.mean():+.4f} d_af2 mean={m.d_af2.mean():+.4f}")
    print(f"  [+control] leverage retention: of3 median={m.retention_of3.median():.3f} "
          f"af2 median={m.retention_af2.median():.3f} (pooled R2: of3~0.84, af2~0.69 of crystal CPI)")

    rows = []
    print("\n=== Spearman(confidence deficit, leverage retention), matched predictor ===")
    tests = [
        ("d_of3", "retention_of3", "OF3 deficit vs OF3 leverage-retention (pred/crystal)"),
        ("d_af2", "retention_af2", "AF2 deficit vs AF2 leverage-retention (pred/crystal)"),
        ("d_of3", "Lrms_of3", "OF3 deficit vs raw OF3 mean|L|rms (proxy)"),
        ("d_af2", "Lrms_af2", "AF2 deficit vs raw AF2 mean|L|rms (proxy)"),
    ]
    for xcol, ycol, label in tests:
        rho, lo, hi, pgt, n = spearman_boot(m[xcol], m[ycol], rng)
        # d is SIGNED (more-negative = larger deficit); retention high = leverage survived. So a
        # POSITIVE rho means big-deficit complexes also have low retention -> the two DEGRADE TOGETHER.
        verdict = ("degrade-together (bigger deficit <-> lower leverage retention)" if lo > 0 else
                   "inverse (bigger deficit <-> higher retention)" if hi < 0 else
                   "null / separable (CI spans 0), direction=degrade-together" if rho > 0 else
                   "null / separable (CI spans 0), direction=inverse")
        print(f"  {label:52s} rho={rho:+.3f} [{lo:+.3f},{hi:+.3f}] P(>0)={pgt:.3f}  n={n}  -> {verdict}")
        rows.append(dict(test=label, x=xcol, y=ycol, spearman=round(rho, 4),
                         lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(pgt, 4), n=n, verdict=verdict))

    # committed deliverable table (per-complex), + a stats sidecar
    out = m[["complex_id", "d_of3", "d_af2"]].copy()
    out["leverage_metric"] = m["retention_of3"]
    out["metric_name"] = "leverage_retention_of3 = mean|L|rms(OF3)/mean|L|rms(crystal), position-matched"
    out["retention_af2"] = m["retention_af2"]
    out["Lrms_of3"] = m["Lrms_of3"]
    out["Lrms_af2"] = m["Lrms_af2"]
    out["Lrms_crystal"] = m["Lrms_crystal"]
    out["d_crystal"] = m["d_crystal"]
    out["seed"] = SEED
    out.to_csv(a.out, index=False)
    st = pd.DataFrame(rows); st["seed"] = SEED
    st.to_csv(a.out.replace(".csv", "_stats.csv"), index=False)
    print(f"\n[wrote] {a.out} ({len(out)} complexes) and "
          f"{a.out.replace('.csv','_stats.csv')} ({len(st)} tests)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyse the CFG-steering sweep (results/cfg_steer.csv): per-alpha ESM-IF1 leverage for the L arm vs
the random control, with COMPLEX-CLUSTERED bootstrap 95% CIs (nan-aware), the paired L-minus-random
specificity contrast, the interface native-recovery cost, the non-interface localization control, and
the pre-registered sweet-spot alpha. Writes results/cfg_steer_summary.csv.

  python3 src/cfg_steer_analyse.py --in results/cfg_steer.csv --out results/cfg_steer_summary.csv
"""
import argparse
import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 5000


def cluster_boot_mean(per_cx_vals, groups, rng, nboot=NBOOT):
    """Complex-clustered bootstrap of a mean over per-complex values. nan-aware.
    Returns (mean, lo, hi, n_used, n_boot_dropped)."""
    vals = np.asarray(per_cx_vals, float)
    groups = np.asarray(groups)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    point = np.nanmean(vals)
    boots, dropped = [], 0
    for _ in range(nboot):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        m = np.nanmean(vals[idx])
        if np.isfinite(m):
            boots.append(m)
        else:
            dropped += 1
    boots = np.array(boots)
    return (float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
            int(np.isfinite(vals).sum()), dropped)


def paired_diff_boot(df_alpha, rng, nboot=NBOOT):
    """Per-complex (L - random) ESM-IF1 leverage at one alpha; cluster-bootstrap the mean diff."""
    piv = df_alpha.pivot_table(index="complex_id", columns="direction", values="meanL_esmif")
    piv = piv.dropna(subset=["L", "random"])
    d = (piv["L"] - piv["random"]).to_numpy()
    ids = piv.index.to_numpy()
    point = float(np.mean(d))
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(nboot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), \
        float(np.mean(np.array(boots) > 0)), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/cfg_steer.csv")
    ap.add_argument("--out", default="results/cfg_steer_summary.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.inp)
    alphas = sorted(df.alpha.unique())
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"[cfg-analyse] {df.complex_id.nunique()} complexes, alphas={alphas}, "
          f"K rows={len(df)}\n")

    # per (direction, alpha): ESM-IF1 leverage, interface recovery, non-interface recovery, MPNN L
    for direction in ("L", "random"):
        for al in alphas:
            g = df[(df.direction == direction) & (df.alpha == al)]
            le, le_lo, le_hi, n, drp = cluster_boot_mean(g.meanL_esmif, g.complex_id, rng)
            ir, ir_lo, ir_hi, _, _ = cluster_boot_mean(g.int_recovery, g.complex_id, rng)
            nr, nr_lo, nr_hi, _, _ = cluster_boot_mean(g.noninterface_recovery, g.complex_id, rng)
            lm = float(np.nanmean(g.meanL_mpnn))
            rows.append(dict(direction=direction, alpha=al, n_cx=n,
                             meanL_esmif=round(le, 5), esmif_lo=round(le_lo, 5), esmif_hi=round(le_hi, 5),
                             int_recovery=round(ir, 4), int_rec_lo=round(ir_lo, 4), int_rec_hi=round(ir_hi, 4),
                             noninterface_recovery=round(nr, 4), noni_lo=round(nr_lo, 4), noni_hi=round(nr_hi, 4),
                             meanL_mpnn=round(lm, 4), boot_dropped=drp))

    # paired L - random specificity contrast per alpha (the anti-magnitude control)
    print("=== ESM-IF1 leverage: L arm vs random control (complex-clustered 95% CI) ===")
    for al in alphas:
        gL = [r for r in rows if r["direction"] == "L" and r["alpha"] == al][0]
        gR = [r for r in rows if r["direction"] == "random" and r["alpha"] == al][0]
        dpt, dlo, dhi, pgt, npair = paired_diff_boot(df[df.alpha == al], rng)
        rows.append(dict(direction="L_minus_random", alpha=al, n_cx=npair,
                         meanL_esmif=round(dpt, 5), esmif_lo=round(dlo, 5), esmif_hi=round(dhi, 5),
                         p_gt0=round(pgt, 4)))
        print(f"  a={al:<5}  L={gL['meanL_esmif']:+.4f}[{gL['esmif_lo']:+.4f},{gL['esmif_hi']:+.4f}]  "
              f"random={gR['meanL_esmif']:+.4f}[{gR['esmif_lo']:+.4f},{gR['esmif_hi']:+.4f}]  "
              f"L-rand={dpt:+.4f}[{dlo:+.4f},{dhi:+.4f}] P(>0)={pgt:.3f}  "
              f"int_rec: L={gL['int_recovery']:.3f} rand={gR['int_recovery']:.3f}")

    # sweet spot: largest alpha where L-arm ESM-IF1 leverage CI>0 AND int_recovery >= 0.5 * baseline
    Lrows = {r["alpha"]: r for r in rows if r["direction"] == "L"}
    base_ir = Lrows[0.0]["int_recovery"] if 0.0 in Lrows else np.nan
    sweet = None
    for al in alphas:
        r = Lrows[al]
        if al > 0 and r["esmif_lo"] > 0 and r["int_recovery"] >= 0.5 * base_ir:
            sweet = al
    print(f"\n[sweet-spot] baseline interface recovery (a=0) = {base_ir:.3f}; "
          f"largest a with ESM-IF1 leverage CI>0 AND recovery>=50% baseline = {sweet}")

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out["nboot"] = NBOOT
    out["sweet_spot_alpha"] = sweet
    out.to_csv(a.out, index=False)
    print(f"\n[wrote] {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

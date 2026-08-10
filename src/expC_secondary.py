"""Exp C SECONDARY (PREREG_expC §6, exploratory — no falsifier attaches).

Binding-relevant readout: recompute the partial Spearman correlation between experimental SKEMPI
ΔΔG_bind and the ProteinMPNN per-mutation log-odds ℓ(mut) − ℓ(wt), controlling for burial (Cβ
neighbour count), on the partial-diffusion backbones as a function of interface-RMSD. Prediction: the
model's ability to rank experimental binding energy DEGRADES (|ρ| → 0) as the binder backbone becomes
non-native. On the crystal this partial-ρ ≈ −0.247 (hardening F1). Every generated backbone stores
lp_<AA>; the log-odds is read off directly, so this is CPU-only and reuses no extra GPU.

Usage:
  python3 src/expC_secondary.py --positions $SCRATCH/expC/scored_positions.csv \
      --backbones $SCRATCH/expC/scored_backbones.csv --out results/expC_secondary.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc  # noqa: E402

SEED, NBOOT = 20260803, 2000


def partial_spearman(x, y, z):
    """Spearman between x and y controlling for z: rank all three, linearly residualise the ranks of
    x and y on the ranks of z, correlate the residuals. Matches src/hardening.py's partial()."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 6:
        return np.nan
    rx, ry, rz = (pd.Series(v[ok]).rank().values for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(stats.spearmanr(ex, ey)[0])


def cboot_partial(df, seed=SEED, n=NBOOT):
    """Complex-level bootstrap CI of the partial Spearman over rows of df (cols: complex_id,ddG,logodds,nbr)."""
    cids = df["complex_id"].unique()
    if len(cids) < 3:
        return (np.nan, np.nan, np.nan)
    by = {c: df[df["complex_id"] == c] for c in cids}
    rng = np.random.default_rng(seed)
    est = partial_spearman(df["ddG"], df["logodds"], df["nbr"])
    vals = []
    for _ in range(n):
        s = pd.concat([by[cids[i]] for i in rng.choice(len(cids), len(cids), True)], ignore_index=True)
        vals.append(partial_spearman(s["ddG"], s["logodds"], s["nbr"]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return (est, float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--skempi", default=os.path.join(os.environ.get("FTAX_DATA", ""), "skempi_v2.csv"))
    ap.add_argument("--out", default="results/expC_secondary.csv")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    # SKEMPI single mutations -> (complex_id, chain, resnum, wt, mut, ddG)
    sk = fc.parse_skempi(a.skempi)
    sk = sk[sk["n_mut"] == 1].copy()
    recs = []
    for r in sk.itertuples():
        pm = fc.parse_mutation(r.muts[0])
        if pm is None:
            continue
        recs.append(dict(complex_id=f"{r.pdb}_{r.group1}_{r.group2}", chain=pm["chain"],
                         resnum=int(pm["resnum"]), wt=pm["wt"], mut=pm["mut"], ddG=float(r.ddG)))
    mut = pd.DataFrame(recs)
    mut = mut.groupby(["complex_id", "chain", "resnum", "wt", "mut"], as_index=False)["ddG"].mean()

    pos = pd.read_csv(a.positions)
    pos["chain"] = pos["chain"].astype(str)
    # positions CSV already carries partial_T; take only irmsd from backbones to avoid a merge collision
    bb = pd.read_csv(a.backbones)[["backbone_id", "irmsd"]]
    key = ["complex_id", "chain", "resnum"]
    m = pos.merge(mut, on=key, how="inner")
    if m.empty:
        print("[expC_secondary] no SKEMPI single-mutation positions matched the scored backbones")
        pd.DataFrame([]).to_csv(a.out, index=False)
        return
    # log-odds = lp_mut - lp_wt on each generated backbone
    m["logodds"] = [row[f"lp_{row['mut']}"] - row[f"lp_{row['wt']}"] for _, row in m.iterrows()]
    m = m.merge(bb, on="backbone_id", how="left")

    rows = []
    for T, sub in m.groupby("partial_T"):
        # average over the N samples per (complex, mutation) at this noise level
        agg = sub.groupby(["complex_id", "chain", "resnum"], as_index=False).agg(
            ddG=("ddG", "first"), logodds=("logodds", "mean"), nbr=("nbr", "mean"))
        est, lo, hi = cboot_partial(agg)
        ir = sub["irmsd"].median()
        rows.append(dict(partial_T=int(T), median_irmsd=float(ir), n_mut=len(agg),
                         n_cx=agg["complex_id"].nunique(), partial_rho=est, lo=lo, hi=hi))
        print(f"  [ddG-corr T={T:2d}] iRMSD~{ir:.2f}  partial-rho={est:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"(n_mut={len(agg)}, n_cx={agg['complex_id'].nunique()})")

    pd.DataFrame(rows).assign(command=cmd).to_csv(a.out, index=False)
    print(f"[expC_secondary] wrote {a.out}")


if __name__ == "__main__":
    main()

"""Experiment A, Analysis 1: burial-matched hotspot gap on predicted backbones.

REUSES the committed pydssp matched pairs (results/p0_dssp_pairs_{VARIANT}.csv), keyed by
(complex_id, chain, resnum), and looks up each hot/control residue's ProteinMPNN native
log-prob on a PREDICTED-backbone positions CSV and on the CRYSTAL positions CSV. Reports,
per variant, with complex-level bootstrap (10,000 reps, seed 20260803):

  d_pred   = mean_pairs[ logp(hot|pred) - logp(ctl|pred) ]          gap on predicted backbone
  d_cry    = mean_pairs[ logp(hot|cry)  - logp(ctl|cry)  ]          gap on crystal (this env)
  delta    = mean_pairs[ d_pred - d_cry ]                            paired predicted-minus-crystal

A negative gap is the hypothesised direction (hotspots harder). d_cry must reproduce the
committed pair `d_logp` column (env-fidelity self-check). If --confidence is given, every
headline is additionally split at the median of a chosen per-complex confidence metric.

Usage:
  python3 src/expA_gap_reuse_pairs.py \
      --pred-positions results/expA_p0_positions.csv \
      --crystal-positions results/p0_positions.csv \
      --pairs-glob 'results/p0_dssp_pairs_*.csv' \
      --confidence results/expA_confidence.csv --out results/expA_gap
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SEED_BOOT = 20260803
N_BOOT = 10000


def pos_lookup(csv, col="logp_native"):
    """Map (complex_id, chain, resnum) -> col value; blank-icode row wins on collision."""
    df = pd.read_csv(csv, usecols=lambda c: c in
                     ("complex_id", "chain", "resnum", "icode", col))
    df["icode"] = df["icode"].fillna("").astype(str)
    df = df.sort_values("icode").drop_duplicates(
        subset=["complex_id", "chain", "resnum"], keep="first")
    return {(r.complex_id, r.chain, int(r.resnum)): getattr(r, col)
            for r in df.itertuples()}


def complex_bootstrap(df, col, n_boot=N_BOOT, seed=SEED_BOOT):
    """Percentile CI resampling COMPLEXES (the independent unit)."""
    d = df[np.isfinite(df[col])]
    if d.empty:
        return dict(n_pairs=0, n_complexes=0, mean=np.nan, lo=np.nan, hi=np.nan, boot_sd=np.nan)
    rng = np.random.default_rng(seed)
    cids = d["complex_id"].unique()
    by = {c: d.loc[d["complex_id"] == c, col].values for c in cids}
    means = np.array([np.nanmean(np.concatenate([by[cids[i]] for i in
                     rng.choice(len(cids), len(cids), True)])) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return dict(n_pairs=len(d), n_complexes=len(cids), mean=float(np.nanmean(d[col])),
                lo=float(lo), hi=float(hi), boot_sd=float(np.nanstd(means)))


def build_pair_frame(pairs, pred_lp, cry_lp):
    """Per pair: predicted gap, crystal-here gap, committed gap (from the pairs file)."""
    rows = []
    for r in pairs.itertuples():
        h = (r.complex_id, r.hot_chain, int(r.hot_resnum))
        c = (r.complex_id, r.ctl_chain, int(r.ctl_resnum))
        dp = pred_lp.get(h, np.nan) - pred_lp.get(c, np.nan)
        dc = cry_lp.get(h, np.nan) - cry_lp.get(c, np.nan)
        rows.append(dict(complex_id=r.complex_id, d_pred=dp, d_cry=dc,
                         d_committed=getattr(r, "d_logp", np.nan)))
    out = pd.DataFrame(rows)
    out["delta"] = out["d_pred"] - out["d_cry"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-positions", required=True)
    ap.add_argument("--crystal-positions", default="results/p0_positions.csv")
    ap.add_argument("--pairs-glob", default="results/p0_dssp_pairs_*.csv")
    ap.add_argument("--confidence", default=None,
                    help="per-complex CSV with complex_id + confidence metrics")
    ap.add_argument("--strat-metrics", default="plddt_iface,ptm,rmsd_iface_ca")
    ap.add_argument("--out", default="results/expA_gap")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    pred_lp = pos_lookup(a.pred_positions, "logp_native")
    cry_lp = pos_lookup(a.crystal_positions, "logp_native")
    print(f"[gap] predicted positions: {len(pred_lp)} residues; crystal: {len(cry_lp)}")

    conf = None
    if a.confidence and os.path.exists(a.confidence):
        conf = pd.read_csv(a.confidence)
        print(f"[gap] confidence for {len(conf)} complexes, cols={list(conf.columns)}")

    summary = []
    for path in sorted(glob.glob(a.pairs_glob)):
        variant = os.path.basename(path).replace("p0_dssp_pairs_", "").replace(".csv", "")
        pairs = pd.read_csv(path)
        pf = build_pair_frame(pairs, pred_lp, cry_lp)
        n_have = int(np.isfinite(pf["d_pred"]).sum())
        n_drop = int(len(pf) - n_have)

        rp = complex_bootstrap(pf, "d_pred")
        rc = complex_bootstrap(pf, "d_cry")
        rd = complex_bootstrap(pf, "delta")
        cry_ref = float(np.nanmean(pf["d_committed"]))
        selfchk = float(np.nanmax(np.abs(pf["d_cry"] - pf["d_committed"]))) if n_have else np.nan
        print(f"\n[{variant}]  pairs {n_have} (+{n_drop} dropped)  complexes {rp['n_complexes']}")
        print(f"   d_pred  {rp['mean']:+.4f} [{rp['lo']:+.4f},{rp['hi']:+.4f}]")
        print(f"   d_cry   {rc['mean']:+.4f} [{rc['lo']:+.4f},{rc['hi']:+.4f}]  "
              f"(committed {cry_ref:+.4f}; self-check max|Δ|={selfchk:.2e})")
        print(f"   delta   {rd['mean']:+.4f} [{rd['lo']:+.4f},{rd['hi']:+.4f}]  (predicted - crystal)")
        summary.append(dict(variant=variant, arm="all", n_pairs=n_have, n_dropped=n_drop,
                            n_complexes=rp["n_complexes"], d_pred=rp["mean"],
                            d_pred_lo=rp["lo"], d_pred_hi=rp["hi"], d_cry=rc["mean"],
                            d_cry_lo=rc["lo"], d_cry_hi=rc["hi"], d_committed=cry_ref,
                            selfcheck_maxabs=selfchk, delta=rd["mean"],
                            delta_lo=rd["lo"], delta_hi=rd["hi"]))

        # confidence stratification (median split per metric) for the two verdict tiers
        if conf is not None and variant in ("PRIMARY_loose_null", "SECONDARY_B_any_interface"):
            pfc = pf.merge(conf, on="complex_id", how="left")
            for metric in [m for m in a.strat_metrics.split(",") if m in pfc.columns]:
                med = pfc[metric].median()
                for side, sel in (("hi", pfc[metric] >= med), ("lo", pfc[metric] < med)):
                    sub = pfc[sel]
                    rpp = complex_bootstrap(sub, "d_pred")
                    rdd = complex_bootstrap(sub, "delta")
                    summary.append(dict(variant=variant, arm=f"{metric}_{side}(med={med:.3g})",
                                        n_pairs=int(np.isfinite(sub['d_pred']).sum()),
                                        n_complexes=rpp["n_complexes"], d_pred=rpp["mean"],
                                        d_pred_lo=rpp["lo"], d_pred_hi=rpp["hi"],
                                        delta=rdd["mean"], delta_lo=rdd["lo"], delta_hi=rdd["hi"]))

    pd.DataFrame(summary).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"\n[done] wrote {a.out}_summary.csv")


if __name__ == "__main__":
    main()

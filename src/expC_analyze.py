"""Exp C: dose-response analysis — burial-matched gap + KL ΔAUROC vs backbone drift.

Consumes the per-backbone positions/backbones CSVs from src/expC_score.py. For each committed pydssp
pair (restricted to pairs with BOTH positions on the diffused binder), computes d = logp(hot)-logp(ctl)
on every generated backbone, aggregates per complex (mean over that complex's N samples), and reports
the SECONDARY_B gap and the KL ΔAUROC-over-burial (burial = Cβ neighbour count) as a function of
partial_T and of interface-RMSD bin, with complex-level bootstrap. partial_T=0 must reproduce the
committed crystal ~0 deficit (KILL C1). Confound control: restrict to well-formed-interface backbones.

Usage:
  python3 src/expC_analyze.py --positions $SCRATCH/expC/scored_positions.csv \
      --backbones $SCRATCH/expC/scored_backbones.csv --pairs-glob 'results/p0_dssp_pairs_*.csv' \
      --binder-map results/expC_complexes.csv --out results/expC
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SEED, NBOOT = 20260803, 10000


def cboot(df, col, n=NBOOT, seed=SEED):
    """Complex-level bootstrap of the mean of per-complex values in df[col]."""
    d = df[np.isfinite(df[col])]
    if d["complex_id"].nunique() < 2:
        return (np.nan, np.nan, np.nan, len(d))
    rng = np.random.default_rng(seed)
    cids = d["complex_id"].unique()
    by = {c: d.loc[d["complex_id"] == c, col].values for c in cids}
    means = np.array([np.nanmean(np.concatenate([by[cids[i]] for i in rng.choice(len(cids), len(cids), True)]))
                      for _ in range(n)])
    return (float(np.nanmean(d[col])), float(np.nanpercentile(means, 2.5)),
            float(np.nanpercentile(means, 97.5)), int(d["complex_id"].nunique()))


def auc(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    r = pd.Series(s).rank().values
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def paired_dauroc(df, seed=SEED, nboot=2000):
    """ΔAUROC(burial+KL − burial) with complex bootstrap; burial = -nbr, label = is_hot."""
    df = df[np.isfinite(df["kl"]) & np.isfinite(df["nbr"])].copy()
    if df["is_hot"].sum() < 5 or df["complex_id"].nunique() < 3:
        return None
    df["burial"] = df["nbr"].astype(float)
    df["bk"] = 0.5 * (df["burial"].rank() / len(df) + df["kl"].rank() / len(df))
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df[df["complex_id"] == c] for c in cids}
    a0 = auc(df["burial"].values, df["is_hot"].values)
    a1 = auc(df["bk"].values, df["is_hot"].values)
    dd = []
    for _ in range(nboot):
        s = pd.concat([by[cids[i]] for i in rng.choice(len(cids), len(cids), True)], ignore_index=True)
        dd.append(auc(s["bk"].values, s["is_hot"].values) - auc(s["burial"].values, s["is_hot"].values))
    # [C2 hardening] nan-aware: degenerate resamples (a bootstrap complex draw with no hotspot, or a
    # constant score) give AUROC nan; count them and compute p_gt0 over finite reps only. Strictly more
    # correct than treating nan as "not >0"; leaves non-degenerate Exp C numbers unchanged.
    dd = np.array(dd, float)
    fin = np.isfinite(dd)
    frac_degen = float(1.0 - fin.mean()) if len(dd) else np.nan
    lo, hi = np.nanpercentile(dd, [2.5, 97.5])
    p_gt0 = float(np.mean(dd[fin] > 0)) if fin.any() else np.nan
    return dict(auc_burial=a0, auc_bk=a1, dauroc=a1 - a0, lo=float(lo), hi=float(hi),
                p_gt0=p_gt0, frac_degen=frac_degen, n=len(df), n_cx=len(cids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--pairs-glob", default="results/p0_dssp_pairs_*.csv")
    ap.add_argument("--binder-map", default="results/expC_complexes.csv")
    ap.add_argument("--variant", default="SECONDARY_B_any_interface")
    ap.add_argument("--out", default="results/expC")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    pos = pd.read_csv(a.positions)
    pos["icode"] = pos["icode"].fillna("").astype(str)
    bbm = pd.read_csv(a.backbones)[["backbone_id", "partial_T", "irmsd", "interface_ok"]]
    binder = pd.read_csv(a.binder_map).set_index("complex_id")["binder_chains"].astype(str).to_dict()

    # logp lookup per backbone
    lp = {(r.backbone_id, r.complex_id, r.chain, int(r.resnum)): r.logp_native for r in pos.itertuples()}

    # ---- GAP dose-response (restricted to within-binder pairs) ----
    pf = os.path.join("results", f"p0_dssp_pairs_{a.variant}.csv")
    pairs = pd.read_csv(pf)
    per_bb = []
    for bbid in pos["backbone_id"].unique():
        cid = "_".join(bbid.split("_")[:-2])
        bset = set(binder.get(cid, ""))
        p = pairs[pairs.complex_id == cid]
        ds = []
        for r in p.itertuples():
            if str(r.hot_chain) not in bset or str(r.ctl_chain) not in bset:
                continue
            dh = lp.get((bbid, cid, str(r.hot_chain), int(r.hot_resnum)))
            dc = lp.get((bbid, cid, str(r.ctl_chain), int(r.ctl_resnum)))
            if dh is not None and dc is not None:
                ds.append(dh - dc)
        if ds:
            per_bb.append(dict(backbone_id=bbid, complex_id=cid, d=float(np.mean(ds)), n_pairs=len(ds)))
    G = pd.DataFrame(per_bb).merge(bbm, on="backbone_id", how="left")
    # per complex per T (average samples), then bootstrap over complexes
    rows = []
    for T, sub in G.groupby("partial_T"):
        percx = sub.groupby("complex_id")["d"].mean().reset_index()
        m, lo, hi, ncx = cboot(percx, "d")
        ir = bbm[bbm.partial_T == T]["irmsd"].median()
        rows.append(dict(kind="gap_by_T", partial_T=int(T), median_irmsd=float(ir),
                         mean_d=m, lo=lo, hi=hi, n_cx=ncx, n_bb=len(sub)))
        print(f"  [gap T={T:2d}] iRMSD~{ir:.2f}  d={m:+.3f} [{lo:+.3f},{hi:+.3f}]  (n_cx={ncx})")
    # by iRMSD bin
    G["irmsd_bin"] = pd.cut(G["irmsd"], [-0.01, 1, 2, 3, 5, 10, np.inf],
                            labels=["<1", "1-2", "2-3", "3-5", "5-10", ">10"])
    for b, sub in G.groupby("irmsd_bin", observed=True):
        percx = sub.groupby("complex_id")["d"].mean().reset_index()
        m, lo, hi, ncx = cboot(percx, "d")
        rows.append(dict(kind="gap_by_irmsd", irmsd_bin=str(b), mean_d=m, lo=lo, hi=hi, n_cx=ncx, n_bb=len(sub)))
        print(f"  [gap iRMSD {str(b):>4}] d={m:+.3f} [{lo:+.3f},{hi:+.3f}]  (n_cx={ncx}, n_bb={len(sub)})")
    # confound / pre-reg "drop dissolved": well-formed interface only (>=5 hotspot inter-chain contacts).
    # The RFdiffusion binder diffusion frequently DIVERGES at higher partial_T (binder coords blow up to
    # 1e3-1e7 A, interface gone); those are numerical failures, not "non-native backbones", and pooling
    # them buries the signal. The interface-formed subset is the physically-meaningful dose-response.
    Gok = G[G.interface_ok == 1]
    for T, sub in Gok.groupby("partial_T"):
        percx = sub.groupby("complex_id")["d"].mean().reset_index()
        m, lo, hi, ncx = cboot(percx, "d")
        ir = Gok[Gok.partial_T == T]["irmsd"].median()
        rows.append(dict(kind="gap_by_T_ifaceok", partial_T=int(T), median_irmsd=float(ir),
                         mean_d=m, lo=lo, hi=hi, n_cx=ncx, n_bb=len(sub)))
        print(f"  [gapOK T={T:2d}] iRMSD~{ir:.2f}  d={m:+.3f} [{lo:+.3f},{hi:+.3f}]  (n_cx={ncx}, n_bb={len(sub)})")
    for b, sub in Gok.groupby("irmsd_bin", observed=True):
        percx = sub.groupby("complex_id")["d"].mean().reset_index()
        m, lo, hi, ncx = cboot(percx, "d")
        rows.append(dict(kind="gap_by_irmsd_ifaceok", irmsd_bin=str(b), mean_d=m, lo=lo, hi=hi, n_cx=ncx, n_bb=len(sub)))
        print(f"  [gapOK iRMSD {str(b):>4}] d={m:+.3f} [{lo:+.3f},{hi:+.3f}]  (n_cx={ncx}, n_bb={len(sub)})")

    # ---- KL ΔAUROC by partial_T (label = strict hotspot; needs is_hot) ----
    # Restrict to INTERFACE positions (as the committed crystal +0.048 was), from the committed set.
    iface = set()
    icsv = "results/p0_dssp_interface_resid.csv"
    if os.path.exists(icsv):
        idf = pd.read_csv(icsv, usecols=lambda c: c in ("complex_id", "chain", "resnum", "is_interface"))
        if "is_interface" in idf.columns:
            idf = idf[idf["is_interface"] == True]
        iface = {(r.complex_id, str(r.chain), int(r.resnum)) for r in idf.itertuples()}
    if iface:
        pos = pos[[(r.complex_id, r.chain, int(r.resnum)) in iface for r in pos.itertuples()]].copy()
    # is_hot from strict pairs' hot positions
    strict = pd.read_csv("results/p0_dssp_pairs_strict_hot2_null.csv")
    hotpos = {(r.complex_id, str(r.hot_chain), int(r.hot_resnum)) for r in strict.itertuples()}
    pos["is_hot"] = [int((r.complex_id, r.chain, int(r.resnum)) in hotpos) for r in pos.itertuples()]
    okbb = set(bbm[bbm.interface_ok == 1]["backbone_id"])
    for label, psub in [("kl_by_T", pos), ("kl_by_T_ifaceok", pos[pos["backbone_id"].isin(okbb)])]:
        for T, sub in psub.groupby("partial_T"):
            # average per (complex,residue) over samples
            agg = sub.groupby(["complex_id", "chain", "resnum"]).agg(
                kl=("kl", "mean"), nbr=("nbr", "mean"), is_hot=("is_hot", "max")).reset_index()
            r = paired_dauroc(agg)
            if r:
                rows.append(dict(kind=label, partial_T=int(T), **r))
                print(f"  [{label} T={T:2d}] burial {r['auc_burial']:.3f}  b+KL {r['auc_bk']:.3f}  "
                      f"ΔAUROC {r['dauroc']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] (n_cx={r['n_cx']})")

    pd.DataFrame(rows).assign(command=cmd).to_csv(f"{a.out}_dose.csv", index=False)
    G.to_csv(f"{a.out}_gap_perbackbone.csv", index=False)
    print(f"\n[expC_analyze] wrote {a.out}_dose.csv")


if __name__ == "__main__":
    main()

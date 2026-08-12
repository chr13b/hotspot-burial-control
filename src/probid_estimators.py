#!/usr/bin/env python3
"""#4-full CORRECTION: estimator-sensitivity for ProBID-Net's hotspot recovery gap.

Supersedes the interpretation of the `uncontrolled_hot_minus_nonhot` row in
results/probid_gap.csv. A Fable-5 adversarial audit (2026-08-12) found, and this script
independently reproduces from committed CSVs, that the headline "+0.098 opposite-sign"
uncontrolled gap was two artifacts:

  1. COMPLEX-AVERAGING. ProBID-Net's published 0.334/0.472 is a per-residue pooled number;
     our +0.098 was averaged over complexes and is carried by the 52/96 complexes with only
     1-2 measured hotspots. Weighted by hotspot count (like-for-like) the gap is +0.014
     [-0.052,+0.087] (NULL), and in comprehensively-Ala-scanned complexes (>=5 measured
     hotspots, the regime ASEdb/BID represent) their NEGATIVE deficit REPRODUCES:
     -0.113 [-0.208,-0.022], p=0.007. So "we could not reproduce their number" is WITHDRAWN.

  2. AMINO-ACID COMPOSITION. ProBID recall runs 0.17 (R) to 0.98 (P); hotspots are enriched
     in its worst residue types (WYFRMH 47% vs 22%) and depleted in its best (GP 3% vs 12%).
     The burial-matched pairs are 93% AA-mismatched, so -0.038 is an AA-composition comparison,
     not a clean burial control. On the pre-registered AA-matched pairs the gap is +0.120
     [-0.060,+0.300] (n_cx=25). No matched variant (burial / AA / hydrophobicity) shows a
     SIGNIFICANT negative deficit -> the deficit is a composition+burial confound, consistent
     with the thesis, but #4-full is NOT the clean "sixth-architecture burial-artifact" result
     originally claimed.

Reads only committed CSVs. Writes results/probid_gap_estimators.csv.
  python3 src/probid_estimators.py --out results/probid_gap_estimators.csv
"""
import argparse, glob
import numpy as np, pandas as pd

SEED = 20260803
AAs = list("ACDEFGHIKLMNPQRSTVWY")


def cboot(vals_by_cx, w=None, nboot=10000):
    rng = np.random.default_rng(SEED)
    cids = list(vals_by_cx)
    if w is None:
        w = {c: 1.0 for c in cids}

    def m(cs):
        return float(np.sum([vals_by_cx[c] * w[c] for c in cs]) / np.sum([w[c] for c in cs]))

    obs = m(cids)
    b = np.array([m(rng.choice(cids, len(cids), True)) for _ in range(nboot)])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return obs, float(lo), float(hi), float(np.mean(b > 0)), len(cids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/probid_gap_estimators.csv")
    a = ap.parse_args()
    R = a.results
    rows = []

    pos = pd.read_csv(f"{R}/probid_positions.csv")
    pos["resnum"] = pos["resnum"].astype(str)
    rec = {(r.pdb, r.chain, r.resnum): r.recovered for r in pos.itertuples()}

    kl = pd.read_csv(f"{R}/kl_detector_joined.csv")
    kl = kl[kl.is_interface == 1].copy()
    kl["pdb"] = kl.complex_id.str.split("_").str[0]
    kl["resnum"] = kl.resnum.astype(str)
    kl["rec"] = [rec.get((p, c, r)) for p, c, r in zip(kl.pdb, kl.chain, kl.resnum)]
    kk = kl.dropna(subset=["rec"]).copy()

    hot = {c: g[g.is_hot == 1].rec.mean() for c, g in kk.groupby("complex_id") if (g.is_hot == 1).any()}
    non = {c: g[g.is_hot == 0].rec.mean() for c, g in kk.groupby("complex_id") if (g.is_hot == 0).any()}
    nhot = {c: int((g.is_hot == 1).sum()) for c, g in kk.groupby("complex_id")}
    both = [c for c in hot if c in non]
    gap_cx = {c: hot[c] - non[c] for c in both}

    def add(name, g, lo, hi, p, n, extra=""):
        rows.append(dict(analysis=name, gap=round(g, 4), lo=round(lo, 4), hi=round(hi, 4),
                         p_gt0=round(p, 3), n_cx=n, note=extra))
        print(f"  {name:42s} {g:+.4f} [{lo:+.4f},{hi:+.4f}] p>0={p:.3f} n_cx={n} {extra}")

    print("=== uncontrolled estimators ===")
    add("uncontrolled_complex_avg_CONFOUNDED", *cboot(gap_cx), "artifact: 1-2-hotspot complexes dominate")
    rp = kk[kk.is_hot == 1].rec.mean() - kk[kk.is_hot == 0].rec.mean()
    rows.append(dict(analysis="uncontrolled_residue_pooled", gap=round(rp, 4), lo="", hi="",
                     p_gt0="", n_cx="", note=f"hot {kk[kk.is_hot==1].rec.mean():.4f}/non {kk[kk.is_hot==0].rec.mean():.4f}"))
    print(f"  uncontrolled_residue_pooled                {rp:+.4f}")
    add("uncontrolled_hotspot_weighted_LIKE4LIKE", *cboot(gap_cx, w=nhot), "matches their per-residue reporting -> NULL")
    for thr in (2, 3, 5):
        sub = {c: gap_cx[c] for c in both if nhot[c] >= thr}
        if len(sub) >= 3:
            add(f"uncontrolled_strat_nhot_ge{thr}", *cboot(sub),
                f"comprehensively-scanned; hotspots={sum(nhot[c] for c in sub)}")

    print("=== burial/AA/hydro-matched gaps (each committed pairs file) ===")
    for pf in sorted(glob.glob(f"{R}/p0_dssp_pairs_*.csv")):
        pr = pd.read_csv(pf)
        if not {"complex_id", "hot_chain", "hot_resnum", "ctl_chain", "ctl_resnum"}.issubset(pr.columns):
            continue
        pr["pdb"] = pr.complex_id.str.split("_").str[0]
        same_aa = (pr.hot_aa == pr.ctl_aa).mean() if {"hot_aa", "ctl_aa"}.issubset(pr.columns) else np.nan
        d = {}
        for r in pr.itertuples():
            rh = rec.get((r.pdb, str(r.hot_chain), str(r.hot_resnum)))
            rc = rec.get((r.pdb, str(r.ctl_chain), str(r.ctl_resnum)))
            if rh is not None and rc is not None:
                d.setdefault(r.complex_id, []).append(rh - rc)
        if not d:
            continue
        percx = {c: float(np.mean(v)) for c, v in d.items()}
        add(f"matched::{pf.split('/')[-1]}", *cboot(percx), f"same_native_aa={same_aa:.2f}")

    print("=== ProBID recall by native residue type ===")
    rr = pos.groupby("native_aa").recovered.agg(["mean", "size"]).reindex(AAs)
    for aa in AAs:
        if not np.isnan(rr.loc[aa, "mean"]):
            rows.append(dict(analysis=f"recall_restype_{aa}", gap=round(float(rr.loc[aa, "mean"]), 4),
                             lo="", hi="", p_gt0="", n_cx=int(rr.loc[aa, "size"]), note="ProBID recall"))
    print("  " + " ".join(f"{a}:{rr.loc[a,'mean']:.2f}" for a in AAs if not np.isnan(rr.loc[a, "mean"])))

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out["command"] = f"python3 src/probid_estimators.py --out {a.out}"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

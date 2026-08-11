"""Exp C2 KILL C2b — the conditioning leakage control.

PREREG_expC2 §5. The burial-matched gap d = logp(hot) - logp(ctl) is computed on the C2 backbones and
split by which arm the HOT position falls in: `conditioned` (its target contacts were passed to
ppi.hotspot_res) vs `heldout` (never passed). If the geometry conditioning leaked interface information
into sequence recovery, conditioned hotspots would show a SMALLER deficit than held-out. Pre-registered
null: the two arms are statistically indistinguishable (paired-per-complex complex-bootstrap CI of
conditioned - heldout contains zero). Computed on interface-FORMED, noised (partial_T!=0) backbones,
within-binder SECONDARY_B pairs (both hot & ctl on the diffused binder), same tier as C2-PRIMARY.

Usage:
  python3 src/expC2_leakage.py --positions $SCRATCH/expC2/scored_positions.csv \
      --backbones $SCRATCH/expC2/scored_backbones.csv --split results/expC2_hotspot_split.csv \
      --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
      --binder-map results/expC_complexes.csv --out results/expC2_leakage.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SEED, NBOOT = 20260803, 10000


def cboot_mean(values_by_cx, seed=SEED, n=NBOOT):
    """Complex-level bootstrap of the mean of per-complex scalars. values_by_cx: {cid: value}."""
    cids = [c for c, v in values_by_cx.items() if np.isfinite(v)]
    if len(cids) < 2:
        return (np.nanmean([values_by_cx[c] for c in cids]) if cids else np.nan, np.nan, np.nan, len(cids))
    arr = np.array([values_by_cx[c] for c in cids], float)
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.choice(len(arr), len(arr), True)].mean() for _ in range(n)])
    return (float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(cids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--split", default="results/expC2_hotspot_split.csv")
    ap.add_argument("--pairs", default="results/p0_dssp_pairs_SECONDARY_B_any_interface.csv")
    ap.add_argument("--binder-map", default="results/expC_complexes.csv")
    ap.add_argument("--out", default="results/expC2_leakage.csv")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    pos = pd.read_csv(a.positions)
    pos["chain"] = pos["chain"].astype(str)
    lp = {(r.backbone_id, r.chain, int(r.resnum)): r.logp_native for r in pos.itertuples()}
    bbm = pd.read_csv(a.backbones)[["backbone_id", "complex_id", "partial_T", "interface_ok"]]
    binder = pd.read_csv(a.binder_map).set_index("complex_id")["binder_chains"].astype(str).to_dict()
    pairs = pd.read_csv(a.pairs)
    arm = {(r.complex_id, str(r.chain), int(r.resnum)): r.arm for r in pd.read_csv(a.split).itertuples()}

    # interface-formed, noised backbones only
    use = bbm[(bbm.interface_ok == 1) & (bbm.partial_T != 0)]
    # per backbone: mean gap over within-binder pairs, split by the HOT arm
    per_bb = []
    for r in use.itertuples():
        bbid, cid = r.backbone_id, r.complex_id
        bset = set(binder.get(cid, ""))
        p = pairs[pairs.complex_id == cid]
        acc = {"conditioned": [], "heldout": []}
        for pr in p.itertuples():
            if str(pr.hot_chain) not in bset or str(pr.ctl_chain) not in bset:
                continue
            ar = arm.get((cid, str(pr.hot_chain), int(pr.hot_resnum)))
            if ar not in acc:
                continue
            dh = lp.get((bbid, str(pr.hot_chain), int(pr.hot_resnum)))
            dc = lp.get((bbid, str(pr.ctl_chain), int(pr.ctl_resnum)))
            if dh is not None and dc is not None:
                acc[ar].append(dh - dc)
        per_bb.append(dict(backbone_id=bbid, complex_id=cid,
                           d_cond=np.mean(acc["conditioned"]) if acc["conditioned"] else np.nan,
                           d_held=np.mean(acc["heldout"]) if acc["heldout"] else np.nan))
    B = pd.DataFrame(per_bb)

    # per complex: mean over that complex's interface-formed noised backbones, per arm
    cond_cx = {c: np.nanmean(v["d_cond"]) for c, v in B.groupby("complex_id") if np.isfinite(v["d_cond"]).any()}
    held_cx = {c: np.nanmean(v["d_held"]) for c, v in B.groupby("complex_id") if np.isfinite(v["d_held"]).any()}

    rows = []
    # pooled (unpaired) arm gaps -- more powered
    for name, byc in (("conditioned_pooled", cond_cx), ("heldout_pooled", held_cx)):
        m, lo, hi, ncx = cboot_mean(byc)
        rows.append(dict(kind=name, mean=m, lo=lo, hi=hi, n_cx=ncx))
        print(f"  [{name:20s}] gap={m:+.3f} [{lo:+.3f},{hi:+.3f}]  (n_cx={ncx})")

    # paired (per-complex) difference conditioned - heldout, on complexes with BOTH arms
    both = sorted(set(cond_cx) & set(held_cx))
    diff_by = {c: cond_cx[c] - held_cx[c] for c in both}
    m, lo, hi, ncx = cboot_mean(diff_by)
    contains_zero = (lo <= 0 <= hi) if np.isfinite(lo) and np.isfinite(hi) else None
    rows.append(dict(kind="paired_cond_minus_held", mean=m, lo=lo, hi=hi, n_cx=ncx))
    print(f"  [paired cond-held]      diff={m:+.3f} [{lo:+.3f},{hi:+.3f}]  (n_cx={ncx}) "
          f"CI∋0={contains_zero}  -> KILL C2b {'PASS (indistinguishable)' if contains_zero else 'CHECK'}")

    pd.DataFrame(rows).assign(command=cmd).to_csv(a.out, index=False)
    print(f"[expC2_leakage] wrote {a.out}  (paired n_cx={ncx}; pooled cond n_cx={len(cond_cx)}, "
          f"held n_cx={len(held_cx)})")


if __name__ == "__main__":
    main()

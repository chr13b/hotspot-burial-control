#!/usr/bin/env python3
"""Determinism control for the ipTM confirmation: same wt sequence folded under 3 AF2 seeds ->
within-complex ipTM SD (the model's own noise floor, so an effect smaller than it is not over-read).
Also prints the per-complex wt/L/random ipTM for whatever has folded so far (descriptive)."""
import glob
import json
import os
import numpy as np
import pandas as pd

S = os.environ["SCRATCH"]


def seed_iptms(d):
    js = sorted(glob.glob(d + "/*_scores_rank_*.json"))
    ip = [json.load(open(j)).get("iptm") for j in js]
    return [x for x in ip if x is not None]


def main():
    print("=== DETERMINISM (wt sequence, num-seeds=3) ===")
    sds = []
    for d in sorted(glob.glob(f"{S}/ftax/iptm/det_out/*__wt")):
        ip = seed_iptms(d)
        if len(ip) >= 2:
            cid = os.path.basename(d)[:-4]
            print(f"  {cid:14s} ipTM seeds={[round(x, 3) for x in ip]} SD={np.std(ip):.4f}")
            sds.append(np.std(ip))
    if sds:
        print(f"  mean within-complex ipTM SD across seeds = {np.mean(sds):.4f} (n={len(sds)} complexes)")

    det_rows = []
    for d in sorted(glob.glob(f"{S}/ftax/iptm/det_out/*__wt")):
        ip = seed_iptms(d)
        if len(ip) >= 2:
            det_rows.append(dict(complex_id=os.path.basename(d)[:-4], n_seeds=len(ip),
                                 iptm_mean=round(float(np.mean(ip)), 4), iptm_sd=round(float(np.std(ip)), 4)))
    pd.DataFrame(det_rows).to_csv("results/iptm_determinism.csv", index=False)

    print("\n=== per-complex mean-over-k ipTM (folded so far) ===")
    d = pd.read_csv("results/iptm_steer.csv")
    piv = d.groupby(["complex_id", "direction"]).iptm.mean().unstack("direction")
    for c in ("wt", "L", "random"):
        if c not in piv:
            piv[c] = np.nan
    piv = piv.dropna(subset=["wt", "L", "random"])
    for cid, r in piv.iterrows():
        print(f"  {cid:14s} wt={r['wt']:.3f} L={r['L']:.3f} random={r['random']:.3f}  "
              f"L-random={r['L'] - r['random']:+.3f}  L-wt={r['L'] - r['wt']:+.3f}")
    if len(piv):
        dlr = (piv["L"] - piv["random"])
        print(f"  n_analyzable={len(piv)}  mean L-random ipTM={dlr.mean():+.4f}  "
              f"({int((dlr > 0).sum())}/{len(piv)} complexes favour L)")


if __name__ == "__main__":
    main()

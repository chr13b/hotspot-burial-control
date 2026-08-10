"""Experiment C setup: fix the L<=400 complex set, binder/target assignment, and residue maps.

For each pair complex in results/pair_complexes.txt: load the crystal, compute total length L and
per-group lengths; gather the committed labelled positions (union over all p0_dssp_pairs_*.csv) and
assign the BINDER = the chain group carrying the labelled hot/control positions (the group to diffuse),
TARGET = the other group (held fixed as motif). Keep L<=400. Emit:
  --out-csv  results/expC_complexes.csv  (complex_id, binder_chains, target_chains, L, L_binder, ...)
  --out-map  results/expC_resmap.json    ({cid: {chain: [[resnum, icode, aa], ...ordered]}})

Usage:
  python3 src/expC_setup.py --data-dir $FTAX_DATA --lmax 400
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", default="results/pair_complexes.txt")
    ap.add_argument("--data-dir", default=os.environ.get("FTAX_DATA", os.path.expanduser("~/ftax/data")))
    ap.add_argument("--pairs-glob", default="results/p0_dssp_pairs_*.csv")
    ap.add_argument("--lmax", type=int, default=400)
    ap.add_argument("--out-csv", default="results/expC_complexes.csv")
    ap.add_argument("--out-map", default="results/expC_resmap.json")
    a = ap.parse_args()

    # labelled positions per complex, per chain (union over all committed pair variants)
    labelled = {}
    for pf in sorted(glob.glob(a.pairs_glob)):
        d = pd.read_csv(pf)
        for r in d.itertuples():
            labelled.setdefault(r.complex_id, {}).setdefault(str(r.hot_chain), set()).add(int(r.hot_resnum))
            labelled.setdefault(r.complex_id, {}).setdefault(str(r.ctl_chain), set()).add(int(r.ctl_resnum))

    cids = [l.strip() for l in open(a.complexes) if l.strip()]
    rows, resmap, skipped = [], {}, []
    for cid in cids:
        pdb, g1, g2 = cid.split("_")
        cx = fc.load_complex(os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb"), pdb, g1, g2)
        if cx is None:
            skipped.append((cid, "load fail")); continue
        L = cx.n
        if L > a.lmax:
            skipped.append((cid, f"L={L}>lmax")); continue
        g1c, g2c = list(g1), list(g2)
        lab = labelled.get(cid, {})
        n_g1 = sum(len(lab.get(c, set())) for c in g1c)
        n_g2 = sum(len(lab.get(c, set())) for c in g2c)
        if n_g1 == 0 and n_g2 == 0:
            skipped.append((cid, "no labelled pairs")); continue
        if n_g1 >= n_g2:
            binder, target, nb, no = g1, g2, n_g1, n_g2
        else:
            binder, target, nb, no = g2, g1, n_g2, n_g1
        Lb = int(np.isin(cx.chains, list(binder)).sum())
        rows.append(dict(complex_id=cid, pdb=pdb, binder_chains=binder, target_chains=target,
                         L=L, L_binder=Lb, L_target=L - Lb,
                         n_labelled_binder=nb, n_labelled_target_spanning=no))
        resmap[cid] = {c: [[int(cx.resnums[i]), str(cx.icodes[i]), str(cx.seq[i])]
                           for i in np.flatnonzero(cx.chains == c)] for c in g1c + g2c}

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    json.dump(resmap, open(a.out_map, "w"))
    print(f"[expC_setup] {len(df)} complexes with L<=400 and labelled pairs "
          f"(of {len(cids)}); skipped {len(skipped)}")
    if len(df):
        print(f"  L: median {int(df.L.median())} range [{df.L.min()},{df.L.max()}]; "
              f"L_binder median {int(df.L_binder.median())} range [{df.L_binder.min()},{df.L_binder.max()}]")
        print(f"  labelled pos split onto the held target (spanning) in "
              f"{(df.n_labelled_target_spanning > 0).sum()} complexes "
              f"(median spanning {int(df.n_labelled_target_spanning.median())})")
    print(f"[expC_setup] wrote {a.out_csv} and {a.out_map}")


if __name__ == "__main__":
    main()

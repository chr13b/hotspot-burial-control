"""Diagnostic (positive control): why is ppi.hotspot_res empty for most complexes at 5 A Cbeta?

For each CONDITIONED binder hotspot (results/expC2_hotspot_split.csv, arm==conditioned), compute the
nearest target-chain virtual-Cbeta distance in the crystal, and the number of target contacts at a
ladder of Cbeta cutoffs {5,6,8,10,12} A. Confirms the geometry is sane (distances are interface-scale,
not garbage from a chain mix-up) BEFORE trusting the 35/55 empty result or changing the cutoff. Also
tabulates, per cutoff, how many complexes would get >=1 hotspot_res (i.e. become pinnable).

Usage (under $SCRATCH/ftax/env.sh):
  python3 src/expC2_contact_diag.py --data-dir $FTAX_DATA
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc  # noqa: E402

CUTOFFS = [5.0, 6.0, 8.0, 10.0, 12.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes-csv", default="results/expC_complexes.csv")
    ap.add_argument("--split", default="results/expC2_hotspot_split.csv")
    ap.add_argument("--data-dir", default=os.environ.get("FTAX_DATA", os.path.expanduser("~/ftax/data")))
    ap.add_argument("--out", default="results/expC2_contact_diag.csv")
    a = ap.parse_args()

    meta = pd.read_csv(a.complexes_csv).set_index("complex_id")
    split = pd.read_csv(a.split)
    cond = split[split.arm == "conditioned"]

    rows = []
    for cid, grp in cond.groupby("complex_id"):
        if cid not in meta.index:
            continue
        pdb = meta.loc[cid, "pdb"]
        binder = set(str(meta.loc[cid, "binder_chains"])); target = set(str(meta.loc[cid, "target_chains"]))
        g1, g2 = cid.split("_")[1], cid.split("_")[2]
        cx = fc.load_complex(os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb"), pdb, g1, g2)
        if cx is None:
            continue
        idx_by = {}
        for i in range(cx.n):
            idx_by.setdefault((str(cx.chains[i]), int(cx.resnums[i])), []).append(i)
        tgt_idx = [i for i in range(cx.n) if str(cx.chains[i]) in target]
        tgt_cb = cx.CB[tgt_idx] if tgt_idx else np.zeros((0, 3))
        for r in grp.itertuples():
            hits = idx_by.get((str(r.chain), int(r.resnum)), [])
            if not hits or not tgt_idx:
                rows.append(dict(complex_id=cid, chain=r.chain, resnum=r.resnum, aa=r.aa,
                                 nearest_tgt_cb=np.nan, **{f"n_{int(c)}A": 0 for c in CUTOFFS}))
                continue
            dmin = np.inf; counts = {c: 0 for c in CUTOFFS}
            for i in hits:
                dist = np.linalg.norm(tgt_cb - cx.CB[i], axis=1)
                dmin = min(dmin, float(dist.min()))
                for c in CUTOFFS:
                    counts[c] = max(counts[c], int((dist < c).sum()))
            rows.append(dict(complex_id=cid, chain=r.chain, resnum=r.resnum, aa=r.aa,
                             nearest_tgt_cb=round(dmin, 2), **{f"n_{int(c)}A": counts[c] for c in CUTOFFS}))
    D = pd.DataFrame(rows)
    D.to_csv(a.out, index=False)

    print(f"[contact_diag] {len(D)} conditioned hotspots over {D.complex_id.nunique()} complexes")
    print(f"  nearest target Cbeta distance (A): "
          f"median {D.nearest_tgt_cb.median():.2f}, "
          f"pctiles 10/25/50/75/90 = "
          f"{np.nanpercentile(D.nearest_tgt_cb,[10,25,50,75,90]).round(2).tolist()}")
    print("  per-hotspot: has >=1 target contact within cutoff:")
    for c in CUTOFFS:
        frac = (D[f"n_{int(c)}A"] > 0).mean()
        print(f"    {int(c):2d} A: {(D[f'n_{int(c)}A']>0).sum():3d}/{len(D)} hotspots  ({frac:.0%})")
    print("  per-complex: >=1 hotspot_res (pinnable) at cutoff:")
    for c in CUTOFFS:
        pc = D.groupby("complex_id")[f"n_{int(c)}A"].sum()
        # union of contacts is >= max per hotspot; conservative lower bound uses per-hotspot sum>0
        pinnable = (D.groupby("complex_id")[f"n_{int(c)}A"].max() > 0).sum()
        print(f"    {int(c):2d} A: {pinnable:2d}/{D.complex_id.nunique()} complexes pinnable "
              f"(median tokens/complex ~{int(pc.median())})")


if __name__ == "__main__":
    main()

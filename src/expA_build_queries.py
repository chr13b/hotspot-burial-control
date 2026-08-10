"""Experiment A, step 1: build OpenFold3 prediction queries from crystal sequences.

For each complex in pair_complexes.txt, extract the per-chain amino-acid sequences EXACTLY
as ftax_common.load_complex sees them in the SKEMPI cleaned PDB (same residue set, same
order, missing/non-standard residues already dropped). Emit:

  --out-json  OpenFold3 query set: {"queries": {cid: {"chains": [{molecule_type, chain_ids, sequence}, ...]}}}
  --out-map   residue map: {cid: {chain: [[resnum, icode, aa], ...ordered...]}}

The residue map is what lets a predicted structure (numbered 1..L per chain) be re-keyed to
the crystal (chain, resnum, icode) so SKEMPI labels and the committed matched pairs transfer.

Usage:
  python3 src/expA_build_queries.py --complexes results/pair_complexes.txt \
      --data-dir $FTAX_DATA --out-json results/expA_queries.json --out-map results/expA_resmap.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", default="results/pair_complexes.txt")
    ap.add_argument("--data-dir", default=os.environ.get("FTAX_DATA", os.path.expanduser("~/ftax/data")))
    ap.add_argument("--out-json", default="results/expA_queries.json")
    ap.add_argument("--out-map", default="results/expA_resmap.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated complex ids to keep")
    a = ap.parse_args()

    cids = [l.strip() for l in open(a.complexes) if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        cids = [c for c in cids if c in keep]
    if a.limit:
        cids = cids[: a.limit]

    queries, resmap, skipped = {}, {}, []
    for cid in cids:
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            skipped.append((cid, "no pdb")); continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None:
            skipped.append((cid, "load_complex None")); continue

        chains_out, cmap = [], {}
        for c in list(g1) + list(g2):
            sel = np.flatnonzero(cx.chains == c)
            if len(sel) == 0:
                continue
            seq = "".join(cx.seq[sel].tolist())
            chains_out.append({"molecule_type": "protein", "chain_ids": [c], "sequence": seq})
            cmap[c] = [[int(cx.resnums[i]), str(cx.icodes[i]), str(cx.seq[i])] for i in sel]
        if not chains_out:
            skipped.append((cid, "no chains")); continue
        queries[cid] = {"chains": chains_out}
        resmap[cid] = cmap

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump({"queries": queries}, f, indent=1)
    with open(a.out_map, "w") as f:
        json.dump(resmap, f)

    tot_res = sum(len(v) for cm in resmap.values() for v in cm.values())
    tot_chains = sum(len(cm) for cm in resmap.values())
    print(f"[build_queries] {len(queries)} complexes, {tot_chains} chains, {tot_res} residues")
    print(f"[build_queries] skipped {len(skipped)}: {skipped[:8]}{'...' if len(skipped) > 8 else ''}")
    print(f"[build_queries] wrote {a.out_json} and {a.out_map}")


if __name__ == "__main__":
    main()

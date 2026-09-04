#!/usr/bin/env python3
"""Parse ColabFold (AF2-multimer) outputs for the CFG-steered ipTM folds -> results/iptm_steer.csv.

For each fold <fid> = '<cid>__wt' | '<cid>__L__k{0,1,2}' | '<cid>__random__k{0,1,2}', read the rank_001
scores JSON and record: iptm, ptm (global), interface_plddt (mean pLDDT over interface residues),
interface_pae (mean predicted-aligned-error over cross-interface g1<->g2 interface residue pairs, both
directions). The folded residue order is the crystal complex order (g1 then g2), so pLDDT/PAE indices ==
crystal-complex residue indices; the interface set is the committed leverage_skempi_positions.csv.

  python3 src/parse_iptm.py --out-dir $SCRATCH/ftax/iptm/af2_out --out results/iptm_steer.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc
import leverage_decomposition as LD

DATA = LD.DATA


def parse_fid(fid):
    if fid.endswith("__wt"):
        return fid[:-4], "wt", -1
    cid, direction, kk = fid.rsplit("__", 2)
    return cid, direction, int(kk[1:])


def load_scores(pred_dir, fid):
    pdbs = sorted(glob.glob(f"{pred_dir}/{fid}/{fid}_unrelaxed_rank_001_*.pdb"))
    if not pdbs:
        return None
    tag = os.path.basename(pdbs[0]).replace("_unrelaxed_", "_scores_").replace(".pdb", ".json")
    sj = f"{pred_dir}/{fid}/{tag}"
    if not os.path.exists(sj):
        return None
    d = json.load(open(sj))
    pae = d.get("predicted_aligned_error", d.get("pae"))
    if pae is None:                                     # colabfold sometimes writes a separate PAE file
        pj = sorted(glob.glob(f"{pred_dir}/{fid}/{fid}_predicted_aligned_error_*.json"))
        if pj:
            pae = json.load(open(pj[0])).get("predicted_aligned_error")
    return dict(plddt=np.asarray(d.get("plddt", []), float),
                ptm=d.get("ptm"), iptm=d.get("iptm"),
                pae=np.asarray(pae, float) if pae is not None else None)


def interface_indices(cid):
    """(g1_iface_idx, g2_iface_idx, all_iface_idx) into the crystal-complex residue order."""
    pdb, g1, g2 = cid.split("_")
    cx = fc.load_complex(f"{DATA}/PDBs/{pdb}.pdb", pdb, g1, g2)
    if cx is None:
        return None
    return cx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ["SCRATCH"] + "/ftax/iptm/af2_out")
    ap.add_argument("--subset", default="results/iptm_subset.txt")
    ap.add_argument("--positions", default="results/leverage_skempi_positions.csv")
    ap.add_argument("--out", default="results/iptm_steer.csv")
    a = ap.parse_args()

    pos = pd.read_csv(a.positions, low_memory=False)
    pos["icode"] = pos.icode.fillna("").astype(str)
    pos = pos[pos.is_interface == True]                                              # noqa: E712
    iface = {}
    for r in pos.itertuples():
        iface.setdefault(r.complex_id, set()).add((r.chain, int(r.resnum), r.icode))

    cids = [ln.strip() for ln in open(a.subset) if ln.strip()]
    rows, missing = [], 0
    for cid in cids:
        cx = interface_indices(cid)
        if cx is None:
            continue
        keys = [(cx.chains[j], int(cx.resnums[j]), cx.icodes[j]) for j in range(cx.n)]
        ik = iface.get(cid, set())
        g1i = np.array([j for j in range(cx.n) if keys[j] in ik and cx.group[j] == 1])
        g2i = np.array([j for j in range(cx.n) if keys[j] in ik and cx.group[j] == 2])
        alli = np.array([j for j in range(cx.n) if keys[j] in ik])
        for direction, k in [("wt", -1)] + [("L", k) for k in (0, 1, 2)] + \
                            [("random", k) for k in (0, 1, 2)]:
            fid = f"{cid}__wt" if direction == "wt" else f"{cid}__{direction}__k{k}"
            sc = load_scores(a.out_dir, fid)
            if sc is None:
                missing += 1
                continue
            ipae = np.nan
            if sc["pae"] is not None and len(g1i) and len(g2i):
                P = sc["pae"]
                if P.shape[0] == cx.n:                       # index safety
                    ipae = float(np.concatenate([P[np.ix_(g1i, g2i)].ravel(),
                                                 P[np.ix_(g2i, g1i)].ravel()]).mean())
            iplddt = float(sc["plddt"][alli].mean()) if (sc["plddt"].size == cx.n and len(alli)) else np.nan
            rows.append(dict(complex_id=cid, direction=direction, k=k,
                             iptm=sc["iptm"], ptm=sc["ptm"],
                             interface_pae=round(ipae, 4) if np.isfinite(ipae) else np.nan,
                             interface_plddt=round(iplddt, 4) if np.isfinite(iplddt) else np.nan,
                             n_iface=int(len(alli))))
    df = pd.DataFrame(rows)
    df["seed"] = 20260803
    df.to_csv(a.out, index=False)
    nfold = df.groupby("complex_id").size()
    print(f"[parse] {len(df)} folds parsed over {df.complex_id.nunique()} complexes "
          f"({missing} folds missing); complexes with all 7 = {(nfold == 7).sum()}")
    print(f"[parse] wt interface ipTM: median={df[df.direction=='wt'].iptm.median():.3f} "
          f"(sanity ~0.6-0.9); wrote {a.out}")


if __name__ == "__main__":
    main()

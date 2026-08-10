"""Exp C: build clean RFdiffusion partial-diffusion inputs from crystal complexes.

Per complex (results/expC_complexes.csv): relabel the BINDER chains -> A,B,... (renumbered 1..Lb_i,
to be DIFFUSED) and the TARGET chains -> the next letters (renumbered 1..Lt_i, HELD as motif). Write a
backbone-only input PDB (N/CA/C/O) and the partial-diffusion contig string. Record the binder
output-chain -> crystal (chain,resnum,icode,aa) mapping so a generated backbone can be re-keyed for
scoring (partial diffusion preserves registration, so this is positional).

Emit:
  <out-dir>/<cid>_input.pdb       clean backbone input
  --out-manifest results/expC_inputs.csv   complex_id, input_pdb, contig, binder_out_chains, Lb, Lt
  --out-map results/expC_outmap.json       {cid: {out_chain: [[crystal_chain, resnum, icode, aa], ...]}}

Usage:  python3 src/expC_prep_inputs.py --data-dir $FTAX_DATA --out-dir $SCRATCH/expC/inputs
"""
import argparse
import json
import os
import string
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

ONE2THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
             "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
             "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
             "Y": "TYR", "V": "VAL"}


def build_input(cx, binder_chains, target_chains, out_pdb):
    """Write backbone-only PDB with relabelled/renumbered chains; return (contig, outmap, Lb, Lt)."""
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdbio
    order = list(binder_chains) + list(target_chains)          # crystal chain letters, binder first
    letters = list(string.ascii_uppercase)
    out_letter = {c: letters[i] for i, c in enumerate(order)}  # crystal chain -> output letter
    coords = {"N": cx.N, "CA": cx.CA, "C": cx.C, "O": cx.O}

    recs = []                      # (out_chain, out_resnum, resname3, atom, elem, x,y,z)
    outmap = {}
    counters = {c: 0 for c in order}
    for i in range(cx.n):
        c = str(cx.chains[i])
        if c not in out_letter:
            continue
        counters[c] += 1
        oc, rn = out_letter[c], counters[c]
        outmap.setdefault(oc, []).append([c, int(cx.resnums[i]), str(cx.icodes[i]), str(cx.seq[i])])
        rname = ONE2THREE.get(str(cx.seq[i]), "GLY")
        for atom in ("N", "CA", "C", "O"):
            x, y, z = (float(v) for v in coords[atom][i])
            recs.append((oc, rn, rname, atom, atom[0], x, y, z))

    # RFdiffusion partial diffusion asserts the motif occupies the SAME index in input and output
    # (hal_idx0 == ref_idx0). The contig lists binder spans first then target motifs, so the written
    # file must be ordered binder-chains-then-target-chains. recs were appended in crystal atom order
    # (which need not be binder-first); stable-sort by output chain letter (binder -> A.. < target)
    # to enforce contig order while preserving intra-chain residue/atom order.
    recs.sort(key=lambda r: r[0])

    n = len(recs)
    arr = struc.AtomArray(n)
    arr.coord = np.array([[r[5], r[6], r[7]] for r in recs], dtype=np.float32)
    arr.chain_id = np.array([r[0] for r in recs])
    arr.res_id = np.array([r[1] for r in recs])
    arr.res_name = np.array([r[2] for r in recs])
    arr.atom_name = np.array([r[3] for r in recs])
    arr.element = np.array([r[4] for r in recs])
    pf = pdbio.PDBFile(); pf.set_structure(arr); pf.write(out_pdb)

    # contig: diffuse each binder chain (length-matched, no chain letter), hold each target chain
    diff = [f"{counters[c]}-{counters[c]}" for c in binder_chains]
    hold = [f"{out_letter[c]}1-{counters[c]}" for c in target_chains]
    contig = "/0 ".join(diff + hold)
    Lb = sum(counters[c] for c in binder_chains)
    Lt = sum(counters[c] for c in target_chains)
    return contig, outmap, Lb, Lt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes-csv", default="results/expC_complexes.csv")
    ap.add_argument("--data-dir", default=os.environ.get("FTAX_DATA", os.path.expanduser("~/ftax/data")))
    ap.add_argument("--out-dir", default=os.path.expandvars("$SCRATCH/expC/inputs"))
    ap.add_argument("--out-manifest", default="results/expC_inputs.csv")
    ap.add_argument("--out-map", default="results/expC_outmap.json")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.complexes_csv)
    rows, outmaps, skipped = [], {}, []
    for r in df.itertuples():
        cid, pdb = r.complex_id, r.pdb
        g1, g2 = cid.split("_")[1], cid.split("_")[2]
        cx = fc.load_complex(os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb"), pdb, g1, g2)
        if cx is None:
            skipped.append((cid, "load fail")); continue
        out_pdb = os.path.join(a.out_dir, f"{cid}_input.pdb")
        try:
            contig, outmap, Lb, Lt = build_input(cx, str(r.binder_chains), str(r.target_chains), out_pdb)
        except Exception as e:
            skipped.append((cid, f"{type(e).__name__}: {e}")); continue
        binder_out = "".join(sorted(k for k in outmap if k < list(string.ascii_uppercase)[len(str(r.binder_chains))]))
        rows.append(dict(complex_id=cid, input_pdb=out_pdb, contig=contig,
                         binder_out_chains="".join(list(string.ascii_uppercase)[:len(str(r.binder_chains))]),
                         Lb=Lb, Lt=Lt))
        outmaps[cid] = outmap

    pd.DataFrame(rows).to_csv(a.out_manifest, index=False)
    json.dump(outmaps, open(a.out_map, "w"))
    print(f"[expC_prep] wrote {len(rows)} inputs to {a.out_dir}; skipped {len(skipped)}: {skipped[:6]}")
    if rows:
        print("  example contig:", rows[0]["complex_id"], "->", rows[0]["contig"])


if __name__ == "__main__":
    main()

"""Experiment A, step 3: OpenFold3 predicted CIF -> crystal-keyed PDB + confidence.

For each complex, pick the top-ranked OF3 sample (max sample_ranking_score over all seeds/samples),
read its all-atom mmCIF, map predicted residues (per chain, in sequence order) back to the crystal
(chain, resnum, icode) via results/expA_resmap.json, and write a PDB whose chain IDs / residue
numbers / insertion codes match the crystal exactly. The existing pipeline (load_complex, keyed by
chain group + (chain,resnum,icode)) then ingests the predicted backbone unchanged, so SKEMPI labels
and the committed matched pairs transfer.

Also emits a per-complex confidence CSV:
  avg_plddt, ptm, iptm, interface_plddt (mean pLDDT over crystal-interface residues),
  rmsd_ca_global, rmsd_ca_interface  (Kabsch fit of predicted CA onto crystal CA; bookkeeping only).

Runs in the stab/ftax env (biotite + biopython + numpy). Predicted coords are used for scoring;
the Kabsch fit is never fed to ProteinMPNN.

Usage:
  python3 src/expA_cif_to_pdb.py --pred-dir $SCRATCH/ftax/predicted/of3_out \
      --resmap results/expA_resmap.json --crystal-dir $FTAX_DATA/PDBs \
      --interface-csv results/p0_dssp_interface_resid.csv \
      --out-pdb-dir $SCRATCH/ftax/predicted/PDBs --out-conf results/expA_confidence.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc


def pick_top_sample(pred_dir, cid):
    """Return (cif_path, aggregated_conf_dict) for the highest sample_ranking_score."""
    aggs = glob.glob(os.path.join(pred_dir, cid, "seed_*",
                                  f"{cid}_seed_*_sample_*_confidences_aggregated.json"))
    best, best_score = None, -np.inf
    for ap in aggs:
        try:
            d = json.load(open(ap))
        except Exception:
            continue
        s = d.get("sample_ranking_score", d.get("ptm", -np.inf))
        if s is not None and s > best_score:
            best_score, best = s, (ap, d)
    if best is None:
        return None, None
    cif = best[0].replace("_confidences_aggregated.json", "_model.cif")
    return (cif if os.path.exists(cif) else None), best[1]


def read_cif(cif_path):
    """AtomArray of protein atoms with per-atom b_factor (= pLDDT for AF-class outputs)."""
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure as struc
    f = pdbx.CIFFile.read(cif_path)
    arr = pdbx.get_structure(f, model=1, extra_fields=["b_factor"], use_author_fields=True)
    return arr[struc.filter_amino_acids(arr)]


def chain_residue_order(arr, chain_id):
    """Ordered unique (res_id, ins_code) for a chain, plus its 1-letter sequence."""
    m = arr.chain_id == chain_id
    seen, order, seq = set(), [], []
    for rid, ins, rname in zip(arr.res_id[m], arr.ins_code[m], arr.res_name[m]):
        key = (int(rid), str(ins))
        if key in seen:
            continue
        seen.add(key); order.append(key)
        seq.append(fc.THREE2ONE.get(str(rname).strip().upper(), "X"))
    return order, "".join(seq)


def kabsch_rmsd(P, Q):
    """RMSD after optimal superposition of P onto Q (both [N,3], matched)."""
    if len(P) < 3:
        return np.nan
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    return float(np.sqrt(((Pc @ R - Qc) ** 2).sum(1).mean()))


def write_pdb(records, path):
    """records: list of (chain, resnum, icode, resname3, atomname, element, x, y, z, bfac)."""
    with open(path, "w") as fh:
        serial = 0
        for (ch, rn, ic, rname, aname, elem, x, y, z, b) in records:
            serial += 1
            an = aname if len(aname) >= 4 or aname[:1].isdigit() else f" {aname}"
            fh.write(f"ATOM  {serial:5d} {an:<4.4s}{rname:>3.3s} {ch:1.1s}{rn:4d}{ic:1.1s}   "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}          {elem:>2.2s}\n")
        fh.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="OF3 output_dir root (contains {cid}/seed_*/)")
    ap.add_argument("--resmap", default="results/expA_resmap.json")
    ap.add_argument("--crystal-dir", default=os.path.join(os.environ.get("FTAX_DATA", ""), "PDBs"))
    ap.add_argument("--interface-csv", default="results/p0_dssp_interface_resid.csv")
    ap.add_argument("--out-pdb-dir", required=True)
    ap.add_argument("--out-conf", default="results/expA_confidence.csv")
    ap.add_argument("--complexes", default="results/pair_complexes.txt")
    a = ap.parse_args()

    resmap = json.load(open(a.resmap))
    cids = [l.strip() for l in open(a.complexes) if l.strip()]
    os.makedirs(a.out_pdb_dir, exist_ok=True)

    iface = set()
    if os.path.exists(a.interface_csv):
        import pandas as pd
        idf = pd.read_csv(a.interface_csv, usecols=lambda c: c in
                          ("complex_id", "chain", "resnum", "is_interface"))
        if "is_interface" in idf.columns:
            idf = idf[idf["is_interface"] == True]
        iface = {(r.complex_id, r.chain, int(r.resnum)) for r in idf.itertuples()}

    rows, skipped = [], []
    for cid in cids:
        if cid not in resmap:
            skipped.append((cid, "no resmap")); continue
        cif, conf = pick_top_sample(a.pred_dir, cid)
        if cif is None:
            skipped.append((cid, "no sample")); continue
        pdb, g1, g2 = cid.split("_")
        try:
            arr = read_cif(cif)
        except Exception as e:
            skipped.append((cid, f"cif read: {type(e).__name__}")); continue

        # map each predicted chain -> crystal chain by exact sequence match
        pred_chains = list(dict.fromkeys(arr.chain_id.tolist()))
        cry_seqs = {c: "".join(x[2] for x in resmap[cid][c]) for c in resmap[cid]}
        assign, used = {}, set()
        for pc in pred_chains:
            order, seq = chain_residue_order(arr, pc)
            for cc, cs in cry_seqs.items():
                if cc not in used and seq == cs:
                    assign[pc] = (cc, order); used.add(cc); break
        if len(assign) != len(cry_seqs):
            skipped.append((cid, f"chain map {len(assign)}/{len(cry_seqs)}")); continue

        # per-atom remap: (pred_chain, pred_res_id, ins) -> (crystal chain, resnum, icode)
        remap = {}
        for pc, (cc, order) in assign.items():
            for j, (rid, ins) in enumerate(order):
                cr_rn, cr_ic, _aa = resmap[cid][cc][j]
                remap[(pc, int(rid), ins)] = (cc, int(cr_rn), str(cr_ic))

        n = arr.array_length()
        new_ch = np.array(["A"] * n, dtype="U4")
        new_rn = np.zeros(n, dtype=int)
        new_ic = np.array([""] * n, dtype="U1")
        keep = np.zeros(n, dtype=bool)
        for i in range(n):
            tgt = remap.get((str(arr.chain_id[i]), int(arr.res_id[i]), str(arr.ins_code[i])))
            if tgt is not None:
                new_ch[i], new_rn[i], new_ic[i] = tgt[0], tgt[1], tgt[2]
                keep[i] = True
        sub = arr[keep].copy()
        sub.chain_id = new_ch[keep]
        sub.res_id = new_rn[keep]
        sub.ins_code = new_ic[keep]

        import biotite.structure.io.pdb as pdbio
        out_pdb = os.path.join(a.out_pdb_dir, f"{pdb}.pdb")
        pf = pdbio.PDBFile()
        pf.set_structure(sub)
        pf.write(out_pdb)

        ca = sub[sub.atom_name == "CA"]
        ca_pred = {(str(c), int(r), str(ic)): np.asarray(xyz, float)
                   for c, r, ic, xyz in zip(ca.chain_id, ca.res_id, ca.ins_code, ca.coord)}
        plddt_by_key = {(str(c), int(r), str(ic)): float(b)
                        for c, r, ic, b in zip(ca.chain_id, ca.res_id, ca.ins_code, ca.b_factor)}

        # RMSD to crystal (matched CA), global + interface
        cx = fc.load_complex(os.path.join(a.crystal_dir, f"{pdb}.pdb"), pdb, g1, g2)
        rmsd_g = rmsd_i = np.nan
        if cx is not None:
            keys = [(cx.chains[k], int(cx.resnums[k]), cx.icodes[k]) for k in range(cx.n)]
            shared = [k for k in keys if k in ca_pred]
            if len(shared) >= 3:
                cry = {(cx.chains[k], int(cx.resnums[k]), cx.icodes[k]): cx.CA[k] for k in range(cx.n)}
                P = np.array([ca_pred[k] for k in shared]); Q = np.array([cry[k] for k in shared])
                rmsd_g = kabsch_rmsd(P, Q)
                isub = [k for k in shared if (cid, k[0], k[1]) in iface]
                if len(isub) >= 3:
                    Pi = np.array([ca_pred[k] for k in isub]); Qi = np.array([cry[k] for k in isub])
                    rmsd_i = kabsch_rmsd(Pi, Qi)

        ipl = [plddt_by_key[k] for k in plddt_by_key if (cid, k[0], k[1]) in iface]
        rows.append(dict(complex_id=cid, cif=os.path.basename(cif),
                         avg_plddt=conf.get("avg_plddt"), ptm=conf.get("ptm"),
                         iptm=conf.get("iptm"), ranking=conf.get("sample_ranking_score"),
                         interface_plddt=float(np.mean(ipl)) if ipl else np.nan,
                         n_res=len(ca_pred), rmsd_ca_global=rmsd_g, rmsd_ca_interface=rmsd_i))
        print(f"  {cid}: plddt {conf.get('avg_plddt'):.1f} ptm {conf.get('ptm'):.3f} "
              f"iptm {conf.get('iptm'):.3f} rmsd_g {rmsd_g:.2f} rmsd_i {rmsd_i:.2f}", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(a.out_conf, index=False)
    print(f"\n[cif_to_pdb] wrote {len(rows)} PDBs to {a.out_pdb_dir}, conf -> {a.out_conf}")
    if skipped:
        print(f"[cif_to_pdb] skipped {len(skipped)}: {skipped[:10]}")


if __name__ == "__main__":
    main()

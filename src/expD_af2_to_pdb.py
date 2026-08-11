"""Experiment D, step 3: AF2-multimer (ColabFold) top-ranked PDB -> crystal-keyed PDB + confidence.

The AF2 analogue of src/expA_cif_to_pdb.py. ColabFold already ranks its 5 models; we take
rank_001 (best 0.8*iptm+0.2*ptm). For each complex, read the all-atom PDB, map predicted chains to
crystal chains by exact sequence match (as Exp A), re-key residues to crystal (chain, resnum, icode)
via results/expA_resmap.json, and write a PDB the existing pipeline ingests unchanged. Confidence
(avg/interface pLDDT, pTM, ipTM, Kabsch RMSD-to-crystal) -> results/expD_confidence.csv, same schema
as results/expA_confidence.csv so the two predictors are directly comparable.

Usage (ftax env):
  python3 src/expD_af2_to_pdb.py --pred-dir $SCRATCH/ftax/expD/af2_out \
      --resmap results/expA_resmap.json --crystal-dir $FTAX_DATA/PDBs \
      --interface-csv results/p0_dssp_interface_resid.csv \
      --out-pdb-dir $SCRATCH/ftax/expD/PDBs --out-conf results/expD_confidence.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc


def pick_top_pdb(pred_dir, cid):
    """(pdb_path, conf_dict) for ColabFold rank_001; conf from the matching scores JSON."""
    pdbs = sorted(glob.glob(os.path.join(pred_dir, cid, f"{cid}_unrelaxed_rank_001_*.pdb")))
    if not pdbs:
        pdbs = sorted(glob.glob(os.path.join(pred_dir, cid, f"{cid}_relaxed_rank_001_*.pdb")))
    if not pdbs:
        return None, None
    pdb = pdbs[0]
    tag = os.path.basename(pdb).replace("_unrelaxed_", "_scores_").replace("_relaxed_", "_scores_")
    sj = os.path.join(pred_dir, cid, tag.replace(".pdb", ".json"))
    conf = {}
    if os.path.exists(sj):
        try:
            d = json.load(open(sj))
            plddt = np.asarray(d.get("plddt", []), float)
            ptm, iptm = d.get("ptm"), d.get("iptm")
            conf = dict(avg_plddt=float(np.mean(plddt)) if plddt.size else None,
                        ptm=float(ptm) if ptm is not None else None,
                        iptm=float(iptm) if iptm is not None else None,
                        ranking=(0.8 * float(iptm) + 0.2 * float(ptm))
                        if (iptm is not None and ptm is not None) else None)
        except Exception:
            conf = {}
    return pdb, conf


def read_pdb(pdb_path):
    """AtomArray of protein atoms with per-atom b_factor (= pLDDT for AF-class outputs)."""
    import biotite.structure.io.pdb as pdbio
    import biotite.structure as struc
    f = pdbio.PDBFile.read(pdb_path)
    arr = f.get_structure(model=1, extra_fields=["b_factor"])
    return arr[struc.filter_amino_acids(arr)]


def chain_residue_order(arr, chain_id):
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
    if len(P) < 3:
        return np.nan
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    return float(np.sqrt(((Pc @ R - Qc) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="ColabFold output root (contains {cid}/*.pdb)")
    ap.add_argument("--resmap", default="results/expA_resmap.json")
    ap.add_argument("--crystal-dir", default=os.path.join(os.environ.get("FTAX_DATA", ""), "PDBs"))
    ap.add_argument("--interface-csv", default="results/p0_dssp_interface_resid.csv")
    ap.add_argument("--out-pdb-dir", required=True)
    ap.add_argument("--out-conf", default="results/expD_confidence.csv")
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
        pdb_in, conf = pick_top_pdb(a.pred_dir, cid)
        if pdb_in is None:
            skipped.append((cid, "no pdb")); continue
        pdb, g1, g2 = cid.split("_")
        try:
            arr = read_pdb(pdb_in)
        except Exception as e:
            skipped.append((cid, f"pdb read: {type(e).__name__}")); continue

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
        ap_ = conf.get("avg_plddt"); pt_ = conf.get("ptm"); ip_ = conf.get("iptm")
        rows.append(dict(complex_id=cid, pdb=os.path.basename(pdb_in),
                         avg_plddt=ap_, ptm=pt_, iptm=ip_, ranking=conf.get("ranking"),
                         interface_plddt=float(np.mean(ipl)) if ipl else np.nan,
                         n_res=len(ca_pred), rmsd_ca_global=rmsd_g, rmsd_ca_interface=rmsd_i))
        print(f"  {cid}: plddt {ap_ if ap_ is None else round(ap_,1)} ptm {pt_} iptm {ip_} "
              f"rmsd_g {rmsd_g:.2f} rmsd_i {rmsd_i:.2f}", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(a.out_conf, index=False)
    print(f"\n[af2_to_pdb] wrote {len(rows)} PDBs to {a.out_pdb_dir}, conf -> {a.out_conf}")
    if skipped:
        print(f"[af2_to_pdb] skipped {len(skipped)}: {skipped[:12]}")


if __name__ == "__main__":
    main()

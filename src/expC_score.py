"""Exp C: score generated (backbone-only) backbones — logp + KL + geometry — re-keyed to crystal.

Consumes out-chain-labelled backbone PDBs named <cid>_T<partial_T>_<sample>.pdb (RFdiffusion output,
or the crystal input for partial_T=0), re-keys each to crystal (chain,resnum,icode) via
results/expC_outmap.json, builds a crystal-keyed ComplexStruct, and computes per residue:
ProteinMPNN teacher-forced native log-prob (8-order mean + per-order), unconditional log-prob, all-20
log-probs (for the log-odds secondary), Cβ neighbour count (burial proxy), and the KL detector
(P = uncond(complex), Q = uncond(chain-deleted)). Also per backbone: interface Cα-RMSD to crystal
(superposed on the held target) and inter-chain contacts at labelled hotspots (interface QC).

Backbone-only: no SASA (all-atom); burial = neighbour count. Runs under the ftax stab env.

Usage:
  python3 src/expC_score.py --backbones-dir $SCRATCH/expC/backbones --outmap results/expC_outmap.json \
      --crystal-dir $FTAX_DATA/PDBs --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
      --mpnn-weights $PROTEINMPNN_DIR/vanilla_model_weights/v_48_020.pt --out $SCRATCH/expC/scored --threads 8
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

BB = ("N", "CA", "C", "O")
NAME_RE = re.compile(r"^(?P<cid>.+?)_T(?P<T>\d+)_(?P<s>\d+)\.pdb$")


_COORD_RE = re.compile(r"-?\d+\.\d{3}")


def read_backbone(pdb_path):
    """out-chain PDB -> {(out_chain, out_resnum): {atom: xyz}} in file order.

    Manual parse (NOT biopython): at high partial_T the diffused binder can drift past |coord|>=1000 A,
    whose %.3f field is 9+ chars and overflows PDB's 8-col x/y/z, running columns together so biopython
    raises PDBConstructionException and the whole (high-drift) backbone is dropped — a noise-correlated
    bias. Here name/chain/resnum come from fixed columns and x/y/z are the first three 3-decimal floats
    after col 30 (occupancy/B are 2-decimal, so they never match), which is robust to the overflow."""
    out, order = {}, []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom not in BB:
                continue
            chain = line[21]
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            floats = _COORD_RE.findall(line[30:])
            if len(floats) < 3:
                continue
            key = (chain, rn)
            d = out.get(key)
            if d is None:
                d = {}
                out[key] = d
                order.append(key)
            d[atom] = np.array([float(floats[0]), float(floats[1]), float(floats[2])], dtype=float)
    out = {k: v for k, v in out.items() if all(b in v for b in BB)}
    order = [k for k in order if k in out]
    return out, order


def build_cx(bb, order, cmap, pdb_id):
    """cmap: {out_chain: [[crystal_chain, resnum, icode, aa], ...ordered]}. Build crystal-keyed cx."""
    chains, resnums, icodes, seq, group = [], [], [], [], []
    Ns, CAs, Cs, Os = [], [], [], []
    # crystal group membership: binder out-chains map to group per their crystal chain; use crystal chain
    for oc in sorted(cmap):
        rows = cmap[oc]
        for j, (cc, crn, cic, aa) in enumerate(rows):
            key = (oc, j + 1)  # out-chain PDB was renumbered 1..L per chain
            if key not in bb:
                continue
            d = bb[key]
            chains.append(cc); resnums.append(int(crn)); icodes.append(str(cic)); seq.append(str(aa))
            Ns.append(d["N"]); CAs.append(d["CA"]); Cs.append(d["C"]); Os.append(d["O"])
    if not seq:
        return None
    N = np.array(Ns, float); CA = np.array(CAs, float); C = np.array(Cs, float); O = np.array(Os, float)
    # group: 1 for the first crystal group letters, else 2 — infer from crystal chain via the complex id
    return fc.ComplexStruct(pdb=pdb_id, group1="", group2="", chains=np.array(chains),
                            resnums=np.array(resnums), icodes=np.array(icodes), seq=np.array(seq),
                            group=np.ones(len(seq), int), N=N, CA=CA, C=C, O=O,
                            CB=fc._virtual_cb(N, CA, C), bfac=np.full(len(seq), np.nan), n=len(seq))


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
    ap.add_argument("--backbones-dir", required=True)
    ap.add_argument("--outmap", default="results/expC_outmap.json")
    ap.add_argument("--complexes-csv", default="results/expC_complexes.csv")
    ap.add_argument("--crystal-dir", default=os.path.join(os.environ.get("FTAX_DATA", ""), "PDBs"))
    ap.add_argument("--pairs", default="results/p0_dssp_pairs_SECONDARY_B_any_interface.csv")
    ap.add_argument("--interface", default="results/p0_dssp_interface_resid.csv")
    ap.add_argument("--mpnn-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/N: process only backbones with index%N==i (parallel scoring)")
    a = ap.parse_args()
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    cmd = "python3 " + " ".join(sys.argv)

    import torch
    torch.set_num_threads(a.threads)
    import pandas as pd
    model, _ = fc.load_mpnn(a.mpnn_weights)
    outmap = json.load(open(a.outmap))
    meta = pd.read_csv(a.complexes_csv).set_index("complex_id")
    # crystal-group membership: which crystal chains are binder vs target
    binder_chain_set = {cid: set(str(meta.loc[cid, "binder_chains"])) for cid in meta.index}

    # labelled hotspot positions per complex (for interface-contact QC)
    pairs = pd.read_csv(a.pairs)
    hot_by = {}
    for r in pairs.itertuples():
        hot_by.setdefault(r.complex_id, set()).add((str(r.hot_chain), int(r.hot_resnum)))

    # interface positions (crystal chain,resnum) -> pre-registered dose variable = interface Cα-RMSD
    iface_by = {}
    if os.path.exists(a.interface):
        idf = pd.read_csv(a.interface)
        if "is_interface" in idf.columns:
            idf = idf[idf["is_interface"] == True]
        for r in idf.itertuples():
            iface_by.setdefault(r.complex_id, set()).add((str(r.chain), int(r.resnum)))
    print(f"[expC_score] interface set: {len(iface_by)} complexes")

    pos_fh = open(f"{a.out}_positions.csv", "w", newline="")
    bb_fh = open(f"{a.out}_backbones.csv", "w", newline="")
    pos_w, bb_w = None, None
    files = sorted(glob.glob(os.path.join(a.backbones_dir, "*.pdb")))
    print(f"[expC_score] {len(files)} backbones")
    for fi, path in enumerate(files):
        if fi % shard_n != shard_i:
            continue
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        cid, T, s = m["cid"], int(m["T"]), int(m["s"])
        if cid not in outmap:
            continue
        pdb, g1, g2 = cid.split("_")
        try:
            bb, order = read_backbone(path)
            cx = build_cx(bb, order, outmap[cid], pdb)
            if cx is None or cx.n < 5:
                continue
            # set group from binder/target crystal chains
            bset = binder_chain_set.get(cid, set())
            cx.group = np.array([1 if c in bset else 2 for c in cx.chains])
            lp = fc.mpnn_conditional_logprobs(model, cx, seeds=range(8))
            lp_mean = lp.mean(axis=0)
            lp_unc = fc.mpnn_unconditional_logprobs(model, cx)
            nbr = fc.neighbour_counts(cx)
            nat = np.array([fc.MPNN_ALPHABET.index(x) for x in cx.seq])
            nat_per_order = lp[:, np.arange(cx.n), nat]
            # KL: P complex uncond, Q chain-deleted (per group)
            P = _dists(lp_unc)
            Q = np.full_like(P, np.nan)
            for grp in (1, 2):
                sel = np.flatnonzero(cx.group == grp)
                if len(sel) < 5:
                    continue
                mono = _submono(cx, sel)
                Qm = _dists(fc.mpnn_unconditional_logprobs(model, mono))
                for k, j in enumerate(sel):
                    Q[j] = Qm[k]
            ok = np.isfinite(Q).all(axis=1)
            eps = 1e-12
            kl = np.where(ok, (P * (np.log(P + eps) - np.log(Q + eps))).sum(1), np.nan)

            # crystal ref for iRMSD: superpose gen onto crystal by the HELD target, then RMSD on the
            # binder. Primary dose = interface binder Cα-RMSD (pre-registered); binder-wide = robustness.
            cxc = fc.load_complex(os.path.join(a.crystal_dir, f"{pdb}.pdb"), pdb, g1, g2)
            irmsd = np.nan; irmsd_binder = np.nan; tgt_rmsd = np.nan; ncontact = 0
            if cxc is not None:
                cryca = {(str(cxc.chains[k]), int(cxc.resnums[k])): cxc.CA[k] for k in range(cxc.n)}
                genca = {(str(cx.chains[k]), int(cx.resnums[k])): cx.CA[k] for k in range(cx.n)}
                ifset = iface_by.get(cid, set())
                tgt = [k for k in genca if k in cryca and k[0] not in bset]
                bind = [k for k in genca if k in cryca and k[0] in bset]
                bind_if = [k for k in bind if k in ifset]
                if len(tgt) >= 3 and len(bind) >= 3:
                    try:
                        Pt = np.array([genca[k] for k in tgt]); Qt = np.array([cryca[k] for k in tgt])
                        Pc, Qc = Pt - Pt.mean(0), Qt - Qt.mean(0)
                        V, _, Wt = np.linalg.svd(Pc.T @ Qc); d = np.sign(np.linalg.det(V @ Wt))
                        R = V @ np.diag([1, 1, d]) @ Wt

                        def _rmsd(keys):
                            if len(keys) < 3:
                                return np.nan
                            Pb = np.array([genca[k] for k in keys]) - Pt.mean(0)
                            Qb = np.array([cryca[k] for k in keys]) - Qt.mean(0)
                            return float(np.sqrt(((Pb @ R - Qb) ** 2).sum(1).mean()))

                        tgt_rmsd = float(np.sqrt(((Pc @ R - Qc) ** 2).sum(1).mean()))  # ~0 => target held
                        irmsd = _rmsd(bind_if)       # interface binder Cα (pre-registered dose variable)
                        irmsd_binder = _rmsd(bind)   # binder-wide Cα (robustness)
                    except np.linalg.LinAlgError:
                        pass  # extreme drift -> SVD didn't converge; leave iRMSD nan, still score logp/KL
                # inter-chain contacts at labelled hotspots (on generated backbone)
                g = cx.group
                dcb = np.linalg.norm(cx.CB[:, None] - cx.CB[None], axis=-1)
                cross = g[:, None] != g[None, :]
                hotset = hot_by.get(cid, set())
                for k in range(cx.n):
                    if (str(cx.chains[k]), int(cx.resnums[k])) in hotset:
                        ncontact += int((dcb[k][cross[k]] < 10.0).sum())

            for i in range(cx.n):
                row = dict(backbone_id=f"{cid}_T{T}_{s}", complex_id=cid, partial_T=T, sample=s,
                           chain=cx.chains[i], resnum=int(cx.resnums[i]), icode=cx.icodes[i], aa=cx.seq[i],
                           group=int(cx.group[i]), nbr=int(nbr[i]),
                           logp_native=float(lp_mean[i, nat[i]]),
                           logp_native_unc=float(lp_unc[i, nat[i]]), kl=float(kl[i]))
                for o in range(8):
                    row[f"logp_native_o{o}"] = float(nat_per_order[o, i])
                for ai, aa in enumerate(fc.MPNN_ALPHABET[:20]):
                    row[f"lp_{aa}"] = float(lp_mean[i, ai])
                if pos_w is None:
                    pos_w = csv.DictWriter(pos_fh, fieldnames=list(row.keys())); pos_w.writeheader()
                pos_w.writerow(row)
            brow = dict(backbone_id=f"{cid}_T{T}_{s}", complex_id=cid, partial_T=T, sample=s,
                        n_res=cx.n, irmsd=irmsd, irmsd_binder=irmsd_binder, tgt_rmsd=tgt_rmsd,
                        hot_contacts=ncontact, interface_ok=int(ncontact >= 5))
            if bb_w is None:
                bb_w = csv.DictWriter(bb_fh, fieldnames=list(brow.keys()) + ["command"]); bb_w.writeheader()
            brow["command"] = cmd; bb_w.writerow(brow)
            pos_fh.flush(); bb_fh.flush()
            if (fi + 1) % 20 == 0:
                print(f"  {fi+1}/{len(files)} scored", flush=True)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {type(e).__name__}: {e}", flush=True)
    pos_fh.close(); bb_fh.close()
    print(f"[expC_score] wrote {a.out}_positions.csv and {a.out}_backbones.csv")


def _dists(lp20):
    z = lp20[:, :20]; z = z - z.max(1, keepdims=True); p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def _submono(cx, sel):
    return fc.ComplexStruct(pdb=cx.pdb, group1="", group2="", chains=cx.chains[sel],
                            resnums=cx.resnums[sel], icodes=cx.icodes[sel], seq=cx.seq[sel],
                            group=np.ones(len(sel), int), N=cx.N[sel], CA=cx.CA[sel], C=cx.C[sel],
                            O=cx.O[sel], CB=cx.CB[sel], bfac=cx.bfac[sel], n=len(sel))


if __name__ == "__main__":
    main()

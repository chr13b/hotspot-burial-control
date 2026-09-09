#!/usr/bin/env python3
"""Predicted-backbone steering (pre-reg: results/PREREG_predicted_steer.md). Does the +alpha*L tilt on a frozen
ProteinMPNN still raise an INDEPENDENT model's binding-leverage vs a matched random direction when leverage is
COMPUTED AND APPLIED ON THE PREDICTED BACKBONE (the staged-design regime), not the crystal?

One pass, all arms (L / random / naive) + a wt reference, all judges (ESM-IF1 / MIF anti-circular; ProteinMPNN self,
kept for reference), for a chosen backbone SOURCE. Reuses cfg_steer / cfg_steer_naive core verbatim
(mpnn_steer.draw, seq_to_chains, load_L, the naive construction, the matched-magnitude random permutation) — only the
backbone PDB, the leverage caches, and the interface set are swapped per source, so the `crystal` source reproduces
the committed crystal judge numbers as a positive control on the SAME complex subset.

  python3 src/cfg_steer_predicted.py --source of3 --subset results/predicted_steer_subset.txt \
      --K 64 --alphas 0,2 --out results/_cfg_pred_of3.csv --seqs-out results/_cfg_pred_of3_seqs.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "models"), os.path.join(HERE, "decoding")):
    sys.path.insert(0, p)
import ftax_common as fc                 # noqa: E402
import leverage_decomposition as LD      # noqa: E402
import mpnn_steer as ms                  # noqa: E402
import cfg_steer as cs                   # noqa: E402  (load_L, seq_to_chains)
from cfg_steer_naive import load_lP      # noqa: E402  (complex-conditioned confidence for the naive tilt)
SEED, AA20, IDX = LD.SEED, LD.AA20, LD.IDX

SRC = {   # source -> (pdb dir, positions file, mpnn pq, esmif pq, mif pq)
    "crystal": (os.path.expanduser("~/ftax/data/PDBs"), "results/leverage_skempi_positions.csv",
                "results/leverage_pq_skempi.csv", "results/leverage_pq_skempi_esmif.csv",
                "results/leverage_pq_skempi_mif.csv"),
    "of3": (os.path.expandvars("$SCRATCH/ftax/predicted/PDBs"), "results/leverage_predicted_of3_positions.csv",
            "results/leverage_pq_predicted_of3.csv", "results/leverage_pq_predicted_of3_esmif.csv",
            "results/leverage_pq_predicted_of3_mif.csv"),
    "af2": (os.path.expandvars("$SCRATCH/ftax/expD/PDBs"), "results/leverage_predicted_af2_positions.csv",
            "results/leverage_pq_predicted_af2.csv", "results/leverage_pq_predicted_af2_esmif.csv",
            "results/leverage_pq_predicted_af2_mif.csv"),
}


def interface_set(posfile):
    p = pd.read_csv(posfile, low_memory=False)
    p["icode"] = p.icode.fillna("").astype(str)
    p = p[p.is_interface.isin([True, 1, "True", "1"])]
    out = {}
    for r in p.itertuples():
        out.setdefault(r.complex_id, set()).add((r.chain, int(r.resnum), r.icode))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SRC))
    ap.add_argument("--subset", default="", help="file of complex_ids (the shared predicted-steer set)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--alphas", default="0,2")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--dump-k", type=int, default=3)
    ap.add_argument("--max-res", type=int, default=700)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs-out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    alphas = [float(x) for x in a.alphas.split(",")]
    pdbdir, posfile, mpnn_pq, esmif_pq, mif_pq = SRC[a.source]

    Lm = cs.load_L(mpnn_pq)                                  # ProteinMPNN leverage — the steering direction + magnitude
    Le = cs.load_L(esmif_pq)                                 # ESM-IF1 leverage — anti-circular judge
    Lmif = cs.load_L(mif_pq) if os.path.exists(mif_pq) else {}   # MIF leverage — anti-circular judge (if present)
    lPm = load_lP(mpnn_pq)                                   # complex-conditioned confidence for the naive tilt
    iface = interface_set(posfile)
    cids = sorted(set(Lm) & set(Le) & set(lPm) & set(iface))
    if a.subset:
        keep = {ln.strip() for ln in open(a.subset) if ln.strip()}
        cids = [c for c in cids if c in keep]
    if a.limit:
        cids = cids[:a.limit]

    model, _ = fc.load_mpnn(LD.MPNN_W)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[pred-steer:{a.source}] {len(cids)} complexes, K={a.K}, alphas={alphas}, device={device}, "
          f"mif={'yes' if Lmif else 'no'}", flush=True)
    rng = np.random.default_rng(SEED)
    rows, seqrows = [], []
    for ci, cid in enumerate(cids):
        pdb, g1, g2 = cid.split("_")
        path = f"{pdbdir}/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None or cx.n > a.max_res:
            continue
        keys = list(zip(cx.chains, [int(x) for x in cx.resnums], cx.icodes))
        wt = np.array([IDX.get(s, -1) for s in cx.seq])
        Lm_pos = np.zeros((cx.n, 20)); Le_pos = np.zeros((cx.n, 20)); Lmif_pos = np.zeros((cx.n, 20))
        lP_pos = np.zeros((cx.n, 20)); usable = np.zeros(cx.n, bool)
        for i, k in enumerate(keys):
            if (k in iface.get(cid, ()) and k in Lm[cid] and k in Le[cid] and k in lPm[cid] and wt[i] >= 0):
                Lm_pos[i] = Lm[cid][k]; Le_pos[i] = Le[cid][k]; lP_pos[i] = lPm[cid][k]; usable[i] = True
                if cid in Lmif and k in Lmif[cid]:
                    Lmif_pos[i] = Lmif[cid][k]
        nu = int(usable.sum())
        if nu < 3:
            continue
        noni = (~usable) & (wt >= 0)
        Rperm = np.array([rng.permutation(Lm_pos[i]) for i in range(cx.n)])       # matched-magnitude random direction
        conf = lP_pos[usable] - lP_pos[usable].mean(1, keepdims=True)             # naive = confidence dir, matched |L_i|
        cn = np.linalg.norm(conf, axis=1, keepdims=True)
        u = conf / np.where(cn > 1e-8, cn, 1.0)
        Ln = np.linalg.norm(Lm_pos[usable], axis=1, keepdims=True)
        naive_pos = np.zeros((cx.n, 20)); naive_pos[usable] = (Ln * u)
        judges = {"mpnn": Lm_pos, "esmif": Le_pos}
        if Lmif and cid in Lmif:
            judges["mif"] = Lmif_pos
        seqrows.append(dict(complex_id=cid, direction="wt", alpha=float("nan"), k=-1,
                            chains=cs.seq_to_chains(cx, wt)))
        for direction, D in [("L", Lm_pos), ("random", Rperm), ("naive", naive_pos)]:
            for al in alphas:
                B = np.zeros((cx.n, 21), np.float32)
                B[usable, :20] = (al * D[usable]).astype(np.float32)
                S, _ = ms.draw(model, cx, a.K, a.K, order=None, temperature=a.temp, seed=SEED,
                               use_patch=False, featurize=fc.featurize, bias_by_res=B)      # [K, L]
                Su = S[:, usable]
                int_rec = float((Su == wt[usable][None]).mean())
                noni_rec = float((S[:, noni] == wt[noni][None]).mean()) if noni.any() else float("nan")

                def meanL(Lpos):
                    idx = np.clip(Su, 0, 19)
                    vals = Lpos[usable][np.arange(nu)[None, :], idx]
                    return float(np.where(Su < 20, vals, 0.0).mean())
                row = dict(complex_id=cid, source=a.source, direction=direction, alpha=al, n_int=nu,
                           int_recovery=round(int_rec, 4), noninterface_recovery=round(noni_rec, 4))
                for jk, Lp in judges.items():
                    row[f"meanL_{jk}"] = round(meanL(Lp), 4)
                rows.append(row)
                if al == alphas[-1]:
                    for k in range(min(a.dump_k, S.shape[0])):
                        s = wt.copy(); s[usable] = S[k, usable]
                        seqrows.append(dict(complex_id=cid, direction=direction, alpha=al, k=k,
                                            chains=cs.seq_to_chains(cx, s)))
        print(f"[pred-steer:{a.source}] {ci+1}/{len(cids)} {cid} n_int={nu}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(seqrows).to_csv(a.seqs_out, index=False)
    print(f"[pred-steer:{a.source}] wrote {len(rows)} rows -> {a.out}; {len(seqrows)} seqs -> {a.seqs_out}", flush=True)


if __name__ == "__main__":
    main()

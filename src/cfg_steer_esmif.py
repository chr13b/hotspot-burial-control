#!/usr/bin/env python3
"""Phase 2 (SET-B) — steer FROZEN ESM-IF1 by +alpha*L and sample. Pre-reg: results/PREREG_esmif_steer.md.

Symmetric analog of cfg_steer.py: there ProteinMPNN is steered by ProteinMPNN's OWN leverage and judged by
ESM-IF1; here ESM-IF1 is steered by ESM-IF1's OWN leverage and judged by ProteinMPNN (anti-circular).

One-shot biased native-conditional sampler (disclosed in the pre-reg): ESM-IF1's per-position complex-conditional
log-probabilities lP and its leverage L are BOTH already in the committed cache results/leverage_pq_skempi_esmif.csv
(lP = P(a|complex); L(a) = (lP(a)-lP(wt))-(lQ(a)-lQ(wt))). At each interface position form biased logits
lP(a) + alpha*L(a), sample K residues independently (temperature T), keep wt elsewhere. No live ESM-IF1 pass is
needed (and none is possible to get out of alignment) — pure numpy over the cache + the crystal backbone.

  python3 src/cfg_steer_esmif.py --subset results/iptm_subset.txt --alphas 0,2 --K 64 \
      --seqs-out results/cfg_steer_esmif_seqs.csv --out results/cfg_steer_esmif.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "models")):
    sys.path.insert(0, p)
import ftax_common as fc            # noqa: E402
import leverage_decomposition as LD  # noqa: E402
import cfg_steer as cs              # noqa: E402  reuse load_L / interface_set / seq_to_chains

SEED, DATA, AA20, IDX = LD.SEED, LD.DATA, LD.AA20, LD.IDX


def load_lP(pqf):
    """complex_id -> {(chain,resnum,icode): lP[20]} = P(a|complex) log-probs (already logdists-normalised)."""
    d = pd.read_csv(pqf, low_memory=False)
    d["icode"] = d.icode.fillna("").astype(str)
    lP = d[[f"lP_{a}" for a in AA20]].to_numpy()
    ok = np.isfinite(lP).all(1)
    d, lP = d[ok].reset_index(drop=True), lP[ok]
    out = {}
    for i, r in enumerate(d.itertuples()):
        out.setdefault(r.complex_id, {})[(r.chain, int(r.resnum), r.icode)] = lP[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="results/iptm_subset.txt")
    ap.add_argument("--alphas", default="0,2")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--dump-k", type=int, default=3)
    ap.add_argument("--seqs-out", default="results/cfg_steer_esmif_seqs.csv")
    ap.add_argument("--out", default="results/cfg_steer_esmif.csv")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    Le = cs.load_L("results/leverage_pq_skempi_esmif.csv")     # ESM-IF1's own leverage = steering direction
    Lm = cs.load_L("results/leverage_pq_skempi.csv")           # ProteinMPNN leverage = the anti-circular judge
    lPe = load_lP("results/leverage_pq_skempi_esmif.csv")      # ESM-IF1 conditional logits to be biased
    iface = cs.interface_set()
    subset = [l.strip() for l in open(a.subset) if l.strip()]
    cids = [c for c in subset if c in Le and c in Lm and c in lPe and c in iface]
    print(f"[esmif-steer] {len(cids)}/{len(subset)} subset complexes have all caches; K={a.K}, alphas={alphas}")

    rng = np.random.default_rng(SEED)
    seqrows, rows = [], []
    for ci, cid in enumerate(cids):
        pdb, g1, g2 = cid.split("_")
        path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            print(f"  [skip] {cid}: no PDB at {path}")
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None or cx.n > 700:
            continue
        keys = list(zip(cx.chains, [int(x) for x in cx.resnums], cx.icodes))
        wt = np.array([IDX.get(s, -1) for s in cx.seq])
        Le_pos = np.zeros((cx.n, 20))
        Lm_pos = np.zeros((cx.n, 20))
        lP_pos = np.zeros((cx.n, 20))
        usable = np.zeros(cx.n, bool)
        for i, k in enumerate(keys):
            if (k in iface.get(cid, ()) and k in Le[cid] and k in Lm[cid] and k in lPe[cid] and wt[i] >= 0):
                Le_pos[i] = Le[cid][k]
                Lm_pos[i] = Lm[cid][k]
                lP_pos[i] = lPe[cid][k]
                usable[i] = True
        nu = int(usable.sum())
        if nu < 3:
            continue
        Rperm = np.array([rng.permutation(Le_pos[i]) for i in range(cx.n)])   # matched-magnitude random dir
        seqrows.append(dict(complex_id=cid, direction="wt", alpha=float("nan"), k=-1,
                            chains=cs.seq_to_chains(cx, wt)))
        for direction, D in [("L", Le_pos), ("random", Rperm)]:
            for al in alphas:
                logits = (lP_pos[usable] + al * D[usable]) / a.temp        # [nu,20] biased conditional
                logits = logits - logits.max(1, keepdims=True)
                pu = np.exp(logits)
                pu /= pu.sum(1, keepdims=True)
                cum = pu.cumsum(1)
                r = rng.random((a.K, nu))
                Su = np.clip((r[:, :, None] > cum[None, :, :]).sum(2), 0, 19)   # [K,nu] sampled aa idx

                def meanL(Lpos):                                            # judge leverage at sampled residues
                    vals = Lpos[usable][np.arange(nu)[None, :], Su]         # [K,nu]
                    return float(vals.mean())
                int_rec = float((Su == wt[usable][None]).mean())
                rows.append(dict(complex_id=cid, direction=direction, alpha=al, n_int=nu,
                                 int_recovery=round(int_rec, 4),
                                 meanL_mpnn=round(meanL(Lm_pos), 4),        # anti-circular judge (ProteinMPNN)
                                 meanL_esmif=round(meanL(Le_pos), 4)))      # self (reference only, circular)
                if al == alphas[-1]:
                    for k in range(min(a.dump_k, a.K)):
                        s = wt.copy()
                        s[usable] = Su[k]
                        seqrows.append(dict(complex_id=cid, direction=direction, alpha=al, k=k,
                                            chains=cs.seq_to_chains(cx, s)))
        print(f"[esmif-steer] {ci+1}/{len(cids)} {cid} n_int={nu}", flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.DataFrame(seqrows).to_csv(a.seqs_out, index=False)
    print(f"[esmif-steer] wrote {len(seqrows)} sequences -> {a.seqs_out}; {len(rows)} score rows -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Is the CFG direction ACTIONABLE? (pre-registered: results/PREREG_cfg_steer.md)

Tilt ProteinMPNN sampling by +alpha*L at interface positions (L = its own leverage / CFG-guidance direction)
and ask whether the sampled sequences bind better -- measured by a DIFFERENT model's leverage (ESM-IF1), so the
test is not "steer by L, measure L". Control: a random direction of matched per-position magnitude.

  python3 src/cfg_steer.py --limit 3 --K 16 --out results/_smoke_cfg.csv     # smoke test
  python3 src/cfg_steer.py --out results/cfg_steer.csv                        # full sweep
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "models"), os.path.join(HERE, "decoding")):
    sys.path.insert(0, p)
import ftax_common as fc
import leverage_decomposition as LD
import mpnn_steer as ms
SEED, DATA, AA20, IDX = LD.SEED, LD.DATA, LD.AA20, LD.IDX


def load_L(pqf):
    """complex_id -> {(chain,resnum,icode): L[20]} where L(a)=(lP(a)-lP(wt))-(lQ(a)-lQ(wt))."""
    d = pd.read_csv(pqf, low_memory=False); d["icode"] = d.icode.fillna("").astype(str)
    lP = d[[f"lP_{a}" for a in AA20]].to_numpy(); lQ = d[[f"lQ_{a}" for a in AA20]].to_numpy()
    wi = d.aa.map(IDX).to_numpy()
    ok = np.isfinite(wi.astype(float)) & np.isfinite(lP).all(1) & np.isfinite(lQ).all(1)
    d, lP, lQ, wi = d[ok].reset_index(drop=True), lP[ok], lQ[ok], wi[ok].astype(int)
    ar = np.arange(len(d))
    L = (lP - lP[ar, wi][:, None]) - (lQ - lQ[ar, wi][:, None])
    out = {}
    for i, r in enumerate(d.itertuples()):
        out.setdefault(r.complex_id, {})[(r.chain, int(r.resnum), r.icode)] = L[i]
    return out


def interface_set():
    p = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    p = p[p.is_interface == True]; p["icode"] = p.icode.fillna("").astype(str)     # noqa: E712
    out = {}
    for r in p.itertuples():
        out.setdefault(r.complex_id, set()).add((r.chain, int(r.resnum), r.icode))
    return out


def seq_to_chains(cx, sidx):
    """Full complex sequence split into per-chain 'C:SEQ|...' strings for external folding."""
    pc = {}
    for i in range(cx.n):
        pc.setdefault(cx.chains[i], []).append(AA20[int(sidx[i])] if 0 <= int(sidx[i]) < 20 else "X")
    return "|".join(f"{c}:{''.join(s)}" for c, s in pc.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--alphas", default="0,0.5,1,2,4"); ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default="results/cfg_steer.csv")
    ap.add_argument("--dump-seqs", action="store_true"); ap.add_argument("--seqs-out", default="results/cfg_steer_seqs.csv")
    ap.add_argument("--dump-k", type=int, default=3)
    a = ap.parse_args()
    import torch; torch.set_num_threads(4)
    alphas = [float(x) for x in a.alphas.split(",")]
    Lm, Le, iface = load_L("results/leverage_pq_skempi.csv"), load_L("results/leverage_pq_skempi_esmif.csv"), interface_set()
    cids = [c for c in sorted(set(Lm) & set(Le) & set(iface))]
    if a.limit:
        cids = cids[:a.limit]
    model, _ = fc.load_mpnn(LD.MPNN_W)
    device = "cuda" if torch.cuda.is_available() else "cpu"      # optional GPU speed; CPU path unchanged
    model = model.to(device)
    print(f"[cfg-steer] {len(cids)} complexes, K={a.K}, alphas={alphas}, temp={a.temp}, device={device}", flush=True)
    rng = np.random.default_rng(SEED); rows = []; seqrows = []
    for ci, cid in enumerate(cids):
        pdb, g1, g2 = cid.split("_"); path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None or cx.n > 700:
            continue
        keys = list(zip(cx.chains, [int(x) for x in cx.resnums], cx.icodes))
        wt = np.array([IDX.get(s, -1) for s in cx.seq])
        # per-position L (mpnn, esmif) and interface mask, aligned to cx order
        Lm_pos = np.zeros((cx.n, 20)); Le_pos = np.zeros((cx.n, 20)); usable = np.zeros(cx.n, bool)
        for i, k in enumerate(keys):
            if k in iface.get(cid, ()) and k in Lm[cid] and k in Le[cid] and wt[i] >= 0:
                Lm_pos[i] = Lm[cid][k]; Le_pos[i] = Le[cid][k]; usable[i] = True
        if usable.sum() < 3:
            continue
        noni = (~usable) & (wt >= 0)
        Rperm = np.array([rng.permutation(Lm_pos[i]) for i in range(cx.n)])          # matched-magnitude random dir
        if a.dump_seqs:                                                              # wt reference fold, once per complex
            seqrows.append(dict(complex_id=cid, direction="wt", alpha=float("nan"), k=-1, chains=seq_to_chains(cx, wt)))
        for direction, D in [("L", Lm_pos), ("random", Rperm)]:
            for al in alphas:
                B = np.zeros((cx.n, 21), np.float32)
                B[usable, :20] = (al * D[usable]).astype(np.float32)
                S, _ = ms.draw(model, cx, a.K, a.K, order=None, temperature=a.temp, seed=SEED,
                               use_patch=False, featurize=fc.featurize, bias_by_res=B)     # [K, L]
                Su = S[:, usable]                                                     # sampled interface residues
                int_rec = float((Su == wt[usable][None]).mean())
                noni_rec = float((S[:, noni] == wt[noni][None]).mean()) if noni.any() else float("nan")
                # binding-leverage of the sampled residues (0 for wt or X); mean over K x interface positions
                def meanL(Lpos):
                    idx = np.clip(Su, 0, 19)                                            # [K, n_int]
                    vals = Lpos[usable][np.arange(int(usable.sum()))[None, :], idx]     # Lpos_usable[j, Su[k,j]]
                    return float(np.where(Su < 20, vals, 0.0).mean())
                rows.append(dict(complex_id=cid, direction=direction, alpha=al, n_int=int(usable.sum()),
                                 int_recovery=round(int_rec, 4), noninterface_recovery=round(noni_rec, 4),
                                 meanL_mpnn=round(meanL(Lm_pos), 4), meanL_esmif=round(meanL(Le_pos), 4)))
                if a.dump_seqs and al == alphas[-1]:                                # steered fold: wt background + steered interface
                    for k in range(min(a.dump_k, S.shape[0])):
                        s = wt.copy(); s[usable] = S[k, usable]
                        seqrows.append(dict(complex_id=cid, direction=direction, alpha=al, k=k, chains=seq_to_chains(cx, s)))
        print(f"[cfg-steer] {ci+1}/{len(cids)} {cid} n_int={int(usable.sum())}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    if a.dump_seqs:
        pd.DataFrame(seqrows).to_csv(a.seqs_out, index=False)
        print(f"[dump] wrote {len(seqrows)} sequences -> {a.seqs_out}", flush=True)
    # summary: mean over complexes per (direction, alpha)
    g = df.groupby(["direction", "alpha"]).agg(int_rec=("int_recovery", "mean"),
        noni_rec=("noninterface_recovery", "mean"), Lmpnn=("meanL_mpnn", "mean"),
        Lesmif=("meanL_esmif", "mean"), n=("complex_id", "nunique")).reset_index()
    print("\n=== summary (mean over complexes) ===")
    print(g.to_string(index=False))
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()

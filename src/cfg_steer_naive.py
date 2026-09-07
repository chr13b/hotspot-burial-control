#!/usr/bin/env python3
"""Naive-guidance SPECIFICITY arm (pre-registered: results/PREREG_naive.md).

Third steering arm alongside the committed L and random arms of cfg_steer.py. Tilt ProteinMPNN sampling at
interface positions by +alpha*naive, where naive points along the model's OWN complex-conditioned confidence
(what it already prefers), scaled to the SAME per-position magnitude as the +alpha*L tilt. So it differs from L
ONLY in direction -- like the random arm, but pointed at confidence rather than at random:

    conf_i  = lP_i - mean_a(lP_i)                     # centered complex-conditioned log-probs (confidence dir)
    naive_i = ||L_i|| * conf_i / ||conf_i||           # matched magnitude to L_i  (assert ||naive_i|| == ||L_i||)
    bias_i  = alpha * naive_i                          # fed to mpnn_steer.draw(bias_by_res=...), exactly like L/random

The point: if L beats naive under an anti-circular judge (ESM-IF1 / MIF), the gain needs the *binding* direction,
not merely a confident one. Reuses cfg_steer's load_L / interface_set / seq_to_chains and the identical ms.draw
protocol (K, order=None, temp, SEED) so decoding-order variance cancels in the paired L-naive / naive-random
contrasts. Generates ONLY the naive arm (wt/L/random are committed and reused).

  python3 src/cfg_steer_naive.py --limit 2 --K 8  --out results/_smoke_naive.csv --seqs-out results/_smoke_naive_seqs.csv
  python3 src/cfg_steer_naive.py                    --out results/cfg_steer_naive.csv --seqs-out results/cfg_steer_naive_seqs.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "models"), os.path.join(HERE, "decoding")):
    sys.path.insert(0, p)
import ftax_common as fc               # noqa: E402
import leverage_decomposition as LD    # noqa: E402
import mpnn_steer as ms                # noqa: E402
import cfg_steer as cs                 # noqa: E402  (load_L, interface_set, seq_to_chains)
SEED, DATA, AA20, IDX = LD.SEED, LD.DATA, LD.AA20, LD.IDX


def load_lP(pqf):
    """complex_id -> {(chain,resnum,icode): lP[20]} = complex-conditioned log-probs P(a|complex)."""
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
    ap.add_argument("--subset", default="")           # optional file of complex_ids; default = full SET-A
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--alphas", default="0,2")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--dump-k", type=int, default=3)
    ap.add_argument("--out", default="results/cfg_steer_naive.csv")
    ap.add_argument("--seqs-out", default="results/cfg_steer_naive_seqs.csv")
    a = ap.parse_args()
    import torch; torch.set_num_threads(4)
    alphas = [float(x) for x in a.alphas.split(",")]

    Lm = cs.load_L("results/leverage_pq_skempi.csv")            # ProteinMPNN leverage (magnitude source)
    Le = cs.load_L("results/leverage_pq_skempi_esmif.csv")      # ESM-IF1 leverage  (anti-circular judge)
    lPm = load_lP("results/leverage_pq_skempi.csv")             # ProteinMPNN complex-conditioned confidence
    Lmif = cs.load_L("results/leverage_pq_skempi_mif.csv") if os.path.exists("results/leverage_pq_skempi_mif.csv") else {}
    iface = cs.interface_set()
    cids = sorted(set(Lm) & set(Le) & set(lPm) & set(iface))
    if a.subset:
        keep = {l.strip() for l in open(a.subset) if l.strip()}
        cids = [c for c in cids if c in keep]
    if a.limit:
        cids = cids[:a.limit]

    model, _ = fc.load_mpnn(LD.MPNN_W)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[naive-steer] {len(cids)} complexes, K={a.K}, alphas={alphas}, temp={a.temp}, device={device}", flush=True)

    rows, seqrows = [], []
    for ci, cid in enumerate(cids):
        pdb, g1, g2 = cid.split("_"); path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None or cx.n > 700:
            continue
        keys = list(zip(cx.chains, [int(x) for x in cx.resnums], cx.icodes))
        wt = np.array([IDX.get(s, -1) for s in cx.seq])
        Lm_pos = np.zeros((cx.n, 20)); Le_pos = np.zeros((cx.n, 20))
        Lmif_pos = np.zeros((cx.n, 20)); lP_pos = np.zeros((cx.n, 20)); usable = np.zeros(cx.n, bool)
        for i, k in enumerate(keys):
            if k in iface.get(cid, ()) and k in Lm[cid] and k in Le[cid] and k in lPm[cid] and wt[i] >= 0:
                Lm_pos[i] = Lm[cid][k]; Le_pos[i] = Le[cid][k]; lP_pos[i] = lPm[cid][k]; usable[i] = True
                if cid in Lmif and k in Lmif[cid]:
                    Lmif_pos[i] = Lmif[cid][k]
        nu = int(usable.sum())
        if nu < 3:
            continue
        noni = (~usable) & (wt >= 0)
        # naive tilt: confidence direction, matched to ||L_i|| per position (differs from L only in direction)
        conf = lP_pos[usable] - lP_pos[usable].mean(1, keepdims=True)
        cn = np.linalg.norm(conf, axis=1, keepdims=True)
        u = conf / np.where(cn > 1e-8, cn, 1.0)
        Ln = np.linalg.norm(Lm_pos[usable], axis=1, keepdims=True)
        naive_u = (Ln * u).astype(np.float64)                                  # [nu,20]
        assert np.allclose(np.linalg.norm(naive_u, axis=1), Ln[:, 0], atol=1e-6), \
            "naive per-position magnitude must equal ||L_i||"
        naive_pos = np.zeros((cx.n, 20)); naive_pos[usable] = naive_u

        judges = {"mpnn": Lm_pos, "esmif": Le_pos}
        if cid in Lmif:
            judges["mif"] = Lmif_pos
        for al in alphas:
            B = np.zeros((cx.n, 21), np.float32)
            B[usable, :20] = (al * naive_pos[usable]).astype(np.float32)
            S, _ = ms.draw(model, cx, a.K, a.K, order=None, temperature=a.temp, seed=SEED,
                           use_patch=False, featurize=fc.featurize, bias_by_res=B)     # [K, L]
            Su = S[:, usable]
            int_rec = float((Su == wt[usable][None]).mean())
            noni_rec = float((S[:, noni] == wt[noni][None]).mean()) if noni.any() else float("nan")

            def meanL(Lpos):
                idx = np.clip(Su, 0, 19)
                vals = Lpos[usable][np.arange(nu)[None, :], idx]
                return float(np.where(Su < 20, vals, 0.0).mean())
            row = dict(complex_id=cid, direction="naive", alpha=al, n_int=nu,
                       int_recovery=round(int_rec, 4), noninterface_recovery=round(noni_rec, 4))
            for jk, Lp in judges.items():
                row[f"meanL_{jk}"] = round(meanL(Lp), 4)
            rows.append(row)
            if al == alphas[-1]:
                for k in range(min(a.dump_k, S.shape[0])):
                    s = wt.copy(); s[usable] = S[k, usable]
                    seqrows.append(dict(complex_id=cid, direction="naive", alpha=al, k=k,
                                        chains=cs.seq_to_chains(cx, s)))
        print(f"[naive-steer] {ci+1}/{len(cids)} {cid} n_int={nu}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    pd.DataFrame(seqrows).to_csv(a.seqs_out, index=False)
    print(f"[naive-steer] wrote {len(df)} score rows -> {a.out}; {len(seqrows)} sequences -> {a.seqs_out}")
    if len(df):
        g = df.groupby("alpha").agg(int_rec=("int_recovery", "mean"),
            Lmpnn=("meanL_mpnn", "mean"), Lesmif=("meanL_esmif", "mean"),
            n=("complex_id", "nunique")).reset_index()
        print("\n=== naive summary (mean over complexes) ===")
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()

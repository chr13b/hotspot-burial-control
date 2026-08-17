#!/usr/bin/env python3
"""W2 (the decisive scope test): does the leverage binding signal survive OFF-CRYSTAL backbones?

The abstract prescribes 'read binding from the mixed derivative', but designers condition on PREDICTED /
generated backbones, and this project's own expC2 shows a near-analogue of L collapses off the native
manifold. Here we jitter the crystal backbone to a target interface RMSD, re-score P and Q, and recompute
CPI(L|geometry) and Spearman(L, ΔΔG) at each rung — the dose law.

Design honesty:
 * sigma=0 is a POSITIVE CONTROL — it must reproduce the committed crystal CPI(L|geom) ~ +0.059 (rule 6).
 * The monomer inherits the SAME per-residue jitter as the complex (partner removed), so the double
   difference stays a clean partner ablation and the only thing changing is the backbone's distance
   from native.
 * Noised crystals are a LOWER BOUND on the damage: expC2 shows independent reconstructions collapse
   HARDER than distance-matched noise. So survival here is SUGGESTIVE (not proof of design-time survival);
   collapse here is DECISIVE (scope the claim to crystal backbones).

  python3 src/leverage_noise_ladder.py --limit 8 --sigmas 0.0            # smoke / positive control
  python3 src/leverage_noise_ladder.py --sigmas 0.0,0.5,1.0,1.5 --out results/leverage_noise_ladder.csv
"""
import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc
import leverage_decomposition as LD

SEED = 20260803
DATA = LD.DATA
IDX = LD.IDX
COORD_ATTRS = ("N", "CA", "C", "O")


def jitter_inplace(cx, sd, rng):
    for attr in COORD_ATTRS:
        a = getattr(cx, attr, None)
        if a is not None:
            a = np.asarray(a, float)
            setattr(cx, attr, a + rng.normal(0.0, sd, a.shape))


def score_L(model, path, pdb, g1, g2, muts, sd, rng):
    """{key:(wt,mut)} -> {key: L} at a backbone jittered by per-coord std `sd` (0 = crystal)."""
    cx = fc.load_complex(path, pdb, g1, g2)
    if cx is None:
        return {}
    if sd > 0:
        jitter_inplace(cx, sd, rng)
    cxmap = {(cx.chains[j], int(cx.resnums[j]), cx.icodes[j]): j for j in range(cx.n)}
    lP = LD.logdists(fc.mpnn_unconditional_logprobs(model, cx))
    lQ = np.full_like(lP, np.nan)
    for chains in (g1, g2):
        if not chains:
            continue
        mono = fc.load_complex(path, pdb, chains, "", require_both=False)
        if mono is None or mono.n < 5:
            continue
        if sd > 0:                       # give the monomer the SAME jitter as the complex (partner removed)
            for k in range(mono.n):
                j = cxmap.get((mono.chains[k], int(mono.resnums[k]), mono.icodes[k]))
                if j is None:
                    continue
                for attr in COORD_ATTRS:
                    ac = getattr(cx, attr, None); am = getattr(mono, attr, None)
                    if ac is not None and am is not None:
                        np.asarray(am)[k] = np.asarray(ac)[j]
        lQm = LD.logdists(fc.mpnn_unconditional_logprobs(model, mono))
        im = {(c, int(r), i): k for k, (c, r, i)
              in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
        for j in range(cx.n):
            k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
            if k is not None:
                lQ[j] = lQm[k]
        del lQm, mono
    out = {}
    for j in range(cx.n):
        key = (cx.chains[j], int(cx.resnums[j]), cx.icodes[j])
        if key in muts and np.isfinite(lQ[j]).all():
            for wt, mut in muts[key]:          # L depends on the MUTANT identity, not just the position
                if wt in IDX and mut in IDX and cx.seq[j] == wt:
                    out[(key[0], key[1], key[2], mut)] = float(
                        (lP[j, IDX[mut]] - lP[j, IDX[wt]]) - (lQ[j, IDX[mut]] - lQ[j, IDX[wt]]))
    del cx, lP, lQ
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.0,0.5,1.0,1.5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/leverage_noise_ladder.csv")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)

    mut = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    if "is_interface" in mut.columns:
        mut = mut[mut.is_interface == 1]
    mut = mut.dropna(subset=["burial", "nbr", "drsasa", "ddG", "wt", "mut"]).copy()
    mut["icode"] = mut.icode.fillna("").astype(str)
    mut["destab"] = (mut.ddG >= LD.HOT_DDG).astype(int)
    cids = sorted(mut.complex_id.unique())
    if a.limit:
        cids = cids[:a.limit]
        mut = mut[mut.complex_id.isin(cids)].copy()
    print(f"[noise-ladder] {len(mut)} interface mutations, {len(cids)} complexes")
    model, _ = fc.load_mpnn(LD.MPNN_W)

    rows = []
    for sigma in [float(s) for s in a.sigmas.split(",")]:
        sd = sigma / np.sqrt(3.0)                 # per-coord std -> per-atom RMSD ~= sigma
        rng = np.random.default_rng(SEED + int(round(sigma * 100)))
        Lv = {}
        for ci, cid in enumerate(cids):
            pdb, g1, g2 = cid.split("_")
            path = f"{DATA}/PDBs/{pdb}.pdb"
            if not os.path.exists(path):
                continue
            g = mut[mut.complex_id == cid]
            muts = {}
            for r in g.itertuples():
                muts.setdefault((r.chain, int(r.resnum), r.icode), []).append((r.wt, r.mut))
            try:
                for k4, L in score_L(model, path, pdb, g1, g2, muts, sd, rng).items():
                    Lv[(cid,) + k4] = L         # (complex, chain, resnum, icode, mut)
            except Exception as e:
                print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
        mut["Ln"] = [Lv.get((r.complex_id, r.chain, int(r.resnum), r.icode, r.mut), np.nan)
                     for r in mut.itertuples()]
        d = mut.dropna(subset=["Ln"]).reset_index(drop=True)
        y = d.destab.to_numpy().astype(float); grp = d.complex_id.to_numpy()
        for c in ["burial", "nbr", "drsasa", "Ln"]:
            d[c + "z"] = LD.zs(d[c])
        Z = d[["burialz", "nbrz", "drsasaz"]].to_numpy()
        cpi_v, lo, hi, p, _, _ = LD.cpi(y, grp, Z, d["Lnz"].to_numpy().copy(), rng)
        sp = stats.spearmanr(d.Ln, d.ddG, nan_policy="omit").correlation
        print(f"  sigma={sigma:.2f} Å (iRMSD~{sigma:.2f}): n={len(d)} "
              f"CPI(L|geom)={cpi_v:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  Spearman(L,ΔΔG)={sp:+.4f}",
              flush=True)
        rows.append(dict(sigma_A=sigma, approx_iRMSD_A=sigma, n_mut=len(d),
                         cpi_L_geom=round(cpi_v, 5), lo=round(lo, 5), hi=round(hi, 5),
                         p_gt0=round(p, 3), spearman_L_ddG=round(float(sp), 4)))
    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = ("W2 backbone-noise dose law: does leverage survive off-crystal backbones. sigma=0 is the "
                   "crystal positive control (~+0.059). Noised crystals are a LOWER BOUND on damage (expC2).")
    out["command"] = "python3 src/leverage_noise_ladder.py --sigmas " + a.sigmas
    out.to_csv(a.out, index=False)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

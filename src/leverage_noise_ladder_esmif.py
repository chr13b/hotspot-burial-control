#!/usr/bin/env python3
"""#9: does the backbone-error DOSE LAW replicate under a SECOND model family (ESM-IF1)? The paper frames the
~1 A cliff as a property every method reading the same mixed derivative inherits; this tests it empirically on
a GVP-transformer. Same jitter + CPI machinery as leverage_noise_ladder, ESM-IF1 scorer swapped in.

  python3 src/leverage_noise_ladder_esmif.py --sigmas 0.0,0.5,1.0 --limit 60 --out results/leverage_noise_ladder_esmif.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "models"))
import ftax_common as fc
import leverage_decomposition as LD
import leverage_noise_ladder as NL
import ftax_esmif as fe
from leverage_esmif import esmif_lp
SEED = LD.SEED; DATA = LD.DATA; IDX = LD.IDX


def score_L_esmif(model, alphabet, path, pdb, g1, g2, muts, sd, rng, max_residues=1000):
    cx = fc.load_complex(path, pdb, g1, g2)
    if cx is None or (max_residues and cx.n > max_residues):
        return {}
    if sd > 0:
        NL.jitter_inplace(cx, sd, rng)
    cxmap = {(cx.chains[j], int(cx.resnums[j]), cx.icodes[j]): j for j in range(cx.n)}
    lP = LD.logdists(esmif_lp(model, alphabet, cx))
    lQ = np.full_like(lP, np.nan)
    for chains in (g1, g2):
        if not chains:
            continue
        mono = fc.load_complex(path, pdb, chains, "", require_both=False)
        if mono is None or mono.n < 5:
            continue
        if sd > 0:                                    # monomer inherits the complex's jitter (clean ablation)
            for k in range(mono.n):
                j = cxmap.get((mono.chains[k], int(mono.resnums[k]), mono.icodes[k]))
                if j is None:
                    continue
                for attr in NL.COORD_ATTRS:
                    ac, am = getattr(cx, attr, None), getattr(mono, attr, None)
                    if ac is not None and am is not None:
                        np.asarray(am)[k] = np.asarray(ac)[j]
        lQm = LD.logdists(esmif_lp(model, alphabet, mono))
        im = {(c, int(r), i): k for k, (c, r, i) in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
        for j in range(cx.n):
            k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
            if k is not None:
                lQ[j] = lQm[k]
        del lQm, mono
    out = {}
    for j in range(cx.n):
        key = (cx.chains[j], int(cx.resnums[j]), cx.icodes[j])
        if key in muts and np.isfinite(lQ[j]).all():
            for wt, mut in muts[key]:
                if wt in IDX and mut in IDX and cx.seq[j] == wt:
                    out[(key[0], key[1], key[2], mut)] = float(
                        (lP[j, IDX[mut]] - lP[j, IDX[wt]]) - (lQ[j, IDX[mut]] - lQ[j, IDX[wt]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.0,0.5,1.0")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-residues", dest="max_residues", type=int, default=1000)
    ap.add_argument("--out", default="results/leverage_noise_ladder_esmif.csv")
    a = ap.parse_args()
    import torch; torch.set_num_threads(a.threads)
    mut = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    if "is_interface" in mut.columns:
        mut = mut[mut.is_interface == 1]
    mut = mut.dropna(subset=["burial", "nbr", "drsasa", "ddG", "wt", "mut"]).copy()
    mut["icode"] = mut.icode.fillna("").astype(str)
    mut["destab"] = (mut.ddG >= LD.HOT_DDG).astype(int)
    cids = sorted(mut.complex_id.unique())[: a.limit or None]
    mut = mut[mut.complex_id.isin(cids)].copy()
    print(f"[esmif-ladder] {len(mut)} interface mutations, {len(cids)} complexes", flush=True)
    model, alphabet = fe.load_esmif(device="cpu")
    rows = []
    for sigma in [float(s) for s in a.sigmas.split(",")]:
        sd = sigma / np.sqrt(3.0)
        rng = np.random.default_rng(SEED + int(round(sigma * 100)))
        Lv = {}
        for cid in cids:
            pdb, g1, g2 = cid.split("_"); path = f"{DATA}/PDBs/{pdb}.pdb"
            if not os.path.exists(path):
                continue
            g = mut[mut.complex_id == cid]; muts = {}
            for r in g.itertuples():
                muts.setdefault((r.chain, int(r.resnum), r.icode), []).append((r.wt, r.mut))
            try:
                for k4, L in score_L_esmif(model, alphabet, path, pdb, g1, g2, muts, sd, rng, a.max_residues).items():
                    Lv[(cid,) + k4] = L
            except Exception as e:
                print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
        mut["Ln"] = [Lv.get((r.complex_id, r.chain, int(r.resnum), r.icode, r.mut), np.nan) for r in mut.itertuples()]
        d = mut.dropna(subset=["Ln"]).reset_index(drop=True)
        y = d.destab.to_numpy().astype(float); grp = d.complex_id.to_numpy()
        for c in ["burial", "nbr", "drsasa", "Ln"]:
            d[c + "z"] = LD.zs(d[c])
        Z = d[["burialz", "nbrz", "drsasaz"]].to_numpy()
        cpi_v, lo, hi, p, _, _ = LD.cpi(y, grp, Z, d["Lnz"].to_numpy().copy(), rng)
        sp = stats.spearmanr(d.Ln, d.ddG, nan_policy="omit").correlation
        print(f"  sigma={sigma:.2f} A: n={len(d)} CPI(L|geom)={cpi_v:+.5f} [{lo:+.5f},{hi:+.5f}] "
              f"P(>0)={p:.3f}  Spearman(L,ddG)={sp:+.4f}", flush=True)
        rows.append(dict(sigma_A=sigma, n_mut=len(d), cpi_L_geom=round(cpi_v, 5), lo=round(lo, 5),
                         hi=round(hi, 5), p_gt0=round(p, 3), spearman_L_ddG=round(float(sp), 4)))
    out = pd.DataFrame(rows); out["seed"] = SEED; out["model"] = "ESM-IF1"
    out["command"] = f"python3 src/leverage_noise_ladder_esmif.py --sigmas {a.sigmas} --limit {a.limit}"
    out.to_csv(a.out, index=False); print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

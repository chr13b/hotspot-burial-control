#!/usr/bin/env python3
"""Big Idea 1 — does an inverse-folding model rank SSM substitutions by BINDING, or only by fold?

Pre-registered in results/PREREG_knows_where.md (P1/P2/P3, seed 20260803). For each SSM position we take the
19 non-native substitutions, label each 'retains binding' (kd_lb < cap) vs 'abolishes', and ask whether the
model's p(aa|complex backbone) ranks the binders above the non-binders — stratified core (stability, the
positive control) / surface / interface (binding), and complex- vs binder-conditioned. Baselines the model
must beat: BLOSUM62, side-chain volume similarity, hydropathy match. ProteinMPNN unconditional, sequence-free.

  python3 src/bennett_knows_where.py --out results/bennett_knows_where.csv
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc
import bennett_kl_detector as bkd

SEED = 20260803
MPNN20 = "ACDEFGHIKLMNPQRSTVWY"      # index order of the 20-dim distribution (MPNN_ALPHABET[:20])
IDX = {a: i for i, a in enumerate(MPNN20)}
BEN = os.path.expanduser("~/ftax/bennett/x/supplemental_files")
LIBS = ["ALK_SSM1", "ALK_SSM2", "IL2Ra_SSM1", "IL2Ra_SSM2", "LTK_SSM1", "LTK_SSM2", "IL10Ra_SSM"]
AAset = set(MPNN20)
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1,
       "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0,
       "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}


def dists(lp):
    z = lp[:, :20]; z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def full_PQ(model, pdb, parent):
    cx = fc.load_complex(pdb, parent, "A", "B", require_both=False)
    if cx is None or not (cx.group == 1).any():
        return None
    P = dists(fc.mpnn_unconditional_logprobs(model, cx))
    mono = fc.load_complex(pdb, parent, "A", "", require_both=False)
    if mono is None:
        return None
    Q = dists(fc.mpnn_unconditional_logprobs(model, mono))
    im = {(c, int(r), i): k for k, (c, r, i) in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
    out = {}
    for j in range(cx.n):
        if cx.group[j] != 1:
            continue
        k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
        if k is not None:
            out[int(cx.resnums[j])] = (P[j], Q[k], cx.seq[j])
    return out


def per_sub_labels(aff):
    """(parent,pos) -> ({sub: binds}, native_aa)."""
    pr = aff["description"].map(bkd.parse_desc)
    aff = aff.assign(parent=[p[0] for p in pr], pos=[p[1] for p in pr], mut=[p[2] for p in pr])
    aff = aff[aff.parent.notna()].copy()
    aff["kd"] = pd.to_numeric(aff["kd_lb"], errors="coerce")
    cap = pd.to_numeric(aff["highest_conc"], errors="coerce").median()
    cap = cap if np.isfinite(cap) else 1000.0
    aff["binds"] = np.isfinite(aff["kd"]) & (aff["kd"] < cap)
    g = aff.groupby(["parent", "pos", "mut"]).binds.max().reset_index()
    lab = {}
    for (parent, pos), sub in g.groupby(["parent", "pos"]):
        tested = set(sub.mut)
        if len(tested) < 8:
            continue
        miss = AAset - tested
        native = miss.pop() if len(miss) == 1 else None
        lab[(parent, int(pos))] = (dict(zip(sub.mut, sub.binds)), native)
    return lab


def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y)
    m = np.isfinite(score); score, y = score[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bennett_knows_where.csv")
    a = ap.parse_args()
    from Bio.Align import substitution_matrices
    B = substitution_matrices.load("BLOSUM62")

    pdb_index = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{BEN}/design_models_ssm_natives/*/*.pdb")}
    model, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))

    recs = []
    for lib in LIBS:
        af = f"{BEN}/ngs_data_analysis/affinities/{lib}.sc"
        if not os.path.exists(af):
            continue
        lab = per_sub_labels(pd.read_csv(af, sep=r"\s+", engine="python"))
        for parent in sorted({p for (p, _) in lab}):
            pdb = pdb_index.get(parent)
            if pdb is None:
                continue
            try:
                pq = full_PQ(model, pdb, parent)
                bi = bkd.burial_interface(pdb, parent)
            except Exception as e:
                print(f"  skip {parent[:24]}: {type(e).__name__}"); continue
            if not pq:
                continue
            for (par, pos), (subs, native) in lab.items():
                if par != parent or pos not in pq or pos not in bi or native is None or native not in IDX:
                    continue
                Pv, Qv, restype = pq[pos]
                if native != restype:                       # mapping guard (excluded aa == PDB native)
                    continue
                sc, sm = bi[pos]["sasa_complex"], bi[pos]["sasa_mono"]
                dsasa = sm - sc
                rsasa = fc.relative_sasa(sc, restype)
                layer = ("interface" if dsasa > 5 else
                         ("core" if (dsasa <= 1 and np.isfinite(rsasa) and rsasa < 0.15) else
                          ("surface" if (dsasa <= 1 and np.isfinite(rsasa) and rsasa > 0.40) else None)))
                if layer is None:
                    continue
                for sub, binds in subs.items():
                    if sub not in IDX or sub == native:
                        continue
                    recs.append(dict(design=parent, layer=layer, sub=sub, binds=int(binds),
                                     P=float(Pv[IDX[sub]]), Q=float(Qv[IDX[sub]]),
                                     blosum=float(B[native, sub]),
                                     vol=-abs(VOL[sub] - VOL[native]),
                                     hydro=-abs(fc.KD_HYDRO[sub] - fc.KD_HYDRO[native]),
                                     dsasa=float(dsasa), sub_vol=VOL[sub], nat_vol=VOL[native],
                                     rsasa=float(rsasa) if np.isfinite(rsasa) else np.nan))
    d = pd.DataFrame(recs)
    d.to_csv(a.out.replace(".csv", "_pairs.csv"), index=False)
    print(f"pairs: {len(d)}  layers={d.groupby('layer').size().to_dict()}  "
          f"designs={d.design.nunique()}  overall bind-rate={d.binds.mean():.2f}")

    scores = ["P", "Q", "blosum", "vol", "hydro"]
    layers = ["core", "surface", "interface"]
    rng = np.random.default_rng(SEED)
    rows = []
    boot_store = {}
    for layer in layers:
        dl = d[d.layer == layer]
        designs = dl.design.unique()
        idx_by = {g: dl.index[dl.design == g].to_numpy() for g in designs}
        Y = dl.binds.to_numpy(); base = dl.reset_index(drop=True)
        pos = {g: np.where(dl.design.values == g)[0] for g in designs}   # positions within dl
        resamp = [np.concatenate([pos[g] for g in rng.choice(designs, len(designs), True)]) for _ in range(2000)]
        yv = dl.binds.to_numpy()
        for s in scores:
            sv = dl[s].to_numpy()
            o = auc(sv, yv)
            b = np.array([auc(sv[ix], yv[ix]) for ix in resamp])
            boot_store[(layer, s)] = b
            rows.append(dict(layer=layer, score=s, auroc=round(o, 4),
                             lo=round(float(np.nanpercentile(b, 2.5)), 4),
                             hi=round(float(np.nanpercentile(b, 97.5)), 4),
                             n_pairs=len(dl), n_bind=int(yv.sum()), n_designs=len(designs)))
        print(f"\n[{layer}] n={len(dl)} pairs, {int(yv.sum())} retain-binding, {len(designs)} designs")
        for s in scores:
            r = [x for x in rows if x["layer"] == layer and x["score"] == s][0]
            print(f"    AUROC {s:7s} = {r['auroc']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")

    # pre-registered tests (paired on the SAME bootstrap resamples)
    print("\n=== PRE-REGISTERED TESTS ===")
    def paired(b1, b2):
        d_ = b1 - b2
        return float(np.nanpercentile(d_, 2.5)), float(np.nanpercentile(d_, 97.5)), float(np.mean(d_ > 0))
    # P1: interface AUROC(P) > 0.5
    iP = [x for x in rows if x["layer"] == "interface" and x["score"] == "P"][0]
    print(f"  P1  interface AUROC(P) = {iP['auroc']:.3f} [{iP['lo']:.3f},{iP['hi']:.3f}]  "
          f"{'>0.5 ✓' if iP['lo'] > 0.5 else '(CI includes 0.5)'}")
    rows.append(dict(layer="TEST", score="P1_interface_P_gt_0.5", auroc=iP['auroc'], lo=iP['lo'], hi=iP['hi']))
    # NB core/interface use different bootstrap resamples (different designs); compare point + CIs
    cP = [x for x in rows if x["layer"] == "core" and x["score"] == "P"][0]
    print(f"  P2  core AUROC(P) {cP['auroc']:.3f} vs interface {iP['auroc']:.3f}  "
          f"(dissociation if core > interface): Δ = {cP['auroc']-iP['auroc']:+.3f}")
    rows.append(dict(layer="TEST", score="P2_core_minus_interface_P", auroc=round(cP['auroc'] - iP['auroc'], 4)))
    # P3: interface P vs Q (paired bootstrap)
    lo, hi, p = paired(boot_store[("interface", "P")], boot_store[("interface", "Q")])
    print(f"  P3  interface AUROC(P) − AUROC(Q) = {iP['auroc']-[x for x in rows if x['layer']=='interface' and x['score']=='Q'][0]['auroc']:+.3f} "
          f"[{lo:+.3f},{hi:+.3f}] P(>0)={p:.3f}  {'complex-conditioning ADDS' if lo>0 else 'null (occlusion, not energetics)'}")
    rows.append(dict(layer="TEST", score="P3_interface_P_minus_Q", auroc=round(iP['auroc'] - [x for x in rows if x['layer'] == 'interface' and x['score'] == 'Q'][0]['auroc'], 4), lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(p, 3)))

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["command"] = "python3 src/bennett_knows_where.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

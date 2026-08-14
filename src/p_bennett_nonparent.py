#!/usr/bin/env python3
"""T2 — break circularity: does a NON-PARENT inverse-folding model (ESM-IF1) also rank Bennett interface
substitutions by binding, and add beyond the all-atom occlusion baseline?

Pre-registered in results/PREREG_bennett_hardening.md. The SSM parents are ProteinMPNN outputs, so P (from
ProteinMPNN) scores substitutions around the model's own mode — kill-shot #3 (circularity). We re-run the
Big-Idea-1 / T1 analysis with ESM-IF1 (GVP-transformer, 142M params, multichain), which did NOT generate the
parents. P_esm = p(aa | complex backbone), Q_esm = p(aa | binder-alone backbone), both over the SSM position
on the de-novo binder (chain A). Geometry (all-atom clash, contact, ΔSASA, vol) is merged from T1's
bennett_occlusion_allatom_pairs.csv. Design-clustered bootstrap, seed 20260803.

DECISION: circularity DEFUSED if interface AUROC(P_esm) > 0.5 (CI excludes 0.5), P_esm−Q_esm > 0 (CI
excludes 0), and ΔAUROC(P_esm over all-atom geometry) keeps the sign (P adds). Sign flips => parent-specific.

  python3 src/p_bennett_nonparent.py --out results/bennett_nonparent.csv
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "models"))
import ftax_common as fc
import ftax_panel as panel
import bennett_knows_where as bkw
import bennett_kl_detector as bkd

SEED = 20260803
MODEL = "esmif"


def softmax20(mix):
    z = mix - mix.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def full_PQ_panel(handle, pdb, parent):
    cx = fc.load_complex(pdb, parent, "A", "B", require_both=False)
    if cx is None or not (cx.group == 1).any():
        return None
    P = softmax20(fc.order_mixture_logprobs(panel.score(MODEL, handle, cx))[:, :20])
    mono = fc.load_complex(pdb, parent, "A", "", require_both=False)
    if mono is None:
        return None
    Q = softmax20(fc.order_mixture_logprobs(panel.score(MODEL, handle, mono))[:, :20])
    im = {(c, int(r), i): k for k, (c, r, i) in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
    out = {}
    for j in range(cx.n):
        if cx.group[j] != 1:
            continue
        k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
        if k is not None:
            out[int(cx.resnums[j])] = (P[j], Q[k], cx.seq[j])
    return out


def boot_paired(auc_fn, y, g, rng, n=3000):
    ids = np.unique(g); pos = {u: np.where(g == u)[0] for u in ids}
    out = []
    for _ in range(n):
        idx = np.concatenate([pos[u] for u in rng.choice(ids, len(ids), True)])
        out.append(auc_fn(idx))
    return np.array([v for v in out if np.isfinite(v)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bennett_nonparent.csv")
    ap.add_argument("--geom", default="results/bennett_occlusion_allatom_pairs.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    pdb_index = {os.path.basename(p)[:-4]: p for p in
                 glob.glob(f"{bkw.BEN}/design_models_ssm_natives/*/*.pdb")}
    handle = panel.load(MODEL)
    print(f"loaded {MODEL}; scoring {len(pdb_index)} designs (complex + binder-alone) ...", flush=True)

    recs = []
    for lib_name in bkw.LIBS:
        af = f"{bkw.BEN}/ngs_data_analysis/affinities/{lib_name}.sc"
        if not os.path.exists(af):
            continue
        lab = bkw.per_sub_labels(pd.read_csv(af, sep=r"\s+", engine="python"))
        for parent in sorted({p for (p, _) in lab}):
            pdb = pdb_index.get(parent)
            if pdb is None:
                continue
            try:
                pq = full_PQ_panel(handle, pdb, parent)
                bi = bkd.burial_interface(pdb, parent)
            except Exception as e:
                print(f"  skip {parent[:22]}: {type(e).__name__} {e}"); continue
            if not pq:
                continue
            for (par, pos), (subs, native) in lab.items():
                if par != parent or pos not in pq or pos not in bi or native is None or native not in bkw.IDX:
                    continue
                Pv, Qv, restype = pq[pos]
                if native != restype:
                    continue
                dsasa = bi[pos]["sasa_mono"] - bi[pos]["sasa_complex"]
                if dsasa <= 5:
                    continue
                for sub, binds in subs.items():
                    if sub not in bkw.IDX or sub == native:
                        continue
                    recs.append(dict(design=parent, resnum=int(pos), sub=sub, binds=int(binds),
                                     P_esm=float(Pv[bkw.IDX[sub]]), Q_esm=float(Qv[bkw.IDX[sub]])))
    e = pd.DataFrame(recs)
    geo = pd.read_csv(a.geom)[["design", "resnum", "sub", "aa_clash", "dsasa", "vol", "contact",
                               "sub_vol", "nat_vol"]]
    d = e.merge(geo, on=["design", "resnum", "sub"], how="inner").reset_index(drop=True)
    print(f"interface pairs (ESM-IF1 ∩ T1 geometry): {len(d)}  designs {d.design.nunique()}  "
          f"bind-rate {d.binds.mean():.2f}  (merge {len(d)}/{len(e)})")
    y = d.binds.to_numpy(); g = d.design.to_numpy()

    aP = bkw.auc(d.P_esm.values, y); aQ = bkw.auc(d.Q_esm.values, y)
    bP = boot_paired(lambda ix: bkw.auc(d.P_esm.values[ix], y[ix]), y, g, rng)
    bPQ = boot_paired(lambda ix: bkw.auc(d.P_esm.values[ix], y[ix]) - bkw.auc(d.Q_esm.values[ix], y[ix]), y, g, rng)
    P_lo, P_hi = np.percentile(bP, [2.5, 97.5])
    PQ_lo, PQ_hi, PQ_p = np.percentile(bPQ, 2.5), np.percentile(bPQ, 97.5), float(np.mean(bPQ > 0))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    Z = lambda c: ((d[c] - d[c].mean()) / (d[c].std() + 1e-9)).to_numpy()
    d["clash_c"] = d.aa_clash * d.dsasa
    G = np.column_stack([Z("aa_clash"), Z("clash_c"), Z("dsasa"), Z("vol"), Z("contact")])
    GP = np.column_stack([G, Z("P_esm")])
    o_g = np.zeros(len(y)); o_gp = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(G, y, g):
        o_g[te] = LogisticRegression(max_iter=1000).fit(G[tr], y[tr]).predict_proba(G[te])[:, 1]
        o_gp[te] = LogisticRegression(max_iter=1000).fit(GP[tr], y[tr]).predict_proba(GP[te])[:, 1]
    ag, agp = bkw.auc(o_g, y), bkw.auc(o_gp, y)
    bD = boot_paired(lambda ix: bkw.auc(o_gp[ix], y[ix]) - bkw.auc(o_g[ix], y[ix]), y, g, rng)
    D_lo, D_hi, D_p = np.percentile(bD, 2.5), np.percentile(bD, 97.5), float(np.mean(bD > 0))

    defused = (P_lo > 0.5) and (PQ_lo > 0) and (D_lo > 0)
    verdict = ("circularity DEFUSED (non-parent ESM-IF1 reproduces the signal)" if defused
               else "signal weakens/flips under non-parent scorer — inspect")
    print(f"  interface AUROC(P_esm)   = {aP:.3f} [{P_lo:.3f},{P_hi:.3f}]   (AUROC(Q_esm) {aQ:.3f})")
    print(f"  P_esm − Q_esm            = {aP-aQ:+.3f} [{PQ_lo:+.3f},{PQ_hi:+.3f}] P(>0)={PQ_p:.3f}")
    print(f"  ΔAUROC(P_esm over all-atom geometry) = {agp-ag:+.4f} [{D_lo:+.4f},{D_hi:+.4f}] P(>0)={D_p:.3f}")
    print(f"  VERDICT: {verdict}")

    rows = [dict(metric="interface_auroc_P_esm", value=round(aP, 4), lo=round(P_lo, 4), hi=round(P_hi, 4)),
            dict(metric="interface_auroc_Q_esm", value=round(aQ, 4)),
            dict(metric="P_minus_Q_esm", value=round(aP - aQ, 4), lo=round(PQ_lo, 4), hi=round(PQ_hi, 4), p_gt0=round(PQ_p, 3)),
            dict(metric="dAUROC_Pesm_over_allatom_geometry", value=round(agp - ag, 4), lo=round(D_lo, 4), hi=round(D_hi, 4), p_gt0=round(D_p, 3)),
            dict(metric="verdict", value=verdict)]
    out = pd.DataFrame(rows); out["model"] = MODEL; out["n_pairs"] = len(d); out["n_designs"] = d.design.nunique()
    out["seed"] = SEED; out["command"] = "python3 src/p_bennett_nonparent.py"
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

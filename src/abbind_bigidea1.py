#!/usr/bin/env python3
"""AB-Bind Big-Idea-1 replication — does the model's per-mutation complex-conditioned DISTRIBUTION predict
antibody-antigen binding ΔΔG BEYOND geometry + substitution-similarity? (replicates the POSITIVE on a 2nd
fixture, not just the nugget.)

For each measured single mutation (wt→mut at an interface position) we take the model's log p(mut|complex
backbone) and log p(mut|binder-alone) (partner-conditioning), and test whether they rank mutations by
measured ΔΔG (destabilising binding, ΔΔG≥1) beyond a geometry+similarity baseline (burial, ΔSASA, BLOSUM62,
volume). ProteinMPNN unconditional; complex-clustered bootstrap, seed 20260803.
  python3 src/abbind_bigidea1.py --out results/abbind_bigidea1.csv
"""
import argparse, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ftax_common as fc
import abbind_nugget as AB

SEED = 20260803
MPNN20 = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(MPNN20)}
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1,
       "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0,
       "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}


def dists(lp):
    z = lp[:, :20]; z = z - z.max(1, keepdims=True); p = np.exp(z); return np.log(p / p.sum(1, keepdims=True))


def auc(s, y):
    from scipy import stats
    s = np.asarray(s, float); y = np.asarray(y); m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/abbind_bigidea1.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    from Bio.Align import substitution_matrices
    BL = substitution_matrices.load("BLOSUM62")

    d = pd.read_csv(f"{AB.ABDIR}/AB-Bind_experimental_data.csv", encoding="latin-1").rename(
        columns={"#PDB": "pdb", "Partners(A_B)": "partners", "ddG(kcal/mol)": "ddg"})
    d["ddg"] = pd.to_numeric(d["ddg"], errors="coerce")
    single = d[~d["Mutation"].astype(str).str.contains(",")].copy()
    muts = []
    for _, r in single.iterrows():
        m = AB.MUT_RE.match(str(r["Mutation"]).strip())
        if m and np.isfinite(r["ddg"]):
            muts.append(dict(pdb=r["pdb"], chain=m.group(1), wt=m.group(2), resnum=int(m.group(3)),
                             mut=m.group(4), ddg=float(r["ddg"])))
    muts = pd.DataFrame(muts)
    partners = d.dropna(subset=["partners"]).groupby("pdb").partners.first().to_dict()

    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    recs = []
    for pdb, part in sorted(partners.items()):
        g1, g2 = part.split("_"); path = f"{AB.ABDIR}/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
            if cx is None or cx.n > 2500:
                continue
            P = dists(fc.mpnn_unconditional_logprobs(model, cx))
            Qmap = {}
            for grp in (g1, g2):
                mono = fc.load_complex(path, pdb, grp, "", require_both=False)
                if mono is None:
                    continue
                Qm = dists(fc.mpnn_unconditional_logprobs(model, mono))
                for k in range(mono.n):
                    Qmap[(mono.chains[k], int(mono.resnums[k]))] = Qm[k]
            pos = {(cx.chains[j], int(cx.resnums[j])): j for j in range(cx.n)}
            mm = muts[muts.pdb == pdb]
            for _, r in mm.iterrows():
                j = pos.get((r.chain, r.resnum))
                if j is None or r.mut not in IDX or r.wt not in IDX:
                    continue
                q = Qmap.get((r.chain, r.resnum))
                recs.append(dict(pdb=pdb, chain=r.chain, resnum=r.resnum, wt=r.wt, mut=r.mut, ddg=r.ddg,
                                 logP=float(P[j, IDX[r.mut]]),
                                 logodds=float(P[j, IDX[r.mut]] - q[IDX[r.mut]]) if q is not None else np.nan,
                                 blosum=float(BL[r.wt, r.mut]), dvol=-abs(VOL[r.mut] - VOL[r.wt])))
            print(f"  {pdb}: {len(mm)} muts", flush=True)
        except Exception as e:
            print(f"  skip {pdb}: {type(e).__name__}: {e}", flush=True)
    e = pd.DataFrame(recs)
    geo = pd.read_csv("results/abbind_positions.csv")[["pdb", "chain", "resnum", "burial", "drsasa", "is_interface"]]
    e = e.merge(geo, on=["pdb", "chain", "resnum"], how="left")
    ei = e[e.is_interface == 1].dropna(subset=["logP", "burial", "drsasa"]).reset_index(drop=True)
    ei["destab"] = (ei.ddg >= 1.0).astype(int)
    y = ei.destab.to_numpy(); g = ei.pdb.to_numpy()
    from scipy import stats
    print(f"\ninterface mutations: {len(ei)}  complexes {ei.pdb.nunique()}  destabilising(ΔΔG≥1) {int(y.sum())}")

    def boot(score):
        ids = np.unique(g); ix = {c: np.where(g == c)[0] for c in ids}; out = []
        for _ in range(5000):
            t = np.concatenate([ix[c] for c in rng.choice(ids, len(ids), True)])
            v = auc(score[t], y[t])
            if np.isfinite(v):
                out.append(v)
        return np.percentile(out, [2.5, 97.5])

    rows = []
    for name, s in [("neg_logP(complex)", -ei.logP.values), ("neg_logodds(P/Q)", -ei.logodds.values),
                    ("burial", ei.burial.values), ("dSASA", ei.drsasa.values),
                    ("neg_BLOSUM", -ei.blosum.values), ("vol_dissim", -ei.dvol.values)]:
        au = auc(s, y)
        if np.isfinite(au):
            lo, hi = boot(s)
            print(f"  AUROC(destab) {name:18s} = {au:.3f} [{lo:.3f},{hi:.3f}]")
            rows.append(dict(feature=name, auroc=round(au, 4), lo=round(lo, 4), hi=round(hi, 4)))
    sp = stats.spearmanr(ei.logP, ei.ddg, nan_policy="omit")
    print(f"  Spearman(logP, ΔΔG) = {sp.correlation:+.3f}  (expect NEGATIVE: low prob → destabilising)")

    # does the model distribution add BEYOND geometry + substitution similarity?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    Z = lambda c: ((ei[c] - ei[c].mean()) / (ei[c].std() + 1e-9)).to_numpy()
    base = np.column_stack([Z("burial"), Z("drsasa"), Z("blosum"), Z("dvol")])
    baseP = np.column_stack([base, Z("logP")])
    ob = np.zeros(len(y)); obp = np.zeros(len(y))
    for tr, te in GroupKFold(min(5, ei.pdb.nunique())).split(base, y, g):
        ob[te] = LogisticRegression(max_iter=1000).fit(base[tr], y[tr]).predict_proba(base[te])[:, 1]
        obp[te] = LogisticRegression(max_iter=1000).fit(baseP[tr], y[tr]).predict_proba(baseP[te])[:, 1]
    ab_, abp = auc(ob, y), auc(obp, y)
    ids = np.unique(g); ix = {c: np.where(g == c)[0] for c in ids}; dd = []
    for _ in range(5000):
        t = np.concatenate([ix[c] for c in rng.choice(ids, len(ids), True)]); yy = y[t]
        if 0 < yy.sum() < len(yy):
            dd.append(auc(obp[t], yy) - auc(ob[t], yy))
    dd = np.array(dd); lo, hi, p = np.percentile(dd, 2.5), np.percentile(dd, 97.5), float(np.mean(dd > 0))
    print(f"\n  geometry+similarity AUROC {ab_:.3f}  |  +model logP {abp:.3f}")
    print(f"  ΔAUROC(logP over geometry+similarity) = {abp-ab_:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}")
    verdict = "POSITIVE replicates (distribution adds beyond geometry)" if lo > 0 else "does not add (CI spans 0)"
    print(f"  -> {verdict}")
    rows += [dict(feature="spearman_logP_ddg", auroc=round(float(sp.correlation), 4)),
             dict(feature="geom_sim_baseline", auroc=round(ab_, 4)),
             dict(feature="geom_sim+logP", auroc=round(abp, 4)),
             dict(feature="dAUROC_logP_over_geom_sim", auroc=round(abp - ab_, 4), lo=round(float(lo), 4),
                  hi=round(float(hi), 4), p_gt0=round(p, 3), verdict=verdict)]
    out = pd.DataFrame(rows); out["seed"] = SEED; out["n"] = len(ei); out["n_destab"] = int(y.sum())
    out["command"] = "python3 src/abbind_bigidea1.py"
    out.to_csv(a.out, index=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

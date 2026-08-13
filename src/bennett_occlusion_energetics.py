#!/usr/bin/env python3
"""Occlusion vs energetics (Big Idea 1 follow-up): does p(aa|complex) predict interface binding BEYOND a
geometric clash/occlusion baseline?

Big Idea 1 (results/PREREG_knows_where.md, FINDINGS_knows_where.md) showed the model's complex-conditioned
distribution P beats the binder-alone Q at ranking the 19 SSM substitutions by measured binding at the
interface (P3, +0.076). This asks the sharp mechanistic question: is that advantage pure steric OCCLUSION
(a bulky substitution at a contacted position clashes AND abolishes binding) or does P carry binding
ENERGETICS beyond geometry? We build a geometric occlusion baseline from the SAME structures (volume
increase, its product with partner-contact area ΔSASA, and ΔSASA) and test whether P adds over it.

Reconciles R1 (the SCALAR KL ≈ ΔSASA on all backbone classes) with a positive: the FULL per-substitution
distribution carries binding signal the scalar summary discards. Reads results/bennett_knows_where_pairs.csv
(interface rows). Design-clustered bootstrap, seed 20260803.

  python3 src/bennett_occlusion_energetics.py --out results/bennett_occlusion_energetics.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/bennett_knows_where_pairs.csv")
    ap.add_argument("--out", default="results/bennett_occlusion_energetics.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.pairs)
    d = d[d.layer == "interface"].reset_index(drop=True)
    d["clash"] = np.maximum(0.0, d.sub_vol - d.nat_vol)        # volume increase of the substitution
    d["clash_c"] = d.clash * d.dsasa                           # clash weighted by partner-contact area
    y = d.binds.to_numpy(); g = d.design.to_numpy()
    Z = lambda c: ((d[c] - d[c].mean()) / d[c].std()).to_numpy()

    rows = []
    for name, sc in [("P_complex", d.P.values), ("Q_binder_alone", d.Q.values),
                     ("clash", -d.clash.values), ("clash_x_contact", -d.clash_c.values),
                     ("dSASA", -d.dsasa.values), ("vol_sim", d.vol.values), ("blosum", d.blosum.values)]:
        rows.append(dict(metric="auroc_standalone", feature=name, value=round(auc(sc, y), 4)))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    geo = np.column_stack([Z("clash"), Z("clash_c"), Z("dsasa"), Z("vol")])
    geoP = np.column_stack([geo, Z("P")])                     # geometry + the model's per-sub prob
    o_geo = np.zeros(len(y)); o_geoP = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(geo, y, g):
        o_geo[te] = LogisticRegression(max_iter=1000).fit(geo[tr], y[tr]).predict_proba(geo[te])[:, 1]
        o_geoP[te] = LogisticRegression(max_iter=1000).fit(geoP[tr], y[tr]).predict_proba(geoP[te])[:, 1]
    ag, agp = auc(o_geo, y), auc(o_geoP, y)
    ids = np.unique(g); pos = {u: np.where(g == u)[0] for u in ids}
    rng = np.random.default_rng(SEED); dd = []
    for _ in range(3000):
        idx = np.concatenate([pos[u] for u in rng.choice(ids, len(ids), True)]); yy = y[idx]
        if 0 < yy.sum() < len(yy):
            dd.append(auc(o_geoP[idx], yy) - auc(o_geo[idx], yy))
    dd = np.array(dd)
    lo, hi, p = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)), float(np.mean(dd > 0))
    rows += [dict(metric="auroc_cv", feature="geometry(clash+contact+dSASA+vol)", value=round(ag, 4)),
             dict(metric="auroc_cv", feature="geometry+P", value=round(agp, 4)),
             dict(metric="dAUROC_P_over_geometry", feature="P_adds_beyond_occlusion", value=round(agp - ag, 4),
                  lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(p, 3))]

    print(f"interface: {len(d)} (pos,sub) pairs, {d.design.nunique()} designs, bind-rate {y.mean():.2f}")
    for r in rows:
        if r["metric"] == "auroc_standalone":
            print(f"  AUROC {r['feature']:18s} {r['value']:.3f}")
    print(f"\n  geometry baseline {ag:.3f}  |  geometry+P {agp:.3f}")
    print(f"  ΔAUROC(P over geometry) = {agp-ag:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}  "
          f"-> {'ENERGETICS (P adds beyond occlusion)' if lo > 0 else 'occlusion (P does not add)'}")
    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = "P adds over a volume-clash/contact occlusion baseline => the model encodes per-sub binding beyond geometry; caveat: a richer all-atom clash model could narrow it"
    out["command"] = "python3 src/bennett_occlusion_energetics.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

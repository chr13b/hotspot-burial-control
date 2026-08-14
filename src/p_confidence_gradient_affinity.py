#!/usr/bin/env python3
"""EXPLORATORY power-up of the binding-permanence gradient — continuous WT affinity over ALL SKEMPI.

The category version (p_confidence_gradient.py) was underpowered: 3 coarse Hold_out_type bins over 50
complexes. Here we use a CONTINUOUS binding-dominance axis — WT binding affinity pKd = -log10(Kd) — over
ALL SKEMPI complexes with a measured affinity (~140, not 50), burial-controlled. Tighter/more-permanent
complexes (higher pKd) are more binding-dominated → the constraint-vs-leverage theory predicts confidence
should track interface hotspots MORE there. Test: logistic is_hot ~ z(burial) + z(conf) + z(conf)·z(pKd),
complex-clustered bootstrap on the interaction coefficient. Same-fixture (no de-novo pooling → no fixture
confound). Still EXPLORATORY (post-hoc; supports pre-registered T3, not a fresh confirmatory law).
  python3 src/p_confidence_gradient_affinity.py --out results/confidence_gradient_affinity.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/confidence_gradient_affinity.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    j = pd.read_csv("results/kl_detector_joined.csv"); j = j[j.is_interface == 1].copy()
    j["pdb"] = j.complex_id.str.split("_").str[0].str.upper()
    sk = pd.read_csv("/home/chris/ftax/data/skempi_v2.csv", sep=";", low_memory=False)
    sk["pdb"] = sk["#Pdb"].astype(str).str.split("_").str[0].str.upper()
    sk["kd"] = pd.to_numeric(sk["Affinity_wt_parsed"], errors="coerce")
    aff = sk[np.isfinite(sk.kd) & (sk.kd > 0)].groupby("pdb").kd.median()   # per-complex WT Kd (M)
    j["kd"] = j.pdb.map(aff)
    j = j.dropna(subset=["logp_native", "is_hot", "burial", "kd"]).copy()
    j["pkd"] = -np.log10(j.kd)                                              # higher = tighter = more binding-dominated
    n_cx = j.complex_id.nunique()
    print(f"SKEMPI interface positions with WT affinity: {len(j)}  complexes {n_cx}  "
          f"pKd range [{j.pkd.min():.1f},{j.pkd.max():.1f}] median {j.pkd.median():.1f}")

    # descriptive: affinity tertiles (per complex) -> conf-AUROC + burial
    cx = j.drop_duplicates("complex_id")[["complex_id", "pkd"]]
    cx["tert"] = pd.qcut(cx.pkd, 3, labels=["weak", "mid", "tight"])
    j = j.merge(cx[["complex_id", "tert"]], on="complex_id")
    rows = []
    print("affinity tertile: conf-AUROC, mean burial, n_hot, n_cx")
    for t in ["weak", "mid", "tight"]:
        d = j[j.tert == t]
        auc_c = auc(d.logp_native.values, d.is_hot.values)
        print(f"  {t:5s} conf-AUROC={auc_c:.3f}  mean_burial={d.burial.mean():+.3f}  "
              f"n_hot={int(d.is_hot.sum())} n_cx={d.complex_id.nunique()} pKd_med={d.pkd.median():.1f}")
        rows.append(dict(metric="tertile_auroc", tertile=t, conf_auroc=round(auc_c, 4),
                         mean_burial=round(float(d.burial.mean()), 4), n_hot=int(d.is_hot.sum()),
                         n_cx=int(d.complex_id.nunique())))

    # burial-controlled continuous interaction
    from sklearn.linear_model import LogisticRegression
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    j["zb"], j["zc"], j["zp"] = z(j.burial), z(j.logp_native), z(j.pkd)
    j["zc_zp"] = j.zc * j.zp
    X = j[["zb", "zc", "zp", "zc_zp"]].to_numpy(); y = j.is_hot.to_numpy().astype(float)
    g = j.complex_id.to_numpy()

    def fit(idx):
        return LogisticRegression(max_iter=3000).fit(X[idx], y[idx]).coef_[0][3]   # conf·pKd interaction
    coef = fit(np.arange(len(y)))
    cids = np.unique(g); pos = {c: np.where(g == c)[0] for c in cids}
    bs = []
    for _ in range(4000):
        idx = np.concatenate([pos[c] for c in rng.choice(cids, len(cids), True)])
        if 0 < y[idx].sum() < len(idx):
            try:
                bs.append(fit(idx))
            except Exception:
                pass
    bs = np.array(bs); lo, hi, p = np.percentile(bs, 2.5), np.percentile(bs, 97.5), float(np.mean(bs > 0))
    verdict = ("POWERED: confidence tracks hotspots MORE in tighter/more-binding-dominated complexes, "
               "beyond burial" if lo > 0 else "still spans 0")
    print(f"\n  interaction conf·pKd | burial = {coef:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}")
    print(f"  corr(pKd, burial) per position = {stats.spearmanr(j.pkd, j.burial).correlation:+.3f} "
          f"(confound check; interaction controls for burial)")
    print(f"  -> {verdict}   (n_cx={n_cx}, {int(y.sum())} hot)")
    rows.append(dict(metric="interaction_conf_x_pkd_given_burial", conf_auroc=round(coef, 4),
                     lo=round(float(lo), 4), hi=round(float(hi), 4), p_gt0=round(p, 3),
                     mean_burial=round(float(stats.spearmanr(j.pkd, j.burial).correlation), 4),
                     n_hot=int(y.sum()), n_cx=n_cx, note=verdict))
    out = pd.DataFrame(rows); out["seed"] = SEED
    out["framing"] = "EXPLORATORY continuous-affinity power-up of the binding-permanence gradient; supports pre-registered T3"
    out["command"] = "python3 src/p_confidence_gradient_affinity.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

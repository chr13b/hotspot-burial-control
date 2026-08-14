#!/usr/bin/env python3
"""EXPLORATORY — the binding-permanence gradient (supports the pre-registered T3 constraint-vs-leverage finding).

NOT a second pre-registered law: SKEMPI has no obligate endpoint (it measures Kd → transient by construction),
and these category AUROCs were seen during the feasibility check, so this is EXPLORATORY support for the
pre-registered de-novo-vs-natural contrast (T3, bennett_conf_fork.csv), not a fresh confirmatory test.

Theory (constraint-vs-leverage): confidence estimates positional CONSTRAINT; hotspot-ness is binding LEVERAGE;
they coincide only when selection is binding-dominated → confidence should predict interface hotspots better
in more binding-dominated regimes. A-priori binding-permanence ordering by textbook affinity (independent of
the AUROCs): TCR/pMHC (μM) < AB/AG (nM) < Pr/PI (sub-nM/quasi-permanent) < de-novo (optimised). We report
per-category confidence-AUROC (complex-bootstrap CIs) + mean burial, and a BURIAL-CONTROLLED interaction test:
does confidence's hotspot-predictiveness rise with permanence rank, controlling for burial (the flagged
confound)? Complex-clustered bootstrap, seed 20260803.
  python3 src/p_confidence_gradient.py --out results/confidence_gradient.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803
DENOVO_AUROC = 0.596   # Bennett de-novo endpoint (T3, logp_native); reported descriptively (different fixture)
# a-priori permanence rank from textbook affinity (set BEFORE looking at AUROCs); 'other' unclassified -> excluded from trend
RANK = {"TCR/pMHC": 0, "AB/AG": 1, "Pr/PI": 2}


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_auc(d, rng, n=5000):
    cids = d.complex_id.unique(); idx = {c: np.where(d.complex_id.values == c)[0] for c in cids}
    y = d.is_hot.values; s = d.logp_native.values; out = []
    for _ in range(n):
        t = np.concatenate([idx[c] for c in rng.choice(cids, len(cids), True)])
        a = auc(s[t], y[t])
        if np.isfinite(a):
            out.append(a)
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/confidence_gradient.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    j = pd.read_csv("results/kl_detector_joined.csv"); j = j[j.is_interface == 1].copy()
    sk = pd.read_csv("/home/chris/ftax/data/skempi_v2.csv", sep=";", low_memory=False)
    sk["pdb"] = sk["#Pdb"].astype(str).str.split("_").str[0].str.upper()
    ht = sk.drop_duplicates("pdb").set_index("pdb")["Hold_out_type"]
    j["pdb"] = j.complex_id.str.split("_").str[0].str.upper()
    j["cat"] = j.pdb.map(ht).fillna("other").replace({"AB/AG,Pr/PI": "Pr/PI"})
    j = j.dropna(subset=["logp_native", "is_hot", "burial"])

    rows = []
    print("per-category SKEMPI interface hotspot confidence-AUROC (complex-bootstrap CI):")
    for c in ["TCR/pMHC", "AB/AG", "other", "Pr/PI"]:
        d = j[j.cat == c]
        if len(d) == 0:
            continue
        a_ = auc(d.logp_native.values, d.is_hot.values); lo, hi = boot_auc(d, rng)
        print(f"  {c:9s} conf-AUROC={a_:.3f} [{lo:.3f},{hi:.3f}]  mean_burial={d.burial.mean():+.3f}  "
              f"n_hot={int(d.is_hot.sum())} n_cx={d.complex_id.nunique()}  rank={RANK.get(c,'—')}")
        rows.append(dict(regime=c, rank=RANK.get(c, np.nan), conf_auroc=round(a_, 4), lo=round(lo, 4),
                         hi=round(hi, 4), mean_burial=round(float(d.burial.mean()), 4),
                         n_hot=int(d.is_hot.sum()), n_cx=int(d.complex_id.nunique())))
    print(f"  de-novo   conf-AUROC={DENOVO_AUROC:.3f} (Bennett/T3 endpoint, different fixture)  rank=3")
    rows.append(dict(regime="de-novo", rank=3, conf_auroc=DENOVO_AUROC, mean_burial=np.nan,
                     n_hot="", n_cx="", note="Bennett T3 endpoint"))

    # --- burial-controlled interaction (the confound killer): is_hot ~ z_bur + z_conf + z_conf:rank ---
    from sklearn.linear_model import LogisticRegression
    lab = j[j.cat.isin(RANK)].copy()
    lab["rk"] = lab.cat.map(RANK).astype(float)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    lab["zb"], lab["zc"] = z(lab.burial), z(lab.logp_native)
    lab["zc_rk"] = lab.zc * lab.rk
    X = lab[["zb", "zc", "zc_rk"]].to_numpy(); y = lab.is_hot.to_numpy().astype(float)
    g = lab.complex_id.to_numpy()

    def fit_interaction(idx):
        m = LogisticRegression(max_iter=2000).fit(X[idx], y[idx])
        return m.coef_[0][2]                       # coefficient on zc:rank
    coef = fit_interaction(np.arange(len(y)))
    cids = np.unique(g); pos = {c: np.where(g == c)[0] for c in cids}
    bs = []
    for _ in range(3000):
        idx = np.concatenate([pos[c] for c in rng.choice(cids, len(cids), True)])
        if 0 < y[idx].sum() < len(idx):
            try:
                bs.append(fit_interaction(idx))
            except Exception:
                pass
    bs = np.array(bs); clo, chi, cp = np.percentile(bs, 2.5), np.percentile(bs, 97.5), float(np.mean(bs > 0))
    verdict = ("gradient survives burial control (confidence predicts hotspots MORE in higher-permanence "
               "regimes, beyond burial)") if clo > 0 else "interaction CI spans 0 (underpowered / weak)"
    print(f"\n  burial-controlled interaction (conf × permanence-rank | burial): "
          f"{coef:+.4f} [{clo:+.4f},{chi:+.4f}] P(>0)={cp:.3f}\n  -> {verdict}")
    print(f"  (labeled-category n={len(lab)}, {int(y.sum())} hot, {lab.complex_id.nunique()} complexes)")
    rows.append(dict(regime="INTERACTION_conf_x_rank_given_burial", conf_auroc=round(coef, 4),
                     lo=round(float(clo), 4), hi=round(float(chi), 4), p_gt0=round(cp, 3), note=verdict))

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["framing"] = "EXPLORATORY support for pre-registered T3; binding-permanence axis (NOT obligate/transient; SKEMPI has no obligate endpoint)"
    out["command"] = "python3 src/p_confidence_gradient.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

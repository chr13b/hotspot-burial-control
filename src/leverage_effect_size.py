#!/usr/bin/env python3
"""Effect-size context for the leverage (extension C): is CPI +0.059 / Spearman -0.30 "a lot"?

Answer it head-to-head against the best CHEAP SUPERVISED baseline a practitioner would build. The
punchline we expect: the leverage L is a ZERO-SHOT readout of a model NEVER trained on binding, yet it
rivals and adds to a supervised geometry+substitution classifier fit directly on the binding labels.

On SKEMPI interface single mutations (results/leverage_skempi_mutations.csv), predict `destab`
(ΔΔG_bind ≥ 1). Cross-fit (GroupKFold by complex, so no complex leaks train->test) AUROC of:
  (1) geometry            [burial, nbr, ΔSASA]                       -- free geometry
  (2) geometry+substitution [ +BLOSUM, Δvol, Δhydro ]               -- the cheap SUPERVISED baseline
  (3) geometry+substitution+L                                        -- the lift from leverage
  (4) L ALONE, zero-shot  (rank by -L, NO fitting on binding labels) -- the honest comparison
Plus Spearman(feature, ΔΔG). Complex-clustered bootstrap CIs, seed 20260803.
  python3 src/leverage_effect_size.py --out results/leverage_effect_size.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 20260803


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def xfit_scores(X, y, g):
    """Out-of-fold predicted scores from a GroupKFold logistic on X."""
    eta = np.zeros(len(y))
    nf = int(min(5, len(np.unique(g))))
    for tr, te in GroupKFold(nf).split(X, y, g):
        m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        eta[te] = X[te] @ m.coef_[0] + m.intercept_[0]
    return eta


def boot_auc(s, y, g, rng, n=2000):
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    out = []
    for _ in range(n):
        t = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        out.append(auc(s[t], y[t]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/leverage_skempi_mutations.csv")
    ap.add_argument("--out", default="results/leverage_effect_size.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(a.csv)
    if "is_interface" in d.columns:
        d = d[d.is_interface == 1]
    need = ["burial", "nbr", "drsasa", "blosum", "dvol", "dhydro", "L", "ddG", "destab"]
    d = d.dropna(subset=need).reset_index(drop=True)
    y = d.destab.to_numpy().astype(float)
    g = d.complex_id.to_numpy()
    for c in ["burial", "nbr", "drsasa", "blosum", "dvol", "dhydro", "L"]:
        d[c + "z"] = zs(d[c])
    print(f"[effect-size] {len(d)} interface mutations, {len(np.unique(g))} complexes, "
          f"{int(y.sum())} destabilising ({100*y.mean():.1f}%)")

    geo = ["burialz", "nbrz", "drsasaz"]
    sub = geo + ["blosumz", "dvolz", "dhydroz"]
    models = [("geometry", geo, True),
              ("geometry+substitution (supervised baseline)", sub, True),
              ("geometry+substitution+leverage L", sub + ["Lz"], True),
              ("leverage L ALONE (zero-shot, unfit)", ["Lz"], False)]
    rows = []
    for name, cols, fit in models:
        if fit:
            s = xfit_scores(d[cols].to_numpy(), y, g)
        else:
            s = -d["Lz"].to_numpy()      # zero-shot: high destab risk = LOW L, no training on y
        a0 = auc(s, y)
        lo, hi = boot_auc(s, y, g, rng)
        print(f"  AUROC[{name:44s}] = {a0:.4f} [{lo:.4f},{hi:.4f}]")
        rows.append(dict(model=name, auroc=round(float(a0), 4), lo=round(lo, 4), hi=round(hi, 4),
                         fit_on_binding_labels=fit, n=len(d)))

    # Spearman of each raw feature with experimental ΔΔG (context for the -0.30)
    print("  --- Spearman(feature, ΔΔG_bind) ---")
    for f, lab in [("L", "leverage L (zero-shot)"), ("blosum", "BLOSUM62"), ("burial", "burial"),
                   ("drsasa", "ΔSASA"), ("conf", "confidence")]:
        if f not in d.columns:
            continue
        sp = stats.spearmanr(d[f], d.ddG, nan_policy="omit").correlation
        print(f"    {lab:26s}: {sp:+.4f}")
        rows.append(dict(model=f"spearman_{lab}_vs_ddG", auroc=round(float(sp), 4), n=len(d)))

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = ("effect-size context (C): zero-shot leverage L vs the best cheap supervised "
                   "geometry+substitution baseline, cross-fit AUROC for destabilising (ΔΔG_bind≥1)")
    out["command"] = "python3 src/leverage_effect_size.py"
    out.to_csv(a.out, index=False)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

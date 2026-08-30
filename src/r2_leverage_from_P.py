"""R1-killer + non-vacuity, threshold-free. The reviewer's decisive objection: the no-go tests three
hand-picked *scalars* of P (confidence, negentropy, KL) entered *linearly*; a *learned* function of the whole
bound distribution might recover the mixed derivative. So we regress leverage on the ENTIRE 20-vector P with a
flexible learner (gradient boosting) and report out-of-sample R^2(L | P). 1 - R^2 is the fraction of the
mixed derivative that is *irreducible* from the bound distribution — a model-comparable statistic with no TV
threshold, so it also replaces the ESM-IF1-confounded matched-pair ratio in nonvacuity.csv.

Pre-registration (before running): the complex one-pass log-odds vector logP(a)-logP(wt) IS a deterministic
function of P, so a flexible learner will recover *that* component of L; we predict a MODERATE R^2 (the paper's
own corr(one-pass, L) = +0.64 upper-bounds the linear part at ~0.4), leaving a MAJORITY of L irreducible from
P. Falsifier: if R^2(L_rms|P) > 0.7 for either model, leverage is largely a function of P and the Proposition's
non-vacuity is weak -- report it honestly, do not spin.

  python3 src/r2_leverage_from_P.py --out results/r2_leverage_from_P.csv
"""
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
AA = "ACDEFGHIKLMNPQRSTVWY"; IDX = {a: i for i, a in enumerate(AA)}; SEED = 20260803


def frame(pqf):
    d = pd.read_csv(pqf, low_memory=False)
    d["icode"] = d.icode.fillna("").astype(str)
    lP = d[[f"lP_{a}" for a in AA]].to_numpy()
    lQ = d[[f"lQ_{a}" for a in AA]].to_numpy()
    wi = d.aa.map(IDX).to_numpy().astype(float)
    ok = np.isfinite(wi) & np.isfinite(lP).all(1) & np.isfinite(lQ).all(1)
    d, lP, lQ, wi = d[ok].reset_index(drop=True), lP[ok], lQ[ok], wi[ok].astype(int)
    ar = np.arange(len(d))
    Lvec = (lP - lP[ar, wi][:, None]) - (lQ - lQ[ar, wi][:, None])     # L(a) = oc(a) - om(a)
    mask = np.ones_like(Lvec, bool); mask[ar, wi] = False
    Ln = np.where(mask, Lvec, np.nan)
    d["L_rms"] = np.sqrt(np.nanmean(Ln ** 2, axis=1))
    d["L_ala"] = np.where(wi == IDX["A"], np.nan, Lvec[:, IDX["A"]])   # leverage of ->Ala (undef where wt=A)
    return d, lP                                                       # features = the 20 raw log-probs (determine P)


def oos_r2(X, y, groups, model_fn, rng, nb=1000):
    m = np.isfinite(y)
    X, y, groups = X[m], y[m], groups[m]
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        pred[te] = model_fn().fit(X[tr], y[tr]).predict(X[te])
    ybar = y.mean()
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - ybar) ** 2)
    # complex-clustered bootstrap CI on R^2 (resample groups over the OOS predictions)
    gl = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    comps = list(gl); boot = []
    for _ in range(nb):
        idx = np.concatenate([gl[c] for c in rng.choice(comps, len(comps), replace=True)])
        yy, pp = y[idx], pred[idx]
        ss_tot = np.sum((yy - yy.mean()) ** 2)
        boot.append(1.0 - np.sum((yy - pp) ** 2) / ss_tot if ss_tot > 0 else np.nan)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(r2), float(lo), float(hi), int(len(y))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/r2_leverage_from_P.csv"); a = ap.parse_args()
    # Report the MAX OOS R^2 over a small learner family (a reviewer WILL run a RandomForest), so the
    # "irreducible-from-P" number is not an artifact of an undertuned single learner.
    gbm   = lambda: HistGradientBoostingRegressor(max_depth=4, max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=SEED)
    rf    = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=4, random_state=SEED)
    ridge = lambda: Ridge(alpha=1.0)
    flexible = [("gbm", gbm), ("rf", rf)]                                   # a reviewer's RF is the binding one
    rows = []
    for name, pqf in [("ProteinMPNN", "results/leverage_pq_skempi.csv"),
                      ("ESM-IF1", "results/leverage_pq_skempi_esmif.csv")]:
        d, Xp = frame(pqf)
        g = d.complex_id.to_numpy(); ncx = int(d.complex_id.nunique())
        wi = d.aa.map(IDX).to_numpy()                                       # wt one-hot: the phi(P,wt) fair class
        Wt = np.zeros((len(d), 20)); Wt[np.arange(len(d)), np.clip(wi, 0, 19)] = 1.0
        Xpwt = np.hstack([Xp, Wt])
        for tgt in ["L_rms", "L_ala"]:
            y = d[tgt].to_numpy()
            best = (-9.0, 0.0, 0.0, "")
            for lname, mfn in flexible:
                r2, lo, hi, n = oos_r2(Xp, y, g, mfn, np.random.default_rng(SEED))
                rows.append(dict(model=name, target=tgt, features="P(20)", learner=lname,
                                 r2=round(r2, 4), lo=round(lo, 4), hi=round(hi, 4), n=n, n_complexes=ncx))
                if r2 > best[0]:
                    best = (r2, lo, hi, lname)
            r2l, ll, hl, n = oos_r2(Xp, y, g, ridge, np.random.default_rng(SEED))
            r2w, low, hiw, _ = oos_r2(Xpwt, y, g, rf, np.random.default_rng(SEED))     # +wt identity (info beyond P)
            for feat, ln, (v, l, h) in [("P(20)", "ridge_linear", (r2l, ll, hl)),
                                        ("P+wt(40)", "rf", (r2w, low, hiw)),
                                        ("P(20)", "MAX_FLEXIBLE", best[:3])]:
                rows.append(dict(model=name, target=tgt, features=feat, learner=ln, r2=round(v, 4),
                                 lo=round(l, 4), hi=round(h, 4), n=n, n_complexes=ncx,
                                 irreducible_frac=round(1 - v, 4)))
            print(f"  {name:11s} {tgt:6s}: MAX-flexible R^2(L|P) = {best[0]:.3f} [{best[1]:.3f},{best[2]:.3f}] "
                  f"(best={best[3]}) -> irreducible {100*(1-best[0]):.0f}%   | +wt: {r2w:.3f}  linear: {r2l:.3f}")
    out = pd.DataFrame(rows); out["seed"] = SEED; out["command"] = "python3 src/r2_leverage_from_P.py"
    out.to_csv(a.out, index=False); print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

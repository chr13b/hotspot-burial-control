#!/usr/bin/env python3
"""Proposition 1(ii), measured against GROUND TRUTH instead of by proxy. The error-floor claim
    inf_phi  E[(ddG - phi(P))^2]  >=  E[Var(ddG | P)]  > 0
is currently supported only via Var(L|P) (R^2(L|P)=0.37) plus cycle (i) linking L to ddG. Here we measure
Var(ddG|P) DIRECTLY: regress EXPERIMENTAL ddG_bind on the bound-distribution 20-vector P with the same flexible
learner family (max over GBM + RandomForest, complex-held-out GroupKFold, complex-clustered bootstrap CI), and
report 1 - R^2 = the fraction of the real binding effect that is IRREDUCIBLE from the bound distribution.

Two readouts:
  (A) ->Ala substitutions, features = P(20): directly comparable to R^2(L_ala|P) (same positions, same
      features) on the alanine-scan backbone of SKEMPI.
  (B) all substitutions, features = P(20) + substitution one-hot(20): uses every measured mutation (the one-hot
      is needed because ddG is per-mutation while P is per-position). We also report R^2(ddG | substitution
      one-hot ALONE) so the P-specific contribution (B minus that baseline) is visible.

Expectation (stated before the run; no pre-registered falsifier is moved): experimental ddG carries large
irreducible variance (measurement noise ~1 kcal/mol + physics the sequence-free P does not encode), so R^2 << 1
and 1 - R^2 is large -> the Proposition's floor is real, not an artifact of the L proxy.

  python3 src/r2_ddg_from_P.py --out results/r2_ddg_from_P.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r2_leverage_from_P import oos_r2, AA, IDX, SEED


def load_P(pqf):
    d = pd.read_csv(pqf, low_memory=False)
    d["icode"] = d.icode.fillna("").astype(str)
    lP = d[[f"lP_{a}" for a in AA]].to_numpy()
    ok = np.isfinite(lP).all(1)
    P = d.loc[ok, ["complex_id", "chain", "resnum", "icode"]].reset_index(drop=True)
    for j, a in enumerate(AA):
        P[f"lP_{a}"] = lP[ok, j]
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/r2_ddg_from_P.csv"); a = ap.parse_args()
    gbm = lambda: HistGradientBoostingRegressor(max_depth=4, max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=SEED)
    rf  = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=4, random_state=SEED)
    flexible = [("gbm", gbm), ("rf", rf)]

    P = load_P("results/leverage_pq_skempi.csv")
    m = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    m["icode"] = m.icode.fillna("").astype(str)
    m = m[np.isfinite(m.ddG) & m.mut.isin(list(AA))].copy()
    d = m.merge(P, on=["complex_id", "chain", "resnum", "icode"], how="inner").reset_index(drop=True)
    Xp = d[[f"lP_{a}" for a in AA]].to_numpy()
    onehot = np.zeros((len(d), 20)); onehot[np.arange(len(d)), d.mut.map(IDX).to_numpy()] = 1.0
    y = d.ddG.to_numpy(float); g = d.complex_id.to_numpy()
    rows = []

    def report(tag, X, yy, gg, feat):
        best = (-9.0, 0.0, 0.0, "")
        for lname, mfn in flexible:
            r2, lo, hi, n = oos_r2(X, yy, gg, mfn, np.random.default_rng(SEED))
            rows.append(dict(readout=tag, features=feat, learner=lname, r2=round(r2, 4), lo=round(lo, 4),
                             hi=round(hi, 4), n=n))
            if r2 > best[0]:
                best = (r2, lo, hi, lname)
        ncx = int(pd.Series(gg[np.isfinite(yy)]).nunique())
        rows.append(dict(readout=tag, features=feat, learner="MAX_FLEXIBLE", r2=round(best[0], 4),
                         lo=round(best[1], 4), hi=round(best[2], 4), n=int(np.isfinite(yy).sum()),
                         n_complexes=ncx, irreducible_frac=round(1 - best[0], 4)))
        print(f"  {tag:28s}[{feat:12s}]: MAX R^2(ddG|.) = {best[0]:+.3f} [{best[1]:+.3f},{best[2]:+.3f}] "
              f"-> irreducible {100*(1-best[0]):.0f}%  (best={best[3]}, n={int(np.isfinite(yy).sum())})", flush=True)
        return best

    print(f"[r2-ddG] {len(d)} mutations with ddG, {d.complex_id.nunique()} complexes", flush=True)
    ala = (d.mut == "A").to_numpy()
    report("ddG ->Ala", Xp[ala], y[ala], g[ala], "P(20)")                          # (A) comparable to L_ala
    report("ddG all | subst-only", onehot, y, g, "onehot(20)")                     # (B) baseline
    report("ddG all | P+subst", np.hstack([Xp, onehot]), y, g, "P+onehot(40)")     # (B) full
    pd.DataFrame(rows).assign(seed=SEED, command="python3 src/r2_ddg_from_P.py").to_csv(a.out, index=False)
    print(f"[wrote] {a.out}", flush=True)


if __name__ == "__main__":
    main()

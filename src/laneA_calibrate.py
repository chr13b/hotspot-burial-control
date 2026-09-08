#!/usr/bin/env python3
"""Task 3 — calibrate the zero-shot leverage L to absolute kcal/mol against experimental ΔΔG, and put FoldX on the
same axis as the fitted reference. FoLdX-independent post-processing of committed Lane-A CSVs (no binary).

A SINGLE GLOBAL 2-parameter fit (slope+intercept), cross-validated complex-clustered (no complex in both train and
test), for each predictor:
  ΔΔG_exp ≈ a + b·(−L)      -> b in kcal/mol per unit leverage
  ΔΔG_exp ≈ a + b·(FoldX)   -> reference (the fitted physics tool on the same axis)
We report the global-fit slope b with a complex-clustered 95% CI, the out-of-fold RMSE, and out-of-fold
Pearson/Spearman of the calibrated prediction vs experiment. GUARD (pre-reg): global slope+intercept only — NO
per-position temperatures, NO per-complex params. This is a *reading* (where L lands in kcal/mol), not a retrained
model. The intercept-only (predict-train-mean) OOF RMSE is the floor.

  python3 src/laneA_calibrate.py  ->  results/laneA_calibration.csv
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

SEED, NBOOT, KFOLD = 20260803, 5000, 10


def merged():
    """Exact merge of foldx_laneA_deepen.py: FoldX ddg_bind + experimental ddG + L on complex|chain|resnum|icode|mut."""
    fx = pd.read_csv("results/foldx_ddg_laneA.csv").dropna(subset=["ddg_bind"]).copy()
    fx["icode"] = fx.get("icode", "").fillna("").astype(str)
    mut = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    mut["icode"] = mut.icode.fillna("").astype(str)
    for df in (fx, mut):
        df["key"] = (df.complex_id.astype(str) + "|" + df.chain.astype(str) + "|" +
                     df.resnum.astype(int).astype(str) + "|" + df.icode + "|" + df.mut.astype(str))
    m = fx.merge(mut[["key", "ddG", "L"]], on="key", how="inner").dropna(
        subset=["ddG", "L", "ddg_bind"]).reset_index(drop=True)
    return m


def ols(x, y):
    """Return (intercept a, slope b) for y ≈ a + b·x."""
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def oof_predictions(x, y, cx, ucx, rng):
    """Complex-clustered K-fold OOF: fit a+b·x on train complexes, predict test. Also OOF intercept-only floor."""
    order = rng.permutation(ucx)
    folds = np.array_split(order, KFOLD)
    pred = np.full(len(y), np.nan)
    floor = np.full(len(y), np.nan)
    for f in folds:
        te = np.isin(cx, f); tr = ~te
        a, b = ols(x[tr], y[tr])
        pred[te] = a + b * x[te]
        floor[te] = y[tr].mean()
    return pred, floor


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def calibrate(name, x, exp, cx, ucx, idxby, rng):
    a_glob, b_glob = ols(x, exp)                                  # global-fit slope reported
    pred, floor = oof_predictions(x, exp, cx, ucx, rng)           # OOF error/skill
    oof_rmse, floor_rmse = rmse(exp, pred), rmse(exp, floor)
    oof_pear = pearsonr(pred, exp)[0]
    oof_spear = spearmanr(pred, exp)[0]
    bslopes = []                                                  # complex-clustered bootstrap CI on the slope
    for _ in range(NBOOT):
        pick = rng.choice(ucx, len(ucx), replace=True)
        ii = np.concatenate([idxby[c] for c in pick])
        bslopes.append(ols(x[ii], exp[ii])[1])
    lo, hi = np.percentile(bslopes, [2.5, 97.5])
    p_gt0 = float((np.array(bslopes) > 0).mean())
    return dict(predictor=name, slope_b=round(b_glob, 4), b_lo=round(float(lo), 4), b_hi=round(float(hi), 4),
                p_slope_gt0=round(p_gt0, 4), intercept_a=round(a_glob, 4),
                oof_rmse=round(oof_rmse, 4), intercept_only_rmse=round(floor_rmse, 4),
                oof_pearson=round(float(oof_pear), 4), oof_spearman=round(float(oof_spear), 4))


def main():
    m = merged()
    exp, F, L, cx = (m.ddG.to_numpy(float), m.ddg_bind.to_numpy(float), m.L.to_numpy(float),
                     m.complex_id.to_numpy())
    ucx = np.unique(cx)
    idxby = {c: np.where(cx == c)[0] for c in ucx}
    rng = np.random.default_rng(SEED)

    rows = [
        calibrate("minus_L", -L, exp, cx, ucx, idxby, rng),      # ΔΔG_exp ≈ a + b·(−L)
        calibrate("FoldX_ddg_bind", F, exp, cx, ucx, idxby, rng),  # reference: fitted physics on same axis
    ]
    out = pd.DataFrame(rows).assign(n=len(m), n_complex=len(ucx), kfold=KFOLD, seed=SEED, nboot=NBOOT)
    out.to_csv("results/laneA_calibration.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("-> results/laneA_calibration.csv")


if __name__ == "__main__":
    main()

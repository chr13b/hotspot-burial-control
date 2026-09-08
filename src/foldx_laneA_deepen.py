#!/usr/bin/env python3
"""Lane-A deepening (FoldX-independent post-processing of the committed raw Lane-A CSV): make the
detection lane as rigorous as Lane B, and test the RECOVERED conditional result.

Three things, all complex-clustered (the unit of resampling is the complex, matching Lane B):
  (1) Spearman(FoldX, exp) and Spearman(L, exp) with complex-clustered 95% CIs (was point-only).
  (2) PARTIAL correlation r(exp, L | FoldX): does the zero-shot mixed derivative carry binding
      rank-signal BEYOND the fitted physics energy function? (The scalar-of-P confidence does not —
      it equals geometry, §8; this is the mixed derivative, the binding direction.) On-thesis: it
      extends the "L adds beyond geometry / conservation / one-pass log-odds" ladder to "beyond a
      fitted physics ΔΔG function."
  (3) A leakage-free LEARNED ensemble (10-fold complex-clustered out-of-fold linear stack of
      rank(FoldX) and rank(-L)) vs FoldX alone — the honest test that "L adds orthogonal signal",
      replacing the crude equal-weight rank-average (which was null only because it diluted FoldX).

  python3 src/foldx_laneA_deepen.py  ->  results/foldx_laneA_deepen.csv
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from numpy.linalg import lstsq

SEED, NBOOT, KFOLD = 20260803, 5000, 10


def scorr(a, b):
    return spearmanr(a, b)[0]


def partial(r_ab, r_ac, r_bc):
    return (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))


def main():
    fx = pd.read_csv("results/foldx_ddg_laneA.csv").dropna(subset=["ddg_bind"]).copy()
    fx["icode"] = fx.get("icode", "").fillna("").astype(str)
    mut = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    mut["icode"] = mut.icode.fillna("").astype(str)
    for df in (fx, mut):
        df["key"] = (df.complex_id.astype(str) + "|" + df.chain.astype(str) + "|" +
                     df.resnum.astype(int).astype(str) + "|" + df.icode + "|" + df.mut.astype(str))
    m = fx.merge(mut[["key", "ddG", "L"]], on="key", how="inner").dropna(
        subset=["ddG", "L", "ddg_bind"]).reset_index(drop=True)
    exp, F, L, cx = (m.ddG.to_numpy(), m.ddg_bind.to_numpy(), m.L.to_numpy(),
                     m.complex_id.to_numpy())
    ucx = np.unique(cx)
    idxby = {c: np.where(cx == c)[0] for c in ucx}
    rng = np.random.default_rng(SEED)

    r_eF, r_eL, r_LF = scorr(exp, F), scorr(exp, L), scorr(L, F)
    pr = partial(r_eL, r_eF, r_LF)

    # leakage-free learned OOF ensemble
    order = rng.permutation(ucx)
    folds = np.array_split(order, KFOLD)
    X = np.column_stack([rankdata(F), rankdata(-L)])
    Xs = (X - X.mean(0)) / X.std(0)
    y = rankdata(exp).astype(float)
    oof = np.full(len(m), np.nan)
    for f in folds:
        te = np.isin(cx, f); tr = ~te
        A = np.column_stack([np.ones(tr.sum()), Xs[tr]])
        coef, *_ = lstsq(A, y[tr], rcond=None)
        oof[te] = coef[0] + Xs[te] @ coef[1:]
    rho_ens = scorr(oof, exp)

    # one complex-clustered bootstrap loop, all statistics resampled together
    bF, bL, bP, bU = [], [], [], []
    for _ in range(NBOOT):
        pick = rng.choice(ucx, len(ucx), replace=True)
        ii = np.concatenate([idxby[c] for c in pick])
        e, f_, l_, o = exp[ii], F[ii], L[ii], oof[ii]
        bF.append(scorr(f_, e)); bL.append(scorr(l_, e))
        bP.append(partial(scorr(e, l_), scorr(e, f_), scorr(l_, f_)))
        bU.append(abs(scorr(o, e)) - abs(scorr(f_, e)))
    def ci(v, arr, tail_lt=False):
        a = np.array(arr)
        p = (a < 0).mean() if tail_lt else (a > 0).mean()
        return round(float(v), 4), round(float(np.percentile(a, 2.5)), 4), \
            round(float(np.percentile(a, 97.5)), 4), round(float(p), 4)

    rows = []
    v, lo, hi, p = ci(r_eF, bF); rows.append(dict(
        metric="spearman_FoldX_vs_exp", value=v, lo=lo, hi=hi, p_gt0=p))
    v, lo, hi, p = ci(r_eL, bL, tail_lt=True); rows.append(dict(
        metric="spearman_L_vs_exp", value=v, lo=lo, hi=hi, p_lt0=p))
    v, lo, hi, p = ci(pr, bP, tail_lt=True); rows.append(dict(
        metric="partial_L_given_FoldX", value=v, lo=lo, hi=hi, p_lt0=p))
    v, lo, hi, p = ci(abs(rho_ens) - abs(r_eF), bU); rows.append(dict(
        metric="ensemble_uplift_over_FoldX", value=v, lo=lo, hi=hi, p_gt0=p,
        note=f"|ens|={abs(rho_ens):.4f} vs |FoldX|={abs(r_eF):.4f}; {KFOLD}-fold complex-clustered OOF"))
    out = pd.DataFrame(rows).assign(n=len(m), n_complex=len(ucx), redundancy_Spearman_L_FoldX=round(r_LF, 4),
                                    seed=SEED, nboot=NBOOT)
    out.to_csv("results/foldx_laneA_deepen.csv", index=False)
    print(out.to_string(index=False))
    print("-> results/foldx_laneA_deepen.csv")


if __name__ == "__main__":
    main()

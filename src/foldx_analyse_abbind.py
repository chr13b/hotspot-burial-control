#!/usr/bin/env python3
"""Task 2 analysis — AB-Bind physics detection: Spearman(FoldX,exp), Spearman(L,exp), partial(L,exp|FoldX), all
complex-clustered (unit = pdb). Pre-reg results/PREREG_laneA_deepen.md. Underpowered (~23 complexes) — the win is a
replicated DIRECTION; magnitude secondary. Reports the WT-identity gate (CLAUDE.md rule 6) before any null is trusted.

  python3 src/foldx_analyse_abbind.py  ->  results/foldx_detection_abbind.csv
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED, NBOOT = 20260803, 5000


def scorr(a, b):
    return spearmanr(a, b)[0]


def partial(r_ab, r_ac, r_bc):
    return (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))


def main():
    shards = sorted(glob.glob("results/_foldx_abbind_A_s*.csv"))
    fx = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    fx = fx.drop_duplicates("setid").copy()
    fx.to_csv("results/foldx_ddg_abbind.csv", index=False)          # consolidated raw per-set (committed)
    n_sets = len(fx)
    n_finite = int(fx.ddg_bind.notna().sum())
    tot_used, tot_drop = int(fx.n_used.sum()), int(fx.n_dropped.sum())     # WT-identity gate (per single mutant)
    fx["pdb"] = fx.complex_id.str.split("_").str[0]
    fx["icode"] = fx.get("icode", "").fillna("").astype(str)
    fxf = fx.dropna(subset=["ddg_bind"]).copy()
    fxf["key"] = (fxf.pdb + "|" + fxf.chain.astype(str) + "|" + fxf.resnum.astype(int).astype(str) + "|" +
                  fxf.icode + "|" + fxf.mut.astype(str))

    com = pd.read_csv("results/leverage_abbind_mutations.csv")
    com["icode"] = com.icode.fillna("").astype(str)
    com = com[com.is_interface == 1].copy()
    # the committed fixture carries exact-duplicate rows for a few mutations (e.g. 3BN9 H100H appears 7x, same L/ddG);
    # dedup on the mutation identity so the correlation weights each mutation once (else a big-ddG outlier is 7x-weighted).
    com = com.drop_duplicates(subset=["complex_id", "chain", "resnum", "icode", "wt", "mut"]).copy()
    com["key"] = (com.complex_id.astype(str) + "|" + com.chain.astype(str) + "|" +
                  com.resnum.astype(int).astype(str) + "|" + com.icode + "|" + com.mut.astype(str))

    m = fxf.merge(com[["key", "ddG", "L"]], on="key", how="inner").dropna(
        subset=["ddG", "L", "ddg_bind"]).reset_index(drop=True)
    exp, F, L, cx = (m.ddG.to_numpy(float), m.ddg_bind.to_numpy(float), m.L.to_numpy(float), m.pdb.to_numpy())
    ucx = np.unique(cx)
    idxby = {c: np.where(cx == c)[0] for c in ucx}
    rng = np.random.default_rng(SEED)

    r_eF, r_eL, r_LF = scorr(exp, F), scorr(exp, L), scorr(L, F)
    pr = partial(r_eL, r_eF, r_LF)
    bF, bL, bP = [], [], []
    for _ in range(NBOOT):
        pick = rng.choice(ucx, len(ucx), replace=True)
        ii = np.concatenate([idxby[c] for c in pick])
        e, f_, l_ = exp[ii], F[ii], L[ii]
        bF.append(scorr(f_, e)); bL.append(scorr(l_, e))
        bP.append(partial(scorr(e, l_), scorr(e, f_), scorr(l_, f_)))

    def ci(v, arr, tail_lt=False):
        a = np.array(arr)
        p = (a < 0).mean() if tail_lt else (a > 0).mean()
        return round(float(v), 4), round(float(np.nanpercentile(a, 2.5)), 4), \
            round(float(np.nanpercentile(a, 97.5)), 4), round(float(p), 4)

    rows = []
    v, lo, hi, p = ci(r_eF, bF); rows.append(dict(metric="spearman_FoldX_vs_exp", value=v, lo=lo, hi=hi, p_gt0=p))
    v, lo, hi, p = ci(r_eL, bL, tail_lt=True); rows.append(dict(metric="spearman_L_vs_exp", value=v, lo=lo, hi=hi, p_lt0=p))
    v, lo, hi, p = ci(pr, bP, tail_lt=True); rows.append(dict(metric="partial_L_given_FoldX", value=v, lo=lo, hi=hi, p_lt0=p))
    out = pd.DataFrame(rows).assign(n=len(m), n_complex=len(ucx), redundancy_Spearman_L_FoldX=round(r_LF, 4),
                                    wt_gate_used=tot_used, wt_gate_dropped=tot_drop, n_sets=n_sets,
                                    n_foldx_finite=n_finite, seed=SEED, nboot=NBOOT)
    out.to_csv("results/foldx_detection_abbind.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print(f"[gate] WT-identity: {tot_used} kept / {tot_used + tot_drop} tried "
          f"({tot_used / max(tot_used + tot_drop, 1):.1%}); FoldX finite sets {n_finite}/{n_sets}; "
          f"merged n={len(m)} over {len(ucx)} complexes")
    print("-> results/foldx_detection_abbind.csv")


if __name__ == "__main__":
    main()

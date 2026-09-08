#!/usr/bin/env python3
"""Analyse FoldX ΔΔG_bind for both lanes (pre-reg: results/PREREG_foldx.md). FoldX-independent post-processing:
concatenates the per-shard driver CSVs and computes

  Lane A (detection): Spearman(FoldX ΔΔG_bind, experimental ΔΔG) vs Spearman(L, experimental ΔΔG) on the SAME
    single mutations (joined to results/leverage_skempi_mutations.csv). -> results/foldx_detection.csv
  Lane B (steering): favorability = -ΔΔG_bind (>0 = better binding), aggregate k per (complex, arm) [mean],
    paired complex-clustered bootstrap L-naive / L-random / naive-random. -> results/foldx_steer.csv

  python3 src/foldx_analyse.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20260803
NBOOT = 5000


def concat_shards(pattern):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(pattern)) if os.path.getsize(f) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_boot(piv, a, b, rng, nboot=NBOOT):
    if a not in piv.columns or b not in piv.columns:
        return None
    s = piv[[a, b]].dropna()
    if len(s) < 2:
        return None
    d = (s[a] - s[b]).to_numpy()
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return dict(n=len(s), mean_a=round(float(s[a].mean()), 4), mean_b=round(float(s[b].mean()), 4),
                delta=round(float(d.mean()), 4), lo=round(float(np.percentile(bt, 2.5)), 4),
                hi=round(float(np.percentile(bt, 97.5)), 4), p_gt0=round(float((bt > 0).mean()), 4))


def lane_a(a_glob, a_out):
    fx = concat_shards(a_glob)
    if not len(fx):
        print("[laneA] no FoldX rows yet"); return
    fx = fx.dropna(subset=["ddg_bind"]).copy()
    fx["icode"] = fx.get("icode", "").fillna("").astype(str)
    mut = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    mut["icode"] = mut.icode.fillna("").astype(str)
    for df in (fx, mut):
        df["key"] = (df.complex_id.astype(str) + "|" + df.chain.astype(str) + "|" +
                     df.resnum.astype(int).astype(str) + "|" + df.icode + "|" + df.mut.astype(str))
    m = fx.merge(mut[["key", "ddG", "L"]], on="key", how="inner").dropna(subset=["ddG", "L", "ddg_bind"])
    rho_fx, p_fx = spearmanr(m.ddg_bind, m.ddG)
    rho_l, p_l = spearmanr(m.L, m.ddG)
    print(f"[laneA] n={len(m)} single mutants ({m.complex_id.nunique()} complexes)")
    print(f"[laneA]   Spearman(FoldX ΔΔG_bind, exp ΔΔG) = {rho_fx:+.4f} (p={p_fx:.2e})  [physics SOTA]")
    print(f"[laneA]   Spearman(L, exp ΔΔG)              = {rho_l:+.4f} (p={p_l:.2e})  [zero-shot readout]")
    pd.DataFrame([
        dict(metric="spearman_FoldX_vs_expddG", rho=round(float(rho_fx), 4), p=float(p_fx),
             n=int(len(m)), n_complex=int(m.complex_id.nunique()), seed=SEED),
        dict(metric="spearman_L_vs_expddG", rho=round(float(rho_l), 4), p=float(p_l),
             n=int(len(m)), n_complex=int(m.complex_id.nunique()), seed=SEED),
    ]).to_csv(a_out, index=False)
    print(f"[laneA] -> {a_out}")


def lane_b(b_glob, b_out):
    fx = concat_shards(b_glob)
    if not len(fx):
        print("[laneB] no FoldX rows yet"); return
    fx = fx.dropna(subset=["ddg_bind"]).copy()
    fx["fav"] = -fx.ddg_bind                      # favorability: >0 = more-favorable binding
    piv = fx.groupby(["complex_id", "arm"]).fav.mean().unstack("arm")     # aggregate k (mean)
    rng = np.random.default_rng(SEED)
    print(f"=== Lane B: paired favorability (= -ΔΔG_bind), complex-clustered 95% CI "
          f"({fx.complex_id.nunique()} complexes) ===")
    rows = []
    for a, b in (("L", "naive"), ("L", "random"), ("naive", "random")):
        res = paired_boot(piv, a, b, rng)
        if res:
            rows.append(dict(contrast=f"{a}-{b}", **res))
            print(f"  {a:6s}-{b:6s} Δfav={res['delta']:+.4f} [{res['lo']:+.4f},{res['hi']:+.4f}] "
                  f"P(>0)={res['p_gt0']:.3f} n={res['n']}")
    pd.DataFrame(rows).assign(seed=SEED).to_csv(b_out, index=False)
    ln = [r for r in rows if r["contrast"] == "L-naive"]
    if ln:
        r = ln[0]
        verdict = ("BOUNDED — falsifier FIRES (physics does not show L>naive)"
                   if (r["delta"] <= 0 or r["lo"] <= 0) else
                   "L > naive on a physics, non-IF readout — specificity holds (strongest form)")
        print(f"[laneB] DECISIVE L-naive favorability = {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] -> {verdict}")
    print(f"[laneB] -> {b_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-glob", default="results/_foldx_A_s*.csv")
    ap.add_argument("--b-glob", default="results/_foldx_B_s*.csv")
    ap.add_argument("--a-out", default="results/foldx_detection.csv")
    ap.add_argument("--b-out", default="results/foldx_steer.csv")
    a = ap.parse_args()
    lane_a(a.a_glob, a.a_out)
    print()
    lane_b(a.b_glob, a.b_out)


if __name__ == "__main__":
    main()

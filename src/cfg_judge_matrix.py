#!/usr/bin/env python3
"""Phase 1 — the JUDGE MATRIX: paired L vs random judge-leverage across (steered model x judge), anti-circular.

Reads cfg_steer.csv (ProteinMPNN-steered SET-A) and cfg_steer_esmif.csv (ESM-IF1-steered SET-B). Each has
meanL_<judge> columns per (complex_id, direction, alpha) — the mean binding-leverage of the sampled interface
residues under that judge (judge in {mpnn, esmif, pifold, mif} wherever a leverage cache exists). For each
(steered_model, judge, alpha) it computes the PAIRED L-arm − random-arm per-complex judge-leverage with a
complex-clustered bootstrap 95% CI and P(>0). A judge equal to the steered model is CIRCULAR (is_self=1) and
reported for reference only, NOT as corroboration.

Pre-reg: bundle Phase 1. Falsifier: an anti-circular judge with paired L−random ≤ 0 does not corroborate.
  python3 src/cfg_judge_matrix.py --out results/cfg_judge_matrix.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 5000
JUDGE_NAME = {"mpnn": "ProteinMPNN", "esmif": "ESM-IF1", "pifold": "PiFold", "mif": "MIF"}
# (csv, steered_model_label, self_judge_key)
SOURCES = [("results/cfg_steer.csv", "ProteinMPNN", "mpnn"),
           ("results/cfg_steer_esmif.csv", "ESM-IF1", "esmif")]


def paired(df_a, col, rng, nboot=NBOOT):
    piv = (df_a.pivot_table(index="complex_id", columns="direction", values=col)
                .dropna(subset=["L", "random"]))
    if len(piv) < 2:
        return None
    d = (piv["L"] - piv["random"]).to_numpy()
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return dict(n_cx=len(piv), arm_L=round(float(piv["L"].mean()), 4),
                arm_random=round(float(piv["random"].mean()), 4), delta=round(float(d.mean()), 4),
                lo=round(float(np.percentile(b, 2.5)), 4), hi=round(float(np.percentile(b, 97.5)), 4),
                p_gt0=round(float((b > 0).mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cfg_judge_matrix.csv")
    ap.add_argument("--alphas", default="0,2")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    rng = np.random.default_rng(SEED)
    rows = []
    print("=== JUDGE MATRIX (paired L − random judge-leverage, complex-clustered 95% CI) ===")
    for csv, steered, selfkey in SOURCES:
        if not os.path.exists(csv):
            print(f"[skip] {csv} missing")
            continue
        df = pd.read_csv(csv)
        judge_cols = [c for c in df.columns if c.startswith("meanL_")]
        for al in alphas:
            da = df[df.alpha == al]
            for jc in judge_cols:
                jkey = jc.replace("meanL_", "")
                judge = JUDGE_NAME.get(jkey, jkey)
                res = paired(da, jc, rng)
                if res is None:
                    continue
                is_self = int(jkey == selfkey)
                rows.append(dict(steered_model=steered, judge=judge, is_self=is_self, alpha=al, **res))
                if al == max(alphas):
                    tag = "SELF/circular" if is_self else "anti-circular"
                    print(f"  steer={steered:11s} judge={judge:11s} a={al:g} [{tag:13s}] "
                          f"L−random={res['delta']:+.4f} [{res['lo']:+.4f},{res['hi']:+.4f}] "
                          f"P(>0)={res['p_gt0']:.3f}  (L={res['arm_L']:+.3f} rnd={res['arm_random']:+.3f} n={res['n_cx']})")
    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out.to_csv(a.out, index=False)
    nonself = out[(out.is_self == 0) & (out.alpha == max(alphas))]
    print(f"\n[judge-matrix] {len(out)} rows -> {a.out}")
    fired = nonself[nonself.lo <= 0]
    if len(fired):
        print("FALSIFIER fires (anti-circular, CI includes/≤0):",
              list(zip(fired.steered_model, fired.judge)))
    else:
        print(f"All {len(nonself)} anti-circular judge(s) at α={max(alphas):g}: paired L−random CI>0 "
              f"→ falsifier does NOT fire.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyse predicted-backbone steering (pre-reg: results/PREREG_predicted_steer.md). Paired complex-clustered
bootstrap judge contrasts, the SAME machinery as the crystal src/cfg_judge_matrix.py. Reads the per-source steer
CSVs from cfg_steer_predicted.py and reports, per (source, judge): L-random (primary), L-naive (secondary),
naive-random. The `crystal` source is the positive control (must reproduce a strongly-positive L-random on the
matched subset before the of3/af2 contrasts are trusted).

  python3 src/analyse_predicted_steer.py --out results/cfg_steer_predicted.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED, NBOOT, ALPHA = 20260803, 5000, 2.0
JUDGES = ["esmif", "mif"]                       # anti-circular (non-self); mpnn is self/circular (reported apart)
CONTRASTS = [("L", "random"), ("L", "naive"), ("naive", "random")]


def paired(df, judge, a, b, rng):
    col = f"meanL_{judge}"
    d = df[(df.alpha == ALPHA) & (df[col].notna())]
    piv = d.pivot_table(index="complex_id", columns="direction", values=col, aggfunc="mean")
    if a not in piv or b not in piv:
        return None
    piv = piv[[a, b]].dropna()
    if len(piv) < 3:
        return None
    diff = (piv[a] - piv[b]).to_numpy()
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(NBOOT)])
    return dict(judge=judge, contrast=f"{a}-{b}", n_cx=len(diff),
                mean_a=round(float(piv[a].mean()), 4), mean_b=round(float(piv[b].mean()), 4),
                delta=round(float(diff.mean()), 4), lo=round(float(np.percentile(boots, 2.5)), 4),
                hi=round(float(np.percentile(boots, 97.5)), 4), p_gt0=round(float((boots > 0).mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cfg_steer_predicted.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows, recov = [], []
    for source in ["of3", "af2", "crystal"]:
        f = f"results/_cfg_pred_{source}.csv"
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        for judge in JUDGES:
            if f"meanL_{judge}" not in df.columns:
                continue
            for aa, bb in CONTRASTS:
                r = paired(df, judge, aa, bb, rng)
                if r:
                    rows.append(dict(source=source, **r, seed=SEED, nboot=NBOOT))
        rr = df[df.alpha == ALPHA].groupby("direction").int_recovery.mean()   # native-recovery-preserved control
        for direction, val in rr.items():
            recov.append(dict(source=source, direction=direction, int_recovery=round(float(val), 4)))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.DataFrame(recov).to_csv(a.out.replace(".csv", "_recovery.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\n[recovery]"); print(pd.DataFrame(recov).to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

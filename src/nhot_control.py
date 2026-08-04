"""Is the constellation cost N_hot a HOTSPOT phenomenon, or generic to T = 0.1?

BRIEF.md computes N_hot only at hotspots. But Phase 0 found no burial-matched hotspot
penalty, which makes the obvious control mandatory: recompute the same constellation cost
over the burial-MATCHED CONTROL positions of the same complexes. If the two are equal,
N_hot measures low-temperature sampling in general, not a tax at the positions that make a
binder a binder.

This is an interpretive control, not a pre-registered falsifier. It needs no model calls -
it reuses the per-position distributions Phase 0 already wrote.

Usage:
  python3 src/nhot_control.py --out results/nhot_control
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T = 0.1
N_BOOT = 10000
SEED = 20260803


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nhot_control")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--pairs", default="results/p0_pairs_SECONDARY_B_any_interface.csv")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    pos = pd.read_csv(a.positions,
                      usecols=["complex_id", "chain", "resnum", "aa", "label",
                               "logp_mode_mix", "logp_native_mix", "rsasa_complex"])
    pos["label"] = pos["label"].fillna("null")   # pandas reads the string "null" as NaN
    pos["delta"] = pos["logp_mode_mix"] - pos["logp_native_mix"]

    pr = pd.read_csv(a.pairs)
    key = pos.set_index(["complex_id", "chain", "resnum"])["delta"]

    rows = []
    for cid, sub in pr.groupby("complex_id"):
        dh, dc = [], []
        for _, r in sub.iterrows():
            try:
                dh.append(float(key.loc[(cid, r["hot_chain"], r["hot_resnum"])]))
                dc.append(float(key.loc[(cid, r["ctl_chain"], r["ctl_resnum"])]))
            except (KeyError, TypeError):
                continue
        if not dh:
            continue
        f = 1.0 / (T * np.log(10))
        rows.append(dict(complex_id=cid, k=len(dh),
                         log10_N_hotspot=float(np.sum(dh) * f),
                         log10_N_control=float(np.sum(dc) * f),
                         sum_delta_hot=float(np.sum(dh)),
                         sum_delta_ctl=float(np.sum(dc))))
    df = pd.DataFrame(rows)
    df["diff_log10"] = df["log10_N_hotspot"] - df["log10_N_control"]
    df["command"] = cmd
    df.to_csv(f"{a.out}.csv", index=False)

    rng = np.random.default_rng(SEED)
    v = df["diff_log10"].values
    boot = np.array([np.mean(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"complexes with matched constellations : {len(df)}")
    print(f"median constellation size k            : {df['k'].median():.1f}")
    print(f"median log10 N at HOTSPOT positions    : {df['log10_N_hotspot'].median():.2f}")
    print(f"median log10 N at MATCHED CONTROLS     : {df['log10_N_control'].median():.2f}")
    print(f"mean paired difference (hot - control) : {v.mean():+.2f} log10 "
          f"[95% CI {lo:+.2f}, {hi:+.2f}]  ({N_BOOT} reps, seed {SEED})")
    print(f"fraction of complexes where hotspot constellation costs MORE: "
          f"{(v > 0).mean():.3f}")

    pd.DataFrame([dict(metric="n_complexes", value=len(df)),
                  dict(metric="median_k", value=float(df["k"].median())),
                  dict(metric="median_log10_N_hotspot",
                       value=float(df["log10_N_hotspot"].median())),
                  dict(metric="median_log10_N_control",
                       value=float(df["log10_N_control"].median())),
                  dict(metric="mean_paired_diff_log10", value=float(v.mean())),
                  dict(metric="ci_lo", value=float(lo)),
                  dict(metric="ci_hi", value=float(hi)),
                  dict(metric="frac_hotspot_costs_more", value=float((v > 0).mean())),
                  ]).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"[done] wrote {a.out}.csv and {a.out}_summary.csv")


if __name__ == "__main__":
    main()

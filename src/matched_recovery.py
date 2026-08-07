"""Burial-matched SEQUENCE RECOVERY - ProBID-Net's own metric, under the matched design.

ProBID-Net reported recovery (0.334 hotspot vs 0.472 non-hotspot), not log-probability.
Phase 0's matched-pair test is run on log p(native), which is the more sensitive statistic
but is not what they published. This recomputes the identical matched design on the
recovery metric so the comparison to their number is like-for-like.

Paired, within complex, on the same matched pairs Phase 0 produced; complex-level bootstrap.

Usage:
  python3 src/matched_recovery.py --out results/matched_recovery
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

N_BOOT = 10000
SEED = 20260803


def boot_complex(df, col, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df.loc[df["complex_id"] == c, col].values for c in cids}
    m = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        m[b] = np.nanmean(np.concatenate([by[cids[i]] for i in pick]))
    lo, hi = np.nanpercentile(m, [2.5, 97.5])
    return float(np.nanmean(df[col])), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/matched_recovery")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--pairs-prefix", default="results/p0_dssp")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    pos = pd.read_csv(a.positions,
                      usecols=["complex_id", "chain", "resnum", "aa", "mode_aa"])
    pos["hit"] = (pos["aa"] == pos["mode_aa"]).astype(float)
    key = pos.set_index(["complex_id", "chain", "resnum"])["hit"]

    rows = []
    for f in sorted(glob.glob(f"{a.pairs_prefix}_pairs_*.csv")):
        tag = os.path.basename(f).split("_pairs_")[1][:-4]
        pr = pd.read_csv(f)
        recs = []
        for _, r in pr.iterrows():
            try:
                h = float(key.loc[(r["complex_id"], r["hot_chain"], r["hot_resnum"])])
                c = float(key.loc[(r["complex_id"], r["ctl_chain"], r["ctl_resnum"])])
            except (KeyError, TypeError):
                continue
            recs.append(dict(complex_id=r["complex_id"], hot_hit=h, ctl_hit=c, d=h - c))
        d = pd.DataFrame(recs)
        if len(d) < 5:
            continue
        mean, lo, hi = boot_complex(d, "d")
        rows.append(dict(analysis=tag, n_pairs=len(d),
                         n_complexes=d["complex_id"].nunique(),
                         recovery_hotspot=float(d["hot_hit"].mean()),
                         recovery_control=float(d["ctl_hit"].mean()),
                         paired_diff=mean, lo=lo, hi=hi))
        print(f"{tag:34s} n={len(d):4d} cx={d['complex_id'].nunique():3d}  "
              f"recovery hot={d['hot_hit'].mean():.3f} ctl={d['ctl_hit'].mean():.3f}  "
              f"paired diff={mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    out = pd.DataFrame(rows).assign(command=cmd)
    out.to_csv(f"{a.out}.csv", index=False)
    print(f"\n[done] wrote {a.out}.csv")
    print("ProBID-Net published, UNCONTROLLED: 0.334 hotspot vs 0.472 non-hotspot "
          "(paired diff -0.138)")


if __name__ == "__main__":
    main()

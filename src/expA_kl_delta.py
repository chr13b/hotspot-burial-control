"""Experiment A, Analysis 2: KL detector AUROC on predicted vs crystal backbones.

The headline is `ΔAUROC-over-burial` = AUROC(burial+KL) − AUROC(burial), with a PAIRED complex-level
bootstrap (same resamples for both scores). Reports it on the CRYSTAL joined table (baseline, must
reproduce the committed +0.048) and on the PREDICTED joined table, so the drop is apples-to-apples.
If a confidence CSV is given, the predicted arm is also split at the median of each metric.

Reuses `kl_analysis.paired_auc` / `auc` / `rankavg` so the estimator is identical to finding C5.

Usage:
  python3 src/expA_kl_delta.py --pred-joined $SCRATCH/ftax/predicted/expA_kl_joined.csv \
      --crystal-joined results/kl_detector_joined.csv --confidence results/expA_confidence.csv \
      --out results/expA_kl_delta
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kl_analysis import auc, paired_auc, rankavg


def analyse(joined_csv, tag, conf=None, strat_metrics=()):
    m = pd.read_csv(joined_csv)
    m["icode"] = m["icode"].fillna("").astype(str)
    if "burial" not in m.columns:
        m["burial"] = -m["rsasa_complex"]
    m = m[np.isfinite(m["kl"]) & np.isfinite(m["burial"])].copy()
    m["burial_KL"] = rankavg(m, ["burial", "kl"])
    rows = []

    def row(sub, arm):
        if sub["is_hot"].sum() < 5 or sub["is_hot"].sum() == len(sub):
            return
        r = paired_auc(sub, "burial_KL", "burial")
        rows.append(dict(source=tag, arm=arm, n=len(sub), n_cx=sub["complex_id"].nunique(),
                         n_hot=int(sub["is_hot"].sum()),
                         auc_burial=r["b"], auc_kl=auc(sub["kl"].values, sub["is_hot"].values),
                         auc_burial_KL=r["a"], dAUROC=r["delta"],
                         lo=r["delta_ci"][0], hi=r["delta_ci"][1], p_gt0=r["p_gt0"]))

    row(m, "all")
    if conf is not None:
        mm = m.merge(conf, on="complex_id", how="left")
        for met in [x for x in strat_metrics if x in mm.columns]:
            med = mm[met].median()
            row(mm[mm[met] >= med], f"{met}>=%.3g" % med)
            row(mm[mm[met] < med], f"{met}<%.3g" % med)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-joined", required=True)
    ap.add_argument("--crystal-joined", default="results/kl_detector_joined.csv")
    ap.add_argument("--confidence", default=None)
    ap.add_argument("--strat-metrics", default="ptm,interface_plddt,rmsd_ca_interface")
    ap.add_argument("--out", default="results/expA_kl_delta")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    conf = pd.read_csv(a.confidence) if a.confidence and os.path.exists(a.confidence) else None
    metrics = a.strat_metrics.split(",")
    rows = analyse(a.crystal_joined, "crystal")
    rows += analyse(a.pred_joined, "predicted", conf, metrics)
    df = pd.DataFrame(rows)
    for r in rows:
        print(f"  [{r['source']:9s} {r['arm']:22s}] burial {r['auc_burial']:.3f}  KL {r['auc_kl']:.3f}  "
              f"burial+KL {r['auc_burial_KL']:.3f}  ΔAUROC {r['dAUROC']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] P(>0)={r['p_gt0']:.3f}  (n_cx={r['n_cx']})")
    df.assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"[done] wrote {a.out}_summary.csv")


if __name__ == "__main__":
    main()

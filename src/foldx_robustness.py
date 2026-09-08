#!/usr/bin/env python3
"""Lane-B robustness recomputation from the committed raw per-set CSV (FoldX-independent post-processing).
Two pre-registered / caveat-answering checks, so the §4 robustness numbers trace to a committed CSV:

  (1) run-attrition sweep — FoldX's stochastic step left <5 usable runs on hard multi-mutation sets;
      restrict to sets with >= T usable runs (mut AND wt) and recompute the DECISIVE L-naive contrast.
      If L-naive is stable across T, the attrition is bounded noise, not a confound.
  (2) best-of-k aggregation (pre-registered secondary) — max favorability over k per (complex, arm),
      i.e. what a designer sampling a few sequences and keeping the best actually gets.

  python3 src/foldx_robustness.py  ->  results/foldx_steer_robustness.csv
"""
import numpy as np, pandas as pd

SEED, NBOOT = 20260803, 5000


def paired_boot(piv, a, b, seed=SEED):
    if a not in piv.columns or b not in piv.columns:
        return None
    s = piv[[a, b]].dropna()
    if len(s) < 2:
        return None
    d = (s[a] - s[b]).to_numpy()
    rng = np.random.default_rng(seed)
    bt = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NBOOT)])
    return dict(n=len(s), delta=round(float(d.mean()), 4), lo=round(float(np.percentile(bt, 2.5)), 4),
                hi=round(float(np.percentile(bt, 97.5)), 4), p_gt0=round(float((bt > 0).mean()), 4))


def main():
    b = pd.read_csv("results/foldx_ddg_laneB.csv").dropna(subset=["ddg_bind"]).copy()
    b["fav"] = -b.ddg_bind
    rows = []
    # (1) run-attrition sweep on the decisive L-naive contrast
    for thr in (1, 3, 5):
        sub = b[(b.n_ie_mut >= thr) & (b.n_ie_wt >= thr)]
        piv = sub.groupby(["complex_id", "arm"]).fav.mean().unstack("arm")
        r = paired_boot(piv, "L", "naive")
        if r:
            rows.append(dict(agg="mean_over_k", filter=f"runs>={thr}", contrast="L-naive",
                             sets_kept=int(len(sub)), **r))
    # (2) best-of-k aggregation (all three contrasts)
    piv = b.groupby(["complex_id", "arm"]).fav.max().unstack("arm")
    for a, bn in (("L", "naive"), ("L", "random"), ("naive", "random")):
        r = paired_boot(piv, a, bn)
        if r:
            rows.append(dict(agg="best_of_k", filter="runs>=1", contrast=f"{a}-{bn}",
                             sets_kept=int(len(b)), **r))
    out = pd.DataFrame(rows).assign(seed=SEED, nboot=NBOOT)
    out.to_csv("results/foldx_steer_robustness.csv", index=False)
    print(out.to_string(index=False))
    print("-> results/foldx_steer_robustness.csv")


if __name__ == "__main__":
    main()

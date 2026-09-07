#!/usr/bin/env python3
"""Paired ipTM specificity contrasts for the naive arm (Phase 4 of PREREG_naive.md). Combine the committed
wt/L/random folds (iptm_steer.csv) with the naive folds (iptm_steer_naive.csv), restrict to the naive complex
set, and report naive-random, L-naive, L-random on ipTM / interface pAE / interface pLDDT / composite / global
pTM (complex-clustered bootstrap, same machinery as analyse_iptm.py). Expected: naive ~ random on ipTM/composite.

  python3 src/analyse_iptm_naive.py --out results/iptm_summary_naive.csv
"""
import argparse

import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 5000
COMPOSITE_COLS = [("iptm", True), ("interface_pae", False), ("interface_plddt", True)]
CONTRASTS = [("naive", "random"), ("L", "naive"), ("L", "random")]


def zorient(df, col, hb):
    v = df[col].to_numpy(float)
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    return z if hb else -z


def paired_boot(piv, ca, cb, rng, nboot=NBOOT):
    if ca not in piv.columns or cb not in piv.columns:
        return None
    s = piv[[ca, cb]].dropna()
    if len(s) < 2:
        return None
    d = (s[ca] - s[cb]).to_numpy()
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return dict(n=len(d), mean_a=round(float(s[ca].mean()), 4), mean_b=round(float(s[cb].mean()), 4),
                delta=round(float(d.mean()), 4), lo=round(float(np.percentile(b, 2.5)), 4),
                hi=round(float(np.percentile(b, 97.5)), 4), p_gt0=round(float((b > 0).mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", default="results/iptm_steer.csv")
    ap.add_argument("--naive", default="results/iptm_steer_naive.csv")
    ap.add_argument("--out", default="results/iptm_summary_naive.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    df = pd.concat([pd.read_csv(a.committed), pd.read_csv(a.naive)], ignore_index=True)
    naive_cx = set(df[df.direction == "naive"].complex_id)
    df = df[df.complex_id.isin(naive_cx)].copy()
    # composite = z-mean of the 3 interface metrics over the combined folds (re-z-scored over this set)
    df["composite"] = np.mean([zorient(df, c, hb) for c, hb in COMPOSITE_COLS], axis=0)

    cols = [("iptm", True), ("interface_pae", False), ("interface_plddt", True),
            ("ptm", True), ("composite", True)]
    rows = []
    print(f"=== NAIVE ipTM specificity ({len(naive_cx)} complexes, mean over k, complex-clustered 95% CI) ===")
    for col, hb in cols:
        piv = df.groupby(["complex_id", "direction"])[col].mean().unstack("direction")
        for ca, cb in CONTRASTS:
            res = paired_boot(piv, ca, cb, rng)
            if res is None:
                continue
            rows.append(dict(metric=col, contrast=f"{ca}-{cb}", higher_better=hb, **res))
            note = "  <- localization" if col == "ptm" else ""
            print(f"  {col:15s} {ca:6s}-{cb:6s} Δ={res['delta']:+.4f} "
                  f"[{res['lo']:+.4f},{res['hi']:+.4f}] P(>0)={res['p_gt0']:.3f} n={res['n']}{note}")
    out = pd.DataFrame(rows); out["seed"] = SEED
    out.to_csv(a.out, index=False)
    print(f"\n[wrote] {a.out}")
    # falsifier read-out at the structure level
    def g(metric, contrast):
        r = out[(out.metric == metric) & (out.contrast == contrast)]
        return r.iloc[0] if len(r) else None
    nr, ln, lr = g("iptm", "naive-random"), g("iptm", "L-naive"), g("iptm", "L-random")
    if nr is not None and ln is not None and lr is not None:
        bounded = (nr.delta >= lr.delta) or (ln.hi <= 0)
        print(f"\nipTM: naive-random={nr.delta:+.4f}  L-naive={ln.delta:+.4f}[{ln.lo:+.4f},{ln.hi:+.4f}]  "
              f"L-random={lr.delta:+.4f} -> "
              f"{'BOUNDED (structure-level falsifier)' if bounded else 'specificity holds at the structure level'}")


if __name__ == "__main__":
    main()

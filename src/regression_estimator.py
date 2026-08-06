"""Regression-adjusted hotspot effect: the matched-pair estimator's higher-powered twin.

The matched-pair design throws away every hotspot that has no acceptable control inside
its own complex - 384 pairs from 701 interface hotspots over 129 of 343 complexes. The
same estimand can be had from ALL of them by absorbing the matching variables as
covariates instead of enforcing them as constraints:

    logp_native ~ hotspot + complex fixed effects + burial spline + nbr + SS + AA identity

Complex fixed effects reproduce the "within the same complex" property of the matched
design; the burial spline, neighbour count and SS class reproduce the matching
constraints; AA identity is BRIEF 5.2's pre-registered fixed effect. Inference is a
complex-level cluster bootstrap, the same independent unit as everywhere else.

This is a robustness estimator, NOT a replacement for the pre-registered primary - it is
reported alongside it. Its value is power: it uses ~1.8x more hotspots and every complex.

Usage:
  python3 src/regression_estimator.py --out results/regression
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_BOOT, SEED = 2000, 20260803


def ns_basis(x, df=4):
    """Natural cubic spline basis on quantile knots (no intercept)."""
    x = np.asarray(x, float)
    kn = np.nanquantile(x, np.linspace(0, 1, df + 1)[1:-1])
    bk = (np.nanmin(x), np.nanmax(x))
    def d(k):
        num = np.maximum(x - k, 0) ** 3 - np.maximum(x - bk[1], 0) ** 3
        return num / (bk[1] - k) if bk[1] != k else np.zeros_like(x)
    cols = [x]
    for k in kn:
        cols.append(d(k) - d(bk[0]))
    return np.column_stack(cols)


def demean_by_group(M, groups):
    """Absorb group fixed effects by within-group demeaning."""
    M = np.asarray(M, float)
    out = M.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(axis=0, keepdims=True)
    return out


def fit(df, hot_col="is_hot", y_col="logp_native"):
    """Return the coefficient on `hot_col` after absorbing complex FE + covariates."""
    y = df[y_col].values
    parts = [df[hot_col].values.reshape(-1, 1),
             ns_basis(df["rsasa_complex"].values),
             df["nbr"].values.reshape(-1, 1).astype(float),
             pd.get_dummies(df["ss"], drop_first=True).values.astype(float),
             pd.get_dummies(df["aa"], drop_first=True).values.astype(float)]
    X = np.column_stack(parts)
    g = df["complex_id"].values
    Xd, yd = demean_by_group(X, g), demean_by_group(y.reshape(-1, 1), g).ravel()
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return float(beta[0])


def cluster_boot(df, n_boot=N_BOOT, seed=SEED, **kw):
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df[df["complex_id"] == c] for c in cids}
    pt = fit(df, **kw)
    out = []
    for b in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        d = pd.concat([by[cids[i]] for i in pick], ignore_index=True)
        # relabel so resampled copies of one complex stay distinct clusters
        d["complex_id"] = np.repeat(np.arange(len(pick)),
                                    [len(by[cids[i]]) for i in pick])
        try:
            out.append(fit(d, **kw))
        except Exception:
            continue
    out = np.array(out)
    return pt, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(out.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/regression")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    pos = pd.read_csv(a.positions,
                      usecols=["complex_id", "chain", "resnum", "aa", "ss", "nbr",
                               "rsasa_complex", "logp_native", "is_interface", "label"])
    pos["label"] = pos["label"].fillna("null")
    d = pos[pos["is_interface"] & np.isfinite(pos["rsasa_complex"])
            & np.isfinite(pos["logp_native"])].copy()

    rows = []
    print(f"interface positions: {len(d)}  complexes: {d['complex_id'].nunique()}")
    for tag, hotset in [("loose+strict", ["hot_loose", "hot_strict"]),
                        ("strict only", ["hot_strict"])]:
        s = d.copy()
        s["is_hot"] = s["label"].isin(hotset).astype(float)
        # keep complexes that actually contain both classes, as the FE absorbs the rest
        keep = s.groupby("complex_id")["is_hot"].transform(lambda x: 0 < x.sum() < len(x))
        s = s[keep]
        pt, lo, hi, se = cluster_boot(s)
        mde80 = 2.802 * se
        n_hot = int(s["is_hot"].sum())
        print(f"\n  {tag}: {n_hot} hotspots, {len(s)} positions, "
              f"{s['complex_id'].nunique()} complexes")
        print(f"    hotspot effect = {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  SE {se:.4f}")
        print(f"    MDE at 80% power = {mde80:.4f} nats")
        rows.append(dict(analysis=f"regression_{tag}", n_hot=n_hot, n_pos=len(s),
                         n_complexes=s["complex_id"].nunique(), effect=pt, lo=lo, hi=hi,
                         se=se, mde80=mde80))

    pd.DataFrame(rows).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"\n[done] wrote {a.out}_summary.csv")


if __name__ == "__main__":
    main()

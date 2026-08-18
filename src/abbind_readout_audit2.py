#!/usr/bin/env python3
"""AB-Bind audit part 2 — the leakage benchmark, without which part 1 proves nothing.

Part 1 found logP's within-mutant-identity AUROC = 0.593 [0.529, 0.660] (vs the committed
dAUROC = +0.008, n.s.). But logP is strongly burial-correlated (buried positions have peaked
distributions), so a coarse stratification leaves residual geometry inside each cell and ANY
geometry-correlated quantity scores >0.5. The benchmark that decides it is the within-stratum
AUROC of the GEOMETRY SCORE ITSELF in the same strata. logP only survives if it beats that.

Strata sweep: mutant identity x (cross-fitted geometry score) quantile bins, resolution swept.
Also reports the identity-normalised logP_minus_wt, whose identity confound is removed by
construction (same position, same distribution, difference of two entries).

  python3 src/abbind_readout_audit2.py --out results/abbind_readout_audit2.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 20260803
NBOOT = 2000
CACHE = "results/abbind_readout_audit_positions.csv"


def sauc(val, y, k):
    val = np.asarray(val, float); ok = np.isfinite(val)
    val, y, k = val[ok], np.asarray(y, float)[ok], np.asarray(k)[ok]
    order = np.lexsort((val, k)); ks, vs, ys = k[order], val[order], y[order]; n = len(ks)
    if n == 0:
        return np.nan
    gs = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    gid = np.zeros(n, np.int64); gid[gs[1:]] = 1; gid = np.cumsum(gid)
    r = np.arange(n) - gs[gid] + 1.0
    nb = np.r_[True, (ks[1:] != ks[:-1]) | (vs[1:] != vs[:-1])]
    bs = np.flatnonzero(nb); bz = np.diff(np.r_[bs, n])
    r = np.repeat((r[bs] + (r[bs] + bz - 1)) / 2.0, bz)
    ng = int(gid[-1]) + 1
    n1 = np.bincount(gid, weights=ys, minlength=ng)
    n0 = np.bincount(gid, minlength=ng).astype(float) - n1
    U = np.bincount(gid, weights=r * ys, minlength=ng) - n1 * (n1 + 1) / 2
    ok2 = (n1 > 0) & (n0 > 0); den = (n1[ok2] * n0[ok2]).sum()
    return U[ok2].sum() / den if den else np.nan


def z(v):
    v = np.asarray(v, float); s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 1e-12 else v * 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/abbind_readout_audit2.csv")
    a = ap.parse_args()
    e = pd.read_csv(CACHE)
    ei = e[e.is_interface == 1].dropna(subset=["logP", "burial", "drsasa"]).reset_index(drop=True)
    ei["destab"] = (ei.ddg >= 1.0).astype(int)
    y = ei.destab.to_numpy(float); g = ei.pdb.to_numpy()
    ids = np.unique(g); idx_by = {c: np.where(g == c)[0] for c in ids}
    rng = np.random.default_rng(SEED)
    print(f"interface mutations {len(ei)}  complexes {len(ids)}  destabilising {int(y.sum())}")
    print(f"corr(logP, burial) = {stats.spearmanr(ei.logP, ei.burial).correlation:+.3f}   "
          f"corr(logP_minus_wt, burial) = "
          f"{stats.spearmanr(ei.logP_minus_wt, ei.burial).correlation:+.3f}")

    # cross-fitted GEOMETRY-ONLY score (burial + ΔSASA) -> the leakage benchmark
    Xg = np.column_stack([z(ei.burial), z(ei.drsasa)])
    sg = np.zeros(len(y))
    for tr, te in GroupKFold(min(5, len(ids))).split(Xg, y, g):
        m = LogisticRegression(max_iter=2000).fit(Xg[tr], y[tr])
        sg[te] = Xg[te] @ m.coef_[0] + m.intercept_[0]

    rows = []
    for nb in (2, 3, 5, 8):
        gb = pd.qcut(pd.Series(sg).rank(method="first"), nb, labels=False).astype(str)
        k = pd.factorize(ei["mut"] + "|" + gb)[0].astype(np.int64)
        u = pd.DataFrame({"k": k, "y": y}).groupby("k").y.agg(["sum", "count"])
        ninf = int(((u["sum"] > 0) & (u["sum"] < u["count"])).sum())
        leak = sauc(sg, y, k)
        print(f"\n--- strata = mut x {nb} geometry bins: {len(u)} cells, {ninf} informative "
              f"| LEAKAGE benchmark (geometry score in its own strata) = {leak:.4f}")
        for f, sgn in [("logP", -1), ("logP_minus_wt", -1), ("rank_mut", -1), ("logodds", -1)]:
            v = sgn * ei[f].to_numpy(float)
            pt = sauc(v, y, k)
            bv, bd = [], []
            for _ in range(NBOOT):
                t = np.concatenate([idx_by[c] for c in rng.choice(ids, len(ids), True)])
                a1, a2 = sauc(v[t], y[t], k[t]), sauc(sg[t], y[t], k[t])
                if np.isfinite(a1) and np.isfinite(a2):
                    bv.append(a1); bd.append(a1 - a2)
            lo, hi = np.percentile(bv, [2.5, 97.5]); dl, dh = np.percentile(bd, [2.5, 97.5])
            vd = ("BEATS leakage" if dl > 0 else "not beyond leakage")
            v05 = "excl 0.5" if lo > 0.5 else "spans 0.5"
            print(f"    {f:14s} sAUROC={pt:.4f} [{lo:.4f},{hi:.4f}] ({v05})  "
                  f"minus leakage {pt-leak:+.4f} [{dl:+.4f},{dh:+.4f}]  {vd}")
            rows.append(dict(strata=f"mut_x_{nb}geobins", quantity=f, sauroc=round(pt, 4),
                             lo=round(lo, 4), hi=round(hi, 4), leakage_sauroc=round(leak, 4),
                             minus_leakage=round(pt - leak, 4), ml_lo=round(dl, 4), ml_hi=round(dh, 4),
                             p_gt_leak=round(float(np.mean(np.array(bd) > 0)), 4), verdict=vd,
                             n_cells=len(u), n_informative=ninf, n=len(ei), n_destab=int(y.sum()),
                             n_complex=len(ids)))

    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_boot"] = NBOOT
    out["note"] = ("leakage-benchmarked within-(mutant identity x geometry) AUROC; the benchmark is "
                   "the cross-fitted geometry score's own within-stratum AUROC")
    out["command"] = "python3 src/abbind_readout_audit2.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

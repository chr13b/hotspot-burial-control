#!/usr/bin/env python3
"""KL audit, part 2 — the two controls part 1 was missing.

(A) LEAKAGE BENCHMARK for the within-stratum AUROC. Coarse geometry strata leave residual
    geometry inside each cell, so ANY geometry-correlated feature scores >0.5 within stratum.
    The benchmark is the within-stratum AUROC of the geometry score ITSELF. KL only "adds"
    if it beats that benchmark. Swept over stratum resolution.

(B) CALIBRATED NULL for the committed estimator. src/kl_geometry_control.py:76-80 compares
    dAUROC(full+X - full) against ZERO. Part 1 showed that estimator's noise floor is about
    -0.022 (crystal) / -0.026 (Bennett): adding a PURE NOISE feature with unit weight to a
    unit-weight 3-term composite costs that much. Here we bootstrap the noise floor properly
    and re-decide KL against it.

  python3 src/kl_readout_audit2.py --out results/kl_readout_audit2.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 20260803
NBOOT = 2000
sys_auc = None


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sauc(val, y, k):
    order = np.lexsort((val, k)); ks, vs, ys = k[order], val[order], y[order]; n = len(ks)
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
    ok = (n1 > 0) & (n0 > 0); den = (n1[ok] * n0[ok]).sum()
    return U[ok].sum() / den if den else np.nan


def z(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else v * 0


def cv_score(X, y, g, nfold=5):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(min(nfold, len(np.unique(g)))).split(X, y, g):
        m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        out[te] = X[te] @ m.coef_[0] + m.intercept_[0]
    return out


def run(name, df, gcol, rows):
    df = df.reset_index(drop=True)
    y = df.y.to_numpy(float); g = df[gcol].to_numpy()
    ids = np.unique(g); idx_by = {c: np.where(g == c)[0] for c in ids}
    rng = np.random.default_rng(SEED)
    Xg = np.column_stack([z(df[c]) for c in ["burial", "nbr", "dsasa"]])
    sg = cv_score(Xg, y, g)
    kl = df.kl.to_numpy(float)
    print(f"\n########## {name}: {len(df)} pos, {len(ids)} groups, {int(y.sum())} hot")

    # ---------------- (A) leakage benchmark, swept over stratum resolution -------------
    print("  (A) within-stratum AUROC vs the LEAKAGE BENCHMARK (geometry score in its own strata)")
    for nb in (10, 20, 40, 80, 160):
        k = pd.qcut(pd.Series(sg).rank(method="first"), nb, labels=False).to_numpy().astype(np.int64)
        a_kl, a_geo = sauc(kl, y, k), sauc(sg, y, k)
        bk, bg, bd = [], [], []
        for _ in range(NBOOT):
            t = np.concatenate([idx_by[c] for c in rng.choice(ids, len(ids), True)])
            v1, v2 = sauc(kl[t], y[t], k[t]), sauc(sg[t], y[t], k[t])
            if np.isfinite(v1) and np.isfinite(v2):
                bk.append(v1); bg.append(v2); bd.append(v1 - v2)
        lk, hk = np.percentile(bk, [2.5, 97.5]); ld, hd = np.percentile(bd, [2.5, 97.5])
        vd = "KL BEATS leakage" if ld > 0 else "not beyond leakage"
        print(f"    {nb:3d} bins (~{len(y)//nb:4d}/bin): KL={a_kl:.4f} [{lk:.4f},{hk:.4f}]  "
              f"geom-leak={a_geo:.4f}  KL-leak={a_kl-a_geo:+.4f} [{ld:+.4f},{hd:+.4f}]  {vd}")
        rows.append(dict(fixture=name, test=f"leakage_bench_{nb}bins", kl_sauroc=round(a_kl, 4),
                         kl_lo=round(lk, 4), kl_hi=round(hk, 4), leak_sauroc=round(a_geo, 4),
                         diff=round(a_kl - a_geo, 4), diff_lo=round(ld, 4), diff_hi=round(hd, 4),
                         p_gt0=round(float(np.mean(np.array(bd) > 0)), 4), verdict=vd))

    # ---------------- (B) the committed estimator against its CALIBRATED null ----------
    print("  (B) committed estimator dAUROC(full+X - full) vs its own NOISE floor")
    u0 = (z(df.burial) + z(df.nbr) + z(df.dsasa)).astype(float)
    a0 = auc(u0, y)
    d_kl = auc(u0 + z(kl), y) - a0
    NN = 200
    noise = np.array([auc(u0 + z(rng.normal(size=len(y))), y) - a0 for _ in range(NN)])
    nlo, nhi = np.percentile(noise, [2.5, 97.5])
    # paired bootstrap of (dAUROC_KL - dAUROC_noise)
    bd = []
    for _ in range(NBOOT):
        t = np.concatenate([idx_by[c] for c in rng.choice(ids, len(ids), True)])
        yy = y[t]
        if not (0 < yy.sum() < len(yy)):
            continue
        b0 = auc(u0[t], yy)
        dn = np.mean([auc(u0[t] + z(rng.normal(size=len(t))), yy) - b0 for _ in range(3)])
        bd.append((auc(u0[t] + z(kl[t]), yy) - b0) - dn)
    bd = np.array(bd); ld, hd = np.percentile(bd, [2.5, 97.5])
    vd = "KL BEATS noise floor" if ld > 0 else "at/below noise floor"
    print(f"    dAUROC(KL) = {d_kl:+.4f}   NOISE floor = {noise.mean():+.4f} [{nlo:+.4f},{nhi:+.4f}] "
          f"(the estimator's true null, NOT 0)")
    print(f"    calibrated: dAUROC(KL) - dAUROC(noise) = {d_kl-noise.mean():+.4f} "
          f"[{ld:+.4f},{hd:+.4f}] P(>0)={np.mean(bd>0):.3f}   {vd}")
    rows.append(dict(fixture=name, test="committed_estimator_vs_calibrated_null",
                     kl_sauroc=round(d_kl, 4), leak_sauroc=round(float(noise.mean()), 4),
                     diff=round(d_kl - float(noise.mean()), 4), diff_lo=round(ld, 4),
                     diff_hi=round(hd, 4), p_gt0=round(float(np.mean(bd > 0)), 4), verdict=vd,
                     note=f"noise floor CI [{nlo:.4f},{nhi:.4f}] over {NN} draws"))

    # the reductio: adding a DUPLICATE of a geometry feature the composite already contains
    for dup in ["dsasa", "burial"]:
        dd = auc(u0 + z(df[dup]), y) - a0
        print(f"    REDUCTIO: adding a 2nd copy of {dup:6s} (already in the composite) "
              f"gives dAUROC = {dd:+.4f}")
        rows.append(dict(fixture=name, test=f"reductio_duplicate_{dup}", kl_sauroc=round(dd, 4),
                         verdict="an estimator that penalises duplicating its own feature is "
                                 "measuring the combiner, not the feature"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kl_readout_audit2.csv")
    a = ap.parse_args()
    rows = []
    j = pd.read_csv("results/kl_detector_joined.csv")
    j = j[j.is_interface == 1].copy(); j["icode"] = j.icode.fillna("").astype(str)
    pos = pd.read_csv("results/p0_positions.csv",
                      usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    pos["icode"] = pos.icode.fillna("").astype(str)
    j = j.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left")
    j = j.rename(columns={"drsasa": "dsasa", "is_hot": "y"}).dropna(
        subset=["dsasa", "kl", "burial", "nbr", "y"])
    run("SKEMPI_crystal", j, "complex_id", rows)

    b = pd.read_csv("results/bennett_kl_positions.csv")
    b = b[(b.native_match == 1) & (b.is_interface == 1)].copy()
    b["y"] = (b.restr >= 0.75).astype(int)
    run("Bennett_denovo", b, "parent", rows)

    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_boot"] = NBOOT
    out["command"] = "python3 src/kl_readout_audit2.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

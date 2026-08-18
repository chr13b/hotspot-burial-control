#!/usr/bin/env python3
"""AUDIT of the KL demotion (results/kl_geometry_control{,_predicted}.csv).

The demotion rests on ONE decision variable: dAUROC(full+KL - full) where
    full    = z(burial) + z(nbr) + z(dsasa)          # FIXED UNIT WEIGHTS, never fitted
    full_KL = full + z(kl)                           # KL forced to weight 1/4 of the composite
(src/kl_geometry_control.py:65-67). Three things that readout cannot do:

1. WEIGHTS. The composite is an unfitted equal-weight z-sum. Adding a 4th unit-weight term
   both (a) forces KL's weight and (b) dilutes the three geometry weights. A feature can carry
   real conditional information and still move an equal-weight composite by ~0 (or negative,
   as on Bennett: -0.012). This is a property of the combiner, not of the feature.

2. POWER. dAUROC over a 0.73-0.77 baseline is compressive in exactly the way
   dAUROC-over-an-0.853-one-hot-baseline was in src/catalytic_audit.py. We calibrate the
   detection floor with a synthetic positive control: inject a feature with a KNOWN
   within-geometry-stratum AUROC and read out what dAUROC it produces.

3. QUANTITY. Only the scalar `kl` was tested. The same file carries `jsd`, `dH` (partner-induced
   entropy change) and `H_complex` (the determinacy quantity the catalytic audit showed is the
   right one). None were run through the geometry control.

Correct readouts computed here: (a) FITTED cross-validated nested logistic dAUROC,
(b) WITHIN-GEOMETRY-STRATUM AUROC (model-free strata; composition-removal analogue),
(c) the synthetic power calibration, (d) the optimal-weight diagnostic.

  python3 src/kl_readout_audit.py --out results/kl_readout_audit.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 20260803
NBOOT = 2000


# ---------------------------------------------------------------- AUROC machinery
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sauc(val, y, k):
    """Pooled WITHIN-STRATUM AUROC. Verbatim from src/catalytic_audit.py:101 (validated there
    against brute force on 597 random cases). Between-stratum contrasts contribute nothing, so
    stratifying on geometry removes geometry exactly."""
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


def qbin(v, q):
    """Model-free quantile bins, robust to ties/degenerate columns."""
    try:
        return pd.qcut(pd.Series(v).rank(method="first"), q, labels=False).to_numpy()
    except Exception:
        return np.zeros(len(v), int)


def cv_score(X, y, g, nfold=5):
    """Cross-fitted logistic linear predictor (grouped by complex) — FITTED weights."""
    out = np.zeros(len(y))
    for tr, te in GroupKFold(min(nfold, len(np.unique(g)))).split(X, y, g):
        m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        out[te] = X[te] @ m.coef_[0] + m.intercept_[0]
    return out


# ---------------------------------------------------------------- per-fixture audit
def run(name, df, gcol, feats, rows, rng):
    df = df.reset_index(drop=True)
    y = df.y.to_numpy(float); g = df[gcol].to_numpy()
    ids = np.unique(g); idx_by = {c: np.where(g == c)[0] for c in ids}
    GEO = ["burial", "nbr", "dsasa"]
    Xg = np.column_stack([z(df[c]) for c in GEO])

    def boot_idx():
        return np.concatenate([idx_by[c] for c in rng.choice(ids, len(ids), True)])

    print(f"\n########## {name}: {len(df)} positions, {len(ids)} groups, {int(y.sum())} hot")

    # ---- model-free geometry strata: burial quintile x nbr tertile x dSASA quintile
    kgeo = pd.factorize(pd.Series(
        [f"{a}|{b}|{c}" for a, b, c in zip(qbin(df.burial, 5), qbin(df.nbr, 3), qbin(df.dsasa, 5))]
    ))[0].astype(np.int64)
    # ---- cross-fitted geometry-score strata (20 bins), the nugget_cpi.py stratification
    sg = cv_score(Xg, y, g)
    kfit = qbin(sg, 20).astype(np.int64)
    nstr = len(np.unique(kgeo))
    usable = pd.DataFrame({"k": kgeo, "y": y}).groupby("k").y.agg(["sum", "count"])
    nus = int(((usable["sum"] > 0) & (usable["sum"] < usable["count"])).sum())
    print(f"  model-free strata: {nstr} cells, {nus} informative (both classes)")

    geo_cv_auc = auc(sg, y)
    print(f"  AUROC geometry (FITTED, cross-validated) = {geo_cv_auc:.4f}   "
          f"(unfitted z-sum = {auc(z(df.burial)+z(df.nbr)+z(df.dsasa), y):.4f})")

    for f in feats:
        if f not in df or not np.isfinite(df[f]).any():
            continue
        v = df[f].to_numpy(float)
        raw = auc(v, y)

        # (b) WITHIN-GEOMETRY-STRATUM AUROC, two stratifications, complex bootstrap
        pt_mf = sauc(v, y, kgeo); pt_ft = sauc(v, y, kfit)
        bmf, bft = [], []
        for _ in range(NBOOT):
            t = boot_idx()
            a1 = sauc(v[t], y[t], kgeo[t]); a2 = sauc(v[t], y[t], kfit[t])
            if np.isfinite(a1): bmf.append(a1)
            if np.isfinite(a2): bft.append(a2)
        lo_mf, hi_mf = np.percentile(bmf, [2.5, 97.5]); lo_ft, hi_ft = np.percentile(bft, [2.5, 97.5])

        # (a) FITTED nested logistic dAUROC (geometry vs geometry+f), cross-validated
        Xf = np.column_stack([Xg, z(v)])
        sf = cv_score(Xf, y, g)
        d_fit = auc(sf, y) - geo_cv_auc
        # (unfitted z-sum dAUROC, the committed readout, for direct comparison)
        u0 = z(df.burial) + z(df.nbr) + z(df.dsasa); u1 = u0 + z(v)
        d_uw = auc(u1, y) - auc(u0, y)
        bf, bu = [], []
        for _ in range(NBOOT):
            t = boot_idx(); yy = y[t]
            if not (0 < yy.sum() < len(yy)):
                continue
            bf.append(auc(sf[t], yy) - auc(sg[t], yy))
            bu.append(auc(u1[t], yy) - auc(u0[t], yy))
        lf, hf = np.percentile(bf, [2.5, 97.5]); lu, hu = np.percentile(bu, [2.5, 97.5])

        vd = "ADDS" if lo_mf > 0.5 else ("ANTI" if hi_mf < 0.5 else "chance")
        print(f"  {f:11s} rawAUROC={raw:.3f} | within-geom sAUROC(model-free)={pt_mf:.4f} "
              f"[{lo_mf:.4f},{hi_mf:.4f}] {vd:6s} (fitstrata {pt_ft:.4f} [{lo_ft:.4f},{hi_ft:.4f}]) "
              f"| dAUROC fitted={d_fit:+.4f} [{lf:+.4f},{hf:+.4f}]  unweighted={d_uw:+.4f} [{lu:+.4f},{hu:+.4f}]")
        rows.append(dict(fixture=name, quantity=f, raw_auroc=round(raw, 4),
                         within_geom_sauroc=round(pt_mf, 4), wg_lo=round(lo_mf, 4), wg_hi=round(hi_mf, 4),
                         wg_verdict=vd,
                         within_geom_sauroc_fitstrata=round(pt_ft, 4), wgf_lo=round(lo_ft, 4),
                         wgf_hi=round(hi_ft, 4),
                         dauroc_FITTED=round(d_fit, 4), dfit_lo=round(lf, 4), dfit_hi=round(hf, 4),
                         dfit_p=round(float(np.mean(np.array(bf) > 0)), 4),
                         dauroc_unweighted_zsum=round(d_uw, 4), duw_lo=round(lu, 4), duw_hi=round(hu, 4),
                         n_pos=len(df), n_groups=len(ids), n_hot=int(y.sum()),
                         geom_auroc_fitted=round(geo_cv_auc, 4)))

    # ---- (c) POWER CALIBRATION of the committed readout ---------------------------
    # Inject a synthetic feature with a KNOWN within-geometry-stratum AUROC, read out what the
    # committed dAUROC(full+X - full) estimator reports. Construction: X = rank-noise + delta*y
    # inside each stratum, sweeping delta.
    print(f"  --- power calibration of dAUROC(full+X - full), unfitted z-sum ---")
    u0 = (z(df.burial) + z(df.nbr) + z(df.dsasa)).astype(float)
    rs = np.random.default_rng(SEED)
    for delta in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
        ds, dus, dfs = [], [], []
        for _ in range(12):
            X = rs.normal(size=len(y)) + delta * y
            ds.append(sauc(X, y, kgeo))
            dus.append(auc(u0 + z(X), y) - auc(u0, y))
            dfs.append(auc(cv_score(np.column_stack([Xg, z(X)]), y, g), y) - geo_cv_auc)
        print(f"    within-geom sAUROC={np.mean(ds):.3f}  ->  dAUROC unweighted={np.mean(dus):+.4f}"
              f"   dAUROC fitted={np.mean(dfs):+.4f}")
        rows.append(dict(fixture=name, quantity=f"POSCONTROL_delta={delta}",
                         within_geom_sauroc=round(float(np.mean(ds)), 4),
                         dauroc_unweighted_zsum=round(float(np.mean(dus)), 4),
                         dauroc_FITTED=round(float(np.mean(dfs)), 4),
                         n_pos=len(df), n_groups=len(ids), n_hot=int(y.sum())))

    # ---- (d) OPTIMAL-WEIGHT diagnostic: is unit weight the right weight for KL? ----
    best = max(((auc(u0 + w * z(df.kl), y), w) for w in np.arange(-1.0, 2.01, 0.05)))
    print(f"  --- optimal-weight: AUROC(full + w*z(KL)) maximised at w={best[1]:+.2f} "
          f"(AUROC {best[0]:.4f}); committed readout forces w=+1.00 (AUROC {auc(u0+z(df.kl), y):.4f}); "
          f"w=0 gives {auc(u0, y):.4f})")
    rows.append(dict(fixture=name, quantity="OPTWEIGHT_kl_in_zsum", raw_auroc=round(best[0], 4),
                     dauroc_unweighted_zsum=round(best[0] - auc(u0, y), 4),
                     wg_verdict=f"argmax_w={best[1]:+.2f}", n_pos=len(df), n_groups=len(ids),
                     n_hot=int(y.sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kl_readout_audit.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []

    # ---- SKEMPI crystal (identical loading to src/kl_geometry_control.py:90-96)
    j = pd.read_csv("results/kl_detector_joined.csv")
    j = j[j.is_interface == 1].copy(); j["icode"] = j.icode.fillna("").astype(str)
    pos = pd.read_csv("results/p0_positions.csv",
                      usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    pos["icode"] = pos.icode.fillna("").astype(str)
    j = j.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left")
    j = j.rename(columns={"drsasa": "dsasa", "is_hot": "y"}).dropna(
        subset=["dsasa", "kl", "burial", "nbr", "y"])
    j["negH"] = -j.H_complex                      # determinacy (the catalytic-audit quantity)
    j["conf"] = j.logp_native                     # the "confidence is not competence" scalar
    run("SKEMPI_crystal", j, "complex_id",
        ["kl", "jsd", "dH", "negH", "conf", "dsasa"], rows, rng)

    # ---- Bennett de-novo (identical loading to src/kl_geometry_control.py:99-102)
    b = pd.read_csv("results/bennett_kl_positions.csv")
    b = b[(b.native_match == 1) & (b.is_interface == 1)].copy()
    b["y"] = (b.restr >= 0.75).astype(int)
    run("Bennett_denovo", b, "parent", ["kl", "dsasa"], rows, rng)

    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_boot"] = NBOOT
    out["note"] = ("audit of kl_geometry_control.csv: FITTED cross-validated nested logistic dAUROC and "
                   "within-geometry-stratum AUROC replace the unfitted equal-weight z-sum dAUROC; "
                   "POSCONTROL rows calibrate the committed readout's detection floor")
    out["command"] = "python3 src/kl_readout_audit.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

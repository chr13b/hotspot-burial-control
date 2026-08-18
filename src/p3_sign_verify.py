#!/usr/bin/env python3
"""Verify the Fable-5 audit's priority-1 claim (sign-recovery is a class-imbalance artifact) and
priority-2 (swapped-order double-counting), and compute the HONEST replacements."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEED = 20260803
rng = np.random.default_rng(SEED)
d = pd.read_csv("results/p3_coupling.csv")
c = d.dropna(subset=["C_lev", "g"]).copy()
c["hit"] = (np.sign(-c.C_lev) == np.sign(c.g)).astype(int)

def mcc(yhat, y):
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn - fp*fn)/den if den > 0 else 0.0

def bal_acc(yhat, y):
    y = np.asarray(y); yhat = np.asarray(yhat)
    tpr = (yhat[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (yhat[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return np.nanmean([tpr, tnr])

print("=== P1: sign accuracy vs the trivial majority-class baseline (audit claim) ===")
print(f"{'subset':12s} {'n':>4s} {'model':>6s} {'major':>6s} {'lift':>7s} {'balAcc':>7s} {'MCC':>7s}")
# model prediction of sign(g): predict g<0 iff C_lev>0 (since C~-g). y = 1[g<0]
c["ypred_gneg"] = (c.C_lev > 0).astype(int)
c["y_gneg"] = (c.g < 0).astype(int)
for lab, sub in [("all", c), ("|g|>0.5", c[c.g.abs() > .5]), ("|g|>1.0", c[c.g.abs() > 1]),
                 ("|g|>1.5", c[c.g.abs() > 1.5]), ("|g|>2.0", c[c.g.abs() > 2]),
                 ("|C|>p75", c[c.C_lev.abs() >= c.C_lev.abs().quantile(.75)]),
                 ("|C|>p90", c[c.C_lev.abs() >= c.C_lev.abs().quantile(.90)])]:
    y = sub.y_gneg.values; yhat = sub.ypred_gneg.values
    maj = max(y.mean(), 1 - y.mean())
    print(f"{lab:12s} {len(sub):>4d} {sub.hit.mean():>6.3f} {maj:>6.3f} {sub.hit.mean()-maj:>+7.3f} "
          f"{bal_acc(yhat,y):>7.3f} {mcc(yhat,y):>+7.3f}")

print("\n=== the HONEST sign channel: partial rank-corr(C_lev, direction) controlling |g| AND distance ===")
def partial_multi(x, y, Zcols):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    for z in Zcols: m &= np.isfinite(z)
    x, y = x[m], y[m]; Z = np.column_stack([np.ones(m.sum())] + [stats.rankdata(z[m]) for z in Zcols])
    rx = stats.rankdata(x) - Z @ np.linalg.lstsq(Z, stats.rankdata(x), rcond=None)[0]
    ry = stats.rankdata(y) - Z @ np.linalg.lstsq(Z, stats.rankdata(y), rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1], m
def boot_multi(sub, x, y, Zcols):
    r, _ = partial_multi(sub[x], sub[y], [sub[z] for z in Zcols])
    ids = sub.complex_id.unique(); by = {k: sub.index[sub.complex_id == k].to_numpy() for k in ids}
    bs = []
    for _ in range(3000):
        ix = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)]); s = sub.loc[ix]
        rr, _ = partial_multi(s[x], s[y], [s[z] for z in Zcols])
        if np.isfinite(rr): bs.append(rr)
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for lab, sub in [("all", c), ("cross", c[c.cross_interface == 1])]:
    sub = sub.reset_index(drop=True); sub["gneg"] = (sub.g < 0).astype(int); sub["absg"] = sub.g.abs()
    r, lo, hi = boot_multi(sub, "C_lev", "gneg", ["absg", "dist_cb"])
    print(f"  {lab:6s}: partial(C_lev, 1[g<0] | |g|,dist) = {r:+.3f} [{lo:+.3f},{hi:+.3f}]  n={len(sub)}")

print("\n=== audit's skew-artifact test: sign-stratified partial rho(C_lev,g|dist) (both should be NEG) ===")
def partial(sub):
    x, y, z = sub.C_lev.values, sub.g.values, sub.dist_cb.values
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); x, y, z = x[m], y[m], z[m]
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z)); Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1], m.sum()
for lab, sub in [("g<0 all", c[c.g < 0]), ("g>0 all", c[c.g > 0]),
                 ("g<0 cross", c[(c.g < 0) & (c.cross_interface == 1)]),
                 ("g>0 cross", c[(c.g > 0) & (c.cross_interface == 1)])]:
    r, n = partial(sub); print(f"  {lab:10s}: partial rho={r:+.3f}  n={n}")

print("\n=== P2: swapped-order double-counting ===")
tri = d.copy()
tri["canon"] = tri.apply(lambda r: r.complex_id + "|" + ",".join(sorted([r.m1, r.m2])), axis=1)
dup = tri.canon.duplicated(keep=False)
print(f"  duplicate physical pairs: {int(dup.sum())} rows collapse to {tri[dup].canon.nunique()} pairs "
      f"(all cross={int(tri[dup].cross_interface.all())})")
# verify C_complex bit-identical under swap
for k, grp in tri[dup].groupby("canon"):
    vals = grp.C_complex.round(6).unique()
    if len(vals) > 1: print(f"    WARN {k}: C_complex differs {vals}")
dedup = tri.drop_duplicates("canon", keep="first").copy()
print(f"  dedup: {len(tri)} -> {len(dedup)} rows")
for lab, sub in [("all", dedup), ("cross", dedup[dedup.cross_interface == 1])]:
    sub = sub.dropna(subset=["C_lev"]).reset_index(drop=True); r, n = partial(sub)
    print(f"    dedup partial(C_lev,g|dist) {lab}: {r:+.3f}  n={n}")
# SKEMPI reproducibility floor from the duplicates (same physical pair, different g)
gg = tri[dup].groupby("canon").g.agg(["min", "max"]); gg["spread"] = gg["max"] - gg["min"]
print(f"  SKEMPI reproducibility floor on these pairs: mean|Δg|={gg.spread.mean():.3f}, max={gg.spread.max():.3f} kcal/mol")

print("\n=== P4: dose-response, distance-controlled (audit: pooled is distance-inflated) ===")
dose = {}
for lab, sub in [("all", c), ("cross", c[c.cross_interface == 1])]:
    sub = sub.reset_index(drop=True); sub["absC"] = sub.C_lev.abs(); sub["absg"] = sub.g.abs()
    r, _ = partial_multi(sub["absC"], sub["absg"], [sub["dist_cb"]])
    dose[lab] = r; print(f"  {lab:6s}: partial Spearman(|C|,|g| | dist) = {r:+.3f}")

# --- committed CSV of the HONEST sign analysis (rule 4) ---
out = []
for lab, sub in [("all", c), ("|g|>1.0", c[c.g.abs() > 1]), ("|C|>p75", c[c.C_lev.abs() >= c.C_lev.abs().quantile(.75)]),
                 ("|C|>p90", c[c.C_lev.abs() >= c.C_lev.abs().quantile(.90)])]:
    y = (sub.g < 0).astype(int).values; yhat = (sub.C_lev > 0).astype(int).values
    out.append(dict(metric="sign_accuracy", subset=lab, n=len(sub), model=round(sub.hit.mean(), 4),
                    majority_baseline=round(max(y.mean(), 1 - y.mean()), 4),
                    balanced_acc=round(bal_acc(yhat, y), 4), mcc=round(mcc(yhat, y), 4)))
for lab, sub in [("all", c), ("cross", c[c.cross_interface == 1])]:
    s2 = sub.reset_index(drop=True); s2["gneg"] = (s2.g < 0).astype(int); s2["absg"] = s2.g.abs()
    r, lo, hi = boot_multi(s2, "C_lev", "gneg", ["absg", "dist_cb"])
    out.append(dict(metric="partial_sign_channel|absg,dist", subset=lab, n=len(s2), model=round(r, 4),
                    majority_baseline=round(lo, 4), balanced_acc=round(hi, 4), mcc=np.nan))
for lab, sub in [("g<0_all", c[c.g < 0]), ("g>0_all", c[c.g > 0]),
                 ("g<0_cross", c[(c.g < 0) & (c.cross_interface == 1)]), ("g>0_cross", c[(c.g > 0) & (c.cross_interface == 1)])]:
    r, n = partial(sub); out.append(dict(metric="skew_strata_partial(C,g|dist)", subset=lab, n=n,
                    model=round(r, 4), majority_baseline=np.nan, balanced_acc=np.nan, mcc=np.nan))
for lab in ("all", "cross"):
    out.append(dict(metric="dose_partial(|C|,|g||dist)", subset=lab, n=np.nan, model=round(dose[lab], 4),
                    majority_baseline=np.nan, balanced_acc=np.nan, mcc=np.nan))
pd.DataFrame(out).to_csv("results/p3_sign_verify.csv", index=False)
print("\n[wrote] results/p3_sign_verify.csv  (majority_baseline/balanced_acc columns hold CI-lo/hi for the partial channel)")

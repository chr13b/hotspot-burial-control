#!/usr/bin/env python3
"""Two questions on the Phase 3 coupling result:
 (A) Did the max_residues=800 exclusion (14 complexes / 28 triangles) BIAS the estimate?
 (B) Is the weak per-pair SIGN accuracy (~53%) recoverable on a principled subset?

(A) The cut is on complex SIZE — a pre-outcome STRUCTURAL property, decided before any coupling was
computed — so it can only bias the estimate if size correlates with the coupling<->epistasis
relationship. We test: (a1) is the experimental epistasis distribution of the DROPPED triangles
different from the RETAINED? (a2) within the retained set, does partial-Spearman(C_lev,g|dist) depend
on complex size? If the g-distributions match and the effect is size-independent, the exclusion is an
honest coverage limitation, not a bias.

(B) Near-additive pairs (small |g|) have a sign that is dominated by measurement + estimation noise, so
a global per-pair sign accuracy mixes real interactions with noise. We test whether sign accuracy rises
with the magnitude of the interaction (|g|) and with the model's own coupling magnitude (|C_lev|).
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3_coupling as P
import ftax_common as fc

SEED = 20260803
DROPPED = ['1BD2_ABC_DE','1YY9_CD_A','2NYY_DC_A','3D3V_ABC_DE','3LZF_AB_HL','3QDG_ABC_DE','3QDJ_ABC_DE',
           '3VR6_ABCDEF_GH','4CVW_A_C','4FTV_ABC_DE','4GNK_A_B','4GXU_ABCDEF_MN','4K71_A_BC','4L3E_ABC_DE']

d = pd.read_csv("results/p3_coupling.csv")            # 562 retained
tri_all = P.build_triangles()                          # 590 (pre-guard)
tri_drop = tri_all[tri_all.complex_id.isin(DROPPED)].copy()
rng = np.random.default_rng(SEED)
rows = []

def partial(x, y, z):
    x, y, z = map(np.asarray, (x, y, z)); m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z)); Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1]), int(m.sum())

def partial_boot(sub):
    r, n = partial(sub.C_lev, sub.g, sub.dist_cb)
    ids = sub.complex_id.unique(); by = {k: sub.index[sub.complex_id == k].to_numpy() for k in ids}
    bs = []
    for _ in range(2000):
        ix = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        rr, _ = partial(sub.loc[ix].C_lev, sub.loc[ix].g, sub.loc[ix].dist_cb)
        if np.isfinite(rr): bs.append(rr)
    bs = np.array(bs); return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), n, len(ids)

print("=" * 78)
print("(A) EXCLUSION-BIAS CHECK")
print("=" * 78)
# a1: epistasis distribution, dropped vs retained
print("\n[a1] experimental epistasis g of DROPPED vs RETAINED triangles (is the cut ~random in g?)")
for lab, gg in [("retained", d.g.values), ("dropped", tri_drop.g.values)]:
    print(f"    {lab:9s}: n={len(gg):3d}  mean g={np.mean(gg):+.3f}  mean|g|={np.mean(np.abs(gg)):.3f}  "
          f"std={np.std(gg):.3f}  |g|>1.0: {int((np.abs(gg)>1).sum())} ({100*np.mean(np.abs(gg)>1):.0f}%)")
ks = stats.ks_2samp(np.abs(d.g.values), np.abs(tri_drop.g.values))
mw = stats.mannwhitneyu(np.abs(d.g.values), np.abs(tri_drop.g.values), alternative="two-sided")
print(f"    KS(|g|) stat={ks.statistic:.3f} p={ks.pvalue:.3f} ; Mann-Whitney |g| p={mw.pvalue:.3f}  "
      f"(large p => dropped g-distribution NOT different => cut ~random in the outcome)")
cross_drop = tri_drop.assign(cross=tri_drop.m1.str[1] != tri_drop.m2.str[1]).cross.mean()
print(f"    dropped set: {len(tri_drop)} triangles in {tri_drop.complex_id.nunique()} complexes, "
      f"~{100*cross_drop:.0f}% cross-chain (mostly TCR/pMHC + antibody Fabs)")
rows.append(dict(check="A1_g_dropped_vs_retained", stat=round(ks.statistic, 4), p=round(ks.pvalue, 4),
                 n_retained=len(d), n_dropped=len(tri_drop),
                 note=f"meanabs_g retained={np.mean(np.abs(d.g.values)):.3f} dropped={np.mean(np.abs(tri_drop.g.values)):.3f}"))

# a2: size-dependence within retained -> need n per retained complex
print("\n[a2] does the effect depend on complex SIZE within the retained set? (load n per complex)")
nmap = {}
for cid in d.complex_id.unique():
    pdb, g1, g2 = cid.split("_"); path = f"{P.DATA}/PDBs/{pdb}.pdb"
    try:
        cx = fc.load_complex(path, pdb, g1, g2); nmap[cid] = cx.n; del cx
    except Exception:
        nmap[cid] = np.nan
d["n_res"] = d.complex_id.map(nmap)
med = np.nanmedian(list(nmap.values()))
print(f"    retained complex sizes: median n={int(med)}, range [{int(np.nanmin(list(nmap.values())))},"
      f"{int(np.nanmax(list(nmap.values())))}]")
for lab, sub in [("small (n<=median)", d[d.n_res <= med]), ("large (median<n<=800)", d[d.n_res > med])]:
    r, lo, hi, n, nc = partial_boot(sub.reset_index(drop=True))
    print(f"    {lab:22s}: partial(C_lev,g|dist) = {r:+.3f} [{lo:+.3f},{hi:+.3f}]  n={n} cplx={nc}")
    rows.append(dict(check=f"A2_size_{lab.split()[0]}", stat=round(r, 4), p=np.nan, n_retained=n,
                     n_dropped=np.nan, note=f"CI[{lo:+.3f},{hi:+.3f}] cplx={nc}"))
# largest retained bin is the closest proxy for the dropped (>800) regime
big = d[d.n_res > 600]
if len(big) > 20:
    r, lo, hi, n, nc = partial_boot(big.reset_index(drop=True))
    print(f"    {'largest retained n>600':22s}: partial = {r:+.3f} [{lo:+.3f},{hi:+.3f}]  n={n} cplx={nc}  "
          f"(closest proxy for the dropped >800 regime)")
    rows.append(dict(check="A2_size_n>600_proxy", stat=round(r, 4), p=np.nan, n_retained=n,
                     n_dropped=np.nan, note=f"CI[{lo:+.3f},{hi:+.3f}] cplx={nc}"))

print("\n" + "=" * 78)
print("(B) SIGN RECOVERABILITY  [sign(-C_lev)==sign(g), chance=0.50]")
print("=" * 78)
c = d[d.C_lev.notna() & d.g.notna()].copy()
c["hit"] = (np.sign(-c.C_lev) == np.sign(c.g)).astype(int)

def sign_acc(sub):
    return sub.hit.mean(), len(sub)

print("\n[b1] overall and by strength of the experimental interaction |g|:")
for lab, sub in [("ALL pairs", c),
                 ("|g|>0.5", c[c.g.abs() > 0.5]), ("|g|>1.0", c[c.g.abs() > 1.0]),
                 ("|g|>1.5", c[c.g.abs() > 1.5]), ("|g|>2.0", c[c.g.abs() > 2.0])]:
    a, n = sign_acc(sub)
    # binomial CI
    lo, hi = stats.binomtest(int(sub.hit.sum()), n).proportion_ci() if n else (np.nan, np.nan)
    print(f"    {lab:10s}: sign acc = {a:.3f} [{lo:.3f},{hi:.3f}]  n={n}")
    rows.append(dict(check=f"B_signacc_{lab.replace(' ','')}", stat=round(a, 4), p=np.nan, n_retained=n,
                     n_dropped=np.nan, note=f"CI[{lo:.3f},{hi:.3f}]"))

print("\n[b2] by the model's own coupling magnitude |C_lev| (model-confident couplings):")
for q, lab in [(0.0, "all"), (0.5, "|C|>median"), (0.75, "|C|>p75"), (0.9, "|C|>p90")]:
    thr = c.C_lev.abs().quantile(q); sub = c[c.C_lev.abs() >= thr]
    a, n = sign_acc(sub); lo, hi = stats.binomtest(int(sub.hit.sum()), n).proportion_ci()
    print(f"    {lab:10s} (|C|>={thr:.3f}): sign acc = {a:.3f} [{lo:.3f},{hi:.3f}]  n={n}")
    rows.append(dict(check=f"B_signacc_{lab}", stat=round(a, 4), p=np.nan, n_retained=n, n_dropped=np.nan,
                     note=f"CI[{lo:.3f},{hi:.3f}] thr|C|={thr:.3f}"))

print("\n[b3] BOTH strong: |g|>1.0 AND |C_lev|>median (real interactions the model is confident about):")
sub = c[(c.g.abs() > 1.0) & (c.C_lev.abs() >= c.C_lev.abs().median())]
a, n = sign_acc(sub); lo, hi = stats.binomtest(int(sub.hit.sum()), n).proportion_ci()
print(f"    sign acc = {a:.3f} [{lo:.3f},{hi:.3f}]  n={n}")
rows.append(dict(check="B_signacc_bothstrong", stat=round(a, 4), p=np.nan, n_retained=n, n_dropped=np.nan,
                 note=f"CI[{lo:.3f},{hi:.3f}] |g|>1 & |C|>median"))

print("\n[b4] |g|-weighted sign agreement (weight each pair by the size of its real interaction):")
w = c.g.abs().values; wacc = float(np.sum(w * c.hit.values) / np.sum(w))
print(f"    weighted sign acc = {wacc:.3f}  (unweighted {c.hit.mean():.3f})")
rows.append(dict(check="B_signacc_gweighted", stat=round(wacc, 4), p=np.nan, n_retained=len(c),
                 n_dropped=np.nan, note=f"unweighted={c.hit.mean():.3f}"))

out = "results/p3_coupling_biascheck.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\n[wrote] {out}")

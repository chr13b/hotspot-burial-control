"""Diagnostic: is the C2-PRIMARY physical slope (-1.248) a real dose-response or a low-iRMSD leverage
artifact? The binned interface-formed gaps are flat & positive (+0.17..+0.55), which is inconsistent with
a -1.25/decade slope unless a few very-low-iRMSD (near-crystal) points are anchoring the OLS line.

Tests: slope vs a rising iRMSD lower bound; per-complex slopes; between-complex slope; Theil-Sen (robust)
vs OLS; and the lowest-iRMSD physical points themselves. Also dumps the KL rows actually in expC2_dose.csv
(to see why interface-formed KL for T5-30 did not print). Read-only."""
import numpy as np
import pandas as pd
from scipy import stats

g = pd.read_csv("results/expC2_gap_perbackbone.csv")
g = g[np.isfinite(g["irmsd"]) & (g["irmsd"] > 0)].copy()
g["x"] = np.log10(g["irmsd"])
phys = g[(g["interface_ok"] == 1) & (g["irmsd"] <= 8) & (g["partial_T"] != 0)].copy()

print(f"=== physical set: n_bb={len(phys)}  n_cx={phys.complex_id.nunique()} ===")
print("iRMSD pctiles [0,5,10,25,50,75,90,100]:",
      np.percentile(phys.irmsd, [0, 5, 10, 25, 50, 75, 90, 100]).round(3).tolist())
print("d     pctiles [0,10,25,50,75,90,100]  :",
      np.percentile(phys.d, [0, 10, 25, 50, 75, 90, 100]).round(3).tolist())

print("\n=== OLS + Theil-Sen slope vs rising iRMSD lower bound (leverage test) ===")
for lo in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0]:
    s = phys[phys.irmsd >= lo]
    if len(s) > 3 and s.x.nunique() > 1:
        ols = np.polyfit(s.x, s.d, 1)[0]
        p = stats.linregress(s.x, s.d).pvalue
        ts = stats.theilslopes(s.d.values, s.x.values)[0]
        print(f"  iRMSD>={lo:>3}: n={len(s):3d} n_cx={s.complex_id.nunique():2d}  "
              f"OLS={ols:+.3f} (p={p:.4f})  TheilSen={ts:+.3f}")

print("\n=== per-complex own-slope (d vs x within each complex) ===")
sl = []
for c, sub in phys.groupby("complex_id"):
    if len(sub) > 3 and sub.x.nunique() > 1:
        sl.append((c, np.polyfit(sub.x, sub.d, 1)[0], len(sub),
                   round(sub.irmsd.min(), 2), round(sub.irmsd.max(), 2)))
sl = pd.DataFrame(sl, columns=["cx", "slope", "n", "irmsd_min", "irmsd_max"])
print(sl.to_string(index=False))
print(f"  per-complex slope: mean={sl.slope.mean():+.3f} median={sl.slope.median():+.3f}")

cm = phys.groupby("complex_id").agg(md=("d", "mean"), mx=("x", "mean"), n=("d", "size"))
if len(cm) > 3:
    bs = np.polyfit(cm.mx, cm.md, 1)[0]
    bp = stats.linregress(cm.mx, cm.md).pvalue
    print(f"\n=== between-complex slope (mean d vs mean logRMSD, {len(cm)} pts): {bs:+.3f} (p={bp:.4f}) ===")

print("\n=== lowest-iRMSD physical backbones (leverage suspects) ===")
print(phys.nsmallest(15, "irmsd")[["backbone_id", "partial_T", "irmsd", "d"]].to_string(index=False))
print("\n=== highest-iRMSD physical backbones ===")
print(phys.nlargest(8, "irmsd")[["backbone_id", "partial_T", "irmsd", "d"]].to_string(index=False))

print("\n=== KL rows present in results/expC2_dose.csv ===")
d = pd.read_csv("results/expC2_dose.csv")
kl = d[d["kind"].astype(str).str.startswith("kl")]
cols = [c for c in ["kind", "partial_T", "auc_burial", "auc_bk", "dauroc", "lo", "hi", "n_cx", "frac_degen"] if c in kl.columns]
print(kl[cols].to_string(index=False))

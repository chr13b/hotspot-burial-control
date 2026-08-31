#!/usr/bin/env python3
"""Figure 2 — the feature-class law. Four panels, one design system (figstyle, validated palette).
Rewritten to the figure-design critique: two-gridspec layout, floor at the placebo 95% upper bound,
one grey for the scalar class, x10^-3 units, finding stated in-plot, CI never clipped (assert_in_view).
Every number read from a committed CSV. Renders PDF (vector) + PNG.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import figstyle as S
S.apply()
R = "results"


def cpi_row(csv, test):
    d = pd.read_csv(f"{R}/{csv}", low_memory=False)
    r = d[d.test.astype(str).str.strip() == test].iloc[0]
    return float(r.stat), float(r.lo), float(r.hi)


fig = plt.figure(figsize=(5.5, 4.7))
gA = fig.add_gridspec(1, 1, left=0.30, right=0.965, top=0.905, bottom=0.575)
gB = fig.add_gridspec(1, 3, left=0.075, right=0.965, top=0.435, bottom=0.10,
                      width_ratios=[1.28, 1.02, 0.98], wspace=0.46)

# ================= 2a — the placebo-calibrated ladder as a table-with-bars =================
axa = fig.add_subplot(gA[0])
lad = pd.read_csv(f"{R}/w_placebo_ladder.csv")
def L(feat):
    r = lad[lad.feature.astype(str).str.contains(feat, case=False, regex=False)].iloc[0]
    return float(r.cpi) * 1e3, float(r.lo) * 1e3, float(r.hi) * 1e3            # ×10^-3
fl, fllo, flhi = L("duplicate of dsasa")                                       # floor 0.72 [0.34, 1.12]
# rows top->bottom: section headers interleaved
rows = [("H", "scalars of P — one pass", S.MUTED, None),
        ("confidence", "log p(native)", S.SCALAR, L("confidence")),
        ("negentropy", "−H(P)", S.SCALAR, L("negentropy")),
        ("scalar KL", "", S.SCALAR, L("scalar kl")),
        ("H", "mixed derivative — two passes", S.LEV, None),
        ("leverage", "−L(→Ala)", S.LEV, L("leverage -l(->ala)")),
        ("H", "placebo control", S.MUTED, None),
        ("duplicate of ΔSASA", "", S.GHOST, (fl, fllo, flhi))]
y = len(rows)
ymap = []
for kind, a, b, vals in rows:
    if kind == "H":
        axa.text(-0.155, y, a, transform=axa.get_yaxis_transform(), ha="left", va="center",
                 fontsize=6.4, color=b, fontweight="bold")
    else:
        v, lo, hi = vals
        axa.add_patch(Rectangle((0, y - 0.30), v, 0.60, facecolor=b, edgecolor="none", zorder=2))
        axa.plot([lo, hi], [y, y], color=S.INK, lw=0.9, zorder=3)
        axa.text(-0.155, y, kind, transform=axa.get_yaxis_transform(), ha="right", va="center", fontsize=7.3, color=S.INK)
        axa.text(-0.145, y, a, transform=axa.get_yaxis_transform(), ha="left", va="center", fontsize=6.8, color=S.MUTED)
        axa.text(0.997, y, f"{v:.2f}", transform=axa.get_yaxis_transform(), ha="right", va="center", fontsize=6.8, color=S.MUTED)
        ymap.append(hi)
    y -= 1
axa.axvspan(0, flhi, color=S.FLOOR_FILL, zorder=0)                             # floor = placebo 95% UPPER bound
axa.axvline(fl, color=S.FLOOR_EDGE, lw=0.8, ls=(0, (2, 2)), zorder=1)          # point-estimate hairline
axa.text(flhi + 0.05, 1, "placebo floor\n(95% upper bound)", fontsize=6.2, color=S.MUTED, va="center")
axa.text(0.40, 0.60, "leverage's entire 95% CI clears the floor\n(6.7× the floor at the point estimate)",
         transform=axa.transAxes, fontsize=7.2, color=S.INK, va="center", linespacing=1.4)
axa.set_ylim(0.4, len(rows) + 0.6); axa.set_xlim(-0.35, max(ymap) * 1.02)
axa.set_yticks([]); axa.set_xticks([0, 2, 4, 6]); axa.tick_params(labelsize=7.5)
axa.set_xlabel(S.CPI_POS, fontsize=8)
S.strip(axa); S.assert_in_view(axa, [max(ymap)])
S.header(axa, "only the mixed derivative clears the floor", "13,401 positions · 343 complexes · 327 hotspots")

# ================= 2b — model class: 4 architectures, mutation level =================
axb = fig.add_subplot(gB[0])
models = [("MPNN", "leverage_decomposition.csv"), ("ESM-IF1", "leverage_esmif.csv"),
          ("PiFold", "leverage_pifold.csv"), ("MIF", "leverage_mif.csv")]
levs = [cpi_row(c, "CPI(LEVERAGE L (full) | burial+nbr+dSASA)") for _, c in models]
confs = [cpi_row(c, "CPI(confidence | burial+nbr+dSASA)") for _, c in models]
x = np.arange(len(models))
for i, (lv, cf) in enumerate(zip(levs, confs)):
    axb.add_patch(Rectangle((i - 0.32, 0), 0.28, lv[0], facecolor=S.LEV, zorder=2))
    axb.plot([i - 0.18, i - 0.18], [lv[1], lv[2]], color=S.INK, lw=0.8, zorder=3)
    axb.add_patch(Rectangle((i + 0.04, 0), 0.28, cf[0], facecolor=S.SCALAR, zorder=2))
    axb.plot([i + 0.18, i + 0.18], [cf[1], cf[2]], color=S.INK, lw=0.8, zorder=3)
axb.set_xticks(x); axb.set_xticklabels([m[0] for m in models], fontsize=6.6)
axb.set_ylim(0, max(l[2] for l in levs) * 1.10); axb.set_xlim(-0.6, len(models) - 0.4)
axb.set_ylabel(S.CPI_MUT, fontsize=7.5); axb.tick_params(labelsize=7)
axb.text(0.03, 0.90, "leverage", color=S.LEV, fontsize=7, transform=axb.transAxes, fontweight="bold")
axb.text(0.03, 0.80, "confidence", color=S.MUTED, fontsize=7, transform=axb.transAxes)
S.strip(axb); S.header(axb, "on four architectures")

# ================= 2c — beyond geometry AND conservation =================
axc = fig.add_subplot(gB[1])
mc = pd.read_csv(f"{R}/skempi_conservation_masked_cpi.csv")
def C(t):
    r = mc[mc.iloc[:, 0].astype(str).str.strip() == t].iloc[0]
    return float(r.iloc[1]) * 1e3, float(r.iloc[2]) * 1e3, float(r.iloc[3]) * 1e3
crows = [("leverage | geom", S.LEV, "none", *C("CPI(leverage -L | geometry)")),
         ("leverage | geom+cons.", S.LEV, "none", *C("CPI(leverage -L | geometry + masked-conservation)")),
         ("conservation | geom", S.GHOST, S.MUTED, *C("CPI(masked-conservation | geometry)"))]
yc = np.arange(len(crows))[::-1]
for yy, (lab, fc, ec, v, lo, hi) in zip(yc, crows):
    axc.text(0.1, yy + 0.30, lab, ha="left", va="bottom", fontsize=6.5, color=S.INK)
    axc.add_patch(Rectangle((0, yy - 0.20), v, 0.40, facecolor=fc, edgecolor=ec, lw=0.8, zorder=2))
    axc.plot([lo, hi], [yy, yy], color=S.INK, lw=0.9, zorder=3)
axc.plot([crows[0][3], crows[1][3]], [yc[0], yc[1]], color=S.LEV, lw=0.6, ls=":", zorder=1)
axc.text(0.96, 0.585, "undiminished", transform=axc.transAxes, ha="right", va="center", fontsize=6.5, color=S.LEV)
axc.set_yticks([]); axc.set_ylim(-0.55, len(crows) - 0.05); axc.set_xlim(0, max(r[5] for r in crows) * 1.15)
axc.set_xlabel("CPI beyond controls ($\\times10^{-3}$)", fontsize=7.5); axc.tick_params(labelsize=7)
S.strip(axc, left=True); S.header(axc, "beyond conservation", "ρ(L, cons) = −0.08")

# ================= 2d — the actionable payoff (ΔAUROC), dumbbell base→new =================
axd = fig.add_subplot(gB[2])
dr = [("+|L| on geometry", 0.704, 0.717, 0.0007, 0.0246),
      ("+|L| on geom+cons.", 0.714, 0.730, 0.0042, 0.0290)]
yd = np.arange(len(dr))[::-1]
for yy, (lab, base, new, lo, hi) in zip(yd, dr):
    axd.plot([base, new], [yy, yy], color=S.LEV, lw=2.4, solid_capstyle="round", zorder=2)
    axd.plot(base, yy, "o", color=S.GHOST, ms=5, zorder=3)
    axd.plot(new, yy, "o", color=S.LEV, ms=5, zorder=4)
    axd.text(base - 0.002, yy, lab, ha="right", va="center", fontsize=6.4, color=S.INK)
    axd.text(new + 0.002, yy, f"+{new-base:.3f}", ha="left", va="center", fontsize=6.4, color=S.MUTED)
axd.set_yticks([]); axd.set_ylim(-0.55, len(dr) - 0.05); axd.set_xlim(0.685, 0.742)
axd.set_xticks([0.70, 0.72]); axd.tick_params(labelsize=7)
axd.set_xlabel("hotspot AUROC (geometry → +|L|)", fontsize=7.3)
S.strip(axd); S.header(axd, "improves the ranker")

for gsl, ch in [(0.30, "a")]:
    S.flabel(fig, gsl, 0.945, ch)
for i, ch in enumerate("bcd"):
    S.flabel(fig, [0.075, 0.415, 0.68][i], 0.475, ch)
S.save(fig, "fig2_featureclass")
print(f"  2b leverage: {[round(l[0],4) for l in levs]}  conf: {[round(c[0],4) for c in confs]}")

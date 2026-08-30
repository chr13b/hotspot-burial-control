#!/usr/bin/env python3
"""Figure 2 — the feature-class law. Four panels, one design system (validated palette, dataviz skill).
Every number is read from a committed CSV or carries a CSV trace. Renders PDF (vector) + PNG (inspection).

  python3 src/fig2_featureclass.py
"""
import os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- design system (validated: node validate_palette.js "#0B6FA4,#C0561F,#1B9E77,#6D4E9C" --mode light = PASS)
INK, RULE, MUTED = "#1A1A1A", "#4D4D4D", "#6B7379"
CONF, NEGENT, KL = "#9AA3AA", "#7E888F", "#636D74"      # scalars-of-P ladder: greyer = more inert = the thesis
LEV, ESMIF, GEOM, CONS = "#0B6FA4", "#C0561F", "#1B9E77", "#6D4E9C"
FLOOR_FILL, FLOOR_EDGE = "#E4E7E9", "#B4BBC0"
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "figure.dpi": 150,
})
R = "results"


def cpi_row(csv, test):
    d = pd.read_csv(f"{R}/{csv}", low_memory=False)
    r = d[d.test.astype(str).str.strip() == test]
    if not len(r):
        return np.nan, np.nan, np.nan
    r = r.iloc[0]
    return float(r.stat), float(r.get("lo", np.nan)), float(r.get("hi", np.nan))


def strip_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=RULE, labelcolor=INK, length=2.5)


def panel_title(ax, t):
    ax.set_title(t, loc="left", fontsize=8.5, color=INK, pad=6)


def sub_n(ax, s):
    ax.text(0, 1.005, s, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")


def letter(fig, ax, ch):
    ax.text(-0.02, 1.06, ch, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, ha="right", va="bottom")


fig = plt.figure(figsize=(5.5, 4.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.18], hspace=0.62, wspace=0.55,
                      left=0.30, right=0.975, top=0.90, bottom=0.10)

# ================= 2a — the placebo-calibrated ladder (position-level, one sample) =================
axa = fig.add_subplot(gs[0, :])
lad = pd.read_csv(f"{R}/w_placebo_ladder.csv")
def L(feat):  # pull a labelled row from the ladder
    r = lad[lad.feature.astype(str).str.contains(feat, case=False, regex=False)].iloc[0]
    return float(r.cpi), float(r.lo), float(r.hi)
floor = float(lad[lad.kind == "placebo"].iloc[0].cpi)      # duplicate-of-dSASA floor +0.00072
rows = [("confidence  log p(native)", CONF, *L("confidence")),
        ("negentropy  −H(P)", NEGENT, *L("negentropy")),
        ("scalar KL", KL, *L("scalar kl")),
        ("leverage  −L(→Ala)", LEV, *L("leverage -l(->ala)"))]
ys = np.arange(len(rows))[::-1]
axa.axvspan(0, floor, color=FLOOR_FILL, zorder=0)
axa.axvline(floor, color=FLOOR_EDGE, lw=0.8, zorder=1)
axa.text(floor + 0.00004, -0.46, "placebo floor", fontsize=6.5, color=MUTED, va="center", ha="left")
for y, (lab, c, v, lo, hi) in zip(ys, rows):
    axa.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3, solid_capstyle="butt")
    axa.add_patch(Rectangle((0, y - 0.28), v, 0.56, color=c, zorder=2))
    axa.text(-0.00006, y, lab, ha="right", va="center", fontsize=7.5, color=INK)
    axa.text(hi + 0.00006, y, f"+{v:.4f}", va="center", fontsize=6.8, color=MUTED)
# 6.7x bracket floor -> leverage
lv = rows[3][2]
axa.annotate("", xy=(lv, 0.0), xytext=(floor, 0.0),
             arrowprops=dict(arrowstyle="<->", color=RULE, lw=0.7, shrinkA=0, shrinkB=0))
axa.text((floor + lv) / 2, 0.28, f"{lv/floor:.1f}×", ha="center", fontsize=7, color=INK)
axa.set_yticks([]); axa.set_ylim(-0.55, len(rows) - 0.4); axa.set_xlim(-0.00004, lv * 1.28)
axa.set_xticks([0, floor, 0.002, 0.004]); axa.set_xticklabels(["0", "", "0.002", "0.004"], fontsize=7.5)
axa.set_xlabel("CPI beyond geometry (burial + neighbours + ΔSASA), position level", fontsize=8)
strip_axes(axa); panel_title(axa, "only the mixed derivative clears the floor")
sub_n(axa, "n = 13,401 interface positions · 343 complexes · 327 hotspots  → w_placebo_ladder.csv")
letter(fig, axa, "a")

# ================= 2b — model class: 4 architectures, mutation level =================
axb = fig.add_subplot(gs[1, 0])
models = [("ProteinMPNN", "leverage_decomposition.csv"), ("ESM-IF1", "leverage_esmif.csv"),
          ("PiFold", "leverage_pifold.csv"), ("MIF", "leverage_mif.csv")]
levs, confs = [], []
for nm, csv in models:
    levs.append(cpi_row(csv, "CPI(LEVERAGE L (full) | burial+nbr+dSASA)"))
    confs.append(cpi_row(csv, "CPI(confidence | burial+nbr+dSASA)"))
x = np.arange(len(models))
for i, (lv, cf) in enumerate(zip(levs, confs)):
    axb.plot([i - 0.18, i - 0.18], [lv[1], lv[2]], color=INK, lw=0.8)
    axb.add_patch(Rectangle((i - 0.32, 0), 0.28, lv[0], color=LEV, zorder=2))
    axb.plot([i + 0.18, i + 0.18], [cf[1], cf[2]], color=INK, lw=0.8)
    axb.add_patch(Rectangle((i + 0.04, 0), 0.28, cf[0], color=CONF, zorder=2))
axb.axhline(0, color=RULE, lw=0.8)
axb.set_xticks(x); axb.set_xticklabels([m[0] for m in models], fontsize=6.6, rotation=20, ha="right")
axb.set_ylim(0, max(l[2] for l in levs) * 1.12); axb.set_ylabel("CPI beyond geometry", fontsize=8)
axb.text(0.02, 0.94, "leverage", color=LEV, fontsize=7, transform=axb.transAxes, fontweight="bold")
axb.text(0.02, 0.84, "confidence", color=MUTED, fontsize=7, transform=axb.transAxes)
strip_axes(axb); panel_title(axb, "on four architectures")
sub_n(axb, "mutation level")
letter(fig, axb, "b")

# ================= 2c — beyond geometry AND conservation =================
axc = fig.add_subplot(gs[1, 1])
mc = pd.read_csv(f"{R}/skempi_conservation_masked_cpi.csv")
def C(t):
    r = mc[mc.iloc[:, 0].astype(str).str.strip() == t].iloc[0]
    return float(r.iloc[1]), float(r.iloc[2]), float(r.iloc[3])
crows = [("conservation | geom", CONS, *C("CPI(masked-conservation | geometry)")),
         ("leverage | geom", LEV, *C("CPI(leverage -L | geometry)")),
         ("leverage | geom+cons.", LEV, *C("CPI(leverage -L | geometry + masked-conservation)"))]
yc = np.arange(len(crows))[::-1]
for y, (lab, c, v, lo, hi) in zip(yc, crows):
    axc.text(0.0001, y + 0.30, lab, ha="left", va="bottom", fontsize=6.6, color=INK)
    axc.add_patch(Rectangle((0, y - 0.20), v, 0.40, color=c, zorder=2))
    axc.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3)
axc.axvline(0, color=RULE, lw=0.8)
axc.set_yticks([]); axc.set_ylim(-0.55, len(crows) - 0.05); axc.set_xlim(0, max(r[4] for r in crows) * 1.15)
axc.set_xlabel("CPI beyond listed controls", fontsize=8)
strip_axes(axc); panel_title(axc, "beyond conservation")
sub_n(axc, "position level")
letter(fig, axc, "c")

# ================= 2d — the actionable payoff (ΔAUROC) =================
axd = fig.add_subplot(gs[1, 2])
dr = [("+|L| on geometry", 0.0125, 0.0007, 0.0246, "w4_combined_ranker.csv"),
      ("+|L| on geom+cons.", 0.0161, 0.0042, 0.0290, "skempi_conservation.csv")]
yd = np.arange(len(dr))[::-1]
for y, (lab, v, lo, hi, _) in zip(yd, dr):
    axd.text(0.0004, y + 0.30, lab, ha="left", va="bottom", fontsize=6.6, color=INK)
    axd.add_patch(Rectangle((0, y - 0.20), v, 0.40, color=LEV, zorder=2))
    axd.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3)
    axd.text(hi + 0.0008, y, f"+{v:.3f}", va="center", fontsize=6.6, color=MUTED)
axd.axvline(0, color=RULE, lw=0.8)
axd.set_yticks([]); axd.set_ylim(-0.55, len(dr) - 0.05); axd.set_xlim(0, 0.036); axd.set_xticks([0, 0.01, 0.02, 0.03])
axd.tick_params(labelsize=7); axd.set_xlabel("Δ hotspot AUROC (+|L|)", fontsize=8)
strip_axes(axd); panel_title(axd, "improves the ranker")
sub_n(axd, "vs the stronger baseline")
letter(fig, axd, "d")

os.makedirs(f"{R}/figures", exist_ok=True)
fig.savefig(f"{R}/figures/fig2_featureclass.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{R}/figures/fig2_featureclass.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
print("wrote results/figures/fig2_featureclass.{pdf,png}")
print(f"  2b leverage CPIs: {[round(l[0],4) for l in levs]}  conf: {[round(c[0],4) for c in confs]}")

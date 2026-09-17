#!/usr/bin/env python3
"""Figure 2 — the feature-class law. Four panels, one design system (figstyle, validated palette).
Two-gridspec layout, floor at the placebo 95% upper bound, one grey for the scalar class, x10^-3
units, finding stated in-plot, CI never clipped (assert_in_view).

EVERY number is read from a committed CSV — nothing is typed into this file:
  2a  results/w_placebo_ladder.csv            the ladder and the placebo floor (the 'Nx the floor'
                                              multiple is computed, not typed)
      results/leverage_skempi_positions.csv   the position / complex / hotspot counts in the note
  2b  results/leverage_{decomposition,esmif,pifold,mif}.csv     four architectures, mutation level
  2c  results/skempi_conservation_masked_cpi.csv                CPI beyond masked conservation
      + results/skempi_conservation_masked.csv joined to the positions CSV for rho(-L, conservation),
      reproducing src/skempi_conservation_masked.py's own printed Spearman from committed data
  2d  results/w4_combined_ranker.csv          Delta-AUROC of adding |L|_rms to geometry, with its CI
      results/skempi_conservation.csv         the same Delta on top of geometry + conservation

2d is a Delta-AUROC panel, not a base->new dumbbell: only the PAIRED deltas and their complex-
clustered CIs are committed for both rungs (the absolute AUROC of the geometry+conservation ranker
is not in any CSV), and the paired delta is the estimate the CI belongs to. The one absolute level
that IS committed — geometry alone — is named in the panel note.

Renders PDF (vector) + PNG.
"""
import numpy as np, pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch
import figstyle as S
S.apply()
R = "results"
MINUS = "−"


def cpi_row(csv, test):
    d = pd.read_csv(f"{R}/{csv}", low_memory=False)
    r = d[d.test.astype(str).str.strip() == test].iloc[0]
    return float(r.stat), float(r.lo), float(r.hi)


fig = plt.figure(figsize=(5.5, 4.7))
gA = fig.add_gridspec(1, 1, left=0.30, right=0.965, top=0.905, bottom=0.575)
gB = fig.add_gridspec(1, 3, left=0.092, right=0.972, top=0.435, bottom=0.10,
                      width_ratios=[1.28, 1.02, 1.06], wspace=0.46)
# the three lower panel origins, derived from the gridspec rather than typed, so a panel letter can
# never drift away from the panel it names
_WR, _WS, _L0, _R0 = [1.28, 1.02, 1.06], 0.46, 0.092, 0.972
_u = (_R0 - _L0) / (sum(_WR) + 2 * _WS * np.mean(_WR))
LEFTS = [_L0,
         _L0 + _u * (_WR[0] + _WS * np.mean(_WR)),
         _L0 + _u * (_WR[0] + _WR[1] + 2 * _WS * np.mean(_WR))]

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
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]     # section headers cross the floor band
for kind, a, b, vals in rows:
    if kind == "H":
        axa.text(-0.155, y, a, transform=axa.get_yaxis_transform(), ha="left", va="center",
                 fontsize=6.4, color=b, fontweight="bold", zorder=6, path_effects=HALO)
    else:
        lead = kind == "leverage"                            # the finding row carries the emphasis
        v, lo, hi = vals
        axa.add_patch(Rectangle((0, y - 0.30), v, 0.60, facecolor=b, edgecolor="none", zorder=2))
        axa.plot([lo, hi], [y, y], color=S.INK, lw=0.9, zorder=3)
        axa.text(-0.155, y, kind, transform=axa.get_yaxis_transform(), ha="right", va="center",
                 fontsize=7.3, color=S.INK, fontweight="bold" if lead else "normal")
        axa.text(-0.145, y, a, transform=axa.get_yaxis_transform(), ha="left", va="center",
                 fontsize=6.8, color=S.LEV if lead else S.MUTED)
        axa.text(0.997, y, f"{v:.2f}", transform=axa.get_yaxis_transform(), ha="right", va="center",
                 fontsize=7.2 if lead else 6.8, color=S.INK if lead else S.MUTED,
                 fontweight="bold" if lead else "normal")
        ymap.append(hi)
    y -= 1
axa.axvspan(0, flhi, color=S.FLOOR_FILL, zorder=0)                             # floor = placebo 95% UPPER bound
axa.axvline(fl, color=S.FLOOR_EDGE, lw=0.8, ls=(0, (2, 2)), zorder=1)          # point-estimate hairline
axa.text(flhi + 0.06, 1, "placebo floor\n(95% upper bound)", fontsize=6.2, color=S.MUTED, va="center")
lev_v, lev_lo, _ = L("leverage -l(->ala)")
MULT = lev_v / fl                                                              # read, never typed
axa.text(0.415, 0.585, f"leverage's entire 95% CI clears the floor\n"
                       f"({MULT:.1f}× the floor at the point estimate)",
         transform=axa.transAxes, fontsize=7.2, color=S.INK, va="center", linespacing=1.4)
# the value column lives PAST the data range, so a long CI can never run into its own number
XTOP = 2.0 * np.floor(max(ymap) / 2.0)                                         # last round tick under the data
axa.set_ylim(0.4, len(rows) + 0.6); axa.set_xlim(-0.35, max(ymap) * 1.16)
axa.set_yticks([]); axa.set_xticks(np.arange(0, XTOP + 1e-9, 2.0)); axa.tick_params(labelsize=7.5)
axa.spines["bottom"].set_bounds(0, XTOP)                                       # rule under the data only
axa.set_xlabel(S.CPI_POS, fontsize=8)
axa.xaxis.set_label_coords(0.42, -0.135)
S.strip(axa, left=False)                     # the rows are categories; a spine at x<0 would read as a baseline
axa.axvline(0, color=S.RULE, lw=0.8, zorder=1)                                 # the real baseline, at zero
S.assert_in_view(axa, [max(ymap)])
pos = pd.read_csv(f"{R}/leverage_skempi_positions.csv", low_memory=False)
S.header(axa, "only the mixed derivative clears the floor",
         f"{len(pos):,} positions · {pos.complex_id.nunique()} complexes · {int(pos.is_hot.sum())} hotspots")

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
axb.set_ylim(0, max(l[2] for l in levs) * 1.10); axb.set_xlim(-0.62, len(models) - 0.38)
axb.set_ylabel(S.CPI_MUT, fontsize=7.5); axb.tick_params(labelsize=7)
axb.set_xticks(x)                            # AFTER tick_params, whose labelsize would otherwise win
axb.set_xticklabels([m[0] for m in models], fontsize=6.1)
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
# rho(masked conservation, -L) recomputed here from the two committed per-position CSVs, exactly as
# src/skempi_conservation_masked.py computes the number it prints — so the note traces to data, not prose
_mk = pd.read_csv(f"{R}/skempi_conservation_masked.csv")
_j = pos.merge(_mk, on=["complex_id", "chain", "resnum", "icode"], how="inner")
_j = _j[(_j.is_interface == True) & _j.masked_negent.notna() & _j.L_ala.notna()]        # noqa: E712
RHO_LC = float(st.spearmanr(_j.masked_negent, -_j.L_ala).correlation)
S.strip(axc, left=True)
S.header(axc, "beyond conservation",
         f"ρ(conservation, −L) = {RHO_LC:+.2f}".replace("-", MINUS) + f" · n = {len(_j):,}")

# ================= 2d — the actionable payoff: ΔAUROC of adding |L|, with its CI =================
# Both rungs are PAIRED, complex-clustered deltas straight out of a committed CSV. The dumbbell this
# replaces plotted an absolute geometry+conservation baseline that no CSV carries, and silently
# dropped the CIs it had already read.
axd = fig.add_subplot(gB[2])
_w4 = pd.read_csv(f"{R}/w4_combined_ranker.csv")
_w4L = _w4[_w4.ranker.astype(str).str.strip() == "geometry + |L|_rms"].iloc[0]
GEOM_BASE = float(_w4[_w4.ranker.astype(str).str.strip() == "geometry (burial+nbr+dSASA)"].iloc[0].auroc)
_cv = pd.read_csv(f"{R}/skempi_conservation.csv")
_cvL = _cv[_cv.test.astype(str).str.strip()
           == "ranker geom+conservation +|L|_rms vs geom+conservation"].iloc[0]
dr = [("on geometry", float(_w4L.delta_vs_geom), float(_w4L.lo), float(_w4L.hi), float(_w4L.p_gt0)),
      ("on geometry\n+ conservation", float(_cvL.stat), float(_cvL.lo), float(_cvL.hi), float(_cvL.p_gt0))]
yd = np.arange(len(dr))[::-1] * 1.0
axd.axvline(0, ls=(0, (3, 2)), color=S.RULE, lw=0.8, zorder=1)
for yy, (lab, v, lo, hi, p) in zip(yd, dr):
    axd.plot([lo, hi], [yy, yy], "-", color=S.LEV, lw=1.15, solid_capstyle="butt", zorder=3)
    for e in (lo, hi):
        axd.plot([e, e], [yy - 0.11, yy + 0.11], "-", color=S.LEV, lw=1.15, zorder=3)
    axd.plot(v, yy, "o", ms=5.0, color=S.LEV, mec="white", mew=0.8, zorder=4)
    axd.text(0.0, yy + 0.30, lab, ha="left", va="bottom", fontsize=6.4, color=S.INK, linespacing=1.3)
    axd.text(hi + 0.0013, yy, f"+{v:.4f}".rstrip("0"), ha="left", va="center",
             fontsize=6.2, color=S.INK, fontweight="bold")
XD = max(h for _, _, _, h, _ in dr)
axd.set_yticks([]); axd.set_ylim(-0.72, len(dr) - 0.28); axd.set_xlim(-0.0035, XD * 1.40)
axd.set_xticks([0, 0.01, 0.02, 0.03]); axd.set_xticklabels(["0", "+.01", "+.02", "+.03"], fontsize=7)
axd.spines["bottom"].set_bounds(0, 0.03)
axd.tick_params(labelsize=7)
axd.set_xlabel("ΔAUROC from adding $|L|$", fontsize=7.3)
axd.xaxis.set_label_coords(0.42, -0.135)
S.strip(axd, left=False)                    # zero is the reference here, and it is already drawn dashed
S.assert_in_view(axd, [h for _, _, _, h, _ in dr])
S.header(axd, "improves the ranker", f"geometry alone {GEOM_BASE:.3f}")

S.flabel(fig, 0.30, 0.952, "a")
for x, ch in zip(LEFTS, "bcd"):
    S.flabel(fig, x, 0.482, ch)
S.save(fig, "fig2_featureclass")
print(f"  2a floor {fl:.3f} [{fllo:.3f},{flhi:.3f}] · leverage {lev_v:.3f} (CI lo {lev_lo:.3f}) "
      f"-> {MULT:.2f}x the floor  [w_placebo_ladder.csv]")
print(f"  2b leverage: {[round(l[0],4) for l in levs]}  conf: {[round(c[0],4) for c in confs]}")
print(f"  2c rho(masked conservation, -L) = {RHO_LC:+.4f} over n={len(_j):,} interface positions "
      f"[skempi_conservation_masked.csv x leverage_skempi_positions.csv]")
for lab, v, lo, hi, p in dr:
    print(f"  2d {lab.replace(chr(10),' '):26s} dAUROC {v:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}")
print(f"  2d geometry-alone AUROC {GEOM_BASE:.4f} [w4_combined_ranker.csv]; "
      f"conservation rung from skempi_conservation.csv")

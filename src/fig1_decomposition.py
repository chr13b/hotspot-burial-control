#!/usr/bin/env python3
"""Figure 1 — the operator. Confidence is a scalar of ONE distribution; binding is how that
distribution MOVES when the partner is deleted, and no scalar of P can see it.

  1a  schematic: two passes through one frozen f_theta -> P and Q (real 20-simplex for 3SZK/C/44,
      exp+renormalised from leverage_pq_skempi.csv), the confidence read (one coordinate of P) vs
      the leverage read (the whole P->Q move), and the classifier-free-guidance alignment strip.
  1b  identifiability: 13,401 positions, confidence vs |L|_rms; the leverage spread is the SAME
      inside every confidence decile.  leverage_skempi_positions.csv + conf_decile_leverage.csv
  1c  non-vacuity: a flexible learner recovers only ~0.37 of L from ALL of P.  r2_leverage_from_P.csv

Every plotted number is read from a committed CSV. The 1b decile ranges are 5-95th percentiles
computed deterministically from the committed per-position CSV (no resampling, no seed needed).

  python3 src/fig1_decomposition.py  ->  results/figures/fig1_decomposition.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import figstyle as S
S.apply()
R = "results"
AA = list("ACDEFGHIKLMNPQRSTVWY")
EX_HOT, EX_COLD = ("3SZK_AB_C", "C", 44), ("2QJ9_AB_C", "B", 82)

# ------------------------------------------------------------------ data (CSV only)
pos = pd.read_csv(f"{R}/leverage_skempi_positions.csv")
dec = pd.read_csv(f"{R}/conf_decile_leverage.csv").set_index("stat")
IQR_RATIO = float(dec.value["within_decile_IQR_over_overall_IQR"])
RHO_CL = float(dec.value["spearman_conf_vs_Lrms"])

r2 = pd.read_csv(f"{R}/r2_leverage_from_P.csv")
r2 = r2[(r2.target == "L_rms") & (r2.features == "P(20)")]
FLEX = {m: r2[(r2.model == m) & (r2.learner == "MAX_FLEXIBLE")].iloc[0] for m in ("ProteinMPNN", "ESM-IF1")}
LIN = {m: r2[(r2.model == m) & (r2.learner == "ridge_linear")].iloc[0] for m in ("ProteinMPNN", "ESM-IF1")}

_dl = pd.read_csv(f"{R}/leverage_decomposition.csv", low_memory=False)
_sp = _dl[_dl.test.astype(str).str.strip() == "spearman_L_vs_ddG"].iloc[0]
SP_LDDG, SP_N = float(_sp.stat), int(_sp.n)

pq = pd.read_csv(f"{R}/leverage_pq_skempi.csv")
_e = pq[(pq.complex_id == EX_HOT[0]) & (pq.chain == EX_HOT[1]) & (pq.resnum == EX_HOT[2])].iloc[0]
P = np.exp(np.array([_e[f"lP_{a}"] for a in AA])); P /= P.sum()
Q = np.exp(np.array([_e[f"lQ_{a}"] for a in AA])); Q /= Q.sum()
WT, IWT = str(_e.aa), AA.index(str(_e.aa))

def _row(cx, ch, rn):
    return pos[(pos.complex_id == cx) & (pos.chain == ch) & (pos.resnum == rn)].iloc[0]
EH, EC = _row(*EX_HOT), _row(*EX_COLD)

# ------------------------------------------------------------------ 1d/1e data (CSV only)
cfg = pd.read_csv(f"{R}/cfg_steer_summary.csv")
CL = cfg[cfg.direction == "L"].sort_values("alpha").reset_index(drop=True)
CR = cfg[cfg.direction == "random"].sort_values("alpha").reset_index(drop=True)
CFG_A = list(CL.alpha)

# ------------------------------------------------------------------ canvas (3 rows: a / b,c / d,e)
fig = plt.figure(figsize=(5.5, 7.05))
gA = fig.add_gridspec(1, 1, left=0.048, right=0.992, top=0.966, bottom=0.672)
gB = fig.add_gridspec(1, 2, left=0.092, right=0.985, top=0.612, bottom=0.432,
                      width_ratios=[1.16, 1.0], wspace=0.40)
gD = fig.add_gridspec(1, 2, left=0.092, right=0.985, top=0.300, bottom=0.076,
                      width_ratios=[1.0, 1.0], wspace=0.38)

# ================================================================== 1a — the operator
axa = fig.add_subplot(gA[0]); axa.set_axis_off()
axa.set_xlim(0, 128); axa.set_ylim(0, 50); axa.set_aspect("equal", adjustable="box")

def box(x, y, w, h, *, fc="none", ec=S.INK, lw=0.8, r=1.5, z=3, ls="-"):
    """The ONE box primitive — every rectangle in this schematic comes from it."""
    axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                 facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))

def wire(pts, *, color=S.RULE, lw=0.8, head=True, z=2):
    """The ONE connector primitive — an orthogonal polyline, arrowhead on the last segment."""
    p = np.asarray(pts, float)
    axa.plot(p[:, 0], p[:, 1], "-", color=color, lw=lw, solid_capstyle="round",
             solid_joinstyle="miter", zorder=z)
    if head:
        axa.annotate("", xy=tuple(p[-1]), xytext=tuple(p[-2]), zorder=z,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                     shrinkA=0, shrinkB=0, mutation_scale=6))

def txt(x, y, s, *, size=6.6, color=S.INK, ha="center", va="center", z=6, **kw):
    axa.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=z, **kw)

# --- inputs: the complex (two bodies) and the monomer (partner deleted) ------------
for y0, lab, partner in [(33.0, "$X_{\\mathrm{complex}}$", True),
                         (17.5, "$X_{\\mathrm{monomer}}$", False)]:
    box(2, y0, 24, 11.5, fc="white", ec=S.GHOST, lw=0.8, z=3)
    box(5.0, y0 + 2.65, 8.5, 6.2, fc=S.TINT, ec=S.INK, lw=0.9, r=2.6, z=4)
    box(13.5, y0 + 2.65, 8.5, 6.2, fc="white" if partner else "none",
        ec=S.INK if partner else S.GHOST, lw=0.9 if partner else 0.7, r=2.6, z=4,
        ls="-" if partner else (0, (2, 2)))
    txt(9.25, y0 + 5.75, "chain", size=5.8, color=S.MUTED)
    txt(17.75, y0 + 5.75, "partner" if partner else "deleted", size=5.8,
        color=S.MUTED if partner else S.GHOST)
    txt(14, y0 - 1.4, lab, size=7.0, va="top")

# --- the frozen network, drawn ONCE; both passes route through it ------------------
box(33, 23, 18, 17, fc=S.TINT, ec=S.INK, lw=1.0, z=3)
txt(42, 33.6, "$f_{\\theta}$", size=11.0)
txt(42, 27.6, "frozen\ninverse-folding\nmodel", size=5.8, color=S.MUTED, linespacing=1.3)
wire([(26, 38.75), (29.5, 38.75), (29.5, 35.0), (33, 35.0)])
wire([(26, 23.25), (29.5, 23.25), (29.5, 28.0), (33, 28.0)])
txt(29.5, 40.4, "pass 1", size=5.8, color=S.MUTED)
txt(29.5, 21.6, "pass 2", size=5.8, color=S.MUTED)

# --- the two 20-simplexes, back to back on one baseline ---------------------------
X0, XW, BASE, SC = 57.0, 27.0, 31.0, 26.0                      # 26 canvas units == probability 1.0
slot = XW / len(AA)
wire([(51, 34.0), (56.2, 34.0)])
wire([(51, 28.0), (56.2, 28.0)])
for i, (p, q) in enumerate(zip(P, Q)):
    xc = X0 + (i + 0.5) * slot
    axa.add_patch(Rectangle((xc - slot * 0.34, BASE), slot * 0.68, p * SC,
                            facecolor=S.INK, edgecolor="none", zorder=4))
    axa.add_patch(Rectangle((xc - slot * 0.34, BASE - q * SC), slot * 0.68, q * SC,
                            facecolor=S.GHOST, edgecolor="none", zorder=4))
axa.plot([X0, X0 + XW], [BASE, BASE], "-", color=S.RULE, lw=0.7, zorder=5)
txt(74.5, 44.4, "$P=p(\\cdot\\,|\\,X_{\\mathrm{complex}})$", size=6.8)
txt(70.5, 22.4, "$Q=p(\\cdot\\,|\\,X_{\\mathrm{monomer}})$", size=6.8, color=S.SOFT)
txt(70.5, 19.4, "20 amino acids, one position", size=5.8, color=S.MUTED)
txt(X0 + (AA.index("Y") + 0.5) * slot, BASE + P[AA.index("Y")] * SC + 1.0, "Y", size=6.0, va="bottom")
for i in (AA.index("E"), AA.index("K")):
    txt(X0 + (i + 0.5) * slot, BASE - Q[i] * SC - 1.0, AA[i], size=6.0, color=S.MUTED, va="top")

# --- read 1: confidence takes ONE coordinate of P ---------------------------------
XWT, YWT = X0 + (IWT + 0.5) * slot, BASE + P[IWT] * SC
axa.plot(XWT, YWT + 0.1, "o", ms=2.6, color=S.SCALAR, zorder=6)
txt(XWT - 1.4, YWT + 0.1, f"{WT} = wt", size=6.0, ha="right")
wire([(XWT, YWT + 1.2), (XWT, 47.4), (89, 47.4), (89, 41.5), (91, 41.5)], color=S.SCALAR)
box(91, 34, 36, 11, fc="none", ec=S.SCALAR, lw=0.8, z=3)
txt(93.5, 42.2, "confidence", size=7.0, color=S.SOFT, ha="left")
txt(93.5, 39.0, "$\\varphi(P)=\\log P(\\mathrm{wt})$", size=7.0, ha="left")
txt(93.5, 36.0, "one pass · one coordinate of $P$", size=6.0, color=S.MUTED, ha="left")

# --- read 2: leverage takes the whole P -> Q move ---------------------------------
YT, YB = BASE + P.max() * SC + 1.2, BASE - Q.max() * SC - 1.2
axa.plot([86, 86], [YB, YT], "-", color=S.LEV, lw=0.9, zorder=5)
for yy in (YB, YT):
    axa.plot([84.9, 86], [yy, yy], "-", color=S.LEV, lw=0.9, zorder=5)
wire([(86, 32.5), (89, 32.5), (89, 24.0), (91, 24.0)], color=S.LEV, lw=0.9)
box(91, 17, 36, 12, fc=S.rgba(S.LEV, 0.06), ec=S.LEV, lw=1.5, z=3)
txt(93.5, 26.2, "leverage", size=7.0, color=S.LEV, ha="left")
txt(93.5, 23.0, "$L=\\log P-\\log Q$", size=7.0, ha="left")
txt(93.5, 20.0, "two passes · the whole 20-vector", size=6.0, color=S.MUTED, ha="left")
txt(127, 15.0, "Spearman($L$, ΔΔG) = " + f"{SP_LDDG:.2f}".replace("-", "−")
    + f"  ·  n = {SP_N:,} mutations",
    size=6.2, color=S.LEV, ha="right", va="top")

# --- the CFG strip: the same difference, one field over ---------------------------
axa.plot([2, 126], [11.5, 11.5], "-", color=S.GHOST, lw=0.6, zorder=1)
CX, CM, CQ = 42.0, 62.0, 84.0
txt(2, 7.6, "diffusion", size=6.2, color=S.MUTED, ha="left")
txt(2, 3.2, "this work", size=6.2, color=S.LEV, ha="left")
for y, a, b in [(7.6, "$\\epsilon_{\\theta}(x\\,|\\,c)$", "$\\epsilon_{\\theta}(x\\,|\\,\\varnothing)$"),
                (3.2, "$\\log p(\\cdot\\,|\\,X_{\\mathrm{complex}})$",
                      "$\\log p(\\cdot\\,|\\,X_{\\mathrm{monomer}})$")]:
    txt(CX, y, a, size=7.2); txt(CM, y, "$-$", size=7.2); txt(CQ, y, b, size=7.2)
for xx in (CX, CQ):                                            # connect the ALIGNED PAIRS only
    axa.plot([xx, xx], [6.3, 4.5], "-", color=S.LEV, lw=0.7, zorder=2)
txt(126, 6.2, "classifier-free guidance\nis this same difference", size=6.0, color=S.MUTED,
    ha="right", va="center", linespacing=1.35)

S.header(axa, "confidence reads one coordinate of P; leverage reads the whole P → Q move",
         f"example position: 3SZK chain C {WT}{EX_HOT[2]} (a hotspot) · bars are the real 20-simplex")

# ================================================================== 1b — identifiability
axb = fig.add_subplot(gB[0])
hot, cold = pos[pos.is_hot == 1], pos[pos.is_hot == 0]
axb.scatter(cold.conf, cold.L_rms, s=1.1, c=S.GHOST, lw=0, alpha=0.45, rasterized=True, zorder=2)
axb.scatter(hot.conf, hot.L_rms, s=3.4, c=S.INK, lw=0, alpha=0.85, rasterized=True, zorder=3)
q = np.quantile(pos.conf, np.linspace(0, 1, 11))
for i in range(10):
    m = (pos.conf >= q[i]) & ((pos.conf <= q[i + 1]) if i == 9 else (pos.conf < q[i + 1]))
    xc = float(np.median(pos.conf[m]))
    axb.plot([xc, xc], np.percentile(pos.L_rms[m], [5, 95]), "-", color=S.LEV, lw=1.8,
             solid_capstyle="butt", zorder=4)
for e in (EH, EC):
    axb.plot(e.conf, e.L_rms, "o", ms=4.8, mfc="none", mec=S.INK, mew=1.0, zorder=6)
axb.annotate("", xy=(EH.conf, EH.L_rms - 0.32), xytext=(EC.conf, EC.L_rms + 0.32),
             arrowprops=dict(arrowstyle="<->", color=S.INK, lw=0.8, shrinkA=0, shrinkB=0), zorder=6)
axb.annotate("same confidence,\n100× the leverage", xy=(EH.conf - 0.10, 3.1), xytext=(-5.90, 6.75),
             fontsize=6.3, color=S.INK, ha="left", va="top", linespacing=1.35,
             arrowprops=dict(arrowstyle="-", color=S.RULE, lw=0.7, shrinkA=3, shrinkB=3))
axb.text(-5.90, 8.75, f"blue bars: 5–95th pct of |L| per decile\n"
                      f"within-decile IQR = {IQR_RATIO:.2f}× overall\n"
                      f"ρ(confidence, |L|) = {RHO_CL:+.3f}",
         fontsize=6.3, color=S.MUTED, ha="left", va="top", linespacing=1.45)
axb.set_xlim(-6.05, 0.40); axb.set_ylim(-0.20, 8.85)
axb.set_xticks([-6, -4, -2, 0]); axb.set_yticks([0, 2, 4, 6]); axb.tick_params(labelsize=7.2)
axb.set_xlabel("confidence   $\\log P(\\mathrm{wt}\\,|\\,X_{\\mathrm{complex}})$", fontsize=8)
axb.set_ylabel("leverage  $|L|_{\\mathrm{rms}}$", fontsize=8)
axb.xaxis.set_label_coords(0.5, -0.145)
S.strip(axb); S.assert_in_view(axb, [float(pos.L_rms.max())], axis="y")
S.header(axb, "confidence does not pin leverage down",
         f"{len(pos):,} positions · {pos.complex_id.nunique()} complexes · {int(pos.is_hot.sum())} hotspots")

# ================================================================== 1c — non-vacuity
axc = fig.add_subplot(gB[1])
for yy, nm in [(1.0, "ProteinMPNN"), (0.0, "ESM-IF1")]:
    f, l = FLEX[nm], LIN[nm]
    axc.add_patch(Rectangle((0, yy - 0.19), 1.0, 0.38, facecolor=S.FLOOR_FILL, edgecolor="none", zorder=1))
    axc.add_patch(Rectangle((0, yy - 0.19), float(f.r2), 0.38, facecolor=S.LEV, edgecolor="none", zorder=2))
    axc.plot([float(l.r2)] * 2, [yy - 0.19, yy + 0.19], "-", color="white", lw=1.0, zorder=3)
    axc.plot([float(f.lo), float(f.hi)], [yy, yy], color=S.INK, lw=0.9, zorder=4)
    axc.text(float(f.r2) * 0.72, yy, f"{float(f.r2):.2f}", fontsize=6.8, color="white",
             ha="center", va="center", zorder=5)
    axc.text(-0.02, yy + 0.235, nm, fontsize=6.9, color=S.INK, ha="left", va="bottom")
    axc.text(0.99, yy + 0.235, f"{float(f.irreducible_frac)*100:.0f}% irreducible",
             fontsize=6.6, color=S.INK, ha="right", va="bottom")
    axc.text(1.0, yy - 0.245, f"n = {int(f.n):,}", fontsize=6.0, color=S.MUTED, ha="right", va="top")
axc.text(float(LIN["ProteinMPNN"].r2) + 0.025, 1.0 - 0.245,
         f"linear read of P: {float(LIN['ProteinMPNN'].r2):.2f}", fontsize=6.0, color=S.MUTED,
         ha="left", va="top")
axc.plot([1.0, 1.0], [-0.52, 1.46], ls=(0, (3, 2)), color=S.RULE, lw=0.8, zorder=5)
axc.text(0.985, -0.60, "P would determine L", fontsize=6.2, color=S.MUTED, ha="right", va="top")
axc.text(0.015, -0.60, "L unrelated to P", fontsize=6.2, color=S.MUTED, ha="left", va="top")
axc.set_xlim(-0.03, 1.06); axc.set_ylim(-1.00, 1.58); axc.set_yticks([])
axc.set_xticks([0, 0.5, 1.0]); axc.tick_params(labelsize=7.2)
axc.set_xlabel("$R^{2}$ of $|L|_{\\mathrm{rms}}$ from all 20 coordinates of $P$", fontsize=8)
axc.xaxis.set_label_coords(0.5, -0.145)
S.strip(axc, left=False)
S.assert_in_view(axc, [float(FLEX["ProteinMPNN"].hi), float(FLEX["ESM-IF1"].hi)])
S.header(axc, "and neither does all of P", "GBM / random forest, complex-held-out")

# ================================================================== 1d — steer with it: binding rises
axd = fig.add_subplot(gD[0])
xp = np.arange(len(CFG_A))
axd.axhline(0, color=S.RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
axd.plot(xp, CR.meanL_esmif, ls=(0, (2, 2)), color=S.SCALAR, lw=1.1, zorder=2)         # random arm
for x, lo, hi in zip(xp, CR.esmif_lo, CR.esmif_hi):
    axd.plot([x, x], [lo, hi], color=S.SCALAR, lw=0.8, zorder=2)
axd.plot(xp, CR.meanL_esmif, "s", mfc="white", mec=S.SCALAR, mew=1.0, ms=4.0, zorder=3)
axd.plot(xp, CL.meanL_esmif, "-", color=S.LEV, lw=1.6, zorder=4)                        # L arm
for x, lo, hi in zip(xp, CL.esmif_lo, CL.esmif_hi):
    axd.plot([x, x], [lo, hi], color=S.LEV, lw=0.9, zorder=4)
axd.plot(xp, CL.meanL_esmif, "o", color=S.LEV, ms=4.6, zorder=5)
axd.text(xp[3], 0.205, "steer by $L$", color=S.LEV,
         fontsize=6.8, fontweight="bold", ha="center", va="bottom")
axd.text(xp[2], float(CR.meanL_esmif.iloc[2]) - 0.035, "random dir.", color=S.MUTED,
         fontsize=6.6, ha="center", va="top")
gp = float(CL.meanL_esmif.iloc[-1] - CR.meanL_esmif.iloc[-1])
axd.annotate("", xy=(xp[-1] - 0.12, float(CL.meanL_esmif.iloc[-1])),
             xytext=(xp[-1] - 0.12, float(CR.meanL_esmif.iloc[-1])),
             arrowprops=dict(arrowstyle="<->", color=S.INK, lw=0.7, shrinkA=0, shrinkB=0), zorder=6)
axd.text(xp[-1] - 0.24, np.mean([float(CL.meanL_esmif.iloc[-1]), float(CR.meanL_esmif.iloc[-1])]),
         f"+{gp:.2f}\n$P{{>}}0{{=}}1.0$", fontsize=5.9, color=S.INK, ha="right", va="center", linespacing=1.3)
axd.set_xticks(xp); axd.set_xticklabels([f"{a:g}" for a in CFG_A], fontsize=7.0)
axd.set_ylim(-0.63, 0.37); axd.set_yticks([-0.5, -0.25, 0, 0.25]); axd.tick_params(labelsize=7.0)
axd.set_xlim(-0.3, len(CFG_A) - 0.55)
axd.set_xlabel("steering strength  α", fontsize=8); axd.xaxis.set_label_coords(0.5, -0.15)
axd.set_ylabel("ESM-IF1 leverage", fontsize=7.7)
S.strip(axd); S.assert_in_view(axd, list(CL.esmif_hi) + list(CR.esmif_lo), axis="y")
S.header(axd, "steer with it, an independent model agrees",
         f"frozen ProteinMPNN, scored by ESM-IF1 · n={int(CL.n_cx.iloc[0])}")

# ================================================================== 1e — at no cost to native recovery
axe = fig.add_subplot(gD[1])
base = float(CL.int_recovery.iloc[0])
axe.axhline(base, color=S.RULE, lw=0.7, ls=(0, (3, 3)), zorder=1)
axe.plot(xp, CR.int_recovery, ls=(0, (2, 2)), color=S.SCALAR, lw=1.1, zorder=2)
axe.plot(xp, CR.int_recovery, "s", mfc="white", mec=S.SCALAR, mew=1.0, ms=4.0, zorder=3)
axe.plot(xp, CL.int_recovery, "-", color=S.LEV, lw=1.6, zorder=4)
axe.plot(xp, CL.int_recovery, "o", color=S.LEV, ms=4.6, zorder=5)
axe.text(xp[2], float(CL.int_recovery.iloc[2]) + 0.004, "steer by $L$", color=S.LEV,
         fontsize=6.5, fontweight="bold", ha="center", va="bottom")
axe.text(xp[-1], float(CR.int_recovery.iloc[-1]) - 0.004, "random dir.", color=S.MUTED,
         fontsize=6.5, ha="right", va="top")
axe.text(0.02, base + 0.0015, "α=0 baseline", fontsize=5.8, color=S.MUTED, va="bottom", ha="left")
axe.set_xticks(xp); axe.set_xticklabels([f"{a:g}" for a in CFG_A], fontsize=7.0)
axe.set_ylim(0.205, 0.315); axe.set_yticks([0.22, 0.26, 0.30]); axe.tick_params(labelsize=7.0)
axe.set_xlim(-0.3, len(CFG_A) - 0.4)
axe.set_xlabel("steering strength  α", fontsize=8); axe.xaxis.set_label_coords(0.5, -0.15)
axe.set_ylabel("native recovery", fontsize=7.7)
S.strip(axe); S.header(axe, "at no cost to native recovery", "the L direction is native-consistent too")

S.flabel(fig, 0.048, 0.992, "a")
S.flabel(fig, 0.092, 0.642, "b")
S.flabel(fig, 0.590, 0.642, "c")
S.flabel(fig, 0.092, 0.330, "d")
S.flabel(fig, 0.545, 0.330, "e")
S.save(fig, "fig1_decomposition")
print(f"  1a P({WT})={P[IWT]:.3f} Q({WT})={Q[IWT]:.3f}  Spearman(L,ddG)={SP_LDDG:+.3f} n={SP_N}")
print(f"  1b IQR ratio {IQR_RATIO:.3f}  rho(conf,|L|)={RHO_CL:+.4f}  exemplars |L| {EH.L_rms:.2f} vs {EC.L_rms:.3f}")
print(f"  1c R2 {[(m, round(float(FLEX[m].r2),4), round(float(FLEX[m].irreducible_frac),4)) for m in FLEX]}")

#!/usr/bin/env python3
"""Figure 0 — experimental overview (schematic). One glance: WHERE binding lives in an inverse-folding model, the
two things we do with it, and the three independent modalities that confirm it.

  (1) The construct: one inverse-folding likelihood, two readings — confidence (a bound-state scalar, blind to
      binding) vs leverage L (the mixed 2nd derivative = ablate the partner = the CFG guidance direction ∝ −ΔΔG).
  (2) Two uses of L: DETECT (rank experimental ΔΔG) and STEER (+α·L on a frozen ProteinMPNN).
  (3) Three independent readouts agree anti-circularly: inverse-folding judges, structure predictors, physics —
      vs random (null) and naive (folds, doesn't bind) controls, on crystal AND predicted backbones.

Schematic only — asserts no data values (those are Figs. 1–P/L and the tables); it carries the logic, no numbers.
  python3 src/fig_overview.py  ->  results/figures/fig_overview.{pdf,png}
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import figstyle as S
S.apply()

fig = plt.figure(figsize=(5.5, 2.98))
ax = fig.add_axes([0.012, 0.01, 0.976, 0.98]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.1, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=2))


def T(x, y, s, color, size=6.8, ha="left", weight="bold"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="top", fontweight=weight, zorder=5)


def B(x, y, s, color=S.INK, size=5.7, ha="left", w="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="top", linespacing=1.34, zorder=5, fontweight=w)


def arrow(x0, y0, x1, y1, color=S.RULE, lw=1.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                                 zorder=1, shrinkA=0, shrinkB=0))


# stage headers
for x, n, s in [(0.012, "1", "The construct"), (0.365, "2", "Two uses of  L"), (0.645, "3", "Three readouts agree")]:
    ax.text(x, 0.965, n, fontsize=7.8, color="white", ha="center", va="center", zorder=6, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.26", fc=S.INK, ec="none"))
    ax.text(x + 0.026, 0.965, s, fontsize=7.4, color=S.INK, ha="left", va="center", fontweight="bold", zorder=5)

# ---------------- Stage 1 — the construct ----------------
ax.text(0.10, 0.90, "one likelihood  $P(\\mathrm{seq}\\,|\\,\\mathrm{structure})$", fontsize=5.8, color=S.INK,
        ha="left", va="center", zorder=5)
ax.add_patch(Ellipse((0.072, 0.83), 0.082, 0.05, fc=S.rgba(S.LEV, 0.16), ec=S.LEV, lw=1.1, zorder=3))
ax.add_patch(Ellipse((0.128, 0.822), 0.055, 0.043, fc=S.rgba(S.ESMIF, 0.18), ec=S.ESMIF, lw=1.1, zorder=3))
ax.text(0.070, 0.83, "receptor", fontsize=4.7, color=S.LEV, ha="center", va="center", zorder=5)
ax.text(0.130, 0.822, "binder", fontsize=4.7, color=S.ESMIF, ha="center", va="center", zorder=5)
ax.text(0.196, 0.826, "complex", fontsize=5.4, color=S.MUTED, ha="left", va="center", zorder=5)

box(0.012, 0.545, 0.315, 0.145, S.FLOOR_FILL, S.SCALAR)
T(0.026, 0.678, "Confidence  φ(P)", S.MUTED)
B(0.026, 0.642, "any scalar of the bound-state\ndistribution — blind to binding\nby construction  ($L \\perp φ$, proven)", S.MUTED)

box(0.012, 0.315, 0.315, 0.205, S.rgba(S.LEV, 0.10), S.LEV, lw=1.3)
T(0.026, 0.505, "Leverage  L", S.LEV)
B(0.026, 0.470, "$=$ mixed 2nd derivative\n(ablate the partner) — the model's\nCFG guidance direction", S.INK)
B(0.026, 0.362, "∝  −ΔΔG$_{bind}$   (binding free energy)", S.LEV, 6.0, w="bold")

# ---------------- Stage 2 — two uses ----------------
box(0.352, 0.545, 0.255, 0.145, S.TINT, S.RULE)
T(0.365, 0.678, "Detect", S.INK)
B(0.365, 0.642, "rank experimental ΔΔG\n(SKEMPI n=2,948; AB-Bind);\nzero-shot, competitive with\nand beyond FoldX physics", S.INK)

box(0.352, 0.315, 0.255, 0.205, S.TINT, S.RULE)
T(0.365, 0.503, "Steer", S.INK)
B(0.365, 0.467, "$+\\,α\\,L$ on a frozen,\noff-the-shelf ProteinMPNN\n→ new interface sequences", S.INK)
B(0.365, 0.348, "controls:  random · naive", S.MUTED, 5.3)

arrow(0.330, 0.470, 0.350, 0.617)      # L -> detect
arrow(0.330, 0.455, 0.350, 0.455)      # L -> steer

# ---------------- Stage 3 — readouts ----------------
rd = [(0.700, S.ESMIF, "Inverse-folding judges", "ESM-IF1 · MIF · PiFold — leverage ↑"),
      (0.560, S.GEOM, "Structure predictors", "AF2-multimer · Boltz-2 — ipTM ↑"),
      (0.420, S.LEV, "Physics energy function", "FoldX ΔΔG$_{bind}$ — more favorable ↑")]
for y, col, t, b in rd:
    box(0.645, y, 0.343, 0.118, S.rgba(col, 0.10), col, lw=1.2)
    T(0.658, y + 0.104, t, col, 6.3)
    B(0.658, y + 0.066, b, S.INK, 5.5)
box(0.645, 0.315, 0.343, 0.083, S.FLOOR_FILL, S.SCALAR)
B(0.658, 0.388, "steered L beats  random  and  naive\n(folds but does not bind) on all three", S.INK, 5.4)
arrow(0.608, 0.455, 0.643, 0.474)      # steer -> readouts

# ---------------- thesis strip ----------------
box(0.012, 0.035, 0.976, 0.175, S.rgba(S.LEV, 0.07), S.LEV, lw=1.0, r=0.025)
ax.text(0.5, 0.163, "Confidence is not competence — and it recurs at every level.", fontsize=7.0, color=S.LEV,
        ha="center", va="center", fontweight="bold", zorder=5)
ax.text(0.5, 0.107, "an inverse-folding model's confidence and a structure predictor's ipTM are both blind to "
        "binding;", fontsize=5.4, color=S.INK, ha="center", va="center", zorder=5)
ax.text(0.5, 0.070, "the competence always lives in the response to ablating the partner.", fontsize=5.4,
        color=S.INK, ha="center", va="center", zorder=5)

S.save(fig, "fig_overview")

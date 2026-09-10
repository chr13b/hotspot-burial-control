#!/usr/bin/env python3
"""Figure L — the beyond-X ladder in ONE matched metric. L carries binding rank-signal beyond every competing
feature class: geometry, conservation, the one-pass log-odds (the full published feature set), and a fitted physics
energy function. Same metric on every rung (partial rank-correlation of L with experimental ΔΔG controlling for that
class), so bar lengths ARE comparable — complementary to the CPI placebo-floor analysis (§4). Every scalar of the
bound distribution, by contrast, sits at the floor (§4) — the dashed reference at 0.

Every number read from results/beyond_x_ladder.csv (src/leverage_ladder.py). Partial is negative (L = favorability,
ΔΔG = cost); we plot |partial| = binding signal retained beyond the control, whiskers = complex-clustered 95% CI.

  python3 src/fig_ladder.py  ->  results/figures/fig_ladder.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
MINUS = "−"

d = pd.read_csv("results/beyond_x_ladder.csv").set_index("rung")
ORDER = ["geometry", "conservation", "one-pass log-odds", "fitted physics (FoldX)"]
LAB = {"geometry": "geometry\n(burial · ΔSASA · contacts)", "conservation": "conservation\n(BLOSUM · Δvol · Δhydro)",
       "one-pass log-odds": "one-pass log-odds\n(full published feature set)",
       "fitted physics (FoldX)": "fitted physics\n(FoldX ΔΔG$_{bind}$) — this work"}
COL = {"geometry": S.GEOM, "conservation": S.CONS, "one-pass log-odds": S.SCALAR, "fitted physics (FoldX)": S.LEV}

fig = plt.figure(figsize=(5.5, 2.35))
ax = fig.add_axes([0.30, 0.20, 0.66, 0.60])
ys = list(range(len(ORDER)))[::-1]                       # first rung at top
ax.axvline(0, ls=(0, (3, 3)), color=S.RULE, lw=0.9, zorder=2)
for rung, y in zip(ORDER, ys):
    r = d.loc[rung]
    v, lo, hi = abs(float(r.partial)), abs(float(r.hi)), abs(float(r.lo))   # |partial|; hi/lo flip under abs (all<0)
    col = COL[rung]
    ax.barh(y, v, height=0.5, color=col, zorder=3)
    ax.plot([lo, hi], [y, y], "-", color=S.INK, lw=1.3, solid_capstyle="butt", zorder=5)
    for e in (lo, hi):
        ax.plot([e, e], [y - 0.1, y + 0.1], "-", color=S.INK, lw=1.3, zorder=5)
    ax.text(hi + 0.008, y, f"{v:.2f}", fontsize=6.6, color=S.INK, ha="left", va="center",
            fontweight="bold", zorder=6)
    ax.text(-0.012, y, LAB[rung], fontsize=6.2, color=S.INK, ha="right", va="center",
            linespacing=1.25, zorder=6)
ax.set_xlim(0, 0.42)
ax.set_ylim(-0.6, len(ORDER) - 0.30)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(S.RULE)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax.tick_params(colors=S.RULE, labelcolor=S.INK, length=2.5, labelsize=6.0)
ax.set_xlabel("binding signal L retains beyond the control  =  |partial rank-corr(L, ΔΔG | control)|  (95% CI)",
              fontsize=5.9, color=S.MUTED)
S.header(ax, "L clears every control anyone has proposed",
         note="one matched metric · SKEMPI interface mutants · confidence stays at 0 at every rung", tsize=8.0)
S.save(fig, "fig_ladder")
print("rungs:", {k: round(abs(float(d.loc[k].partial)), 3) for k in ORDER})

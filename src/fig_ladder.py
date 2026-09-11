#!/usr/bin/env python3
"""Figure L — the beyond-X ladder in ONE matched metric. L carries binding rank-signal beyond every competing
feature class: geometry, conservation, the one-pass log-odds (the full published feature set), and a fitted physics
energy function. Same metric on every rung (partial rank-correlation of L with experimental ΔΔG controlling for that
class), so bar lengths ARE comparable — complementary to the CPI placebo-floor analysis (§4). Every scalar of the
bound distribution, by contrast, sits at the floor (§4) — the dashed reference at 0.

Every number read from results/beyond_x_ladder.csv (src/leverage_ladder.py). Partial is negative (L = favorability,
ΔΔG = cost); we plot |partial| = binding signal retained beyond the control, whiskers = complex-clustered 95% CI.

Layout note: the rung labels live INSIDE the axes at negative x (the spine is bounded to the data range), so the
finding-title, the bars and the axis caption all hang off one left edge and the panel fills the 5.5in text width.

  python3 src/fig_ladder.py  ->  results/figures/fig_ladder.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
NDASH = "–"                                                    # ranges take an en dash, never the minus sign

d = pd.read_csv("results/beyond_x_ladder.csv").set_index("rung")
ORDER = ["geometry", "conservation", "one-pass log-odds", "fitted physics (FoldX)"]
NAME = {"geometry": "geometry", "conservation": "conservation", "one-pass log-odds": "one-pass log-odds",
        "fitted physics (FoldX)": "fitted physics"}
SUB = {"geometry": "burial · ΔSASA · contacts", "conservation": "BLOSUM · Δvol · Δhydro",
       "one-pass log-odds": "full published feature set",
       "fitted physics (FoldX)": "FoldX ΔΔG$_{bind}$ — this work"}
COL = {"geometry": S.GEOM, "conservation": S.CONS, "one-pass log-odds": S.SCALAR,
       "fitted physics (FoldX)": S.LEV}

# ---- axis geometry, all derived from the committed numbers (nothing about the scale is hardcoded) ----
val = {k: abs(float(d.loc[k].partial)) for k in ORDER}
whi = {k: (abs(float(d.loc[k].hi)), abs(float(d.loc[k].lo))) for k in ORDER}   # abs flips lo/hi (all < 0)
XTOP = 0.1 * np.ceil((max(w[1] for w in whi.values()) + 0.02) / 0.1)           # scale ends on a round tick
XNUM = XTOP + 0.022                                            # effect sizes in a flush column past the scale
XLAB = 0.150                                                   # room at negative x for the rung labels

fig = plt.figure(figsize=(S.FIG_W, 2.56))
ax = fig.add_axes([0.052, 0.215, 0.936, 0.590])
ys = list(range(len(ORDER)))[::-1]                             # first rung at top

for g in np.arange(0.1, XTOP + 1e-9, 0.1):                     # whisper-light scale guides, behind the bars
    ax.axvline(g, color=S.GHOST, lw=0.6, zorder=1)
ax.axvline(0, ls=(0, (3, 3)), color=S.RULE, lw=0.9, zorder=2)

for rung, y in zip(ORDER, ys):
    v, (lo, hi), col = val[rung], whi[rung], COL[rung]
    ax.barh(y, v, height=0.50, color=col, zorder=3)
    ax.plot([lo, hi], [y, y], "-", color=S.INK, lw=1.05, solid_capstyle="butt", zorder=5)   # 95% CI
    for e in (lo, hi):
        ax.plot([e, e], [y - 0.085, y + 0.085], "-", color=S.INK, lw=1.05, zorder=5)
    ax.text(XNUM, y, f"{v:.2f}", fontsize=6.8, color=S.INK, ha="left", va="center",
            fontweight="bold", zorder=6)
    ax.text(-0.011, y + 0.080, NAME[rung], fontsize=6.6, color=S.INK, ha="right", va="bottom", zorder=6)
    ax.text(-0.011, y - 0.080, SUB[rung], fontsize=5.5, ha="right", va="top", zorder=6,
            color=S.LEV if rung == "fitted physics (FoldX)" else S.MUTED)

ax.set_xlim(-XLAB, XNUM + 0.030)
ax.set_ylim(-0.60, len(ORDER) - 0.40)
S.assert_in_view(ax, [XNUM] + [w[1] for w in whi.values()])
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(S.RULE)
ax.spines["bottom"].set_bounds(0, XTOP)                        # the rule spans the data range, not the labels
ax.set_xticks(np.round(np.arange(0, XTOP + 1e-9, 0.1), 2))
ax.tick_params(colors=S.RULE, labelcolor=S.INK, length=2.5, labelsize=6.0, pad=2)

# axis caption: name of the quantity, then how it is measured — centred on the data range, not on the labels
ax.annotate("binding signal L retains beyond the control", xy=(XTOP / 2, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -14.5), textcoords="offset points", fontsize=6.2, color=S.INK, ha="center", va="top")
nb = int(d.nboot.max())
ax.annotate(f"|partial rank-corr(L, ΔΔG | control)|   ·   95% CI, {nb:,} complex-clustered bootstrap replicates",
            xy=(XTOP / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -22.5),
            textcoords="offset points", fontsize=5.4, color=S.MUTED, ha="center", va="top")


def rng(lo, hi, fmt="{:,}"):
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}{NDASH}{fmt.format(hi)}"


n = rng(int(d.n.min()), int(d.n.max()))
nc = rng(int(d.n_complex.min()), int(d.n_complex.max()))
S.header(ax, "L clears every control anyone has proposed",
         note=f"one matched metric · SKEMPI interface mutants, n = {n} ({nc} complexes)"
              f" · confidence stays at 0 at every rung", tsize=8.0)
S.save(fig, "fig_ladder")
print("rungs:", {k: round(val[k], 3) for k in ORDER})

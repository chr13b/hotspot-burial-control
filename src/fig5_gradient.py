#!/usr/bin/env python3
"""Figure 5 — constraint vs leverage across interface types (half-width, ~2.85in).
Takeaway: confidence-AUROC climbs the pre-registered transience order (a fold-constraint signal that tracks
how binding-dominated the interface is), while leverage-AUROC stays flat and high and burial runs the other
way — the two feature classes DIVERGE across a controlled axis, and the confidence gradient is not a burial
artifact. Every number read from a committed CSV. Renders PDF (vector) + PNG.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
R = "results"

# ---- data (CSV-only) ----
tp = pd.read_csv(f"{R}/threepoint_law.csv")
nat = tp[tp.transience_rank.isin([1, 2, 3])].sort_values("transience_rank")     # the 3 natural classes
dn = pd.read_csv(f"{R}/bennett_conf_fork.csv")
dn_conf = dn[dn.feature == "logp_native"].iloc[0]                               # de-novo confidence anchor

labels = ["TCR/\npMHC", "AB/AG", "Pr/PI", "de-novo"]
x = np.array([1, 2, 3, 4])
conf = list(nat.conf_auroc) + [float(dn_conf.auroc)]                           # 3 natural + de-novo
conf_lo = list(nat.lo) + [float(dn_conf.lo)]
conf_hi = list(nat.hi) + [float(dn_conf.hi)]
lev = list(nat.leverage_auroc_Lrms)                                            # natural only (de-novo lev not computed)
lev_lo = list(nat.lev_lo); lev_hi = list(nat.lev_hi)
bur = list(nat.burial_auroc)                                                   # the opposite-running control

fig = plt.figure(figsize=(2.85, 2.6))
ax = fig.add_axes([0.185, 0.205, 0.75, 0.63])

ax.axhline(0.5, color=S.RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)                # chance
ax.text(4.42, 0.492, "chance", fontsize=6.0, color=S.MUTED, va="top", ha="right")

# burial — baseline control (non-monotone: neither tracks transience nor explains the confidence climb)
ax.plot(x[:3], bur, ls=(0, (2, 2)), color=S.GEOM, lw=1.0, zorder=2)
ax.plot(x[:3], bur, "^", mfc="white", mec=S.GEOM, mew=1.1, ms=5, zorder=3)

# confidence — a scalar of P (grey): a rising TREND, so draw the connecting line
ax.plot(x, conf, "-", color=S.SCALAR, lw=1.3, zorder=4)
for xi, lo, hi in zip(x, conf_lo, conf_hi):
    ax.plot([xi, xi], [lo, hi], color=S.SCALAR, lw=0.9, zorder=3)
ax.plot(x, conf, "o", color=S.SCALAR, ms=5, zorder=5)

# leverage — the mixed derivative (blue): flat + high, so NO connecting line (trend-free)
for xi, lo, hi in zip(x[:3], lev_lo, lev_hi):
    ax.plot([xi, xi], [lo, hi], color=S.LEV, lw=0.9, zorder=3)
ax.plot(x[:3], lev, "s", color=S.LEV, ms=5.5, zorder=6)

# direct labels (no boxed legend)
ax.text(1.55, 0.585, "leverage", color=S.LEV, fontsize=7, fontweight="bold", ha="center")
ax.text(4.0, 0.648, "confidence", color=S.MUTED, fontsize=7, ha="center")
ax.text(2.0, 0.785, "burial", color=S.GEOM, fontsize=6.5, ha="center")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.4)
ax.set_ylim(0.34, 0.84); ax.set_xlim(0.55, 4.55)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8]); ax.tick_params(labelsize=7)
ax.set_ylabel("hotspot AUROC", fontsize=8)
ax.set_xlabel("interface class  (pre-registered order →)", fontsize=7.3)
S.strip(ax)
S.assert_in_view(ax, conf_lo + conf_hi + lev_lo + lev_hi, axis="y")
S.header(ax, "confidence climbs the gradient; leverage stays high",
         "hotspot AUROC · de-novo leverage-AUROC not yet computed")
S.save(fig, "fig5_gradient")
print("conf:", [round(c, 3) for c in conf], " lev:", [round(l, 3) for l in lev], " burial:", [round(b, 3) for b in bur])

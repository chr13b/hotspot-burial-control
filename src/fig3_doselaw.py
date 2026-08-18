#!/usr/bin/env python3
"""Figure 3 — the backbone-error dose law (W2). Leverage's binding signal survives accurate backbones
(~0.5 A) and collapses by ~1 A. From results/leverage_noise_ladder.csv (n=2949 mutations, 285 complexes).

  python3 src/fig3_doselaw.py   ->   results/figures/fig3_doselaw.png (+ .pdf)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.linewidth": 0.8})
INK, ACC, ACC2, MUT, GOOD = "#22303a", "#2f6f8f", "#b8562f", "#8a8f94", "#3f7d5a"

d = pd.read_csv("results/leverage_noise_ladder.csv").sort_values("sigma_A")
s, cpi, lo, hi, sp = d.sigma_A.values, d.cpi_L_geom.values, d.lo.values, d.hi.values, d.spearman_L_ddG.values

fig, ax = plt.subplots(figsize=(6.6, 4.2))
# shaded regimes
ax.axvspan(-0.05, 0.55, color=GOOD, alpha=0.07, zorder=0)
ax.axvspan(0.9, 1.6, color=ACC2, alpha=0.06, zorder=0)
ax.text(0.25, 0.0735, "diagnostic\nzone", color=GOOD, fontsize=8.5, ha="center", weight="bold")
ax.text(1.25, 0.0735, "collapsed", color=ACC2, fontsize=8.5, ha="center", weight="bold")

ax.axhline(0, color=INK, lw=0.8, zorder=1)
ax.fill_between(s, lo, hi, color=ACC, alpha=0.15, zorder=2)
ax.plot(s, cpi, "-o", color=ACC, lw=2, ms=6, zorder=4, label="CPI(L | geometry)")
ax.set_xlabel("backbone perturbation  σ  ≈  interface RMSD (Å)", fontsize=10)
ax.set_ylabel("CPI(leverage | geometry)", color=ACC, fontsize=10)
ax.tick_params(axis="y", colors=ACC)
ax.set_xlim(-0.05, 1.6); ax.set_ylim(-0.01, 0.082)
ax.annotate("crystal\n(positive control)", xy=(0.0, cpi[0]), xytext=(0.15, 0.028),
            fontsize=7.6, color=MUT, arrowprops=dict(arrowstyle="-", color=MUT, lw=0.7))

ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
ax2.plot(s, -sp, "--s", color=ACC2, lw=1.6, ms=5, zorder=3, label="−Spearman(L, ΔΔG)")
ax2.set_ylabel("−Spearman(L, ΔΔG)", color=ACC2, fontsize=10); ax2.tick_params(axis="y", colors=ACC2)
ax2.set_ylim(-0.01, 0.42)

h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc="upper right")
ax.set_title("The binding signal is a dose law of backbone error", fontsize=11.5, loc="left", weight="bold")
ax.text(0.0, -0.006, "survives ≤0.5 Å  ·  collapses by ~1 Å  ·  a LOWER bound (real reconstructions collapse harder)",
        fontsize=7.6, color=MUT)

fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
for e in ("png", "pdf"): fig.savefig(f"results/figures/fig3_doselaw.{e}", bbox_inches="tight")
print("wrote fig3_doselaw; CPI:", dict(zip(s, cpi.round(4))), "| -Spearman:", dict(zip(s, (-sp).round(3))))

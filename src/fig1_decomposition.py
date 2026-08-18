#!/usr/bin/env python3
"""Figure 1 — the Confidence-Leverage Decomposition: on natural complexes (SKEMPI) the model DOES know
binding, but only in the MIXED derivative (leverage L), not in confidence. From leverage_decomposition.csv.

  python3 src/fig1_decomposition.py   ->   results/figures/fig1_decomposition.png (+ .pdf)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8})
INK, ACC, ACC2, MUT = "#22303a", "#2f6f8f", "#b8562f", "#8a8f94"

d = pd.read_csv("results/leverage_decomposition.csv")
def stat(fixture, test):
    r = d[(d.fixture == fixture) & (d.test == test)]
    return (float(r.stat.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0])) if len(r) else (np.nan,)*3

NAT = "SKEMPI (natural)"
# (a) mutation-level CPI beyond geometry
feats = [("leverage L\n(mixed derivative)", "CPI(LEVERAGE L (full) | burial+nbr+dSASA)", ACC),
         ("logP(mut|complex)", "CPI(logP(mut|complex) | burial+nbr+dSASA)", INK),
         ("confidence", "CPI(confidence | burial+nbr+dSASA)", MUT),
         ("scalar KL", "CPI(scalar KL | burial+nbr+dSASA)", MUT)]
cpivals = [(lab, stat(NAT, t), col) for lab, t, col in feats]
# (b) within-stratum hotspot AUROC
au = [("leverage −L", "within_stratum_AUROC(-LEVERAGE L (full) | burial+nbr+dSASA)", ACC),
      ("−logP(mut)", "within_stratum_AUROC(-logP(mut|complex) | burial+nbr+dSASA)", INK),
      ("−scalar KL", "within_stratum_AUROC(-scalar KL | burial+nbr+dSASA)", MUT),
      ("confidence", "within_stratum_AUROC(-confidence | burial+nbr+dSASA)", MUT)]
auvals = [(lab, stat(NAT, t), col) for lab, t, col in au]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))

# ---- (a)
for i, (lab, (v, lo, hi), col) in enumerate(cpivals):
    ax[0].barh(i, v, color=col, height=0.62, zorder=3)
    ax[0].plot([lo, hi], [i, i], color=INK, lw=1.4, zorder=4)
ax[0].axvline(0, color=INK, lw=0.9)
ax[0].set_yticks(range(len(cpivals))); ax[0].set_yticklabels([c[0] for c in cpivals], fontsize=9)
ax[0].invert_yaxis(); ax[0].set_xlabel("CPI beyond geometry (mutation level)", fontsize=9.5)
ax[0].set_title("a  the mixed derivative adds binding info; confidence barely does",
                fontsize=10, loc="left", weight="bold")
ax[0].text(0.98, 0.04, "Spearman(L, ΔΔG) = −0.30", transform=ax[0].transAxes, ha="right",
           fontsize=8.2, color=ACC, weight="bold")

# ---- (b)
geom = stat(NAT, "marginal_AUROC(geometry burial+nbr+dSASA)")[0]
for i, (lab, (v, lo, hi), col) in enumerate(auvals):
    ax[1].barh(i, v, color=col, height=0.62, zorder=3)
    if np.isfinite(lo): ax[1].plot([lo, hi], [i, i], color=INK, lw=1.4, zorder=4)
ax[1].axvline(0.5, color=MUT, lw=1.1, ls=":"); ax[1].text(0.5, -0.7, "chance", color=MUT, fontsize=7.6, ha="center")
ax[1].set_yticks(range(len(auvals))); ax[1].set_yticklabels([c[0] for c in auvals], fontsize=9)
ax[1].invert_yaxis(); ax[1].set_xlim(0.38, 0.72)
ax[1].set_xlabel("within-burial-stratum hotspot AUROC", fontsize=9.5)
ax[1].set_title("b  confidence is BELOW chance; leverage predicts", fontsize=10, loc="left", weight="bold")

fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
for e in ("png", "pdf"): fig.savefig(f"results/figures/fig1_decomposition.{e}", bbox_inches="tight")
print("wrote fig1;", [(c[0].split(chr(10))[0], round(c[1][0], 4)) for c in cpivals])
print("  AUROC:", [(c[0], round(c[1][0], 3)) for c in auvals], "geom", round(geom, 3))

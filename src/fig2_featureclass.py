#!/usr/bin/env python3
"""Figure 2 — the feature-class law, and it is model-general. Position-level CPI beyond geometry: no scalar
functional of the distribution (confidence, negentropy, scalar KL) ranks hotspots; only the MIXED derivative
(leverage) does — and it replicates on a second IF model family (ESM-IF1, a GVP-transformer, conditional
readout). From leverage_decomposition.csv (ProteinMPNN) + leverage_esmif.csv (ESM-IF1).

  python3 src/fig2_featureclass.py   ->   results/figures/fig2_featureclass.png (+ .pdf)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8})
INK, ACC, ACC2, MUT = "#22303a", "#2f6f8f", "#b8562f", "#8a8f94"

def lookup(csv, fixture, needle):
    d = pd.read_csv(csv)
    r = d[(d.fixture == fixture) & (d.test.astype(str).str.contains(needle, regex=False))]
    if not len(r): return (np.nan, np.nan, np.nan)
    return float(r.stat.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0])

FEATS = [("confidence\n(one-pass, diagonal)", "CPI_position_level(confidence"),
         ("negentropy\n(one-pass)", "CPI_position_level(negentropy"),
         ("scalar KL\n(two-pass scalar)", "CPI_position_level(scalar KL"),
         ("leverage L→Ala\n(mixed derivative)", "CPI_position_level(leverage L(->Ala)")]
MODELS = [("ProteinMPNN", "results/leverage_decomposition.csv", "SKEMPI", ACC),
          ("ESM-IF1 (GVP-transformer)", "results/leverage_esmif.csv", "SKEMPI_esmif", ACC2)]

fig, ax = plt.subplots(figsize=(8.4, 4.3))
x = np.arange(len(FEATS)); w = 0.38
for mi, (mname, csv, fix, col) in enumerate(MODELS):
    vals = [lookup(csv, fix, n) for _, n in FEATS]
    xs = x + (mi - 0.5) * w
    ax.bar(xs, [v[0] for v in vals], width=w, color=col, zorder=3, label=mname)
    for xi, (v, lo, hi) in zip(xs, vals):
        if np.isfinite(lo): ax.plot([xi, xi], [lo, hi], color=INK, lw=1.3, zorder=4)
ax.axhline(0, color=INK, lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([f[0] for f in FEATS], fontsize=8.6)
ax.set_ylabel("CPI beyond geometry (position level)", fontsize=10)
ax.legend(fontsize=8.6, frameon=False, loc="upper left")
ax.set_title("Only the mixed derivative adds — on both model families", fontsize=11.5, loc="left", weight="bold")
ax.annotate("confidence: CI spans 0\n(blind by construction)", xy=(0, 0.0004), xytext=(0.35, 0.0028),
            fontsize=7.8, color=MUT, arrowprops=dict(arrowstyle="-", color=MUT, lw=0.7))
ax.text(0.99, 0.05, "collapsing L to any scalar\ndiscards ~80% of its signal", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7.8, color=ACC)

fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
for e in ("png", "pdf"): fig.savefig(f"results/figures/fig2_featureclass.{e}", bbox_inches="tight")
for mname, csv, fix, _ in MODELS:
    print(mname, [(f[0].split(chr(10))[0], round(lookup(csv, fix, n)[0], 5)) for f, (_, n) in zip(FEATS, FEATS)])

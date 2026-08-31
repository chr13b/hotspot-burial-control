#!/usr/bin/env python3
"""Figure 3 — the backbone-error dose law, with R2 (real predicted backbones) folded in.
Numbers verified against leverage_noise_ladder{,_075full}.csv, leverage_noise_ladder_esmif_{all285,tail,redraw}.csv,
leverage_predicted.csv (traces in comments). Renders PDF + PNG.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as S
S.apply()

# --- 3a data: CPI(L | geometry), mutation level ---  (→ leverage_noise_ladder*.csv, esmif_all285+tail)
sig_m = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
cpi_m = [0.0575, 0.0588, 0.0474, 0.0321, 0.0024, -0.0012]      # ProteinMPNN, n=2,949
sig_e = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
cpi_e = [0.0362, 0.0350, 0.0266, 0.0115, 0.0177, 0.0020, 0.0011]  # ESM-IF1, n=2,809
spr_m = [0.301, 0.294, 0.293, 0.195, 0.077, 0.060]            # −Spearman(L,ΔΔG)
spr_e = [0.252, 0.170, 0.172, 0.118, 0.169, 0.096, 0.103]
R2 = {"OpenFold3": 0.0389, "AF2-multimer": 0.0320}           # → leverage_predicted.csv (real predicted backbones)
FLOOR = 0.0007

fig = plt.figure(figsize=(5.5, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 0.85], wspace=0.5,
                      left=0.09, right=0.985, top=0.84, bottom=0.20)

# ================= 3a — the two ladders + real predicted backbones =================
axa = fig.add_subplot(gs[0])
axa.axhspan(-0.004, FLOOR, color=S.FLOOR_FILL, zorder=0)          # placebo-floor band
axa.axhline(0, color=S.RULE, lw=0.8, zorder=1)
axa.plot(sig_m, cpi_m, "-o", color=S.LEV, ms=4.5, lw=1.6, zorder=4, label="ProteinMPNN")
axa.plot(sig_e, cpi_e, "--s", color=S.ESMIF, ms=4, lw=1.4, mfc="white", mew=1.2, zorder=4, label="ESM-IF1")
# R2 real predicted backbones (green), placed at their typical interface RMSD band
for i, (nm, v) in enumerate(R2.items()):
    axa.plot(0.9, v, "D", color=S.GEOM, ms=6, zorder=5, mec="white", mew=0.7)
axa.annotate("real predicted\nbackbones (OF3/AF2)\nland here — survive",
             xy=(0.9, 0.036), xytext=(1.15, 0.052), fontsize=6.3, color=S.GEOM, va="center",
             arrowprops=dict(arrowstyle="->", color=S.GEOM, lw=0.8))
axa.text(0.02, 0.0605, "survives ≤ 0.5 Å", fontsize=6.5, color=S.MUTED)
axa.set_xlim(-0.08, 2.08); axa.set_ylim(-0.004, 0.066)
axa.set_xlabel("backbone jitter σ ≈ interface RMSD (Å)", fontsize=8)
axa.set_ylabel("CPI(L | geometry)", fontsize=8); axa.tick_params(labelsize=7.5)
axa.legend(fontsize=6.8, frameon=False, loc="upper right", handlelength=1.6, borderpad=0.2)
S.strip(axa); S.title(axa, "backbone-error dose law")
S.subn(axa, "mutation level · MPNN 2,949 / ESM-IF1 2,809")
S.letter(axa, "a")

# ================= 3b — the model-free readout =================
axb = fig.add_subplot(gs[1])
axb.plot(sig_m, spr_m, "-o", color=S.LEV, ms=4, lw=1.5)
axb.plot(sig_e, spr_e, "--s", color=S.ESMIF, ms=3.5, lw=1.3, mfc="white", mew=1.1)
axb.set_xlim(-0.08, 2.08); axb.set_ylim(0, 0.33)
axb.set_xlabel("σ (Å)", fontsize=8); axb.set_ylabel("−Spearman(L, ΔΔG)", fontsize=8)
axb.tick_params(labelsize=7.5)
axb.annotate("ESM-IF1's raw\ncorrelation is more\njitter-robust than\nits CPI (untested)",
             xy=(1.5, 0.096), xytext=(0.55, 0.24), fontsize=6.0, color=S.MUTED,
             arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=0.7))
S.strip(axb); S.title(axb, "model-free readout")
S.subn(axb, "the honesty panel")
S.letter(axb, "b")

# ================= 3c — realization variance at 1 Å =================
axc = fig.add_subplot(gs[2])
draws = [0.0114, 0.0019, -0.0002]                                # 3 ESM-IF1 jitter draws @σ≈1 Å (200-cx subsample)
axc.axhline(0, color=S.RULE, lw=0.8)
axc.axhspan(-0.004, FLOOR, color=S.FLOOR_FILL, zorder=0)
xs = np.full(len(draws), 0.5) + np.array([-0.12, 0, 0.12])
axc.plot(xs, draws, "s", color=S.ESMIF, ms=6, mfc="white", mew=1.3)
axc.annotate("", xy=(0.5, max(draws)), xytext=(0.5, min(draws)),
             arrowprops=dict(arrowstyle="<->", color=S.MUTED, lw=0.8))
axc.text(0.66, 0.005, "spread ≈ 0.012\n= the estimate\nitself → the 1 Å\nrung straddles\nthe floor",
         fontsize=6.0, color=S.MUTED, va="center")
axc.set_xlim(0, 1.2); axc.set_ylim(-0.004, 0.018); axc.set_xticks([])
axc.set_ylabel("CPI at σ ≈ 1 Å", fontsize=8); axc.tick_params(labelsize=7.5)
S.strip(axc); S.title(axc, "the tail is noise")
S.subn(axc, "3 jitter draws @ 1 Å")
S.letter(axc, "c")

os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/fig3_doselaw.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("results/figures/fig3_doselaw.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
print("wrote results/figures/fig3_doselaw.{pdf,png}")

#!/usr/bin/env python3
"""Figure 3 — the backbone-error dose law. Three panels, one design system (figstyle).

Every number is READ from a committed CSV (no hard-coded values):
  3a ladders  leverage_noise_ladder.csv + _075full.csv           (ProteinMPNN, n=2,949 mut)
              leverage_noise_ladder_esmif_all285.csv + _tail.csv (ESM-IF1,    n=2,809 mut)
     slot     leverage_predicted.csv  CPI_mut(L | burial+nbr+dSASA) for crystal / of3 / af2 (140 cx)
  3b          spearman_L_ddG column of the same ladder CSVs
  3c          leverage_noise_ladder_esmif.csv (sigma 1.0) + _redraw.csv (0.99, 1.01), 200-cx subsample

Design notes: the placebo floor in Fig 2 is a POSITION-level number and this axis is mutation level,
so no floor band is drawn here — significance is read from each rung's own 95% CI band (zero rule).
The predicted-backbone points are NOT placed on the sigma axis (real prediction error is not iid
jitter and no committed CSV gives them an iRMSD); they sit in a broken-off "real" slot.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
R = "results"

# ---------------------------------------------------------------- data (CSV only)
def ladder(*csvs):
    d = pd.concat([pd.read_csv(f"{R}/{c}") for c in csvs]).sort_values("sigma_A")
    return (d.sigma_A.values, d.cpi_L_geom.values, d.lo.values, d.hi.values,
            -d.spearman_L_ddG.values, int(d.n_mut.iloc[0]))

sm, cm, cm_lo, cm_hi, rm, n_m = ladder("leverage_noise_ladder.csv", "leverage_noise_ladder_075full.csv")
se, ce, ce_lo, ce_hi, re_, n_e = ladder("leverage_noise_ladder_esmif_all285.csv",
                                        "leverage_noise_ladder_esmif_tail.csv")

lp = pd.read_csv(f"{R}/leverage_predicted.csv")
lp = lp[lp.metric.astype(str) == "CPI_mut(L | burial+nbr+dSASA)"].set_index("source")
PRED = [(k, float(lp.stat[k]), float(lp.lo[k]), float(lp.hi[k])) for k in ("crystal", "of3", "af2")]
n_pred, ncx_pred = int(lp.n["crystal"]), int(lp.n_cx["crystal"])

rz = pd.concat([pd.read_csv(f"{R}/leverage_noise_ladder_esmif.csv").query("sigma_A == 1.0"),
                pd.read_csv(f"{R}/leverage_noise_ladder_esmif_redraw.csv").query("sigma_A in [0.99, 1.01]")])
n_rz = int(rz.n_mut.iloc[0])

# ---------------------------------------------------------------- canvas
fig = plt.figure(figsize=(5.5, 2.62))
L0, R0, WR, WS = 0.098, 0.985, [1.82, 0.95, 1.12], 0.46
gs = fig.add_gridspec(1, 3, width_ratios=WR, wspace=WS, left=L0, right=R0, top=0.775, bottom=0.185)
u = (R0 - L0) / (sum(WR) + 2 * WS * np.mean(WR))                      # fig-fraction per ratio unit
lefts = [L0, L0 + u * (WR[0] + WS * np.mean(WR)), L0 + u * (WR[0] + WR[1] + 2 * WS * np.mean(WR))]

# ================= 3a — the two ladders + real predicted backbones =================
axa = fig.add_subplot(gs[0])
XB, YLO, YHI = 2.145, -0.007, 0.0760                                  # break x, ylims
XSLOT = [2.27, 2.41, 2.55]                                            # crystal / OF3 / AF2 slot
axa.axhline(0, color=S.RULE, lw=0.8, zorder=1)
for x, y, lo, hi, col, ls, mk, kw in [
        (sm, cm, cm_lo, cm_hi, S.LEV, "-", "o", dict(ms=4.2, mew=0)),
        (se, ce, ce_lo, ce_hi, S.ESMIF, (0, (4, 2)), "s", dict(ms=3.8, mfc="white", mew=1.2))]:
    axa.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0, zorder=2)
    axa.plot(x, y, ls=ls, marker=mk, color=col, lw=1.5, mec=col, zorder=4, **kw)

axa.text(0.02, cm[0] + 0.0045, "ProteinMPNN", color=S.LEV, fontsize=6.8, ha="left", va="bottom")
axa.text(0.03, 0.0190, "ESM-IF1", color=S.ESMIF, fontsize=6.8, ha="left", va="center")
axa.text(0.05, -0.0043, "shaded = 95% CI", fontsize=6.2, color=S.MUTED, ha="left", va="center")

# --- broken-off slot: real predicted backbones (their own 140-complex sample)
for xx, (nm, v, lo, hi) in zip(XSLOT, PRED):
    axa.plot([xx, xx], [lo, hi], color=S.INK, lw=0.9, zorder=4)
    open_ = nm == "crystal"
    axa.plot(xx, v, "D", ms=4.2, color=S.INK, mfc="white" if open_ else S.INK,
             mec=S.INK, mew=1.0, zorder=5)
axa.plot(XSLOT, [p[1] for p in PRED], "-", color=S.INK, lw=0.7, zorder=3)
axa.annotate("real predicted\nbackbones, 140 cx\ncrystal → OF3 → AF2",
             xy=(XSLOT[0] - 0.04, PRED[0][1] + 0.004), xytext=(1.28, 0.0755), fontsize=6.2,
             color=S.INK, va="top", ha="left", linespacing=1.35,
             arrowprops=dict(arrowstyle="-", color=S.RULE, lw=0.7, shrinkA=2, shrinkB=4,
                             connectionstyle="arc3,rad=-0.20"))

axa.set_xlim(-0.10, 2.66); axa.set_ylim(YLO, YHI)
axa.set_xticks([0, 0.5, 1.0, 1.5, 2.0] + XSLOT)
axa.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0", "", "", ""], fontsize=7.2)
axa.set_yticks([0, 0.02, 0.04, 0.06]); axa.tick_params(labelsize=7.2)
axa.text(XSLOT[1], -0.025, "real", transform=axa.get_xaxis_transform(),
         ha="center", va="top", fontsize=7.0, color=S.MUTED)
axa.set_xlabel("backbone jitter σ (Å)", fontsize=8)
axa.xaxis.set_label_coords(0.40, -0.175)
axa.set_ylabel("CPI(L | geometry)", fontsize=8)
S.strip(axa)
h = 0.014 * (YHI - YLO)
axa.plot([XB - 0.060, XB + 0.060], [YLO, YLO], color="white", lw=2.2, zorder=6, clip_on=False)
for xb in (XB - 0.034, XB + 0.034):                                    # axis-break marks
    axa.plot([xb - 0.032, xb + 0.032], [YLO - h, YLO + h], color=S.RULE, lw=0.8, zorder=7, clip_on=False)
S.assert_in_view(axa, list(cm_hi) + list(ce_hi) + [p[3] for p in PRED], axis="y")
S.header(axa, "survives ≤ 0.75 Å, then a cliff",
         f"mutation level · MPNN {n_m:,} · ESM-IF1 {n_e:,} mut", tsize=8.0)

# ================= 3b — the model-free readout =================
axb = fig.add_subplot(gs[1])
axb.plot(sm, rm, "-o", color=S.LEV, ms=3.8, lw=1.4, zorder=4)
axb.plot(se, re_, ls=(0, (4, 2)), marker="s", color=S.ESMIF, ms=3.4, lw=1.3,
         mfc="white", mec=S.ESMIF, mew=1.1, zorder=4)
axb.set_xlim(-0.10, 2.10); axb.set_ylim(0, 0.345)
axb.set_xticks([0, 1, 2]); axb.set_yticks([0, 0.1, 0.2, 0.3])
axb.set_xlabel("σ (Å)", fontsize=8); axb.set_ylabel("−Spearman(L, ΔΔG)", fontsize=8)
axb.xaxis.set_label_coords(0.5, -0.175)
axb.tick_params(labelsize=7.2)
axb.annotate("ESM-IF1's raw ρ\noutlasts its CPI\n(untested why)",
             xy=(1.90, 0.108), xytext=(0.92, 0.315), fontsize=6.1, color=S.MUTED, linespacing=1.35,
             va="top", ha="left",
             arrowprops=dict(arrowstyle="-", color=S.RULE, lw=0.7, shrinkA=3, shrinkB=3,
                             connectionstyle="arc3,rad=0.25"))
S.strip(axb); S.assert_in_view(axb, list(rm) + list(re_), axis="y")
S.header(axb, "same law, no model", "rank corr. with ΔΔG", tsize=8.0)

# ================= 3c — realization variance at σ ≈ 1 Å =================
axc = fig.add_subplot(gs[2])
axc.axhline(0, color=S.RULE, lw=0.8, zorder=1)
xs = np.arange(len(rz)) * 0.26 + 0.62
for xx, (_, r) in zip(xs, rz.iterrows()):
    axc.plot([xx, xx], [r.lo, r.hi], color=S.INK, lw=0.9, zorder=3)
    axc.plot(xx, r.cpi_L_geom, "s", ms=4.6, color=S.ESMIF, mfc="white", mec=S.ESMIF, mew=1.2, zorder=4)
axc.annotate("", xy=(0.42, rz.cpi_L_geom.max()), xytext=(0.42, rz.cpi_L_geom.min()),
             arrowprops=dict(arrowstyle="<->", color=S.MUTED, lw=0.8, shrinkA=0, shrinkB=0))
axc.text(0.34, rz.cpi_L_geom.mean(), "between-draw spread\n≈ the estimate itself", rotation=90,
         fontsize=6.1, color=S.MUTED, va="center", ha="right", linespacing=1.3)
axc.set_xlim(0, 1.30); axc.set_ylim(-0.010, 0.0215); axc.set_xticks([])
axc.set_yticks([0, 0.01, 0.02]); axc.tick_params(labelsize=7.2)
axc.set_ylabel("CPI at σ ≈ 1 Å", fontsize=8)
S.strip(axc); S.assert_in_view(axc, list(rz.hi) + list(rz.lo), axis="y")
S.header(axc, "the 1 Å rung is noise", f"3 draws · {n_rz:,} mut", tsize=8.0)

for x, ch in zip(lefts, "abc"):
    S.flabel(fig, x, 0.955, ch)
S.save(fig, "fig3_doselaw")
print(f"  3a MPNN {np.round(cm,4).tolist()} | ESM {np.round(ce,4).tolist()}")
print(f"  3a slot {[(k, round(v,5)) for k, v, _, _ in PRED]}")
print(f"  3c draws {np.round(rz.cpi_L_geom.values,5).tolist()}")

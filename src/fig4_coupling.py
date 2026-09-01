#!/usr/bin/env python3
"""Figure 4 — the second mixed derivative. Second order predicts binding epistasis: magnitude
clearly, sign barely, and only once the partner is ablated.

  4a  ablation forest       partial Spearman(C, g | Cβ distance)   -> p3_coupling_summary.csv
  4b  magnitude             median |C| in tertiles of measured |g| -> p3_coupling.csv (+ the
                            distance-controlled partial rho from p3_sign_verify.csv)
  4c  the honest sign bound model vs majority-class sign accuracy  -> p3_sign_verify.csv

Palette fix vs the old figure: the |g| tertiles are an ORDINAL magnitude, so they use the
light->dark RAMP_MPNN, never categorical hues. Vermillion means ESM-IF1 and appears nowhere here.

Sign convention (house rule): right = MORE binding signal, so 4a plots MINUS the partial
Spearman — the cycle predicts C proportional to -g, so a correct model sits to the right of 0.

Only 4b's tertile medians are aggregated in-figure (from the committed per-pair CSV); their CI is
a complex-clustered bootstrap, seed 20260803, 2000 replicates, printed on stdout.

  python3 src/fig4_coupling.py  ->  results/figures/fig4_coupling.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
R, SEED, NBOOT = "results", 20260803, 2000

# ---------------------------------------------------------------- data (CSV only)
summ = pd.read_csv(f"{R}/p3_coupling_summary.csv").set_index("set")
sv = pd.read_csv(f"{R}/p3_sign_verify.csv")
cpl = pd.read_csv(f"{R}/p3_coupling.csv").dropna(subset=["C_lev", "g"]).reset_index(drop=True)

def forest(key):                       # -> (-partial, -hi, -lo) so that right = correct sign
    r = summ.loc[key]
    return -float(r.partial_dist), -float(r.pd_hi), -float(r.pd_lo), int(r.n), int(r.n_complex)

def sign_row(subset):
    r = sv[(sv.metric == "sign_accuracy") & (sv.subset == subset)].iloc[0]
    return float(r.model), float(r.majority_baseline), int(r.n)

_ps = sv[(sv.metric == "partial_sign_channel|absg,dist") & (sv.subset == "all")].iloc[0]
PS, PS_LO, PS_HI = float(_ps.model), float(_ps.majority_baseline), float(_ps.balanced_acc)
DOSE_RHO = float(sv[(sv.metric == "dose_partial(|C|,|g||dist)") & (sv.subset == "all")].iloc[0].model)

# ---------------------------------------------------------------- canvas
fig = plt.figure(figsize=(5.5, 2.32))
L0, R0, WR, WS = 0.070, 0.988, [1.20, 0.72, 1.28], 0.55
gs = fig.add_gridspec(1, 3, width_ratios=WR, wspace=WS, left=L0, right=R0, top=0.760, bottom=0.185)
u = (R0 - L0) / (sum(WR) + 2 * WS * np.mean(WR))
lefts = [L0, L0 + u * (WR[0] + WS * np.mean(WR)), L0 + u * (WR[0] + WR[1] + 2 * WS * np.mean(WR))]

# ================= 4a — partner ablation surfaces the coupling =================
axa = fig.add_subplot(gs[0])
ROWS = [("cross-interface, ablated", "cross/C_lev", S.LEV, True),
        ("same-side, ablated", "same/C_lev", S.LEV, False),
        ("same-side, un-ablated", "same/C_complex", S.SCALAR, False)]
ys = np.arange(len(ROWS))[::-1] * 1.0
lims, NS = [], {}
for yy, (lab, key, col, filled) in zip(ys, ROWS):
    v, lo, hi, n, ncx = forest(key)
    lims += [lo, hi]
    axa.text(-0.285, yy + 0.28, f"{lab}  (n = {n})", fontsize=6.4, color=S.INK,
             ha="left", va="bottom")
    NS[key] = (n, ncx)
    axa.plot([lo, hi], [yy, yy], color=S.INK, lw=0.9, zorder=3)
    axa.plot(v, yy, "o", ms=5.2, color=col, mfc=col if filled else "white", mec=col, mew=1.2, zorder=4)
axa.axvline(0, color=S.RULE, lw=0.8, zorder=1)
axa.text(0.0, -0.78, "right of 0 = the sign the cycle predicts", fontsize=6.2, color=S.MUTED,
         ha="center", va="center")
axa.set_yticks([]); axa.set_ylim(-1.05, len(ROWS) - 0.42); axa.set_xlim(-0.29, 0.29)
axa.set_xticks([-0.2, 0, 0.2]); axa.tick_params(labelsize=7.2)
axa.set_xlabel("−partial Spearman(C, g | distance)", fontsize=8)
axa.xaxis.set_label_coords(0.5, -0.205)
S.strip(axa, left=False); S.assert_in_view(axa, lims)
S.header(axa, "ablating the partner surfaces it", "filled ● = 95% CI excludes 0", tsize=8.0)

# ================= 4b — coupling magnitude tracks measured epistasis =================
axb = fig.add_subplot(gs[1])
cpl["tert"] = pd.qcut(cpl.g.abs(), 3, labels=["low", "mid", "high"])
rng = np.random.default_rng(SEED)
ids = cpl.complex_id.unique()
idx = {k: cpl.index[cpl.complex_id == k].to_numpy() for k in ids}
draws = [np.concatenate([idx[k] for k in rng.choice(ids, len(ids), True)]) for _ in range(NBOOT)]
meds, cis, ns = [], [], []
for t in ["low", "mid", "high"]:
    m = cpl.tert == t
    meds.append(float(cpl.C_lev[m].abs().median())); ns.append(int(m.sum()))
    bs = [np.median(np.abs(cpl.C_lev.values[d[cpl.tert.values[d] == t]])) for d in draws]
    cis.append(np.percentile(bs, [2.5, 97.5]))
xb = np.arange(3)
axb.plot(xb, meds, "-", color=S.RAMP_MPNN[2], lw=0.9, zorder=2)
for i, (v, (lo, hi)) in enumerate(zip(meds, cis)):
    axb.plot([i, i], [lo, hi], color=S.INK, lw=0.9, zorder=3)
    axb.plot(i, v, "o", ms=5.4, color=S.RAMP_MPNN[i * 2 + 1], mec=S.LEV, mew=0.8, zorder=4)
axb.set_xticks(xb); axb.set_xticklabels(["low", "mid", "high"], fontsize=7.0)
axb.set_xlim(-0.55, 2.55); axb.set_ylim(0, max(h for _, h in cis) * 1.30)
axb.set_xlabel("tertile of measured |g|", fontsize=8)
axb.set_ylabel("median |C| (model)", fontsize=7.8)
axb.xaxis.set_label_coords(0.5, -0.205); axb.tick_params(labelsize=7.2)
axb.text(0.035, 0.985, f"partial ρ\n(|C|, |g| | dist)\n= {DOSE_RHO:+.2f}", transform=axb.transAxes,
         ha="left", va="top", fontsize=6.3, color=S.MUTED, linespacing=1.35)
S.strip(axb); S.assert_in_view(axb, [h for _, h in cis], axis="y")
S.header(axb, "magnitude tracks |g|", f"{sum(ns)} pairs / {cpl.complex_id.nunique()} cx", tsize=8.0)

# ================= 4c — the honest sign bound =================
axc = fig.add_subplot(gs[2])
SUB = [("all", "all"), ("|g| > 1", "|g|>1.0"), ("|C| > p75", "|C|>p75"), ("|C| > p90", "|C|>p90")]
yc = np.arange(len(SUB))[::-1] * 1.0
xs = []
for yy, (lab, key) in zip(yc, SUB):
    mv, bv, n = sign_row(key); xs += [mv, bv]
    axc.plot([bv, mv], [yy, yy], "-", color=S.GHOST, lw=1.4, solid_capstyle="round", zorder=2)
    axc.plot(bv, yy, "D", ms=4.4, color=S.SCALAR, mfc="white", mec=S.SCALAR, mew=1.1, zorder=3)
    axc.plot(mv, yy, "o", ms=5.0, color=S.LEV, mec=S.LEV, mew=1.2, zorder=4)
    axc.text(0.856, yy, f"n = {n}", fontsize=6.0, color=S.MUTED, ha="right", va="center")
axc.plot([0.5, 0.5], [-0.58, len(SUB) - 0.55], ls=(0, (3, 2)), color=S.RULE, lw=0.8, zorder=1)
axc.text(0.507, -0.32, "chance", fontsize=6.2, color=S.MUTED, ha="left", va="center")
axc.text(0.856, -0.95, f"chance-corrected sign channel\nρ = {PS:+.3f} [{PS_LO:+.3f}, {PS_HI:+.3f}]",
         fontsize=6.2, color=S.MUTED, ha="right", va="top", linespacing=1.35)
axc.set_yticks(yc); axc.set_yticklabels([l for l, _ in SUB], fontsize=6.6)
axc.set_ylim(-1.95, len(SUB) - 0.42); axc.set_xlim(0.478, 0.866)
axc.set_xticks([0.5, 0.6, 0.7, 0.8]); axc.tick_params(labelsize=7.2)
axc.set_xlabel("sign accuracy", fontsize=8)
axc.xaxis.set_label_coords(0.5, -0.205)
S.strip(axc, left=False); axc.tick_params(axis="y", length=0, pad=2, labelsize=6.6)
S.assert_in_view(axc, xs)
S.header(axc, "sign barely survives", "● model   ◇ majority-class", tsize=8.0)

for x, ch in zip(lefts, "abc"):
    S.flabel(fig, x, 0.965, ch)
S.save(fig, "fig4_coupling")
print(f"  4a {[(k, round(forest(k)[0], 4)) for _, k, _, _ in ROWS]}")
print(f"  4b medians {np.round(meds,4).tolist()}  CI {np.round(cis,4).tolist()}  n {ns}"
      f"  (bootstrap {NBOOT} reps, complex-clustered, seed {SEED})")
print(f"  4c {[(k, sign_row(k)[:2]) for _, k in SUB]}")

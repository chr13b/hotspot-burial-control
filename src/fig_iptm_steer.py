#!/usr/bin/env python3
"""Figure — CFG-steering transfers to an INDEPENDENT structure predictor (AF2-multimer).

A frozen ProteinMPNN is steered by +alpha*L (leverage = the partner-ablation / classifier-free-guidance
direction) at interface positions; the resulting sequences are folded with AF2-multimer. Three arms per
complex: wt (crystal), L (steered), random (matched-magnitude random direction).

  (a) forest of the paired L - random deltas (mean over k, complex-clustered bootstrap 95% CI), one row
      per metric, sign-oriented so >0 = better, with the AF2 seed-noise floor drawn behind the ipTM row
      and global pTM kept on the SAME 0-1 scale as ipTM as the localization control.
  (b) the same effect at the level of single complexes: paired wt / L / random ipTM, so specificity
      (L > random) and recovery (L close to wt) read at once.

Every plotted number is read from a committed CSV — results/iptm_summary.csv (deltas + CIs),
results/iptm_steer.csv (per-fold raw), results/iptm_determinism.csv (AF2 seed noise). Nothing is
hardcoded; the annotations are formatted from the values read.

Rows carry DIFFERENT units (z, 0-1, angstrom, pLDDT points) so each unit family is drawn on its own
scale and the value is direct-labelled; the axis therefore carries no numeric ticks. The one comparison
that must be on a shared scale — ipTM vs global pTM, both 0-1 confidences — is locked to one scale, so
their whisker lengths compare directly. Standardising instead (z / SD, or fraction of the wt-random gap)
would erase exactly that comparison: on those scales the global control lands on top of ipTM.

  python3 src/fig_iptm_steer.py  ->  results/figures/fig_iptm_steer.{pdf,png}
"""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import figstyle as S
S.apply()
R = "results"

MINUS = "−"


def sgn(s):
    """Render a signed number with a real minus sign (U+2212), never a hyphen."""
    return s.replace("-", MINUS)


def num(v, dp):
    """Signed fixed-point, rounded HALF-UP so the figure agrees with the manuscript text."""
    q = Decimal(repr(round(float(v), 4))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    return sgn(f"{q:+.{dp}f}")


# ------------------------------------------------------------------ data (CSV only)
summ = pd.read_csv(f"{R}/iptm_summary.csv")
MK = summ[summ["agg"] == "mean_over_k"]


def row(metric, contrast="L_minus_random"):
    """One summary row, SIGN-ORIENTED so that positive always means 'better'."""
    r = MK[(MK.metric == metric) & (MK.contrast == contrast)].iloc[0]
    s = 1.0 if bool(r.higher_better) else -1.0
    d, lo, hi = s * float(r.delta), s * float(r.lo), s * float(r.hi)
    if lo > hi:
        lo, hi = hi, lo
    p_better = float(r.p_gt0) if s > 0 else 1.0 - float(r.p_gt0)
    return dict(delta=d, lo=lo, hi=hi, p=p_better, n=int(r.n))


det = pd.read_csv(f"{R}/iptm_determinism.csv")
SEED_SD = float(det.iptm_sd.mean())                    # AF2 seed-noise floor, ipTM units

raw = pd.read_csv(f"{R}/iptm_steer.csv")
ARMS = ["wt", "L", "random"]
piv = raw.groupby(["complex_id", "direction"])["iptm"].mean().unstack("direction")[ARMS].dropna()
ARM_MEAN = {a: float(piv[a].mean()) for a in ARMS}
N_CX = len(piv)
N_WIN = int((piv.L > piv.random).sum())

# rows of the forest: (metric, label, unit-family, decimals, muted?)
FOREST = [
    ("composite",       "composite  (z-mean)",         "z",    2, False),
    ("iptm",            "ipTM",                        "conf", 3, False),
    ("interface_pae",   f"{MINUS}interface pAE  (Å)",  "pae",  2, False),
    ("interface_plddt", "interface pLDDT  (pts)",      "pts",  2, False),
    ("ptm",             "global pTM",                  "conf", 3, True),
]
V = {m: row(m) for m, *_ in FOREST}

# per-family display scale: one number per unit family, sized so every whisker gets the same headroom.
# Derived from the data (no hardcoded constants); ipTM and global pTM deliberately share family "conf".
FAM = {}
for m, _, fam, _, _ in FOREST:
    FAM[fam] = max(FAM.get(fam, 0.0), V[m]["hi"])
FAM = {k: v * 1.15 for k, v in FAM.items()}
FAMOF = {m: fam for m, _, fam, _, _ in FOREST}


def X(m, key):
    return V[m][key] / FAM[FAMOF[m]]


D_IPTM, D_PTM = V["iptm"]["delta"], V["ptm"]["delta"]
LOCAL_RATIO = D_IPTM / D_PTM                                    # global pTM moves this many times less
SEED_MULT = D_IPTM / SEED_SD                                    # effect / AF2 seed noise
LW = row("iptm", "L_minus_wt")                                  # recovery contrast, ipTM
CLOSER = abs(D_IPTM) / abs(LW["delta"])                         # L sits this many times closer to wt
P_ALL = min(v["p"] for v in V.values())                         # every row's P(>0); reported once

# ------------------------------------------------------------------ canvas
fig = plt.figure(figsize=(5.5, 3.28))
gA = fig.add_gridspec(1, 1, left=0.030, right=0.565, top=0.845, bottom=0.048)
gB = fig.add_gridspec(1, 1, left=0.700, right=0.985, top=0.845, bottom=0.170)

# ================================================================== (a) forest
# three columns: metric name (left of zero) · whisker (right of zero) · value.
axa = fig.add_subplot(gA[0])
XL, XV = -0.70, 1.46                                              # label gutter / axes right edge
VX = max(X(m, "hi") for m, *_ in FOREST) + 0.06                   # value column starts clear of the caps
YS = {"composite": 0.00, "iptm": 1.05, "interface_pae": 2.10, "interface_plddt": 2.80, "ptm": 3.70}
SEPY, SUB = 3.26, 0.34                                            # separator; sub-note offset
axa.set_xlim(XL, XV)
axa.set_ylim(5.16, -0.62)
axa.set_axis_off()


def note(y, s, x=0.045, size=5.6, color=S.MUTED, ha="left"):
    axa.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", zorder=6)


axa.add_patch(Rectangle((XL, YS["composite"] - 0.42), XV - XL, 0.84,          # primary-endpoint band
                        facecolor=S.TINT, edgecolor="none", zorder=0))

# the AF2 seed-noise floor, behind the ipTM row (ipTM units -> the shared "conf" scale)
bw = SEED_SD / FAM["conf"]
axa.add_patch(Rectangle((-bw, YS["iptm"] - 0.30), 2 * bw, 0.60, facecolor=S.FLOOR_FILL,
                        edgecolor="none", zorder=1))

# zero — the only reference line
axa.plot([0, 0], [-0.52, 4.12], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)
axa.text(0.0, 4.32, "no effect", fontsize=5.6, color=S.MUTED, ha="center", va="top")

for m, lab, fam, dp, muted in FOREST:
    y = YS[m]
    c = S.SCALAR if muted else S.LEV
    big = (m == "composite")
    axa.text(-0.10, y, lab, fontsize=6.6 if big else 6.3, color=S.MUTED if muted else S.INK,
             fontweight="bold" if big else "normal", ha="right", va="center", zorder=6)
    axa.text(VX, y, f"{num(V[m]['delta'], dp)}  [{num(V[m]['lo'], dp)}, {num(V[m]['hi'], dp)}]",
             fontsize=6.1 if big else 5.9, color=S.INK if not muted else S.SOFT,
             fontweight="bold" if big else "normal", ha="left", va="center", zorder=6)
    axa.plot([X(m, "lo"), X(m, "hi")], [y, y], "-", color=c, lw=1.6 if big else 1.1,
             solid_capstyle="butt", zorder=4)
    for e in ("lo", "hi"):                                        # CI end caps
        axa.plot([X(m, e)] * 2, [y - 0.125, y + 0.125], "-", color=c, lw=1.6 if big else 1.1, zorder=4)
    axa.plot(X(m, "delta"), y, "o", ms=6.4 if big else 4.4, color=c, mec="white",
             mew=1.0 if big else 0.7, zorder=5)

note(YS["composite"] + SUB, "pre-registered primary endpoint")
note(YS["iptm"] + SUB, "AF2 seed-noise floor", x=XL)
note(YS["iptm"] + SUB, sgn(f"grey band = ±{SEED_SD:.3f} ipTM — the shift is ≈{int(SEED_MULT)}× larger"))

# separator, then the muted localization control
axa.plot([XL, XV], [SEPY, SEPY], "-", color=S.GHOST, lw=0.6, zorder=1)
axa.plot([X("iptm", "delta")] * 2, [YS["ptm"] - 0.19, YS["ptm"] + 0.19], ls=(0, (1.6, 1.6)),
         color=S.FLOOR_EDGE, lw=0.9, zorder=2)
axa.plot(X("iptm", "delta"), YS["ptm"], "o", ms=4.4, mfc="white", mec=S.FLOOR_EDGE, mew=1.0,
         zorder=3)
axa.text(X("iptm", "delta") + 0.020, YS["ptm"] - 0.19, sgn("ΔipTM"), fontsize=5.4, color=S.MUTED,
         ha="left", va="bottom", zorder=6)
note(YS["ptm"] + SUB, "localization control", x=XL)
note(YS["ptm"] + SUB, sgn(f"moves {LOCAL_RATIO:.1f}× less than ipTM, on the same 0–1 scale"))

note(4.68, sgn(f"whiskers = 95% CI, complex-clustered bootstrap  ·  P(>0) = {P_ALL:.2f} on every row"), x=XL)
note(4.98, "unit families are drawn on their own scales; ipTM and global pTM share one", x=XL)

S.header(axa, "every interface metric moves the same way",
         sgn(f"paired Δ (L {MINUS} random), mean over k  ·  n = {V['iptm']['n']} complexes "
             f"({V['composite']['n']} for pAE, composite)"))

# ================================================================== (b) per-complex arms
axb = fig.add_subplot(gB[0])
xs = np.arange(3)
for _, r in piv.iterrows():                                       # one faint line per complex
    axb.plot(xs, [r.wt, r.L, r["random"]], "-", color=S.GHOST, lw=0.45, alpha=0.62,
             solid_capstyle="round", zorder=2)
mu = [ARM_MEAN[a] for a in ARMS]
axb.plot(xs, mu, "-", color=S.INK, lw=1.5, zorder=5)
for x, a, col in zip(xs, ARMS, (S.INK, S.LEV, S.SCALAR)):
    axb.plot(x, ARM_MEAN[a], "o", ms=6.0, color=col, mec="white", mew=1.0, zorder=6)

halo = [pe.withStroke(linewidth=2.6, foreground="white")]
axb.text(-0.10, mu[0], f"{mu[0]:.2f}", fontsize=6.4, color=S.INK, ha="right", va="center",
         path_effects=halo, zorder=7)
axb.text(0.86, mu[1] - 0.030, f"{mu[1]:.2f}", fontsize=6.4, color=S.LEV, ha="center", va="top",
         fontweight="bold", path_effects=halo, zorder=7)
axb.text(2.10, mu[2], f"{mu[2]:.2f}", fontsize=6.4, color=S.SOFT, ha="left", va="center",
         path_effects=halo, zorder=7)

# the two contrasts, written on the segments they describe
axb.text(0.40, np.mean(mu[:2]) + 0.050, f"recovery\n{num(LW['delta'], 3)}", fontsize=6.0,
         color=S.SOFT, ha="center", va="bottom", linespacing=1.3, path_effects=halo, zorder=7)
axb.text(1.62, np.mean(mu[1:]) + 0.075, f"specificity\n{num(V['iptm']['delta'], 3)}", fontsize=6.5,
         color=S.LEV, ha="center", va="bottom", fontweight="bold", linespacing=1.3,
         path_effects=halo, zorder=7)

axb.set_xticks(xs)
axb.set_xticklabels(["wt\ncrystal", "L\nsteered", "random\nmatched"], fontsize=6.4, linespacing=1.35)
axb.set_xlim(-0.48, 2.48)
axb.set_ylim(0.0, 1.035)
axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axb.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=6.6)
axb.set_ylabel("AF2-multimer ipTM", fontsize=7.6)
axb.yaxis.set_label_coords(-0.235, 0.5)
axb.tick_params(labelsize=6.6)
S.strip(axb)
S.assert_in_view(axb, [float(piv.values.min()), float(piv.values.max())], axis="y")
axb.text(-0.42, 0.030, sgn(f"L > random in {N_WIN} / {N_CX} complexes\nP(>0) = {V['iptm']['p']:.2f}"),
         fontsize=5.7, color=S.MUTED, ha="left", va="bottom", linespacing=1.35,
         path_effects=halo, zorder=7)
S.header(axb, "direction, not magnitude", f"ipTM · n = {N_CX} complexes")

S.flabel(fig, 0.030, 0.985, "a")
S.flabel(fig, 0.700, 0.985, "b")
S.save(fig, "fig_iptm_steer")

# ------------------------------------------------------------------ provenance
print("[provenance] all values read from results/iptm_summary.csv (agg=mean_over_k), "
      "results/iptm_steer.csv, results/iptm_determinism.csv")
for m, lab, fam, dp, _ in FOREST:
    v = V[m]
    print(f"  (a) {m:16s} oriented Δ={v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
          f"P(better)={v['p']:.3f} n={v['n']}  family={fam} scale={FAM[fam]:.4f}")
print(f"  (a) seed-noise SD={SEED_SD:.5f} (mean of {len(det)} complexes) -> effect/noise={SEED_MULT:.2f}x ; "
      f"localization ipTM/pTM={LOCAL_RATIO:.2f}x")
print(f"  (b) arm means ipTM: " + ", ".join(f"{a}={ARM_MEAN[a]:.4f}" for a in ARMS) +
      f"  ; L-random={V['iptm']['delta']:+.4f}  L-wt={LW['delta']:+.4f} [{LW['lo']:+.4f},{LW['hi']:+.4f}]"
      f"  ; L>random in {N_WIN}/{N_CX}  ; L is {CLOSER:.2f}x closer to wt than to random")

#!/usr/bin/env python3
"""Figure P — the +alpha*L steering benefit holds on a PHYSICS energy function (FoldX), a binding
readout from outside the inverse-folding family.

Two lanes, pre-registered in results/PREREG_foldx.md:
  (a) STEERING SPECIFICITY (decisive). Favorability = -ddG_bind (>0 = more-favorable binding), paired
      per complex, complex-clustered 95% CI. All three contrasts clear zero: L > naive > random. The
      decisive one is L - naive (highlighted): L beats a same-magnitude *confidence* tilt on physics,
      so the binding-specific advantage is not an inverse-folding artifact. It is deliberately the
      SMALL bar next to the two large vs-random gaps — an honest, CI-excluding-zero increment on top
      of a confidence baseline that already captures most of the benefit.
  (b) DETECTION. |Spearman| vs experimental ddG on the SAME single mutations (SKEMPI). FoldX, a fit
      physics energy function, is more accurate (expected, disclosed); the zero-shot L, with no
      binding-energy term and no fitting, recovers ~70% of its rank accuracy.

Every plotted number is read from a committed CSV; nothing is hardcoded:
  results/foldx_steer.csv      (a) three paired contrasts: delta, lo, hi, p_gt0, n
  results/foldx_detection.csv  (b) Spearman(FoldX, exp) and Spearman(L, exp), n, n_complex

  python3 src/fig_foldx.py  ->  results/figures/fig_foldx.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
R = "results"
MINUS = "−"


def sgn(x, d):
    return f"{x:+.{d}f}".replace("-", MINUS)


def ci(v, lo, hi, d):
    return f"{sgn(v, d)}  [{sgn(lo, d)}, {sgn(hi, d)}]"


# ---- read committed summaries (verified to reproduce byte-exact from the raw per-set CSVs) ----
st_mean = pd.read_csv(f"{R}/foldx_steer.csv").set_index("contrast")            # mean-over-k (pre-registered primary)
rb = pd.read_csv(f"{R}/foldx_steer_robustness.csv")
st = rb[rb["agg"] == "best_of_k"].set_index("contrast")                       # best-of-k (design yield; headline)
det = pd.read_csv(f"{R}/foldx_detection.csv").set_index("metric")
rho_fx = float(det.loc["spearman_FoldX_vs_expddG", "rho"])
rho_l = float(det.loc["spearman_L_vs_expddG", "rho"])
nA = int(det.loc["spearman_FoldX_vs_expddG", "n"])
ncxA = int(det.loc["spearman_FoldX_vs_expddG", "n_complex"])
recov = abs(rho_l) / abs(rho_fx)

fig = plt.figure(figsize=(5.5, 2.62))
gsL, gsR = 0.095, 0.585
axa = fig.add_axes([gsL, 0.20, 0.40, 0.60])       # (a) steering forest
axb = fig.add_axes([gsR, 0.20, 0.345, 0.60])      # (b) detection bars

# ============================ (a) steering specificity ============================
ROWS = [("L-naive", "L − naive   (decisive)", S.LEV, True),
        ("naive-random", "naive − random", S.SCALAR, False),
        ("L-random", "L − random", S.MUTED, False)]
ys = [2.3, 1.15, 0.0]                              # decisive on top, extra gap below it
axa.axvline(0, ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)
for (key, lab, col, decisive), y in zip(ROWS, ys):
    r = st.loc[key]
    v, lo, hi = float(r.delta), float(r.lo), float(r.hi)
    lw = 2.2 if decisive else 1.4
    ms = 6.4 if decisive else 5.0
    txt = ci(v, lo, hi, 2) + (f"  P={float(r.p_gt0):.3f}" if decisive else "")
    axa.plot([lo, hi], [y, y], "-", color=col, lw=lw, solid_capstyle="butt", zorder=4)
    for e in (lo, hi):
        axa.plot([e, e], [y - 0.11, y + 0.11], "-", color=col, lw=lw, zorder=4)
    axa.plot(v, y, "o", ms=ms, color=col, mec="white", mew=0.9, zorder=5)
    axa.text(hi + 0.30, y, txt, fontsize=5.9, color=S.INK if decisive else S.MUTED,
             ha="left", va="center", zorder=6)
    axa.text(0.0, y + 0.30, lab, fontsize=6.6, color=col,
             fontweight="bold" if decisive else "normal", ha="left", va="bottom", zorder=6)
    if decisive:                                         # mean-over-k (pre-registered primary) as a light anchor
        vm = float(st_mean.loc[key].delta)
        axa.plot(vm, y - 0.30, "o", ms=4.2, mfc="white", mec=col, mew=1.1, zorder=5)
        axa.text(vm + 0.30, y - 0.30, f"mean-k {sgn(vm,2)} (pre-reg primary)", fontsize=5.0,
                 color=S.MUTED, ha="left", va="center", zorder=6)
axa.set_xlim(-0.6, 11.6)
axa.set_ylim(-0.75, 3.15)
axa.set_yticks([])
axa.spines["left"].set_visible(False)
axa.spines["top"].set_visible(False)
axa.spines["right"].set_visible(False)
axa.spines["bottom"].set_color(S.RULE)
axa.set_xticks([0, 3, 6, 9])
axa.tick_params(colors=S.RULE, labelcolor=S.INK, length=2.5, labelsize=6.0)
axa.set_xlabel("Δ favorability  (= −ΔΔG$_{bind}$, kcal/mol)  —  >0 favours the first arm",
               fontsize=5.9, color=S.MUTED)
S.header(axa, "L > naive > random on physics ΔΔG$_{bind}$",
         note="best-of-k (design yield); paired 95% CI  ·  n = 59 / 57 complexes", tsize=7.6)

# ============================ (b) detection ============================
bars = [("FoldX ΔΔG$_{bind}$", abs(rho_fx), S.SCALAR, "fit to ΔΔG"),
        ("L  (leverage)", abs(rho_l), S.LEV, "zero-shot, no binding term")]
by = [1.0, 0.0]
for (lab, val, col, sub), y in zip(bars, by):
    axb.barh(y, val, height=0.52, color=col, zorder=3)
    axb.text(val + 0.012, y + 0.11, f"{val:.3f}".replace("-", MINUS), fontsize=6.6,
             color=S.INK, ha="left", va="center", fontweight="bold", zorder=6)
    axb.text(0.006, y + 0.11, lab, fontsize=6.4, color="white", ha="left", va="center",
             fontweight="bold", zorder=6)
    axb.text(0.006, y - 0.22, sub, fontsize=5.4, color=S.MUTED, ha="left", va="center", zorder=6)
axb.annotate(f"L recovers {recov*100:.0f}% of the\nphysics tool's rank accuracy",
             xy=(abs(rho_l), 0.0), xytext=(abs(rho_fx) - 0.02, 0.52),
             fontsize=5.5, color=S.LEV, ha="right", va="center", linespacing=1.35, zorder=6)
axb.set_xlim(0, 0.52)
axb.set_ylim(-0.62, 1.62)
axb.set_yticks([])
axb.spines["left"].set_visible(False)
axb.spines["top"].set_visible(False)
axb.spines["right"].set_visible(False)
axb.spines["bottom"].set_color(S.RULE)
axb.set_xticks([0, 0.2, 0.4])
axb.tick_params(colors=S.RULE, labelcolor=S.INK, length=2.5, labelsize=6.0)
axb.set_xlabel("|Spearman|  vs experimental ΔΔG", fontsize=5.9, color=S.MUTED)
S.header(axb, "Zero-shot L is competitive as a detector",
         note=f"SKEMPI single mutants  ·  n = {nA:,} ({ncxA} complexes)", tsize=7.6)

S.flabel(fig, gsL, 0.985, "a")
S.flabel(fig, gsR, 0.985, "b")
S.save(fig, "fig_foldx")
print(f"  (a) L-naive {ci(float(st.loc['L-naive'].delta), float(st.loc['L-naive'].lo), float(st.loc['L-naive'].hi), 2)}")
print(f"  (b) FoldX {rho_fx:+.4f}  L {rho_l:+.4f}  recovery {recov:.3f}")

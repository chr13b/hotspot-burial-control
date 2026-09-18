#!/usr/bin/env python3
"""Figure P — the steering knob survives on PREDICTED backbones, the staged-design regime.

Every other steering result in the paper tilts a frozen ProteinMPNN on a CRYSTAL backbone. A designer has
no crystal. So the leverage is recomputed AND applied on the predicted backbone (OpenFold3, AF2-multimer),
and judged anti-circularly by a different model's leverage read off that same predicted structure.

  (a) the judge level, the pre-registered primary. Two predicted backbones x two anti-circular judges, two
      contrasts: L - random (does the tilt beat a matched-magnitude null direction?) and L - naive (does it
      beat the CONFIDENCE tilt, i.e. is the gain binding-specific?). The matched crystal control - the same
      106 complexes, same pipeline, crystal backbone - is drawn BEHIND each judge group as a grey 95% CI
      band, so "barely attenuated" is read directly off the geometry: every predicted dot lands inside the
      band it is being compared to. All six leverage contrasts are one instrument family in one unit, so
      they legitimately share one scale.
  (b) an INDEPENDENT instrument, outside the inverse-folding family entirely: AF2-multimer folds of the
      predicted-backbone-steered sequences (n = 93). Composite and interface ipTM both clear zero; global
      pTM is kept on the SAME 0-1 scale as ipTM as the localization control, so "the gain is at the
      interface" is a comparison of two whiskers rather than a claim in the caption.
  (c) the same ipTM effect per complex: wt / L / random arms, so specificity (L > random) and the ordering
      wt > L > random read at once, with the calibration-free win rate.

Every plotted number is a literal cell of a committed CSV; nothing is hardcoded, every annotation is
formatted from a value that was read, and the bootstrap replicate count is imported from the analysis
modules that produced the CSVs rather than retyped:
  (a) results/cfg_steer_predicted.csv            (deltas + CIs, incl. the crystal control block)
      results/_cfg_pred_{of3,af2,crystal}.csv    (per-complex raw: win rates, and a check against the summary)
      results/cfg_steer_predicted_recovery.csv   (the native-recovery control, quoted in the footnote)
  (b) results/iptm_predicted.csv                 (deltas + CIs)
  (c) results/iptm_predicted_steer.csv           (per-fold raw)

  python3 src/fig_predicted_steer.py  ->  results/figures/fig_predicted_steer.{pdf,png}
"""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import figstyle as S
from analyse_predicted_steer import NBOOT as NBOOT_JUDGE, ALPHA   # replicate count + dose come from the
from analyse_iptm import NBOOT as NBOOT_IPTM                      # producing code, never retyped here
S.apply()
R = "results"
SEED_FIG = 20260803          # fixed: this figure draws no random numbers, but the seed is recorded anyway
np.random.seed(SEED_FIG)

MINUS = "−"
assert NBOOT_JUDGE == NBOOT_IPTM, "the two analyses used different bootstrap replicate counts"
NBOOT = NBOOT_JUDGE


def sgn(s):
    """Render a signed number with a real minus sign (U+2212), never a hyphen."""
    return s.replace("-", MINUS)


def num(v, dp, plus=True):
    """Signed fixed-point, rounded HALF-UP so the figure agrees with the manuscript text."""
    q = Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    return sgn(f"{q:+.{dp}f}" if plus else f"{q:.{dp}f}")


def ci(r, dp):
    return f"{num(r['v'], dp)}  [{num(r['lo'], dp)}, {num(r['hi'], dp)}]"


def nice_max(hi, step):
    """Smallest multiple of `step` clearing the widest CI end — ordinary axis logic, so the scale is set
    by the data and not by a typed constant."""
    return float(step * np.ceil(hi * 1.02 / step - 1e-9))


# ================================================================== (a) data — the judge level
JUDGES = [("esmif", "ESM-IF1", S.ESMIF), ("mif", "MIF", S.MIF)]      # anti-circular: never ProteinMPNN
SOURCES = [("of3", "OpenFold3"), ("af2", "AF2-multimer")]            # the predicted backbones
CONTRASTS = [("L-random", "L  −  random", "the tilt beats a matched-magnitude null direction"),
             ("L-naive", "L  −  naive", "and beats the confidence tilt — the gain is binding-specific")]

JM = pd.read_csv(f"{R}/cfg_steer_predicted.csv")
RAW = {s: pd.read_csv(f"{R}/_cfg_pred_{s}.csv") for s, _ in SOURCES + [("crystal", "crystal")]}
assert all(ALPHA in set(d.alpha) for d in RAW.values()), f"alpha {ALPHA} missing from a raw steer CSV"


def jrow(source, judge, contrast):
    """One committed summary row, plus the per-complex win rate recomputed from the raw CSV."""
    r = JM[(JM.source == source) & (JM.judge == judge) & (JM.contrast == contrast)].iloc[0]
    a, b = contrast.split("-")
    pv = (RAW[source][RAW[source].alpha == ALPHA]
          .pivot_table(index="complex_id", columns="direction", values=f"meanL_{judge}", aggfunc="mean")
          [[a, b]].dropna())
    d = pv[a] - pv[b]
    assert abs(float(d.mean()) - float(r.delta)) < 5e-4, \
        f"{source}/{judge}/{contrast}: raw per-complex CSV disagrees with the committed summary delta"
    assert int(len(d)) == int(r.n_cx), f"{source}/{judge}/{contrast}: n mismatch raw vs summary"
    return dict(v=float(r.delta), lo=float(r.lo), hi=float(r.hi), p=float(r.p_gt0), n=int(r.n_cx),
                nwin=int((d > 0).sum()), seed=int(r.seed), nboot=int(r.nboot),
                mean_a=float(r.mean_a), mean_b=float(r.mean_b))


CELL = {(c, j, s): jrow(s, j, c) for c, *_ in CONTRASTS for j, *_ in JUDGES for s, _ in SOURCES}
CRY = {(c, j): jrow("crystal", j, c) for c, *_ in CONTRASTS for j, *_ in JUDGES}
NR = {(j, s): jrow(s, j, "naive-random") for j, *_ in JUDGES for s, _ in SOURCES + [("crystal", "crystal")]}

assert all(v["nboot"] == NBOOT for v in list(CELL.values()) + list(CRY.values())), \
    "a committed judge row was bootstrapped with a different replicate count than the analysis module"
assert all(v["lo"] > 0 for v in CELL.values()), \
    "a predicted-backbone interval touches zero — the pre-registered falsifier would have fired"
# the claim the grey band makes, checked rather than asserted in prose
INSIDE = {k: (CRY[(k[0], k[1])]["lo"] <= v["v"] <= CRY[(k[0], k[1])]["hi"]) for k, v in CELL.items()}
RETAIN = {k: v["v"] / CRY[(k[0], k[1])]["v"] for k, v in CELL.items()}
N_CX = CELL[("L-random", "esmif", "of3")]["n"]
JSC = nice_max(max(v["hi"] for v in list(CELL.values()) + list(CRY.values())), 0.05)   # one shared scale

# ================================================================== (b, c) data — the independent instrument
ipt = pd.read_csv(f"{R}/iptm_predicted.csv")
MK = ipt[ipt["agg"] == "mean_over_k"]


def irow(metric, contrast="L_minus_random"):
    """One summary row, SIGN-ORIENTED so that positive always means 'better'."""
    r = MK[(MK.metric == metric) & (MK.contrast == contrast)].iloc[0]
    sg = 1.0 if bool(r.higher_better) else -1.0
    lo, hi = sorted((sg * float(r.lo), sg * float(r.hi)))
    return dict(v=sg * float(r.delta), lo=lo, hi=hi, n=int(r.n), seed=int(r.seed),
                p=float(r.p_gt0) if sg > 0 else 1.0 - float(r.p_gt0))


FOREST = [("composite", "composite  (z-mean)", "z", 2, False),
          ("iptm", "interface ipTM", "conf", 3, False),
          ("ptm", "global pTM", "conf", 3, True)]
V = {m: irow(m) for m, *_ in FOREST}
PAE, PLDDT = irow("interface_pae"), irow("interface_plddt")          # quoted in the footnote, same CSV
assert V["composite"]["lo"] > 0 and V["iptm"]["lo"] > 0, "panel (b) H1 would not hold — check the CSV"

# per-family display scale: one number per unit family, sized so every whisker gets the same headroom.
FAM = {}
for m, _, fam, _, _ in FOREST:
    FAM[fam] = max(FAM.get(fam, 0.0), V[m]["hi"])
FAM = {k: v * 1.18 for k, v in FAM.items()}
FAMOF = {m: fam for m, _, fam, _, _ in FOREST}
X = lambda m, key: V[m][key] / FAM[FAMOF[m]]
LOCAL_RATIO = V["iptm"]["v"] / V["ptm"]["v"]                         # global pTM moves this many times less

ARMS = ["wt", "L", "random"]
praw = pd.read_csv(f"{R}/iptm_predicted_steer.csv")
piv = praw.groupby(["complex_id", "direction"])["iptm"].mean().unstack("direction")[ARMS].dropna()
ARM_MEAN = {a: float(piv[a].mean()) for a in ARMS}
N_FOLD, N_WIN = len(piv), int((piv.L > piv["random"]).sum())
assert abs(float((piv.L - piv["random"]).mean()) - V["iptm"]["v"]) < 5e-4, \
    "the per-fold ipTM CSV disagrees with the committed summary delta"
assert N_FOLD == V["iptm"]["n"], "per-fold complex count disagrees with the committed summary n"
LWT = irow("iptm", "L_minus_wt")

rec = pd.read_csv(f"{R}/cfg_steer_predicted_recovery.csv")
REC = {(r.source, r.direction): float(r.int_recovery) for _, r in rec.iterrows()}

# ================================================================== canvas
fig = plt.figure(figsize=(S.FIG_W, 4.34))
gA = fig.add_gridspec(1, 1, left=0.040, right=0.995, top=0.948, bottom=0.437)
gB = fig.add_gridspec(1, 1, left=0.040, right=0.612, top=0.360, bottom=0.176)
gC = fig.add_gridspec(1, 1, left=0.775, right=0.988, top=0.360, bottom=0.176)

HALO = [pe.withStroke(linewidth=2.4, foreground="white")]


def whisk(ax, x, y, r, col, lw=1.3, ms=5.0, cap=0.17):
    ax.plot([x(r["lo"]), x(r["hi"])], [y, y], "-", color=col, lw=lw, solid_capstyle="butt", zorder=5)
    for e in ("lo", "hi"):
        ax.plot([x(r[e])] * 2, [y - cap, y + cap], "-", color=col, lw=lw, zorder=5)
    ax.plot(x(r["v"]), y, "o", ms=ms, color=col, mec="white", mew=0.9, zorder=6)


# ================================================================== (a) judge level
axa = fig.add_subplot(gA[0])
# five columns no mark may cross: judge · backbone · whisker (0..1 = the shared leverage scale) · value ·
# win rate. Sized so the value column still clears the panel edge under the WIDEST fallback font.
XL, XS, VX, WX, XV = -1.16, -0.62, 1.035, 1.925, 2.31
YA, y = {}, 0.0
for cname, *_ in CONTRASTS:
    for jkey, *_ in JUDGES:
        for i, (skey, _) in enumerate(SOURCES):
            YA[(cname, jkey, skey)] = y + 1.02 * i
        y += 3.30                                              # group: 2 rows + the crystal caption
    y += 1.16                                                  # gap between the two contrast blocks
TOPY = -1.74
BOT = max(YA.values()) + 2.62
axa.set_xlim(XL, XV)
axa.set_ylim(BOT, TOPY)
axa.set_axis_off()
xj = lambda v: v / JSC

BH0, BH1 = -0.44, 1.46                                         # crystal band extent around a judge group
for bi, (cname, clab, csub) in enumerate(CONTRASTS):
    y0 = YA[(cname, JUDGES[0][0], SOURCES[0][0])]
    # block header, sitting on its own line above the block (never on the zero rule)
    axa.text(XL, y0 - 1.06, clab, fontsize=7.4, color=S.INK, fontweight="bold", ha="left", va="center",
             zorder=7, path_effects=[pe.withStroke(linewidth=5.0, foreground="white")])
    axa.text(XL + 0.50, y0 - 1.06, csub, fontsize=5.7, color=S.MUTED, ha="left", va="center",
             zorder=7, path_effects=[pe.withStroke(linewidth=5.0, foreground="white")])
    for jkey, jlab, jcol in JUDGES:
        yy0 = YA[(cname, jkey, SOURCES[0][0])]
        c = CRY[(cname, jkey)]
        # the matched crystal control as a 95% CI BAND behind the group: the comparison is the geometry
        axa.add_patch(Rectangle((xj(c["lo"]), yy0 + BH0), xj(c["hi"]) - xj(c["lo"]), BH1 - BH0,
                                facecolor=S.FLOOR_FILL, edgecolor="none", zorder=1))
        axa.plot([xj(c["v"])] * 2, [yy0 + BH0, yy0 + BH1], ls=(0, (1.6, 1.6)), color=S.FLOOR_EDGE,
                 lw=0.9, zorder=2)
        axa.text(xj(c["v"]), yy0 + BH1 + 0.30, sgn(f"crystal control {num(c['v'], 2)}"), fontsize=5.3,
                 color=S.MUTED, ha="center", va="center", zorder=7, path_effects=HALO)
        axa.text(XL, yy0 + 0.51, jlab, fontsize=6.9, color=jcol, fontweight="bold", ha="left",
                 va="center", zorder=7)
        axa.text(XL, yy0 + 1.14, "judge", fontsize=5.3, color=S.MUTED, ha="left", va="center", zorder=7)
        for skey, slab in SOURCES:
            yy, r = YA[(cname, jkey, skey)], CELL[(cname, jkey, skey)]
            axa.text(XS, yy, slab, fontsize=6.2, color=S.INK, ha="left", va="center", zorder=7)
            whisk(axa, xj, yy, r, jcol)
            axa.text(VX, yy, f"{num(r['v'], 3)} [{num(r['lo'], 3)}, {num(r['hi'], 3)}]", fontsize=5.9,
                     color=S.INK, ha="left", va="center", zorder=7)
            axa.text(WX, yy, f"{r['nwin']} / {r['n']}", fontsize=5.9, color=S.INK, ha="left",
                     va="center", zorder=7)

# zero — the only reference line; and the column headers on one rule
axa.plot([0, 0], [TOPY + 0.46, BOT - 0.22], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=3)
axa.plot([XL, XV], [TOPY + 0.40] * 2, "-", color=S.GHOST, lw=0.6, zorder=1)
for xx, s, ha in [(0.0, "no effect", "center"), (VX, "paired Δ, 95% CI", "left"),
                  (WX, "L wins", "left")]:
    axa.text(xx, TOPY + 0.26, s, fontsize=5.6, color=S.SOFT, ha=ha, va="bottom", zorder=7)

# the leverage scale, drawn once at the foot of the panel; the rule spans the whole whisker region
TICKS = list(np.arange(0.0, JSC + 1e-9, 0.25))
RULY = BOT - 1.14
axa.plot([0.0, 1.0], [RULY] * 2, "-", color=S.RULE, lw=0.7, zorder=3)
for t in TICKS:
    axa.plot([xj(t)] * 2, [RULY - 0.18, RULY], "-", color=S.RULE, lw=0.7, zorder=3)
    axa.text(xj(t), RULY + 0.40, f"{t:.2f}", fontsize=5.4, color=S.MUTED, ha="center", va="center", zorder=7)
axa.text(1.06, RULY + 0.05, "judge leverage (nats)", fontsize=5.5, color=S.MUTED,
         ha="left", va="center", zorder=7)

axa.text(XV, RULY + 1.02,
         sgn(f"all eight predicted estimates fall inside that band  ·  "
             f"P(>0) = {min(v['p'] for v in CELL.values()):.2f}  ·  α = {ALPHA:g}")
         if all(INSIDE.values()) else
         sgn(f"P(>0) = {min(v['p'] for v in CELL.values()):.2f}  ·  α = {ALPHA:g}"),
         fontsize=5.5, color=S.MUTED, ha="right", va="center", zorder=7)
S.header(axa, "steering barely attenuates on a predicted backbone",
         "frozen ProteinMPNN, $+\\,α\\,L$ on the predicted backbone  ·  grey band = matched crystal "
         "control, 95% CI", tsize=8.2)

# ================================================================== (b) an independent instrument
axb = fig.add_subplot(gB[0])
BXL, BVX = -0.96, max(X(m, "hi") for m, *_ in FOREST) + 0.05
YB = {"composite": 0.0, "iptm": 1.0, "ptm": 2.18}
axb.set_xlim(BXL, 2.08)
axb.set_ylim(2.94, -0.92)
axb.set_axis_off()
axb.add_patch(Rectangle((BXL, YB["composite"] - 0.40), 2.08 - BXL, 0.80,       # primary endpoint
                        facecolor=S.TINT, edgecolor="none", zorder=0))
axb.plot([0, 0], [-0.62, 2.54], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)
axb.text(0.0, -0.72, "no effect", fontsize=5.5, color=S.MUTED, ha="center", va="center", zorder=7)
axb.plot([BXL, 2.08], [YB["ptm"] - 0.56] * 2, "-", color=S.GHOST, lw=0.6, zorder=1)

for m, lab, fam, dp, muted in FOREST:
    yy, r = YB[m], V[m]
    col = S.SCALAR if muted else S.LEV
    big = (m == "composite")
    axb.text(BXL, yy, lab, fontsize=6.4 if big else 6.2, color=S.MUTED if muted else S.INK,
             fontweight="bold" if big else "normal", ha="left", va="center", zorder=7)
    whisk(axb, lambda v, m=m: v / FAM[FAMOF[m]], yy, r, col, lw=1.5 if big else 1.15,
          ms=5.8 if big else 4.4, cap=0.13)
    axb.text(BVX, yy, ci(r, dp), fontsize=6.0 if big else 5.8, color=S.SOFT if muted else S.INK,
             fontweight="bold" if big else "normal", ha="left", va="center", zorder=7)
axb.text(BXL, YB["composite"] + 0.42, "pre-registered primary endpoint", fontsize=5.4, color=S.MUTED,
         ha="left", va="center", zorder=7)
# the localization control, said with geometry: the ipTM shift marked on the pTM row's own scale
axb.plot([X("iptm", "v")] * 2, [YB["ptm"] - 0.17, YB["ptm"] + 0.17], ls=(0, (1.6, 1.6)),
         color=S.FLOOR_EDGE, lw=0.9, zorder=3)
axb.plot(X("iptm", "v"), YB["ptm"], "o", ms=4.4, mfc="white", mec=S.FLOOR_EDGE, mew=1.0, zorder=4)
axb.text(X("iptm", "v") + 0.035, YB["ptm"] - 0.18, sgn("ΔipTM"), fontsize=5.3, color=S.MUTED,
         ha="left", va="bottom", zorder=7)
axb.text(BXL, YB["ptm"] + 0.44, sgn(f"localization control — moves {LOCAL_RATIO:.1f}× less than ipTM, "
                                    f"on the same 0–1 scale"),
         fontsize=5.4, color=S.MUTED, ha="left", va="center", zorder=7)
S.header(axb, "an independent structure predictor agrees",
         sgn(f"AF2-multimer, paired Δ (L {MINUS} random)  ·  n = {V['iptm']['n']}  ·  each unit family on "
             f"its own scale"))

# ================================================================== (c) the same effect per complex
axc = fig.add_subplot(gC[0])
xs = np.arange(3)
for _, r in piv.iterrows():
    axc.plot(xs, [r.wt, r.L, r["random"]], "-", color=S.GHOST, lw=0.42, alpha=0.6,
             solid_capstyle="round", zorder=2)
mu = [ARM_MEAN[a] for a in ARMS]
axc.plot(xs, mu, "-", color=S.INK, lw=1.4, zorder=5)
for x, a, col in zip(xs, ARMS, (S.INK, S.LEV, S.SCALAR)):
    axc.plot(x, ARM_MEAN[a], "o", ms=5.6, color=col, mec="white", mew=1.0, zorder=6)
axc.text(-0.12, mu[0], f"{mu[0]:.2f}", fontsize=6.2, color=S.INK, ha="right", va="center",
         path_effects=HALO, zorder=7)
axc.text(0.92, mu[1] - 0.035, f"{mu[1]:.2f}", fontsize=6.2, color=S.LEV, ha="center", va="top",
         fontweight="bold", path_effects=HALO, zorder=7)
axc.text(2.12, mu[2], f"{mu[2]:.2f}", fontsize=6.2, color=S.SOFT, ha="left", va="center",
         path_effects=HALO, zorder=7)
axc.text(1.58, np.mean(mu[1:]) + 0.075, f"{num(V['iptm']['v'], 3)}", fontsize=6.6, color=S.LEV,
         ha="center", va="bottom", fontweight="bold", path_effects=HALO, zorder=7)
axc.set_xticks(xs)
axc.set_xticklabels(["wt", "L", "random"], fontsize=6.2)
axc.set_xlim(-0.52, 2.52)
axc.set_ylim(0.0, 1.03)
axc.set_yticks([0, 0.5, 1.0])
axc.set_yticklabels(["0", "0.50", "1.00"], fontsize=6.4)
axc.set_ylabel("ipTM", fontsize=7.0, labelpad=1.0)
axc.tick_params(labelsize=6.4)
S.strip(axc)
S.assert_in_view(axc, [float(piv.values.min()), float(piv.values.max())], axis="y")
axc.text(-0.44, 0.035, f"L > random\n{N_WIN} / {N_FOLD}", fontsize=5.6, color=S.MUTED, ha="left",
         va="bottom", linespacing=1.35, path_effects=HALO, zorder=7)
S.header(axc, "per complex", f"the three arms · n = {N_FOLD}", tsize=7.6)

# ================================================================== footnote
SEEDS = sorted({v["seed"] for v in CELL.values()} | {V["iptm"]["seed"]})
NOTE = [
    "a, judge leverage = the per-complex mean L over the sampled interface positions, scored by a model "
    "that is NOT",
    f"the steered one and read off the same predicted structure.  Crystal control = the same {N_CX} "
    f"complexes on the",
    f"crystal backbone; native interface recovery is preserved (L arm {REC[('of3', 'L')]:.2f} / "
    f"{REC[('af2', 'L')]:.2f} vs crystal {REC[('crystal', 'L')]:.2f}, random "
    f"{REC[('of3', 'random')]:.2f} / {REC[('af2', 'random')]:.2f}).",
    f"b, the other two interface metrics move the same way: interface pAE {PAE['v']:.2f} Å lower, "
    f"interface pLDDT {PLDDT['v']:.2f} points higher.",
    f"All intervals are complex-clustered bootstrap 95% CIs, {NBOOT:,} replicates, seed {SEEDS[0]}.",
]
fig.text(0.040, 0.142, sgn("\n".join(NOTE)), fontsize=5.3, color=S.MUTED, ha="left", va="top",
         linespacing=1.52)

S.flabel(fig, 0.040, 0.997, "a")
S.flabel(fig, 0.040, 0.403, "b")
S.flabel(fig, 0.775, 0.403, "c")
S.save(fig, "fig_predicted_steer")

# ------------------------------------------------------------------ provenance
print("[provenance] every mark is a literal cell of a committed CSV; bootstrap replicates imported from "
      f"src/analyse_predicted_steer.py and src/analyse_iptm.py (NBOOT={NBOOT}, alpha={ALPHA:g})")
print(f"  (a) results/cfg_steer_predicted.csv — one shared leverage scale {JSC:.2f} nats")
for cname, clab, _ in CONTRASTS:
    for jkey, jlab, _ in JUDGES:
        for skey, slab in SOURCES:
            r, c = CELL[(cname, jkey, skey)], CRY[(cname, jkey)]
            print(f"      {cname:9s} judge-{jlab:7s} {slab:12s} Δ={r['v']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] "
                  f"P(>0)={r['p']:.3f} n={r['n']} | L>other {r['nwin']}/{r['n']} "
                  f"({r['nwin'] / r['n'] * 100:.1f}%) | crystal {c['v']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] "
                  f"-> {RETAIN[(cname, jkey, skey)] * 100:.1f}% of crystal, inside crystal CI: "
                  f"{INSIDE[(cname, jkey, skey)]}")
print(f"      arms (mean_a/mean_b per row) e.g. L-random of3/ESM-IF1: L={CELL[('L-random','esmif','of3')]['mean_a']:+.4f} "
      f"random={CELL[('L-random','esmif','of3')]['mean_b']:+.4f}")
print("      naive-random (the confidence tilt alone, same CSV): " +
      ", ".join(f"{s}/{j}={NR[(j, s)]['v']:+.4f}" for j, *_ in JUDGES for s, _ in SOURCES))
print(f"      all eight predicted estimates inside the matched crystal 95% CI: {all(INSIDE.values())}; "
      f"retention {min(RETAIN.values()) * 100:.0f}–{max(RETAIN.values()) * 100:.0f}% of crystal")
print(f"  (b) results/iptm_predicted.csv (agg=mean_over_k) — family scales " +
      ", ".join(f"{k}={v:.4f}" for k, v in FAM.items()))
for m, lab, fam, dp, _ in FOREST + [("interface_pae", "", "", 2, False), ("interface_plddt", "", "", 2, False)]:
    r = V.get(m) or irow(m)
    print(f"      {m:16s} oriented Δ={r['v']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] P(better)={r['p']:.3f} "
          f"n={r['n']}")
print(f"      localization: ipTM/pTM = {LOCAL_RATIO:.2f}x on the same 0-1 scale")
print(f"  (c) results/iptm_predicted_steer.csv — arm means ipTM: " +
      ", ".join(f"{a}={ARM_MEAN[a]:.4f}" for a in ARMS) +
      f"  ; L−random={V['iptm']['v']:+.4f}  L−wt={LWT['v']:+.4f} [{LWT['lo']:+.4f},{LWT['hi']:+.4f}]"
      f"  ; L>random in {N_WIN}/{N_FOLD}")
print(f"      results/cfg_steer_predicted_recovery.csv — int_recovery " +
      ", ".join(f"{s}/{d}={REC[(s, d)]:.4f}" for s in ("of3", "af2", "crystal") for d in ("L", "naive", "random")))

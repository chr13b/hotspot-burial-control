#!/usr/bin/env python3
"""Figure R — the +alpha*L steering effect replicates across structure predictors, steering
directions, judge architectures and complexes.

A frozen inverse-folding model is steered by +alpha*L (leverage = the partner-ablation /
classifier-free-guidance direction) at interface positions. The headline contrast is always the same
PAIRED one: L-arm minus a matched-magnitude random-direction arm, per complex, so positions,
per-position magnitude, readout and complex are held fixed and only the DIRECTION differs.

  (a) the readout that is a structure predictor, i.e. an instrument outside the sequence-model family:
      three independent runs — AF2-multimer x steer-ProteinMPNN (n=120), Boltz-2 x steer-ProteinMPNN
      (n=60), AF2-multimer x steer-ESM-IF1 (n=60, the reverse steering direction).
  (b) the readout that is another inverse-folding model, ANTI-CIRCULAR: the steered residues score
      higher binding-leverage under a judge that is NOT the steered model, over four architectures
      and both steering directions. The self-judged diagonal is excluded as trivially circular.

THE ONE THING THIS FIGURE MUST NOT SAY. AF2-multimer and Boltz-2 ipTM are DIFFERENT CALIBRATIONS,
and the composite is z-scored within each run, so "+0.235 (AF2) vs +0.139 (Boltz-2)" is NOT a
magnitude comparison — AF2 is not "better". The layout therefore refuses the comparison structurally
rather than only in the caption: every row in (a) carries its OWN scale bar, drawn at one physical
length and labelled with the number of ipTM points it spans (a different number on each row), so
whisker lengths encode nothing across rows; the only thing the three rows share is the zero line,
which is exactly the claim ("all > 0"). What the reader SHOULD compare is put in instead, and it is
calibration-free: the per-complex win rate (L > random), and where the dot sits between the two
landmarks that bound every row — zero is the random arm, the dotted tick is the wild-type arm, and
the dot lands at the same relative place on all three. Panel (b) is a different unit entirely
(leverage, nats), gets its own axis, and there the cells ARE one instrument family, so they share
one scale legitimately.

Every plotted number is read from a committed CSV; nothing is hardcoded, every annotation is
formatted from a value read, and the bootstrap replicate count is imported from the analysis modules
that produced the CSVs rather than retyped:
  (a) results/iptm_summary_120.csv, iptm_summary_boltz.csv, iptm_summary_esmif.csv   (deltas + CIs)
      results/iptm_steer_120.csv, iptm_steer_boltz.csv, iptm_steer_esmif.csv         (per-fold raw)
  (b) results/cfg_judge_matrix.csv, results/cfg_judge_matrix_pifold.csv              (deltas + CIs)
      results/cfg_judge_dumped_set{A,B}{,_pifold}.csv                                (per-complex raw)

  python3 src/fig_robustness.py  ->  results/figures/fig_robustness.{pdf,png}
"""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import figstyle as S
from analyse_iptm import NBOOT as NBOOT_IPTM          # replicate counts come from the producing code,
from cfg_judge_matrix import NBOOT as NBOOT_JUDGE     # never retyped into the figure
S.apply()
R = "results"

MINUS = "−"
assert NBOOT_IPTM == NBOOT_JUDGE, "the two analyses used different bootstrap replicate counts"
NBOOT = NBOOT_IPTM


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
    """Smallest multiple of `step` that clears the widest CI end with a little headroom — ordinary
    axis logic, so each row's scale is set by its own data and not by a typed constant."""
    return float(step * np.ceil(hi * 1.02 / step - 1e-9))


# ================================================================== (a) data — three instrument runs
def summ(csv, metric, contrast="L_minus_random", agg="mean_over_k"):
    d = pd.read_csv(f"{R}/{csv}")
    r = d[(d["agg"] == agg) & (d.metric == metric) & (d.contrast == contrast)].iloc[0]
    s = 1.0 if bool(r.higher_better) else -1.0                       # sign-orient: >0 always = better
    lo, hi = sorted((s * float(r.lo), s * float(r.hi)))
    return dict(v=s * float(r.delta), lo=lo, hi=hi,
                p=float(r.p_gt0) if s > 0 else 1.0 - float(r.p_gt0),
                n=int(r.n), seed=int(r.seed))


ARMS = ["wt", "L", "random"]
INSTR = [
    dict(key="af2_mpnn",   folder="AF2-multimer", steer="ProteinMPNN", tag="", col=S.LEV,
         summ="iptm_summary_120.csv",   raw="iptm_steer_120.csv"),
    dict(key="boltz_mpnn", folder="Boltz-2",      steer="ProteinMPNN", tag="", col=S.LEV,
         summ="iptm_summary_boltz.csv", raw="iptm_steer_boltz.csv"),
    dict(key="af2_esmif",  folder="AF2-multimer", steer="ESM-IF1", tag=" (reverse)", col=S.ESMIF,
         summ="iptm_summary_esmif.csv", raw="iptm_steer_esmif.csv"),
]
for it in INSTR:
    it["iptm"] = summ(it["summ"], "iptm")
    it["comp"] = summ(it["summ"], "composite")
    it["lwt"] = summ(it["summ"], "iptm", "L_minus_wt")
    piv = (pd.read_csv(f"{R}/{it['raw']}").groupby(["complex_id", "direction"])["iptm"]
           .mean().unstack("direction")[ARMS].dropna())
    it["cx"] = set(piv.index)
    it["nwin"] = int((piv.L > piv["random"]).sum())
    it["ncx"] = len(piv)
    it["span"] = float(piv.wt.mean() - piv["random"].mean())         # the folder's OWN yardstick
    it["frac"] = it["iptm"]["v"] / it["span"]
    assert abs(float((piv.L - piv["random"]).mean()) - it["iptm"]["v"]) < 5e-4, \
        f"{it['key']}: raw per-fold CSV disagrees with the committed summary delta"
    # this row's own scale, ipTM units — sized to hold both the CI and the wild-type landmark
    it["sc"] = nice_max(max(it["iptm"]["hi"], it["span"]), 0.05)
    assert it["span"] / it["sc"] <= 1.0, f"{it['key']}: wild-type landmark falls outside the column"

# the two 60-complex runs are the SAME 60 complexes, nested in the 120 — so the AF2 / Boltz-2
# difference cannot be a complex-set effect, and we say so with the matched-subset number.
BZ, EF, A120 = INSTR[1], INSTR[2], INSTR[0]
SAME60 = BZ["cx"] == EF["cx"] and BZ["cx"] <= A120["cx"]
p120 = (pd.read_csv(f"{R}/{A120['raw']}").groupby(["complex_id", "direction"])["iptm"]
        .mean().unstack("direction")[ARMS].dropna())
p60 = p120.loc[sorted(BZ["cx"])]
D_AF2_ON60 = float((p60.L - p60["random"]).mean())
F_AF2_ON60 = D_AF2_ON60 / float(p60.wt.mean() - p60["random"].mean())
assert all(i["iptm"]["lo"] > 0 and i["comp"]["lo"] > 0 for i in INSTR), \
    "a panel-(a) interval touches zero — the figure's claim would be wrong"

# ================================================================== (b) data — the judge matrix
JCOL = {"ProteinMPNN": S.LEV, "ESM-IF1": S.ESMIF, "PiFold": S.GEOM, "MIF": S.CONS}
JUDGES = ["ProteinMPNN", "ESM-IF1", "PiFold", "MIF"]                 # fixed order -> the two blocks
JKEY = {"ProteinMPNN": "mpnn", "ESM-IF1": "esmif", "PiFold": "pifold", "MIF": "mif"}   # align as a matrix
JM = pd.concat([pd.read_csv(f"{R}/cfg_judge_matrix.csv"),
                pd.read_csv(f"{R}/cfg_judge_matrix_pifold.csv")], ignore_index=True)
METHOD, ALPHA = "dumped3", 2.0
JM = JM[(JM.method == METHOD) & (JM.alpha == ALPHA)]

STEERED = [("ProteinMPNN", "cfg_judge_dumped_setA"), ("ESM-IF1", "cfg_judge_dumped_setB")]
CELL = {}
for st, stem in STEERED:
    raw = pd.read_csv(f"{R}/{stem}.csv").merge(pd.read_csv(f"{R}/{stem}_pifold.csv"),
                                               on=["complex_id", "direction", "alpha"], how="outer")
    for j in JUDGES:
        r = JM[(JM.steered_model == st) & (JM.judge == j)].iloc[0]
        pv = (raw.pivot_table(index="complex_id", columns="direction", values=f"meanL_{JKEY[j]}")
                 .dropna(subset=["L", "random"]))
        d = pv.L - pv["random"]
        assert abs(float(d.mean()) - float(r.delta)) < 5e-4, f"{st}/{j}: raw disagrees with summary"
        CELL[(st, j)] = dict(v=float(r.delta), lo=float(r.lo), hi=float(r.hi), p=float(r.p_gt0),
                             n=int(r.n_cx), self=bool(r.is_self), nwin=int((d > 0).sum()),
                             seed=int(r.seed))
NONSELF = [CELL[(s, j)] for s, _ in STEERED for j in JUDGES if not CELL[(s, j)]["self"]]
assert all(c["lo"] > 0 for c in NONSELF), "an anti-circular judge cell touches zero"
JSC = nice_max(max(c["hi"] for c in NONSELF), 0.05)                  # one shared leverage scale
WORST_WIN = min(c["nwin"] / c["n"] for c in NONSELF)

# ================================================================== canvas
fig = plt.figure(figsize=(5.5, 5.15))
gA = fig.add_gridspec(1, 1, left=0.043, right=0.995, top=0.945, bottom=0.660)
gB = fig.add_gridspec(1, 1, left=0.043, right=0.995, top=0.570, bottom=0.150)

# columns, in axes units: label gutter | whisker (0..1) | value | companion | win rate.
# One unit of whisker = the ROW'S OWN scale in (a), so the same physical length means a different
# number of ipTM points on each row — the scale bar under each row says how many.
XL, VX, CX, WX, XV = -1.52, 1.10, 2.35, 3.60, 4.20


def whisk(ax, x, y, r, col, lw=1.25, ms=4.8, cap=0.135):
    ax.plot([x(r["lo"]), x(r["hi"])], [y, y], "-", color=col, lw=lw, solid_capstyle="butt", zorder=4)
    for e in ("lo", "hi"):
        ax.plot([x(r[e])] * 2, [y - cap, y + cap], "-", color=col, lw=lw, zorder=4)
    ax.plot(x(r["v"]), y, "o", ms=ms, color=col, mec="white", mew=0.9, zorder=5)


def colhead(ax, x, s, y, ha="left", size=5.6):
    ax.text(x, y, s, fontsize=size, color=S.SOFT, ha=ha, va="bottom", linespacing=1.5, zorder=6)


# ================================================================== (a) structure predictors
axa = fig.add_subplot(gA[0])
YA = {it["key"]: 1.45 * i for i, it in enumerate(INSTR)}
BARDY, YTOP, YEND = 0.53, -1.94, max(YA.values()) + 0.53
axa.set_xlim(XL, XV)
axa.set_ylim(YEND + 1.32, YTOP)
axa.set_axis_off()

axa.add_patch(Rectangle((CX - 0.13, -0.86), (WX - 0.22) - (CX - 0.13),
                        max(YA.values()) + 0.45 + 0.86,
                        facecolor=S.TINT, edgecolor="none", zorder=0))      # primary-endpoint column
axa.plot([0, 0], [-0.80, YEND + 0.06], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)  # ONE shared ref
axa.plot([XL, XV], [-0.86] * 2, "-", color=S.GHOST, lw=0.6, zorder=1)                   # header rule
colhead(axa, 0.0, "no effect\n(random arm)", -0.80, ha="center", size=5.5)
colhead(axa, INSTR[0]["span"] / INSTR[0]["sc"], "wild-type", -0.80, ha="center", size=5.3)
colhead(axa, VX + 0.10, "ipTM", -0.80, size=5.9)
colhead(axa, CX, "composite (z)\npre-registered endpoint", -0.80)
colhead(axa, WX, "L > random", -0.80)

for it in INSTR:
    y, sc = YA[it["key"]], it["sc"]
    x = lambda v, sc=sc: v / sc
    axa.text(XL, y - 0.20, it["folder"], fontsize=6.9, color=S.INK, fontweight="bold",
             ha="left", va="center", zorder=6)
    axa.text(XL, y + 0.23, sgn(f"steer {it['steer']}{it['tag']} · n = {it['ncx']}"),
             fontsize=5.75, color=it["col"], ha="left", va="center", zorder=6)
    # the wild-type ARM as a landmark on this row's own axis: zero = the random arm, this tick = the
    # crystal sequence. The dot then reads as "how far from random toward wild-type", which IS
    # comparable across folders — the three dots land at the same relative place on three scales.
    axa.plot([x(it["span"])] * 2, [y - 0.32, y + 0.32], ls=(0, (1.4, 1.4)), color=S.SCALAR,
             lw=1.0, zorder=3)
    whisk(axa, x, y, it["iptm"], it["col"], lw=1.4, ms=5.4)
    axa.text(VX, y, ci(it["iptm"], 3), fontsize=6.0, color=S.INK, ha="left", va="center", zorder=6)
    axa.text(CX, y, ci(it["comp"], 2), fontsize=6.0, color=S.INK, ha="left", va="center", zorder=6)
    axa.text(WX, y - 0.03, f"{it['nwin']} / {it['ncx']}", fontsize=6.0, color=S.INK,
             ha="left", va="bottom", zorder=6)
    axa.text(WX, y + 0.05, f"{it['nwin'] / it['ncx'] * 100:.0f}%", fontsize=5.5, color=S.MUTED,
             ha="left", va="top", zorder=6)
    # the scale bar: SAME physical length, DIFFERENT number of ipTM points. The rows therefore
    # cannot be compared by eye, and the bar — not only the caption — is what says so.
    by = y + BARDY
    axa.plot([0, 1], [by] * 2, "-", color=S.FLOOR_EDGE, lw=0.7, zorder=3)
    for e in (0.0, 1.0):
        axa.plot([e] * 2, [by - 0.085, by + 0.085], "-", color=S.FLOOR_EDGE, lw=0.7, zorder=3)
    axa.text(1.04, by, f"one bar = {sc:.2f} ipTM", fontsize=5.3, color=S.MUTED, ha="left",
             va="center", zorder=6)

axa.text(XL, YEND + 0.55, "each row is read in its OWN calibration — every interval clears zero, "
                          "but the three magnitudes are NOT comparable",
         fontsize=6.0, color=S.INK, ha="left", va="center", zorder=6)
axa.text(XL, YEND + 1.05, sgn(
    "on each row the tilt covers "
    + ",  ".join(f"{it['frac'] * 100:.0f}%" for it in INSTR)
    + " of the random → wild-type distance — the comparison the raw Δ cannot make"),
    fontsize=5.5, color=S.MUTED, ha="left", va="center", zorder=6)
S.header(axa, "a second folder, a second steered model, the same direction",
         sgn(f"paired Δ (L {MINUS} random) per complex, mean over k = 3 folds  ·  "
             f"complex-clustered 95% CI  ·  P(>0) = "
             f"{min(min(i['iptm']['p'], i['comp']['p']) for i in INSTR):.2f} on every row"))

# ================================================================== (b) the judge matrix
axb = fig.add_subplot(gB[0])
YB, y = {}, 0.0
for st, _ in STEERED:
    for j in JUDGES:
        YB[(st, j)] = y
        y += 1.0
    y += 1.55
SEPY = (YB[(STEERED[0][0], JUDGES[-1])] + YB[(STEERED[1][0], JUDGES[0])]) / 2.0
YBB, RULY = max(YB.values()), max(YB.values()) + 0.90
axb.set_xlim(XL, XV)
axb.set_ylim(RULY + 2.05, -1.42)
axb.set_axis_off()
xj = lambda v: v / JSC

axb.plot([0, 0], [-0.86, RULY], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)
axb.plot([XL, XV], [-0.86] * 2, "-", color=S.GHOST, lw=0.6, zorder=1)
axb.plot([XL, XV], [SEPY] * 2, "-", color=S.GHOST, lw=0.6, zorder=1)
colhead(axb, WX, "L > random", -0.80)

for st, _ in STEERED:
    y0 = YB[(st, JUDGES[0])]
    axb.text(XL, y0 - 0.78, f"steer {st}", fontsize=6.9, color=JCOL[st], fontweight="bold",
             ha="left", va="center", zorder=6)
    axb.text(0.06, y0 - 0.78, f"n = {CELL[(st, JUDGES[0])]['n']} complexes", fontsize=5.7,
             color=S.MUTED, ha="left", va="center", zorder=6)
    for j in JUDGES:
        yy, c = YB[(st, j)], CELL[(st, j)]
        if c["self"]:                       # the excluded diagonal, greyed out like a knocked-out cell
            axb.add_patch(Rectangle((XL, yy - 0.42), XV - XL, 0.84, facecolor=S.FLOOR_FILL,
                                    edgecolor="none", zorder=1))
            axb.text(XL, yy, j, fontsize=6.2, color=S.FLOOR_EDGE, ha="left", va="center", zorder=6)
            axb.text(0.06, yy, f"self-judge — circular, excluded   ({num(c['v'], 2)})", fontsize=5.6,
                     color=S.FLOOR_EDGE, ha="left", va="center", zorder=6)
            continue
        axb.text(XL, yy, j, fontsize=6.2, color=JCOL[j], ha="left", va="center", zorder=6)
        whisk(axb, xj, yy, c, JCOL[j], lw=1.25, ms=4.8)
        axb.text(VX, yy, ci(c, 3), fontsize=6.0, color=S.INK, ha="left", va="center", zorder=6)
        axb.text(WX, yy, f"{c['nwin']} / {c['n']}", fontsize=6.0, color=S.INK, ha="left",
                 va="center", zorder=6)

TICKS = list(np.arange(0.0, JSC + 1e-9, 0.25))
axb.plot([xj(TICKS[0]), xj(TICKS[-1])], [RULY] * 2, "-", color=S.RULE, lw=0.7, zorder=3)
for t in TICKS:
    axb.plot([xj(t)] * 2, [RULY, RULY + 0.16], "-", color=S.RULE, lw=0.7, zorder=3)
    axb.text(xj(t), RULY + 0.26, f"{t:.2f}", fontsize=5.4, color=S.MUTED, ha="center", va="top",
             zorder=6)
axb.text(xj(TICKS[-1]) + 0.07, RULY + 0.05, "leverage (nats)", fontsize=5.5, color=S.MUTED,
         ha="left", va="center", zorder=6)
axb.text(XL, RULY + 1.10, sgn(
    f"every non-self cell: L > random in ≥ {WORST_WIN * 100:.0f}% of complexes — four architectures, "
    f"both steering directions, none of them the steered model"),
    fontsize=6.0, color=S.INK, ha="left", va="center", zorder=6)
axb.text(XL, RULY + 1.66, "all six share ONE leverage scale — unlike a, these cells are one "
                          "instrument family, so their lengths do compare",
         fontsize=5.5, color=S.MUTED, ha="left", va="center", zorder=6)
S.header(axb, "and under four judge architectures — never the one that was steered",
         sgn(f"paired Δ judge-leverage (L {MINUS} random), α = {ALPHA:g}  ·  the self-judged diagonal "
             f"is excluded as circular  ·  P(>0) = {min(c['p'] for c in NONSELF):.2f} on all six"))

# ================================================================== footnote
SEEDS = sorted({i["iptm"]["seed"] for i in INSTR} | {c["seed"] for c in NONSELF})
NOTE = [
    f"a, paired Δ per complex, mean over k = 3 folds; complex-clustered bootstrap, {NBOOT:,} "
    f"replicates, seed {SEEDS[0]}. Composite = z-mean of",
    f"(ipTM, {MINUS}interface pAE, interface pLDDT) z-scored WITHIN each run (n = "
    + " / ".join(str(i["comp"]["n"]) for i in INSTR) + f"). The {BZ['ncx']} Boltz-2 and the "
    f"{EF['ncx']} reverse-direction complexes",
    f"are the SAME set, nested in the {A120['ncx']}; on those {BZ['ncx']}, AF2 / steer-ProteinMPNN "
    f"gives {num(D_AF2_ON60, 3)} ({F_AF2_ON60 * 100:.0f}% of its span), so the AF2 / Boltz-2 difference is",
    f"not a complex-set effect.   b, judge-leverage = the per-complex mean L over the sampled "
    f"interface residues; method {METHOD}, same bootstrap.",
]
fig.text(0.043, 0.118, sgn("\n".join(NOTE)), fontsize=5.35, color=S.MUTED, ha="left", va="top",
         linespacing=1.55)

S.flabel(fig, 0.043, 0.994, "a")
S.flabel(fig, 0.043, 0.619, "b")
S.save(fig, "fig_robustness")

# ------------------------------------------------------------------ provenance
print("[provenance] every mark is a literal cell of a committed CSV; "
      f"bootstrap replicates imported from src/analyse_iptm.py and src/cfg_judge_matrix.py (NBOOT={NBOOT})")
for it in INSTR:
    print(f"  (a) {it['folder']:12s} x steer-{it['steer']:11s} [{it['summ']}]  "
          f"ipTM Δ={it['iptm']['v']:+.4f} [{it['iptm']['lo']:+.4f},{it['iptm']['hi']:+.4f}] "
          f"P(>0)={it['iptm']['p']:.3f} n={it['iptm']['n']}  |  "
          f"composite Δ={it['comp']['v']:+.4f} [{it['comp']['lo']:+.4f},{it['comp']['hi']:+.4f}] "
          f"n={it['comp']['n']}")
    print(f"      {'':12s}   [{it['raw']}]  L>random {it['nwin']}/{it['ncx']} "
          f"({it['nwin'] / it['ncx'] * 100:.1f}%)  wt−random span={it['span']:.4f}  "
          f"recovered={it['frac'] * 100:.1f}%  own scale={it['sc']:.2f} ipTM  "
          f"L−wt={it['lwt']['v']:+.4f} [{it['lwt']['lo']:+.4f},{it['lwt']['hi']:+.4f}]")
print(f"      complex sets nested and the two 60-sets identical: {SAME60}; "
      f"AF2 restricted to those {BZ['ncx']}: Δ={D_AF2_ON60:+.4f} ({F_AF2_ON60 * 100:.1f}% of span)")
for st, _ in STEERED:
    for j in JUDGES:
        c = CELL[(st, j)]
        print(f"  (b) steer-{st:11s} judge-{j:11s} [{'SELF/circular' if c['self'] else 'anti-circular'}] "
              f"Δ={c['v']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] P(>0)={c['p']:.3f} "
              f"n={c['n']} L>random {c['nwin']}/{c['n']} ({c['nwin'] / c['n'] * 100:.1f}%)"
              + ("   [not plotted]" if c["self"] else ""))
print(f"      shared leverage scale = {JSC:.2f} nats (method={METHOD}, alpha={ALPHA:g}); "
      f"worst non-self win rate = {WORST_WIN * 100:.1f}%")

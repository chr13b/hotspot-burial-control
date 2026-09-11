#!/usr/bin/env python3
"""Figure 0 — experimental overview (schematic). One glance: WHERE binding lives in an inverse-folding model, the
two things we do with it, and the three independent modalities that confirm it.

  (1) The construct: one inverse-folding likelihood, two readings — confidence (a bound-state scalar, blind to
      binding) vs leverage L (the mixed 2nd derivative = ablate the partner = the CFG guidance direction ∝ −ΔΔG).
  (2) Two uses of L: DETECT (rank experimental ΔΔG) and STEER (+α·L on a frozen ProteinMPNN).
  (3) Three independent readouts agree anti-circularly: inverse-folding judges, structure predictors, physics —
      vs random (null) and naive (folds, doesn't bind) controls, on crystal AND predicted backbones.

Schematic only — asserts no data values (those are Figs. 1–P/L and the tables); it carries the logic, no numbers.

Layout: a three-column grid with constant gutters. Every box is SIZED FROM ITS OWN TEXT (see `_h`), so a box can
never be too small for what it holds, and each column is justified to a common floor — the arrows and the thesis
strip then hang off that same grid. Hue is model/role identity, never decoration: LEV = the construct L and its
path, SCALAR-grey = the blind scalar-of-P class, ESMIF/GEOM/CONS = the three independent readout families.

  python3 src/fig_overview.py  ->  results/figures/fig_overview.{pdf,png}
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import figstyle as S
S.apply()

H = 3.02                                                  # width is fixed at S.FIG_W; height is ours to choose
RECT = [0.012, 0.010, 0.976, 0.980]
fig = plt.figure(figsize=(S.FIG_W, H))
ax = fig.add_axes(RECT); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

XIN, YIN = RECT[2] * S.FIG_W, RECT[3] * H                 # inches spanned by one axes unit, x and y
PX, PY = 1 / (72 * XIN), 1 / (72 * YIN)                   # one typographic point, in axes x / y units
ASP = XIN / YIN                                           # keeps rounded corners ROUND, not squashed

# ---------------- the grid: three columns, constant gutters; every element snaps to it ----------------
COL = [(0.000, 0.300), (0.346, 0.278), (0.670, 0.330)]    # (x, width); gutters 0.046 carry the arrows
RULE_Y, BAND_TOP, FLOOR = 0.934, 0.822, 0.272             # header rule · top of the boxes · common bottom
STRIP_TOP = 0.212
TSZ, BSZ, LS, PAD = 7.0, 5.8, 1.32, 6.0                   # title pt · body pt · linespacing · box padding pt


def rbox(x, y, w, h, fc, ec, lw=1.1, ls="solid", r=0.013):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=ASP, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def _h(p):
    """Height this panel's own text needs, in axes-y units — so a box can never be too small for its content."""
    bs, n = p.get("bsize", BSZ), p["body"].count("\n") + 1
    h = 2 * p.get("pad", PAD) + (n - 1) * bs * LS + bs
    if p.get("title"):
        h += p.get("tsize", TSZ) + 3.4
    if p.get("tail"):
        h += 5.0 + p.get("tailsize", 6.0)
    return h * PY


def draw(p, x, w, ytop, h):
    bs, n = p.get("bsize", BSZ), p["body"].count("\n") + 1
    rbox(x, ytop - h, w, h, p["fc"], p["ec"], p.get("lw", 1.1), p.get("ls", "solid"))
    tx, y = x + 7.0 * PX, ytop - (h - _h(p)) / 2 - p.get("pad", PAD) * PY   # centre text in a justified box
    if p.get("title"):
        ax.text(tx, y, p["title"], fontsize=p.get("tsize", TSZ), color=p.get("tcol", S.INK),
                ha="left", va="top", fontweight="bold", zorder=5)
        y -= (p.get("tsize", TSZ) + 3.4) * PY
    ax.text(tx, y, p["body"], fontsize=bs, color=p.get("bcol", S.INK), ha="left", va="top",
            linespacing=LS, zorder=5)
    if p.get("tail"):
        y -= ((n - 1) * bs * LS + bs + 5.0) * PY
        ax.text(tx, y, p["tail"], fontsize=p.get("tailsize", 6.0), color=p.get("tailcol", S.MUTED),
                ha="left", va="top", fontweight=p.get("tailweight", "normal"), zorder=5)


def stack(specs, col, ytop, ybot, gmin=6.0, gmax=12.0):
    """Lay panels top->bottom so every column ends flush on the same floor. Returns each box's (top, bottom).
    Slack a column cannot spend on gaps goes into the boxes, so no column is ever short of room for its text."""
    x, w = col
    hs = [_h(p) for p in specs]
    gap = min((ytop - ybot - sum(hs)) / (len(specs) - 1), gmax * PY)
    grow = (ytop - ybot - sum(hs) - gap * (len(specs) - 1)) / len(specs)
    assert gap >= gmin * PY, f"[fig_overview] column over-stuffed: gap down to {gap / PY:.1f}pt"
    out, y = [], ytop
    for p, h in zip(specs, hs):
        draw(p, x, w, y, h + grow)
        out.append((y, y - h - grow))
        y -= h + grow + gap
    return out


def fan(xsrc, ysrc, col, targets, color=S.RULE, lw=1.2):
    """One trunk out of the source box, a riser standing in the gutter, one arrowhead into each target.
    The same idiom in both gutters, so 'L forks into two uses' and 'steering is judged three ways' read alike."""
    xb, ys = col[0] - 0.026, [(t[0] + t[1]) / 2 for t in targets]
    ax.plot([xsrc, xb], [ysrc, ysrc], "-", color=color, lw=lw, zorder=3, solid_capstyle="butt")
    ax.plot([xb, xb], [min(ys + [ysrc]), max(ys + [ysrc])], "-", color=color, lw=lw, zorder=3)
    for y in ys:
        ax.add_patch(FancyArrowPatch((xb, y), (col[0], y), arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=lw, zorder=4, shrinkA=0, shrinkB=0))


# ---------------- stage headers ----------------
for (x, w), n, s in zip(COL, "123", ["The construct", "Two uses of  L", "Three readouts agree"]):
    ax.text(x + 0.0152, 0.972, n, fontsize=7.4, color="white", ha="center", va="center", zorder=6,
            fontweight="bold", bbox=dict(boxstyle="circle,pad=0.30", fc=S.INK, ec="none"))
    ax.text(x + 0.041, 0.972, s, fontsize=7.4, color=S.INK, ha="left", va="center", fontweight="bold", zorder=5)
    ax.plot([x, x + w], [RULE_Y, RULE_Y], "-", color=S.RULE, lw=0.9, zorder=3, solid_capstyle="butt")

# ---------------- stage 1 — the construct ----------------
ax.add_patch(Ellipse((0.039, 0.876), 0.078, 0.072, fc=S.rgba(S.LEV, 0.15), ec=S.LEV, lw=1.1, zorder=3))
ax.add_patch(Ellipse((0.097, 0.891), 0.060, 0.056, fc=S.rgba(S.ESMIF, 0.17), ec=S.ESMIF, lw=1.1, zorder=4))
ax.text(0.037, 0.871, "receptor", fontsize=5.0, color=S.LEV, ha="center", va="center", zorder=5)
ax.text(0.099, 0.891, "binder", fontsize=5.0, color=S.ESMIF, ha="center", va="center", zorder=5)
ax.text(0.150, 0.899, "one likelihood", fontsize=6.0, color=S.INK, ha="left", va="center",
        fontweight="bold", zorder=5)
ax.text(0.150, 0.867, "$P(\\mathrm{seq}\\,|\\,\\mathrm{structure})$", fontsize=6.0, color=S.INK,
        ha="left", va="center", zorder=5)

CONF = dict(title="Confidence  φ(P)", tcol=S.MUTED, bcol=S.MUTED, fc=S.FLOOR_FILL, ec=S.SCALAR, lw=0.9,
            ls=(0, (2.6, 2.0)),                                    # dashed = the dead end: no path to binding
            body="any scalar of the bound-state\ndistribution — blind to binding\n"
                 "by construction  ($L \\perp φ$, proven)")
LEVG = dict(title="Leverage  L", tcol=S.LEV, fc=S.rgba(S.LEV, 0.09), ec=S.LEV, lw=1.3,
            body="$=$ mixed 2nd derivative\n(ablate the partner) — the model's\nCFG guidance direction",
            tail="∝  −ΔΔG$_{bind}$   (binding free energy)", tailcol=S.LEV, tailweight="bold")

# ---------------- stage 2 — two uses ----------------
DET = dict(title="Detect", fc=S.TINT, ec=S.FLOOR_EDGE, lw=1.0,     # structural grey: SCALAR-grey is reserved
           body="rank experimental ΔΔG\n(SKEMPI n=2,948; AB-Bind);\nzero-shot, competitive with\n"
                "and beyond FoldX physics")                        # for the blind scalar-of-P class alone
STEER = dict(title="Steer", fc=S.TINT, ec=S.FLOOR_EDGE, lw=1.0,
             body="$+\\,α\\,L$ on a frozen,\noff-the-shelf ProteinMPNN\n→ new interface sequences",
             tail="controls:  random · naive", tailsize=5.4)

# ---------------- stage 3 — readouts ----------------
RD = [(S.ESMIF, "Inverse-folding judges", "ESM-IF1 · MIF · PiFold — leverage ↑"),
      (S.GEOM, "Structure predictors", "AF2-multimer · Boltz-2 — ipTM ↑"),
      (S.CONS, "Physics energy function", "FoldX ΔΔG$_{bind}$ — more favorable ↑")]
READ = [dict(title=t, tcol=c, tsize=6.5, bsize=5.6, body=b, fc=S.rgba(c, 0.09), ec=c, lw=1.15, pad=5.0)
        for c, t, b in RD]
AGREE = dict(fc=S.FLOOR_FILL, ec=S.FLOOR_EDGE, lw=1.0, bsize=5.4, pad=5.0,
             body="steered L beats  random  and  naive\n(folds but does not bind) on all three")

c1 = stack([CONF, LEVG], COL[0], BAND_TOP, FLOOR)
c2 = stack([DET, STEER], COL[1], BAND_TOP, FLOOR)
c3 = stack(READ + [AGREE], COL[2], BAND_TOP, FLOOR)

# ---------------- connectors: L forks into its two uses, steering fans out to all three readouts ----------------
mid = lambda b: (b[0] + b[1]) / 2
fan(COL[0][0] + COL[0][1], mid(c1[1]), COL[1], [c2[0], c2[1]], color=S.LEV)     # L  -> Detect, Steer
fan(COL[1][0] + COL[1][1], mid(c2[1]), COL[2], c3[:3])                         # Steer -> the three readouts

# ---------------- thesis strip ----------------
SH = (2 * 6.0 + 7.3 + 5.0 + 5.5 * 1.38 + 5.5) * PY
rbox(0.0, STRIP_TOP - SH, 1.0, SH, S.rgba(S.LEV, 0.07), S.LEV, lw=1.0, r=0.014)
ax.text(0.5, STRIP_TOP - 6.0 * PY, "Confidence is not competence — and it recurs at every level.",
        fontsize=7.3, color=S.LEV, ha="center", va="top", fontweight="bold", zorder=5)
ax.text(0.5, STRIP_TOP - (6.0 + 7.3 + 5.0) * PY,
        "an inverse-folding model's confidence and a structure predictor's ipTM are both blind to binding;\n"
        "the competence always lives in the response to ablating the partner.",
        fontsize=5.5, color=S.INK, ha="center", va="top", linespacing=1.38, zorder=5)

S.save(fig, "fig_overview")

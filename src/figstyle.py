"""Shared figure design system for the ICLR figure set. One place so all figures stay one coherent system.

A HUE MEANS EXACTLY ONE THING ACROSS THE WHOLE PAPER. Two disjoint groups, never crossed:
  MODEL identities  LEV (ProteinMPNN / the construct L) · ESMIF · PIFOLD · MIF
  CONCEPT hues      GEOM (geometry: burial / ΔSASA / contacts) · CONS (conservation / substitution similarity)
                    PHYS (a fitted physics energy function: the FoldX ΔΔG_bind rung)
PIFOLD and MIF exist because GEOM and CONS used to double as the PiFold and MIF judge identities, which made
teal mean "geometry" in one figure and "PiFold" in another. PHYS exists for the same reason one level down:
fig_ladder's "fitted physics (FoldX)" rung used to wear LEV, so the blue that means ProteinMPNN everywhere else
meant "FoldX" there. Anything that is neither a model nor one of those concepts takes a grey: SCALAR for the
scalar-of-P class, FLOOR_*/GHOST/TINT for structure.

Palette validation (dataviz scripts/validate_palette.js, Machado-Oliveira-Fernandes 2009 at severity 1.0):
  the four MODEL hues, ALL-PAIRS (any two models can sit side by side in a judge list) --
    node validate_palette.js "#0B6FA4,#C0561F,#CA007F,#996BFE" --pairs all --mode light  -> 5/5 PASS
    node validate_palette.js "#0B6FA4,#C0561F,#CA007F,#996BFE" --pairs all --mode dark   -> 5/5 PASS
      worst CVD ΔE 10.0 (#CA007F↔#0B6FA4, protan) · worst normal ΔE 18.3 (#CA007F↔#C0561F)
  the documented 7-slot order, ADJACENT --
    node validate_palette.js "#0B6FA4,#C0561F,#CA007F,#996BFE,#1B9E77,#6D4E9C,#6A6E00" --mode light -> 5/5 PASS
    node validate_palette.js "#0B6FA4,#C0561F,#CA007F,#996BFE,#1B9E77,#6D4E9C,#6A6E00" --mode dark  -> PASS,
      worst adjacent CVD ΔE 12.9 and normal ΔE 18.3 (both PIFOLD↔ESMIF) in either mode, with one
      pre-existing WARN (CONS 2.67:1 on the dark surface) that predates PHYS and is relieved the way the
      house style already requires: every plotted value is direct-labelled. PHYS itself clears contrast in
      both modes (5.32:1 light, 3.19:1 dark) and adds no new WARN.
  fig_ladder's own chromatic rungs, ALL-PAIRS (they are four bars in one chart, so any two are neighbours) --
    node validate_palette.js "#1B9E77,#6D4E9C,#6A6E00" --pairs all --mode light -> 5/5 PASS
    node validate_palette.js "#1B9E77,#6D4E9C,#6A6E00" --pairs all --mode dark  -> PASS (same CONS WARN)
      worst CVD ΔE 13.0 · worst normal ΔE 15.2 (both PHYS↔GEOM). PHYS↔SCALAR, the fourth rung: 17.6 / 18.6.
  CONCEPT hues are additionally >= 15 normal-vision ΔE from every MODEL hue, so nothing reads as "the teal
  one" across figures even though they never share a chart. PHYS vs each shipped hue (CVD / normal):
    LEV 19.9/20.8 · PIFOLD 12.0/30.4 · MIF 31.1/34.9 · GEOM 13.0/15.2 · CONS 21.0/23.9 · SCALAR 17.6/18.6
    ESMIF 1.1/16.4 -- the one soft spot, and it is cross-figure only: a deuteranope sees dark ochre and
      burnt orange converge, but PHYS appears ONLY in fig_ladder and ESMIF ONLY in fig_predicted_steer and
      fig_robustness, so no chart ever asks a reader to tell them apart; every rung is direct-labelled.
  Why PHYS is an ochre and not something livelier: with six hues already placed, an exhaustive step-1 sweep
  of all 16.7M sRGB colors finds ZERO that clear CVD ΔE >= 8 AND normal ΔE >= 15 against all six while
  staying inside both lightness bands, over the chroma floor and over 3:1 on both surfaces (best is
  normal 14.7). Seven hues is past the all-pairs series cap the dataviz skill documents, which is also why
  the shipped set has always validated all-pairs over the four MODEL hues only -- LEV↔CONS (CVD 2.2,
  normal 12.2), PIFOLD↔GEOM (6.9) and PIFOLD↔CONS (5.0) never co-occur and never have. Under the operative
  pairlist the only unclaimed families were ochre, a second blue (reads as LEV) and a pink (reads as
  PIFOLD); ochre is the one that collides with no other hue at normal vision.

Verified house format (ICLR 2026 sty: \textwidth 5.5 true in; NeurIPS demands embedded fonts -> fonttype 42).
"""
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.colors import to_rgba

INK, RULE, MUTED, SOFT = "#1A1A1A", "#4D4D4D", "#6B7379", "#333333"
SCALAR = "#8A9299"                                     # ONE grey for the scalar-of-P class (they are one class)
LEV, ESMIF, PIFOLD, MIF = "#0B6FA4", "#C0561F", "#CA007F", "#996BFE"   # MODEL identities, in palette order
GEOM, CONS = "#1B9E77", "#6D4E9C"                      # CONCEPT hues — geometry, conservation. NEVER a model.
PHYS = "#6A6E00"                                       # CONCEPT hue — a fitted physics energy (FoldX). NEVER a model.
PALETTE = [LEV, ESMIF, PIFOLD, MIF, GEOM, CONS, PHYS]  # the documented slot order the validation above uses
MODEL_HUES = {"ProteinMPNN": LEV, "ESM-IF1": ESMIF, "PiFold": PIFOLD, "MIF": MIF}
FLOOR_FILL, FLOOR_EDGE, GHOST, TINT = "#E4E7E9", "#B4BBC0", "#C8CDD1", "#F4F6F7"
RAMP_MPNN = ["#D6E7F0", "#A9CBDF", "#7BAECC", "#4B90B9", "#0B6FA4", "#08526F"]   # ordinal ramp (never categorical)
FIG_W = 5.5                                            # ICLR single-column text width (exact)

# one name per quantity (never type an axis label inline)
CPI_POS = "CPI beyond geometry  ($\\times10^{-3}$, position level)"
CPI_MUT = "CPI beyond geometry  (mutation level)"


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "savefig.dpi": 600,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.labelcolor": INK, "text.color": INK,
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "patch.linewidth": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderpad": 0.2,
    })


def strip(ax, left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    for s in ("left", "bottom"):
        if ax.spines[s].get_visible():
            ax.spines[s].set_color(RULE)
    ax.tick_params(colors=RULE, labelcolor=INK, length=2.5)


def header(ax, title, note=None, tsize=8.5):
    """Stack the finding-title (and optional grey n-note) in POINT space so they never collide."""
    if note:
        ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2.5),
                    textcoords="offset points", fontsize=6.5, color=MUTED, ha="left", va="bottom")
        tdy = 12.0
    else:
        tdy = 3.0
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, tdy),
                textcoords="offset points", fontsize=tsize, color=INK, ha="left", va="bottom")


def flabel(fig, gs_left, y, ch):
    """Panel letter at the panel's full left extent (not the axes edge), consistent inch offset."""
    fig.text(gs_left - 0.03, y, ch, fontsize=9.5, fontweight="bold", color=INK, ha="left", va="top")


def assert_in_view(ax, xs, axis="x"):
    lo, hi = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())   # sorted: reversed axes too
    bad = [round(x, 6) for x in xs if x is not None and not (lo <= x <= hi)]
    assert not bad, f"[figstyle] clipped {axis}-data {bad} outside {(round(lo,5), round(hi,5))}"


PAD = 0.02
FALLBACK = "DejaVu Sans"      # what a machine with no Helvetica / Arial / Nimbus Sans falls back to


def _tight_w(fig):
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD


def _fallback_w(fig):
    """The tight width this same figure would have under the metric-WIDEST fallback font.

    matplotlib resolves a Text's family when the Text is built, so flipping rcParams here would be
    silently ignored; each Text is therefore re-familied in place and put back afterwards. The file
    that gets written is unaffected — this is a measurement, not a restyle."""
    import matplotlib.text as mtext
    texts = list(fig.findobj(mtext.Text))
    old = [t.get_fontfamily() for t in texts]
    try:
        for t in texts:
            t.set_fontfamily([FALLBACK])
        return _tight_w(fig)
    finally:
        for t, o in zip(texts, old):
            t.set_fontfamily(o)
        fig.canvas.draw()


def save(fig, stem):
    """Write PDF + PNG at true final width. bbox_inches='tight' silently GROWS the saved file when
    a title/note overhangs the axes, which is how a '5.5in' figure ships at 6.25in and gets scaled
    down by \\includegraphics — the exact failure this house style exists to prevent. So measure the
    tight bbox and refuse to write when it overflows: shorten the overhanging text instead.

    The same measurement is repeated under the FALLBACK font, because the width guard is only worth
    anything on the machine that renders the camera-ready: a figure that fits in Nimbus Sans here can
    overflow in DejaVu Sans there, and that overflow is invisible until the page is typeset."""
    import os
    w = fig.get_size_inches()[0]
    assert w <= FIG_W + 0.02, f"[figstyle] figure width {w:.2f}in exceeds {FIG_W}in"
    saved = _tight_w(fig)
    assert saved <= FIG_W + 0.01, (
        f"[figstyle] saved width {saved:.3f}in overflows {FIG_W}in by {saved - FIG_W:+.3f}in — "
        f"an in-plot title/note/label overhangs the figure. Shorten it or widen its panel; "
        f"never render wide and scale down.")
    fb = _fallback_w(fig)
    assert fb <= FIG_W + 0.01, (
        f"[figstyle] under the {FALLBACK} fallback the saved width is {fb:.3f}in, {fb - FIG_W:+.3f}in "
        f"over {FIG_W}in (it fits at {saved:.3f}in in the font installed here). Shorten the longest "
        f"note/label or widen its gutter — the camera-ready machine may have no Helvetica or Nimbus.")
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{stem}.pdf", bbox_inches="tight", pad_inches=PAD)
    fig.savefig(f"results/figures/{stem}.png", bbox_inches="tight", pad_inches=PAD, dpi=200)
    print(f"wrote results/figures/{stem}.{{pdf,png}}  ({saved:.2f} × "
          f"{fig.get_tightbbox(fig.canvas.get_renderer()).height + 2 * PAD:.2f} in"
          f"; {FALLBACK} fallback {fb:.2f} in)")


rgba = to_rgba

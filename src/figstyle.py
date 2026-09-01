"""Shared figure design system for the ICLR figure set. One place so all figures stay one coherent system.
Palette validated: node validate_palette.js "#0B6FA4,#C0561F,#1B9E77,#6D4E9C" --mode light -> all 6 checks PASS.
Verified house format (ICLR 2026 sty: \textwidth 5.5 true in; NeurIPS demands embedded fonts -> fonttype 42).
"""
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.colors import to_rgba

INK, RULE, MUTED, SOFT = "#1A1A1A", "#4D4D4D", "#6B7379", "#333333"
SCALAR = "#8A9299"                                     # ONE grey for the scalar-of-P class (they are one class)
LEV, ESMIF, GEOM, CONS = "#0B6FA4", "#C0561F", "#1B9E77", "#6D4E9C"   # a hue is a MODEL identity or nothing
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
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    bad = [round(x, 6) for x in xs if x is not None and not (lo <= x <= hi)]
    assert not bad, f"[figstyle] clipped {axis}-data {bad} outside {(round(lo,5), round(hi,5))}"


PAD = 0.02


def save(fig, stem):
    """Write PDF + PNG at true final width. bbox_inches='tight' silently GROWS the saved file when
    a title/note overhangs the axes, which is how a '5.5in' figure ships at 6.25in and gets scaled
    down by \\includegraphics — the exact failure this house style exists to prevent. So measure the
    tight bbox and refuse to write when it overflows: shorten the overhanging text instead."""
    import os
    w = fig.get_size_inches()[0]
    assert w <= FIG_W + 0.02, f"[figstyle] figure width {w:.2f}in exceeds {FIG_W}in"
    fig.canvas.draw()
    saved = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD
    assert saved <= FIG_W + 0.01, (
        f"[figstyle] saved width {saved:.3f}in overflows {FIG_W}in by {saved - FIG_W:+.3f}in — "
        f"an in-plot title/note/label overhangs the figure. Shorten it or widen its panel; "
        f"never render wide and scale down.")
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{stem}.pdf", bbox_inches="tight", pad_inches=PAD)
    fig.savefig(f"results/figures/{stem}.png", bbox_inches="tight", pad_inches=PAD, dpi=200)
    print(f"wrote results/figures/{stem}.{{pdf,png}}  ({saved:.2f} × "
          f"{fig.get_tightbbox(fig.canvas.get_renderer()).height + 2 * PAD:.2f} in)")


rgba = to_rgba

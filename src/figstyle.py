"""Shared figure design system for the ICLR figure set (validated palette + rcParams + helpers).
One place to apply the figure-design guidance so all figures stay one coherent system.
Palette validated: node validate_palette.js "#0B6FA4,#C0561F,#1B9E77,#6D4E9C" --mode light -> all 6 checks PASS.
"""
import matplotlib as mpl
mpl.use("Agg")

INK, RULE, MUTED = "#1A1A1A", "#4D4D4D", "#6B7379"
CONF, NEGENT, KL = "#9AA3AA", "#7E888F", "#636D74"     # scalars-of-P ladder: greyer = more inert = the thesis
LEV, ESMIF, GEOM, CONS = "#0B6FA4", "#C0561F", "#1B9E77", "#6D4E9C"
FLOOR_FILL, FLOOR_EDGE, GHOST = "#E4E7E9", "#B4BBC0", "#C8CDD1"
# ProteinMPNN dose ramp (open at crystal -> saturated at 2 A)
RAMP_MPNN = ["#D6E7F0", "#A9CBDF", "#7BAECC", "#4B90B9", "#0B6FA4", "#08526F"]


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "figure.dpi": 150,
        "axes.labelcolor": INK, "text.color": INK,
    })


def strip(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=RULE, labelcolor=INK, length=2.5)


def title(ax, t, size=8.5):
    ax.set_title(t, loc="left", fontsize=size, color=INK, pad=6)


def subn(ax, s):
    ax.text(0, 1.008, s, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")


def letter(ax, ch):
    ax.text(-0.02, 1.06, ch, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, ha="right", va="bottom")

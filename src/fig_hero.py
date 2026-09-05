#!/usr/bin/env python3
"""Hero figure — what the CFG-steered arm buys, on ONE complex, in structure space.

ILLUSTRATIVE. Complex 1GL0_E_I was pre-registered as the MEDIAN-effect complex of the ipTM run: its
arm-mean ipTM delta (L - random) is +0.197, rank 31 of 60, the upper of the two complexes straddling
the +0.183 cohort median (both computed live below). The fold shown for each arm is the k whose ipTM
is CLOSEST to that arm's mean — a representative fold, never best-of-k. The effect size is the cohort
statistic quoted in the footnote, NOT this one picture.

Three panels, wt / L-steered / random-matched, each the Ca backbone trace of the AF2-multimer rank-1
model coloured by per-residue pLDDT (the B-factor column of the committed PDB). The three folds are
superimposed on the wt RECEPTOR chain (chain A = group1 = crystal chain E) and projected with ONE
basis — PC1/PC2 of the wt Ca cloud — so the panels are directly comparable; the binder chain and the
interface residues therefore sit in the same place in all three panels, and the only thing that moves
is the colour. Interface residues (the committed ddSASA interface set, the same 43 positions the
annotated interface pLDDT averages over) carry a larger marker.

Every number is read live:
  results/hero_pdbs/1GL0_E_I_meta.csv     per-arm ipTM, interface pLDDT, arm-mean ipTM, k choice
  results/hero_pdbs/1GL0_E_I__{wt,L,random}.pdb   coordinates + per-residue pLDDT (B-factor)
  results/iptm_steer.csv                  cross-check of every meta value; cohort median delta
  results/leverage_skempi_positions.csv   the interface residue set (is_interface, dSASA > 0.05)

  python3 src/fig_hero.py  ->  results/figures/fig_hero.{pdf,png}
"""
from decimal import Decimal, ROUND_HALF_UP
import textwrap
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import biotite.structure.io.pdb as biopdb
from scipy.spatial import cKDTree
import figstyle as S
S.apply()
R = "results"
CID = "1GL0_E_I"
HERO = f"{R}/hero_pdbs"
ARMS = ["wt", "L", "random"]
# ColabFold chain -> the crystal chain of the complex id (pdb_group1_group2), recorded at fold time
CHAIN2XTAL = {"A": CID.split("_")[1], "B": CID.split("_")[2]}
IFACE_CUT = 5.0                                   # A, heavy-atom, for the cross-check only

MINUS = "−"


def num(v, dp, plus=False):
    """Fixed-point, rounded HALF-UP, real minus sign — so the figure agrees with the text."""
    q = Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    return (f"{q:+.{dp}f}" if plus else f"{q:.{dp}f}").replace("-", MINUS)


# ================================================================== structures
def load(arm):
    """Heavy atoms of the rank-1 model, with the pLDDT that ColabFold wrote to the B-factor column."""
    a = biopdb.PDBFile.read(f"{HERO}/{CID}__{arm}.pdb").get_structure(model=1,
                                                                     extra_fields=["b_factor"])
    return a[a.element != "H"]


def kabsch(P, Q):
    """Rotation+translation taking P onto Q (least-squares, reflection-free)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rt = Vt.T @ D @ U.T
    return Rt, qc - Rt @ pc


ATOM = {a: load(a) for a in ARMS}
CA = {a: ATOM[a][ATOM[a].atom_name == "CA"] for a in ARMS}
REF = CA["wt"]
assert all(len(CA[a]) == len(REF) for a in ARMS), "arms differ in residue count — not superimposable"
assert all((CA[a].chain_id == REF.chain_id).all() and (CA[a].res_id == REF.res_id).all()
           for a in ARMS), "arms differ in chain/residue order"
RECEPTOR = REF.chain_id == "A"                    # the 241-residue partner; the frame we hold fixed

# superimpose every arm on the wt RECEPTOR chain, so a change in the binder's pose stays visible
# instead of being absorbed into the fit
FIT, RMSD = {}, {}
for a in ARMS:
    Rt, t = kabsch(CA[a].coord[RECEPTOR], REF.coord[RECEPTOR])
    FIT[a] = (Rt @ CA[a].coord.T).T + t
    d = FIT[a] - REF.coord
    RMSD[a] = dict(all=float(np.sqrt((d ** 2).sum(1).mean())),
                   rec=float(np.sqrt((d[RECEPTOR] ** 2).sum(1).mean())),
                   bnd=float(np.sqrt((d[~RECEPTOR] ** 2).sum(1).mean())))

# ONE projection basis: PC1/PC2 of the wt Ca cloud, applied unchanged to all three arms
CEN = REF.coord.mean(0)
_, SV, VT = np.linalg.svd(REF.coord - CEN, full_matrices=False)
BASIS = VT[:2]
VAR2 = float((SV[:2] ** 2).sum() / (SV ** 2).sum())
XY = {a: (FIT[a] - CEN) @ BASIS.T for a in ARMS}
PLDDT = {a: np.asarray(CA[a].b_factor, float) for a in ARMS}
# how much of the binding axis survives the projection (a view is only useful if the interface does)
AX3 = REF.coord[RECEPTOR].mean(0) - REF.coord[~RECEPTOR].mean(0)
AX2 = (REF.coord[RECEPTOR].mean(0) - CEN) @ BASIS.T - (REF.coord[~RECEPTOR].mean(0) - CEN) @ BASIS.T
AXKEEP = float(np.linalg.norm(AX2) / np.linalg.norm(AX3))

# ================================================================== the interface set (committed CSV)
# The complex is folded in crystal-complex residue order (group1 then group2) and ColabFold renumbers
# each chain from 1, so fold (chain A, res i) == crystal (chain E, resnum i) here; asserted below.
pos = pd.read_csv(f"{R}/leverage_skempi_positions.csv", low_memory=False)
pos = pos[(pos.complex_id == CID) & (pos.is_interface == True)]                       # noqa: E712
IFACE_KEYS = {(c, int(r)) for c, r in zip(pos.chain, pos.resnum)}
KEY = np.array([(CHAIN2XTAL[c], int(r)) for c, r in zip(REF.chain_id, REF.res_id)], dtype=object)
IS_IFACE = np.array([(k[0], k[1]) in IFACE_KEYS for k in KEY])
assert IS_IFACE.sum() == len(IFACE_KEYS) == len(pos), (
    f"interface set does not map onto the fold: {IS_IFACE.sum()} of {len(pos)} matched")

# independent cross-check of that set: heavy-atom contacts in the wt fold itself
wa, wb = ATOM["wt"][ATOM["wt"].chain_id == "A"], ATOM["wt"][ATOM["wt"].chain_id == "B"]
hits = cKDTree(wa.coord).query_ball_tree(cKDTree(wb.coord), r=IFACE_CUT)
CONTACT = {(CHAIN2XTAL["A"], int(wa.res_id[i])) for i, js in enumerate(hits) if js} | \
          {(CHAIN2XTAL["B"], int(wb.res_id[j])) for js in hits for j in js}
N_AGREE = len(CONTACT & IFACE_KEYS)

# ================================================================== the numbers (committed CSVs)
meta = pd.read_csv(f"{HERO}/{CID}_meta.csv").set_index("arm")
raw = pd.read_csv(f"{R}/iptm_steer.csv")
sub = raw[raw.complex_id == CID]
for a in ARMS:                                    # every annotated value must be a cell of iptm_steer
    m, r = meta.loc[a], sub[(sub.direction == a) & (sub.k == meta.loc[a].k)].iloc[0]
    assert abs(float(m.iptm) - float(r.iptm)) < 5e-4, f"{a}: meta ipTM != iptm_steer.csv"
    assert abs(float(m.interface_plddt) - float(r.interface_plddt)) < 5e-4, f"{a}: meta pLDDT != CSV"
    assert abs(float(m.arm_mean_iptm) - float(sub[sub.direction == a].iptm.mean())) < 5e-4, \
        f"{a}: meta arm mean != mean over k of iptm_steer.csv"
IPTM = {a: float(meta.loc[a].iptm) for a in ARMS}
IPL = {a: float(meta.loc[a].interface_plddt) for a in ARMS}
# global pTM is not in meta.csv; take it from the same iptm_steer.csv row meta selects. It is the
# localization control: the claim is that the INTERFACE loses confidence, not the whole fold.
PTM = {a: float(sub[(sub.direction == a) & (sub.k == meta.loc[a].k)].iloc[0].ptm) for a in ARMS}
AMEAN = {a: float(meta.loc[a].arm_mean_iptm) for a in ARMS}
D_CX = AMEAN["L"] - AMEAN["random"]               # this complex's effect
piv = raw.groupby(["complex_id", "direction"]).iptm.mean().unstack("direction")[ARMS].dropna()
DELTA = (piv.L - piv["random"]).sort_values()
D_MED = float(DELTA.median())                     # the cohort effect the figure must NOT overstate
N_CX = len(DELTA)
RANK = int(DELTA.index.get_loc(CID)) + 1          # where this complex sits in the cohort (1 = worst)
assert sub.n_iface.nunique() == 1 and int(sub.n_iface.iloc[0]) == len(IFACE_KEYS), \
    "the annotated interface pLDDT averages over a different residue set than the one marked"
for a in ARMS:      # the marked residues must REPRODUCE the annotated number, not merely resemble it
    assert abs(float(PLDDT[a][IS_IFACE].mean()) - IPL[a]) < 1e-3, (
        f"{a}: mean pLDDT over the marked residues ({PLDDT[a][IS_IFACE].mean():.4f}) != the "
        f"annotated interface pLDDT ({IPL[a]:.4f}) — the marks and the number disagree")

# ================================================================== canvas
# The three panels share ONE set of limits as well as one basis, so a residue sits at the same place
# in all three and only its colour changes.
LO = np.stack([XY[a].min(0) for a in ARMS]).min(0)
HI = np.stack([XY[a].max(0) for a in ARMS]).max(0)
PADXY = 0.055 * (HI - LO).max()
LO, HI = LO - PADXY, HI + PADXY
ASPECT = (HI[1] - LO[1]) / (HI[0] - LO[0])

L0, R0, GAP = 0.020, 0.988, 0.014
PW = ((R0 - L0) - 2 * GAP) / 3.0                                   # panel width, figure fraction
PH_IN = PW * S.FIG_W * ASPECT                                      # equal aspect -> panel height, in

# vertical budget in INCHES from the top, so nothing has to be re-tuned when the panel aspect moves
NB = "\u00a0"                                      # wrap-proof space: a number never leaves its unit
FOOT = (f"Cα trace of the AF2-multimer rank-1 model, coloured by the pLDDT ColabFold wrote to the "
        f"B-factor column; the larger marks are the {int(IS_IFACE.sum())} interface positions that "
        f"the annotated interface pLDDT averages over. Each arm is the fold whose ipTM is closest "
        f"to that arm's mean — representative, not best-of-k. The three are superimposed on the wt "
        f"receptor chain (Cα RMSD {num(RMSD['L']['all'], 2)} and {num(RMSD['random']['all'], 2)}{NB}"
        f"Å for L and random: they differ in confidence, not in pose) and projected on ONE basis, "
        f"PC1–PC2 of the wt Cα cloud. The effect size is the cohort statistic, not this picture: "
        f"arm-mean ipTM {num(AMEAN['L'], 3)}{NB}(L) vs {num(AMEAN['random'], 3)}{NB}(random) here, "
        f"Δ{NB}{num(D_CX, 3)}, against a median Δ of {num(D_MED, 3)} over {N_CX} complexes.")
FOOT = "\n".join(textwrap.wrap(FOOT, 124)).replace(NB, " ")
Y_TITLE, Y_NOTE = 0.030, 0.205                                     # va="top"
Y_NAME, Y_SUB = 0.460, 0.590                                       # baselines of the arm captions
Y_PANEL = 0.655
Y_IPTM = Y_PANEL + PH_IN + 0.055                                   # va="top"
Y_IPL = Y_IPTM + 0.125
Y_PTM = Y_IPL + 0.108
Y_CBLAB = Y_PTM + 0.285                                            # baseline
Y_CB = Y_CBLAB + 0.045
CB_H = 0.048
Y_FOOT = Y_CB + CB_H + 0.180                                       # va="top"; ticks live in the gap
FIG_H = Y_FOOT + 0.117 * (FOOT.count("\n") + 1) + 0.035


def yf(inches, h=0.0):
    """Figure fraction of a box `h` inches tall whose TOP is `inches` below the figure top."""
    return 1.0 - (inches + h) / FIG_H


fig = plt.figure(figsize=(S.FIG_W, FIG_H))

CMAP = LinearSegmentedColormap.from_list("plddt", S.RAMP_MPNN)     # house ordinal ramp: dark = high
NORM = Normalize(vmin=float(np.floor(min(P.min() for P in PLDDT.values()))), vmax=100.0)
LWC = {"A": 0.95, "B": 2.2}                                        # width marks the chain, not a value
TITLES = {"wt": "wt", "L": "L-steered", "random": "random-matched"}
SUBS = {"wt": "crystal sequence", "L": "leverage direction", "random": "matched-magnitude control"}
halo = [pe.withStroke(linewidth=2.2, foreground="white")]

# the binder label goes OUTWARD along the projected receptor->binder axis, so it never lands on the
# receptor whatever the projection turns out to be
CB2 = XY["wt"][~RECEPTOR].mean(0)
UAX = CB2 - XY["wt"][RECEPTOR].mean(0)
UAX = UAX / np.linalg.norm(UAX)

for i, a in enumerate(ARMS):
    left = L0 + i * (PW + GAP)
    ax = fig.add_axes([left, yf(Y_PANEL, PH_IN), PW, PH_IN / FIG_H])
    ax.set_xlim(LO[0], HI[0])
    ax.set_ylim(LO[1], HI[1])
    ax.set_aspect("equal")
    ax.set_axis_off()

    for ch in ("A", "B"):
        m = REF.chain_id == ch
        p, b = XY[a][m], PLDDT[a][m]
        seg = np.stack([p[:-1], p[1:]], axis=1)
        ax.add_collection(LineCollection(seg, colors=S.FLOOR_FILL, lw=LWC[ch] + 1.2,  # a faint grey
                                         capstyle="round", zorder=2))                 # trace so the
        lc = LineCollection(seg, cmap=CMAP, norm=NORM, lw=LWC[ch],                    # pale (= low
                            capstyle="round", zorder=3)                               # pLDDT) chain
        lc.set_array(0.5 * (b[:-1] + b[1:]))                                          # stays legible
        ax.add_collection(lc)

    m = IS_IFACE
    ax.scatter(XY[a][m][:, 0], XY[a][m][:, 1], c=PLDDT[a][m], cmap=CMAP, norm=NORM,
               s=9.5, linewidths=0.4, edgecolors="white", zorder=5)

    xc = left + PW / 2.0
    fig.text(xc, yf(Y_NAME), TITLES[a], fontsize=7.4, color=S.INK, fontweight="bold",
             ha="center", va="baseline")
    fig.text(xc, yf(Y_SUB), SUBS[a], fontsize=5.8, color=S.MUTED, ha="center", va="baseline")
    fig.text(xc, yf(Y_IPTM), f"ipTM {num(IPTM[a], 2)}", fontsize=7.0, color=S.INK,
             ha="center", va="top", fontweight="bold")
    fig.text(xc, yf(Y_IPL), f"interface pLDDT {num(IPL[a], 1)}", fontsize=6.2, color=S.MUTED,
             ha="center", va="top")
    fig.text(xc, yf(Y_PTM), f"global pTM {num(PTM[a], 2)}", fontsize=5.8, color=S.MUTED,
             ha="center", va="top")
    fig.text(left, yf(Y_NAME), "abc"[i], fontsize=9.5, fontweight="bold", color=S.INK,
             ha="left", va="baseline")

    if a == "wt":                                    # name the two chains once, in the wt panel
        ax.text(0.010, 0.990, f"{CHAIN2XTAL['A']}  ·  receptor, {int(RECEPTOR.sum())} aa",
                transform=ax.transAxes, fontsize=5.8, color=S.SOFT, ha="left", va="top",
                path_effects=halo, zorder=7)
        ax.annotate(f"{CHAIN2XTAL['B']}  ·  binder, {int((~RECEPTOR).sum())} aa",
                    xy=tuple(CB2 + 4.0 * UAX), xytext=(0.990, 0.012), textcoords="axes fraction",
                    fontsize=5.8, color=S.SOFT, ha="right", va="bottom", path_effects=halo,
                    zorder=7, arrowprops=dict(arrowstyle="-", lw=0.6, color=S.FLOOR_EDGE,
                                              shrinkA=2.0, shrinkB=0.0))

# ------------------------------------------------------------------ the key: one row, two halves
# [marker + what the larger marks are]   [the shared pLDDT colourbar]
# The row is MEASURED and then centred, so the two halves can never drift into each other when a
# count or a label changes length.
KEYTXT = f"larger mark  =  one of the {int(IS_IFACE.sum())} interface positions"
CBW, MARKW, GAPK = 0.220, 0.024, 0.055
_probe = fig.text(0, 0, KEYTXT, fontsize=5.8)
fig.canvas.draw()
KEYW = _probe.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / S.FIG_W
_probe.remove()
KX = 0.5 - (MARKW + KEYW + GAPK + CBW) / 2.0                       # left edge of the whole key row
YKC = yf(Y_CB + CB_H / 2.0)

fig.text(KX + MARKW, YKC, KEYTXT, fontsize=5.8, color=S.MUTED, ha="left", va="center")
kax = fig.add_axes([KX, yf(Y_CB, CB_H), MARKW, CB_H / FIG_H])
kax.set_axis_off()
kax.set_xlim(0, 1)
kax.set_ylim(0, 1)
kax.plot([0.30], [0.5], "o", ms=3.3, mfc=S.RAMP_MPNN[4], mec="white", mew=0.45, zorder=3)

CBX = KX + MARKW + KEYW + GAPK
cax = fig.add_axes([CBX, yf(Y_CB, CB_H), CBW, CB_H / FIG_H])
cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.set_ticks([50, 70, 90, 100])
cax.set_xticklabels(["50", "70", "90", "100"], fontsize=5.6)
cax.tick_params(length=1.8, width=0.6, colors=S.RULE, labelcolor=S.INK, pad=1.2)
fig.text(CBX + CBW / 2.0, yf(Y_CBLAB), "per-residue pLDDT", fontsize=6.4, color=S.INK,
         ha="center", va="baseline")

# ------------------------------------------------------------------ title, note, footnote
fig.text(L0, yf(Y_TITLE), "confidence is lost at the interface, not in the fold", fontsize=8.5,
         color=S.INK, ha="left", va="top")
fig.text(L0, yf(Y_NOTE),
         f"ILLUSTRATIVE  ·  {CID}, the pre-registered median-effect complex  ·  one fold per arm, "
         f"not best-of-k", fontsize=6.5, color=S.MUTED, ha="left", va="top")
fig.text(L0, yf(Y_FOOT), FOOT, fontsize=5.6, color=S.MUTED, ha="left", va="top", linespacing=1.5)
S.save(fig, "fig_hero")

# ------------------------------------------------------------------ provenance
print(f"[provenance] structures {HERO}/{CID}__{{{','.join(ARMS)}}}.pdb ; numbers {HERO}/{CID}_meta.csv "
      f"(each asserted equal to its {R}/iptm_steer.csv cell)")
for a in ARMS:
    print(f"  {a:7s} k={int(meta.loc[a].k):2d}  ipTM {IPTM[a]:.2f}  interface_pLDDT {IPL[a]:7.4f}  "
          f"arm-mean ipTM {AMEAN[a]:.4f}  |  Ca pLDDT mean: receptor "
          f"{PLDDT[a][RECEPTOR].mean():.2f} binder {PLDDT[a][~RECEPTOR].mean():.2f} "
          f"interface {PLDDT[a][IS_IFACE].mean():.2f}")
    print(f"          superimposed on wt receptor Ca: RMSD all {RMSD[a]['all']:.3f} A, "
          f"receptor {RMSD[a]['rec']:.3f} A, binder {RMSD[a]['bnd']:.3f} A")
print(f"  effect: this complex L-random = {D_CX:+.4f}, rank {RANK}/{N_CX} of the cohort (it is the "
      f"upper of the two complexes straddling the median); cohort median = {D_MED:+.4f} "
      f"({R}/iptm_steer.csv, mean over k) -> the picture is at the median, not the tail")
print(f"  projection: PC1/PC2 of the {len(REF)} wt Ca ({100*VAR2:.1f}% of the Ca variance), one basis "
      f"for all three panels; {100*AXKEEP:.0f}% of the receptor-binder axis lies in the drawn plane")
print(f"  interface: {len(IFACE_KEYS)} positions from {R}/leverage_skempi_positions.csv "
      f"(is_interface, ΔrSASA>0.05 on the crystal) = the n_iface of iptm_steer.csv; an independent "
      f"{IFACE_CUT:.0f} Å heavy-atom recomputation on the wt fold gives {len(CONTACT)} residues, "
      f"{N_AGREE}/{len(IFACE_KEYS)} of the committed set among them")
print(f"  pLDDT scale: {NORM.vmin:.0f}–{NORM.vmax:.0f} (vmin = floor of the global minimum, so no "
      f"value is clipped)")

#!/usr/bin/env python3
"""Figure — the tax lives in the conditioning set (paper section 6).

On CRYSTAL backbones the burial-matched hotspot gap is ~0 (Fig. deficit_burial). Swap the crystal
for a backbone a designer actually has — one predicted by OpenFold3 or by AlphaFold2-multimer — and
the gap reappears. The claim does not rest on either marginal number but on their AGREEMENT: the two
architecturally-independent predictors find the same complexes hard, per complex.

  (a) the agreement itself: the per-complex burial-matched gap under OpenFold3 (x) against the same
      complex under AF2-multimer (y), all 127 complexes. y = x guide, Spearman with CI, and the
      partial correlation that survives controlling for interface burial. The three complexes whose
      removal most weakens the deficit are the SAME SET under both predictors — highlighted.
  (b) the divergence the decomposition predicts, on those same predicted backbones. The mixed
      derivative (leverage) SURVIVES — CPI(L | geometry) stays positive with CI > 0, retaining
      69-84% of its crystal value — while the confidence-type readout, the matched hotspot gap,
      goes from ~0 on crystal to clearly negative. Confidence degrades exactly where leverage holds.
      The two rows are DIFFERENT quantities, so each block is drawn on its own scale and every value
      is direct-labelled; only the sign and the position of the CI relative to zero are compared.

Every plotted number is read from a committed CSV:
  (a) results/expD_af2_of3_corr_percomplex.csv, results/expD_af2_of3_corr.csv,
      results/expD_leverage.csv (which complexes), results/deficit_burial_residualize.csv
  (b) results/leverage_predicted.csv, results/expD_leverage.csv, results/p0_dssp_summary.csv
Nothing is hardcoded; every annotation is formatted from a value read.

  python3 src/fig_conditioning_tax.py  ->  results/figures/fig_conditioning_tax.{pdf,png}
"""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import figstyle as S
S.apply()
R = "results"

MINUS = "−"


def sgn(s):
    return s.replace("-", MINUS)


def num(v, dp=3, plus=True):
    """Signed fixed-point, rounded HALF-UP so the figure agrees with the manuscript text."""
    q = Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    return sgn(f"{q:+.{dp}f}" if plus else f"{q:.{dp}f}")


def ci(v, lo, hi, dp=3):
    return f"{num(v, dp)}  [{num(lo, dp)}, {num(hi, dp)}]"


# ================================================================== (a) data
px = pd.read_csv(f"{R}/expD_af2_of3_corr_percomplex.csv")
cor = pd.read_csv(f"{R}/expD_af2_of3_corr.csv").set_index("metric")
SP = cor.loc["spearman"]
RHO, RHO_LO, RHO_HI, N_CX = float(SP.estimate), float(SP.lo95), float(SP.hi95), int(SP.n_complexes)
MEAN_OF3, MEAN_AF2 = float(SP.mean_d_of3), float(SP.mean_d_af2)
MEAN_CRY = float(SP.mean_d_crystal)

lev = pd.read_csv(f"{R}/expD_leverage.csv")
BLK = {"of3": "expA_of3_SECONDARY_B_dpred", "af2": "expD_af2_SECONDARY_B_dpred"}


def dropped(pred, subset="drop_top3_supporters"):
    r = lev[(lev.block == BLK[pred]) & (lev.subset == subset)].iloc[0]
    return set(str(r.dropped).split("|")), r


TOP_OF3, _ = dropped("of3")
TOP_AF2, _ = dropped("af2")
SHARED = sorted(TOP_OF3 & TOP_AF2)                      # the claim: the SAME set under both
assert TOP_OF3 == TOP_AF2, "top-3 supporter sets differ — the figure's claim would be wrong"

res = pd.read_csv(f"{R}/deficit_burial_residualize.csv").set_index("metric")
PR = res.loc["partial_rho_given_burial"]
PRHO, PR_LO, PR_HI = float(PR.value), float(PR.lo), float(PR.hi)
PRHO_D3 = float(res.loc["partial_rho_burial_drop_top3"].value)

# ================================================================== (b) data
lp = pd.read_csv(f"{R}/leverage_predicted.csv")
CPI = "CPI_mut(L | burial+nbr+dSASA)"
CPI_C = "CPI_mut(L | burial+nbr+dSASA+confidence)"


def lrow(source, metric=CPI):
    r = lp[(lp.source == source) & (lp.metric == metric)].iloc[0]
    return dict(v=float(r.stat), lo=float(r.lo), hi=float(r.hi), p=float(r.p_gt0),
                n=int(r.n), ncx=int(r.n_cx))


def drow(pred):
    r = lev[(lev.block == BLK[pred]) & (lev.subset == "full")].iloc[0]
    return dict(v=float(r.estimate), lo=float(r.lo95), hi=float(r.hi95), p=float(r.p_lt0),
                n=int(r.n_units), ncx=int(r.n_cx))


dss = pd.read_csv(f"{R}/p0_dssp_summary.csv").set_index("analysis")
CRY = dss.loc["SECONDARY_B_any_interface"]

BACKBONE = [("crystal", "crystal"), ("of3", "OpenFold3"), ("af2", "AF2-multimer")]
LEVR = {k: lrow(k) for k, _ in BACKBONE}
DEFR = {"crystal": dict(v=float(CRY["mean"]), lo=float(CRY.lo), hi=float(CRY.hi), p=np.nan,
                        n=int(CRY.n_pairs), ncx=int(CRY.n_complexes)),
        "of3": drow("of3"), "af2": drow("af2")}
for d in list(LEVR.values()) + list(DEFR.values()):
    d["sig"] = not (d["lo"] <= 0.0 <= d["hi"])
RETAIN = {k: LEVR[k]["v"] / LEVR["crystal"]["v"] for k, _ in BACKBONE}
BEYOND = {k: lrow(k, CPI_C) for k in ("of3", "af2")}

BLOCKS = [("lev", "the mixed derivative:  CPI(L | geometry)", LEVR, S.LEV, 3),
          ("def", "the confidence readout:  the matched gap", DEFR, S.SCALAR, 3)]
# each block gets its own scale (the two carry different quantities): the widest CI end in a
# block maps to |x| = 1, which is the edge of the whisker column
SCALE = {tag: max(max(abs(d[k]) for k in ("lo", "hi", "v")) for d in src.values())
         for tag, _, src, _, _ in BLOCKS}

# ================================================================== canvas
fig = plt.figure(figsize=(5.5, 3.45))
gA = fig.add_gridspec(1, 1, left=0.088, right=0.435, top=0.845, bottom=0.290)
gB = fig.add_gridspec(1, 1, left=0.535, right=0.995, top=0.845, bottom=0.250)

HALO = [pe.withStroke(linewidth=2.4, foreground="white")]

# ================================================================== (a) the two predictors agree
axa = fig.add_subplot(gA[0])
LIM = (-5.50, 3.80)
axa.set_xlim(*LIM)
axa.set_ylim(*LIM)
axa.set_aspect("equal", adjustable="box", anchor="N")

axa.plot(LIM, LIM, "-", color=S.FLOOR_EDGE, lw=0.7, zorder=1)          # y = x
for f in (axa.axhline, axa.axvline):
    f(0.0, color=S.RULE, lw=0.7, ls=(0, (3, 3)), zorder=1)

is_top = px.complex_id.isin(SHARED)
rest = px[~is_top]
axa.plot(rest.d_of3, rest.d_af2, "o", ms=2.6, mfc="white", mec=S.SOFT, mew=0.55, zorder=3)
axa.plot(px.loc[is_top, "d_of3"], px.loc[is_top, "d_af2"], "o", ms=4.6, color=S.LEV,
         mec="white", mew=0.8, zorder=6)
axa.plot(MEAN_OF3, MEAN_AF2, "D", ms=4.6, color=S.INK, mec="white", mew=0.8, zorder=7)

# hand-placed once against the render (offsets in POINTS, so they hold at any panel size)
COFF = {"1JRH_LH_I": (-4.5, -1.0, "right"), "1JTD_A_B": (4.5, -1.5, "left"),
        "1Z7X_W_X": (4.5, 1.5, "left")}
for cid in SHARED:                                                      # direct-label the shared 3
    r = px[px.complex_id == cid].iloc[0]
    dx, dy, ha = COFF[cid]
    axa.annotate(cid.split("_")[0], xy=(r.d_of3, r.d_af2), xytext=(dx, dy),
                 textcoords="offset points", fontsize=5.8, color=S.LEV, fontweight="bold",
                 ha=ha, va="center", path_effects=HALO, zorder=8)
axa.annotate("mean", xy=(MEAN_OF3, MEAN_AF2), xytext=(-5.0, -4.5), textcoords="offset points",
             fontsize=5.8, color=S.INK, ha="right", va="top", path_effects=HALO, zorder=8)
axa.text(LIM[1] - 0.15, LIM[1] - 0.15, "y = x", fontsize=5.6, color=S.MUTED, ha="right",
         va="top", rotation=45, rotation_mode="anchor", path_effects=HALO, zorder=4)

axa.text(-5.30, 3.62, sgn(f"ρ = {num(RHO, 2)}  [{num(RHO_LO, 2)}, {num(RHO_HI, 2)}]"),
         fontsize=7.4, color=S.INK, fontweight="bold", ha="left", va="top",
         path_effects=HALO, zorder=8)
axa.text(-5.30, 2.80, sgn(f"controlling interface burial  {num(PRHO, 2)} "
                          f"[{num(PR_LO, 2)}, {num(PR_HI, 2)}]\n"
                          f"drop the 3 shared complexes  {num(PRHO_D3, 2)}"),
         fontsize=5.5, color=S.MUTED, ha="left", va="top", linespacing=1.5,
         path_effects=HALO, zorder=8)
axa.text(3.60, -5.30, "highlighted = the 3 complexes whose\nremoval most weakens the deficit —\n"
                      "the same set under both predictors",
         fontsize=5.5, color=S.LEV, ha="right", va="bottom", linespacing=1.5,
         path_effects=HALO, zorder=8)

TK = [-4, -2, 0, 2]
axa.set_xticks(TK)
axa.set_yticks(TK)
axa.set_xticklabels([sgn(str(t)) for t in TK], fontsize=6.4)
axa.set_yticklabels([sgn(str(t)) for t in TK], fontsize=6.4)
axa.set_xlabel("OpenFold3 backbone", fontsize=7.2, labelpad=1.6)
axa.set_ylabel("AF2-multimer backbone", fontsize=7.2)
axa.yaxis.set_label_coords(-0.135, 0.5)
axa.tick_params(labelsize=6.4)
S.strip(axa)
S.assert_in_view(axa, list(px.d_of3), axis="x")
S.assert_in_view(axa, list(px.d_af2), axis="y")
S.header(axa, "the same complexes are hard under both",
         sgn(f"burial-matched gap per complex, nats  ·  n = {N_CX}"))

# ================================================================== (b) leverage holds, confidence goes
axb = fig.add_subplot(gB[0])
# three columns that no mark may enter: label gutter | whisker region (|x| <= 1) | value column.
# The whisker region is normalised per block, so the SHARED zero line is the only thing the two
# blocks have in common — which is exactly the comparison being made.
XL, LX, VX, XV = -2.20, -1.06, 1.10, 2.90
YB, y = {}, 0.0
for tag, *_ in BLOCKS:
    for k, _ in BACKBONE:
        YB[(tag, k)] = y
        y += 1.0
    y += 0.90
SEPY = (YB[("lev", "af2")] + YB[("def", "crystal")]) / 2.0
axb.set_xlim(XL, XV)
axb.set_ylim(max(YB.values()) + 0.80, -1.30)
axb.set_xticks([])
axb.set_yticks([])
axb.set_axis_off()
axb.plot([0, 0], [-0.95, max(YB.values()) + 0.55], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)
axb.text(0.0, -1.02, "no effect", fontsize=5.6, color=S.MUTED, ha="center", va="bottom")
axb.plot([XL, XV], [SEPY, SEPY], "-", color=S.GHOST, lw=0.6, zorder=1)

for tag, htitle, src, col, dp in BLOCKS:
    axb.text(XL, YB[(tag, "crystal")] - 0.66, htitle, fontsize=6.2, color=S.INK,
             ha="left", va="center", zorder=6)
    axb.plot([src[k]["v"] / SCALE[tag] for k, _ in BACKBONE],                # crystal -> predicted
             [YB[(tag, k)] for k, _ in BACKBONE], "-", color=S.GHOST, lw=0.7, zorder=3)
    for k, klab in BACKBONE:
        yy, r = YB[(tag, k)], src[k]
        x = lambda v: v / SCALE[tag]
        axb.text(LX, yy, klab, fontsize=6.1, color=S.MUTED if k == "crystal" else S.INK,
                 ha="right", va="center", zorder=6)
        axb.plot([x(r["lo"]), x(r["hi"])], [yy, yy], "-", color=col, lw=1.2,
                 solid_capstyle="butt", zorder=4)
        for e in ("lo", "hi"):
            axb.plot([x(r[e])] * 2, [yy - 0.15, yy + 0.15], "-", color=col, lw=1.2, zorder=4)
        axb.plot(x(r["v"]), yy, "o", ms=4.8, mfc=col if r["sig"] else "white", mec=col,
                 mew=1.0, zorder=5)
        axb.text(VX, yy, ci(r["v"], r["lo"], r["hi"], dp), fontsize=5.6,
                 color=S.INK if r["sig"] else S.SOFT, ha="left", va="center", zorder=6)

for k, klab in BACKBONE[1:]:                                 # what survives, in words
    axb.text(LX, YB[("lev", k)] + 0.36, f"{RETAIN[k] * 100:.0f}% of crystal", fontsize=5.5,
             color=S.MUTED, ha="right", va="center", zorder=6)
axb.text(XV, YB[("lev", "crystal")] - 0.66,
         f"P(>0) = {min(LEVR[k]['p'] for k, _ in BACKBONE):.3f}", fontsize=5.8, color=S.LEV,
         ha="right", va="center", zorder=6)
axb.text(XV, YB[("def", "crystal")] - 0.66,
         f"P(<0) ≥ {min(DEFR[k]['p'] for k in ('of3', 'af2')):.3f}", fontsize=5.8,
         color=S.MUTED, ha="right", va="center", zorder=6)
axb.text(LX, YB[("def", "crystal")] + 0.36, "the crystal reference", fontsize=5.5,
         color=S.MUTED, ha="right", va="center", zorder=6)

S.header(axb, "leverage holds where confidence fails",
         "the same two scorers, re-run on the predicted backbone")

fig.text(0.028, 0.185,
         sgn(f"b, upper block: CPI(L | burial + neighbours + ΔSASA), mutation level, "
             f"{LEVR['crystal']['n']:,} mutations over {LEVR['crystal']['ncx']} shared complexes; "
             f"lower block: the\nburial-matched paired design in nats, {DEFR['of3']['n']} pairs / "
             f"{DEFR['of3']['ncx']} complexes predicted ({DEFR['crystal']['n']} / "
             f"{DEFR['crystal']['ncx']} crystal).  The two blocks are different\nquantities: each "
             f"is on its own scale and every value is printed, so only the sign is compared.  "
             f"95% complex-clustered bootstrap CIs."),
         fontsize=5.6, color=S.MUTED, ha="left", va="top", linespacing=1.5)

S.flabel(fig, 0.088, 0.985, "a")
S.flabel(fig, 0.535, 0.985, "b")
S.save(fig, "fig_conditioning_tax")

# ------------------------------------------------------------------ provenance
print("[provenance] every mark is a literal cell of a committed CSV")
print(f"  (a) {R}/expD_af2_of3_corr_percomplex.csv — {len(px)} complexes plotted "
      f"(d_of3 {px.d_of3.min():+.3f}..{px.d_of3.max():+.3f}, "
      f"d_af2 {px.d_af2.min():+.3f}..{px.d_af2.max():+.3f})")
print(f"      {R}/expD_af2_of3_corr.csv  spearman = {RHO:+.4f} [{RHO_LO:+.4f},{RHO_HI:+.4f}] "
      f"n_cx={N_CX} ; means d_of3={MEAN_OF3:+.4f} d_af2={MEAN_AF2:+.4f} "
      f"d_crystal={MEAN_CRY:+.4f}")
print(f"      {R}/deficit_burial_residualize.csv  partial rho | burial = {PRHO:+.4f} "
      f"[{PR_LO:+.4f},{PR_HI:+.4f}] ; drop shared top-3 = {PRHO_D3:+.4f}")
print(f"      {R}/expD_leverage.csv  drop_top3_supporters: of3 = {sorted(TOP_OF3)} ; "
      f"af2 = {sorted(TOP_AF2)}  -> identical set {SHARED}")
for cid in SHARED:
    r = px[px.complex_id == cid].iloc[0]
    print(f"        {cid:12s} d_of3={r.d_of3:+.4f}  d_af2={r.d_af2:+.4f}  "
          f"d_crystal={r.d_crystal:+.4f}")
print(f"  (b) block scales (own scale per block): " +
      ", ".join(f"{t}={SCALE[t]:.4f}" for t, *_ in BLOCKS))
for tag, htitle, src, _, _ in BLOCKS:
    for k, klab in BACKBONE:
        r = src[k]
        extra = f"  retention={RETAIN[k]*100:.1f}%" if tag == "lev" else ""
        print(f"      {tag}/{klab:13s} {r['v']:+.5f} [{r['lo']:+.5f},{r['hi']:+.5f}] "
              f"n={r['n']:4d} n_cx={r['ncx']:3d} CI_excl_0={r['sig']}{extra}")
print(f"      L beyond geometry AND confidence on the predicted backbone: " +
      ", ".join(f"{k}={BEYOND[k]['v']:+.5f} [{BEYOND[k]['lo']:+.5f},{BEYOND[k]['hi']:+.5f}]"
                for k in ("of3", "af2")))
print(f"      crystal deficit row from {R}/p0_dssp_summary.csv SECONDARY_B_any_interface "
      f"(n_boot={int(CRY.n_boot)}, seed={int(CRY.seed)})")

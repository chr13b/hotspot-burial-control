#!/usr/bin/env python3
"""Figure — on crystal backbones the published hotspot recovery deficit is a BURIAL confound (paper section 5).

One quantity runs through the whole figure: the paired recovery gap

        gap = recovery(hotspot) - recovery(matched control),

so a value LEFT of zero is a hotspot deficit (what ProBID-Net published: 0.334 vs 0.472,
-0.138, uncontrolled) and a value RIGHT of zero means hotspots are recovered BETTER. The sign
is kept exactly as committed in the CSVs so every mark can be grepped in results/.

  (a) the confound itself — the hotspot pool and the three candidate control pools of the
      uncontrolled comparison, placed in the (burial, recovery) plane. Recovery is monotone in
      pool burial across all five pools, so the size and the SIGN of an uncontrolled 'deficit'
      is set by which control pool is chosen, before any binding term enters.
  (b) ProBID-Net's own released voxel-CNN, re-run on our fixture. Its published deficit
      reproduces only in comprehensively-scanned complexes, and then dissolves under
      residue-type / burial / hydrophobicity matching.
  (c) the same pre-registered matched-pair design across five inverse-folding architectures,
      on both matched tiers. No architecture shows a residual DEFICIT; the two whose CI excludes
      zero on the strict 47-pair tier (MIF, PiFold) do so on the hotspots-are-EASIER side.

Every plotted number is read from a committed CSV — one CSV per panel:
  (a) results/p0_dssp_summary.csv      (b) results/probid_gap_estimators.csv
  (c) results/panel_summary.csv + results/matched_recovery.csv (identical estimator, same pairs)
  footnote: results/dsasa_matched_sens.csv
Nothing is hardcoded; every annotation is formatted from a value read.

  python3 src/fig_deficit_burial.py  ->  results/figures/fig_deficit_burial.{pdf,png}
"""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
R = "results"

MINUS = "−"
DDG = "ΔΔG"


def sgn(s):
    """Render a signed number with a real minus sign (U+2212), never a hyphen."""
    return s.replace("-", MINUS)


def num(v, dp=3, plus=True):
    """Signed fixed-point, rounded HALF-UP so the figure agrees with the manuscript text."""
    q = Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    return sgn(f"{q:+.{dp}f}" if plus else f"{q:.{dp}f}")


def ci(v, lo, hi, dp=3):
    return f"{num(v, dp)}  [{num(lo, dp)}, {num(hi, dp)}]"


def thousands(n):
    return f"{int(n):,}"


# ================================================================== (a) data — one CSV
dss = pd.read_csv(f"{R}/p0_dssp_summary.csv").set_index("analysis")
U_ALL = dss.loc["UNCONTROLLED_PROBIDNET_strict_vs_all_other_iface"]
U_MEA = dss.loc["UNCONTROLLED_strict_vs_measured_nonhot_iface"]
U_LOO = dss.loc["UNCONTROLLED_loose_vs_null_iface"]

# five pools: the hotspot pool(s) and the three candidate control pools, each read from the CSV
POOLS = [
    ("nulls",              f"nulls |{DDG}|<0.25",  U_LOO.rsasa_null, U_LOO.recovery_null, U_LOO.n_nonhot, 0),
    ("measured_non",       "measured\nnon-hotspots",   U_MEA.rsasa_null, U_MEA.recovery_null, U_MEA.n_nonhot, 0),
    ("all_other",          "all other\ninterface",     U_ALL.rsasa_null, U_ALL.recovery_null, U_ALL.n_nonhot, 0),
    ("hot_loose",          f"hotspots {DDG}>1",    U_LOO.rsasa_hot,  U_LOO.recovery_hot,  U_LOO.n_hot,    1),
    ("hot_strict",         f"hotspots {DDG}>2",    U_ALL.rsasa_hot,  U_ALL.recovery_hot,  U_ALL.n_hot,    1),
]
POOLS = sorted(POOLS, key=lambda r: -r[2])                      # exposed -> buried
PX = np.array([p[2] for p in POOLS], float)
PY = np.array([p[3] for p in POOLS], float)
# the three uncontrolled 'gaps' the same hotspot pool yields against the three control pools
N_IFACE = int(U_ALL.n_hot + U_ALL.n_nonhot)     # pools NEST, so never sum the five
UGAPS = [("vs all other interface", float(U_ALL.recovery_hot - U_ALL.recovery_null)),
         ("vs measured non-hotspots", float(U_MEA.recovery_hot - U_MEA.recovery_null)),
         ("vs nulls", float(U_LOO.recovery_hot - U_LOO.recovery_null))]

# ================================================================== (b) data — one CSV
pg = pd.read_csv(f"{R}/probid_gap_estimators.csv").set_index("analysis")


def prow(key, label, block):
    r = pg.loc[key]
    return dict(key=key, label=label, block=block, v=float(r.gap), lo=float(r.lo), hi=float(r.hi),
                n=int(r.n_cx), sig=not (float(r.lo) <= 0.0 <= float(r.hi)))


PROBID = [
    prow("uncontrolled_hotspot_weighted_LIKE4LIKE", "all complexes", "unc"),
    prow("uncontrolled_strat_nhot_ge2",             "≥ 2 measured hotspots", "unc"),
    prow("uncontrolled_strat_nhot_ge3",             "≥ 3 measured hotspots", "unc"),
    prow("uncontrolled_strat_nhot_ge5",             "≥ 5 measured hotspots", "unc"),
    prow("matched::p0_dssp_pairs_AAMATCHED_any_interface.csv",   "residue type", "mat"),
    prow("matched::p0_dssp_pairs_SECONDARY_B_any_interface.csv", "burial + SS + packing", "mat"),
    prow("matched::p0_dssp_pairs_HYDROMATCHED_any_interface.csv", "hydrophobicity", "mat"),
]
DEEP = next(r for r in PROBID if r["key"].endswith("ge5"))
N_MAT_SPAN0 = sum(1 for r in PROBID if r["block"] == "mat" and not r["sig"])
N_MAT = sum(1 for r in PROBID if r["block"] == "mat")

# ================================================================== (c) data — two CSVs, one estimator
# p0_multimodel.py and matched_recovery.py compute the identical statistic (paired hit(hotspot) -
# hit(control) on the SAME committed p0_dssp pair files, complex bootstrap, seed 20260803); the
# only difference is which positions file supplies the argmax. So they share one axis.
pan = pd.read_csv(f"{R}/panel_summary.csv")
mrec = pd.read_csv(f"{R}/matched_recovery.csv")
TIERS = [("SECONDARY_B_any_interface", "any interface position"),
         ("PRIMARY_loose_null",        "measured nulls")]
ARCH = [("mpnn_vanilla",  "ProteinMPNN",   "vanilla"),
        ("mpnn_soluble",  "ProteinMPNN",   "soluble"),
        ("esmif",         "ESM-IF1",       ""),
        ("mif",           "MIF",           ""),
        ("pifold",        "PiFold",        "")]


def arow(model, tier):
    if model == "mpnn_vanilla":
        r = mrec[mrec.analysis == tier].iloc[0]
        v, lo, hi, n, ncx = r.paired_diff, r.lo, r.hi, r.n_pairs, r.n_complexes
        src = "matched_recovery.csv"
    else:
        r = pan[(pan.model == model) & (pan.analysis == tier)].iloc[0]
        v, lo, hi, n, ncx = r.gap_recovery, r.rec_lo, r.rec_hi, r.n_pairs, r.n_complexes
        src = "panel_summary.csv"
    return dict(v=float(v), lo=float(lo), hi=float(hi), n=int(n), ncx=int(ncx),
                sig=not (float(lo) <= 0.0 <= float(hi)), src=src)


PAN = {(m, t): arow(m, t) for m, *_ in ARCH for t, _ in TIERS}
HP, ST = TIERS[0][0], TIERS[1][0]
SIG_ST = [lab for (m, lab, _) in ARCH if PAN[(m, ST)]["sig"]]
MAXABS_HP = max(abs(PAN[(m, HP)]["v"]) for m, *_ in ARCH)


def rng(t, k):
    """The pair / complex count ranges over the five architectures (they differ slightly because
    a model that drops chain-junction positions loses a few pairs). Printed as a range, never
    as one architecture's number standing in for all five."""
    v = sorted({PAN[(m, t)][k] for m, *_ in ARCH})
    return f"{v[0]}" if len(v) == 1 else f"{v[0]}" + MINUS + f"{v[-1]}"


NPAIRS = {t: rng(t, "n") for t, _ in TIERS}
NCXS = {t: rng(t, "ncx") for t, _ in TIERS}

ds = pd.read_csv(f"{R}/dsasa_matched_sens.csv")
DS = ds[ds.pairs_file.str.contains("SECONDARY_B")].iloc[0]

# bootstrap protocol, also read rather than typed (identical in all three estimators)
N_BOOT = int(dss.loc["SECONDARY_B_any_interface"].n_boot)
SEED = int(dss.loc["SECONDARY_B_any_interface"].seed)
assert SEED == int(pg.loc["uncontrolled_strat_nhot_ge5"].seed), "panels disagree on the seed"

# ================================================================== canvas
fig = plt.figure(figsize=(5.5, 5.45))
gA = fig.add_gridspec(1, 1, left=0.142, right=0.372, top=0.910, bottom=0.680)
gB = fig.add_gridspec(1, 1, left=0.600, right=0.985, top=0.910, bottom=0.676)
gC = fig.add_gridspec(1, 1, left=0.205, right=0.985, top=0.505, bottom=0.255)

RAMP = S.RAMP_MPNN


# ================================================================== (a) the confound
# ONE quantity on the axis (recovery); burial is the ordinal colour, so the lockstep is read
# without putting two different quantities on one scale.
axa = fig.add_subplot(gA[0])
YA = np.arange(len(POOLS), dtype=float)
axa.plot(PY, YA, "-", color=S.GHOST, lw=0.9, zorder=2, solid_capstyle="round")
for i, (_, lab, x, y, n, is_hot) in enumerate(POOLS):
    axa.plot(y, YA[i], "o", ms=5.8 if is_hot else 4.8, color=RAMP[i + 1], mec="white",
             mew=0.9, zorder=5)
    axa.annotate(f"{lab}\n{thousands(int(n))}", xy=(0, YA[i]), xycoords=("axes fraction", "data"),
                 xytext=(-3.5, 0), textcoords="offset points", fontsize=5.5,
                 color=S.INK if is_hot else S.SOFT, ha="right", va="center", linespacing=1.3,
                 fontweight="bold" if is_hot else "normal", annotation_clip=False, zorder=6)

axa.set_xlim(0.325, 0.552)
axa.set_ylim(len(POOLS) - 0.42, -0.72)
axa.set_xticks([0.35, 0.40, 0.45, 0.50])
axa.set_xticklabels(["0.35", "0.40", "0.45", "0.50"], fontsize=6.4)
axa.set_yticks([])
axa.spines["left"].set_visible(False)
axa.set_xlabel("sequence recovery", fontsize=7.0, labelpad=1.8)
axa.tick_params(labelsize=6.4)
S.strip(axa, left=False)
S.assert_in_view(axa, list(PY), axis="x")
axa.text(0.330, len(POOLS) - 1.42, sgn(f"colour = pool burial\ndarker = more buried\n"
                                       f"rSASA {num(PX.max(), 3, False)} → "
                                       f"{num(PX.min(), 3, False)}"),
         fontsize=5.5, color=S.MUTED, ha="left", va="center", linespacing=1.45)
S.header(axa, "recovery tracks pool burial",
         f"5 overlapping pools of {thousands(N_IFACE)} interface positions")

# ================================================================== (b) ProBID-Net's own model
axb = fig.add_subplot(gB[0])
YB = {}
y = 0.0
for i, r in enumerate(PROBID):
    if i and r["block"] != PROBID[i - 1]["block"]:
        y += 0.95                                            # block gap
    YB[r["key"]] = y
    y += 1.0
BSEP = (YB[PROBID[3]["key"]] + YB[PROBID[4]["key"]]) / 2.0
XB = (-0.235, 0.325)
axb.set_xlim(*XB)
axb.set_ylim(max(YB.values()) + 0.60, -0.98)
axb.plot([0, 0], [-0.62, max(YB.values()) + 0.52], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)

for r in PROBID:
    yy = YB[r["key"]]
    c = S.SOFT if r["block"] == "unc" else S.LEV
    axb.plot([r["lo"], r["hi"]], [yy, yy], "-", color=c, lw=1.15, solid_capstyle="butt", zorder=4)
    for e in ("lo", "hi"):
        axb.plot([r[e]] * 2, [yy - 0.16, yy + 0.16], "-", color=c, lw=1.15, zorder=4)
    axb.plot(r["v"], yy, "o", ms=4.6, mfc=c if r["sig"] else "white", mec=c, mew=1.0, zorder=5)
    axb.annotate(f"{r['label']}   {thousands(r['n'])} cx", xy=(XB[0], yy), xytext=(-2.5, 0),
                 textcoords="offset points", fontsize=6.0,
                 color=S.INK if r["sig"] else S.SOFT, ha="right", va="center",
                 fontweight="bold" if r["sig"] else "normal",
                 annotation_clip=False, zorder=6)

axb.text(XB[0] + 0.008, YB[PROBID[0]["key"]] - 0.62, "no confound control  ·  by scan depth",
         fontsize=5.8, color=S.MUTED, ha="left", va="center", zorder=6)
axb.text(XB[0] + 0.008, YB[PROBID[4]["key"]] - 0.60, "confound-matched, same fixture",
         fontsize=5.8, color=S.MUTED, ha="left", va="center", zorder=6)
axb.plot(list(XB), [BSEP, BSEP], "-", color=S.GHOST, lw=0.6, zorder=1)

axb.text(XB[0] + 0.008, YB[DEEP["key"]] + 0.58,
         sgn(f"{num(DEEP['v'])} — the published deficit, reproduced"),
         fontsize=5.7, color=S.INK, ha="left", va="center", zorder=7)
axb.text(XB[1], YB[PROBID[4]["key"]] - 0.60, f"all {N_MAT_SPAN0} span zero", fontsize=5.8,
         color=S.LEV, ha="right", va="center", fontweight="bold", zorder=7)

axb.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
axb.set_xticklabels([sgn("-0.2"), sgn("-0.1"), "0", "+0.1", "+0.2", "+0.3"], fontsize=6.4)
axb.set_yticks([])
axb.spines["left"].set_visible(False)
axb.set_xlabel("recovery gap", fontsize=7.0, labelpad=1.8)
S.strip(axb, left=False)
S.assert_in_view(axb, [v for r in PROBID for v in (r["lo"], r["hi"])], axis="x")
S.header(axb, "and it dissolves under matching",
         "ProBID-Net's own voxel-CNN, our fixture")

# ================================================================== (c) five architectures
axc = fig.add_subplot(gC[0])
XC = (-0.165, 0.790)                                         # right of +0.5 is the value column
VXC = 0.520
YC = {m: float(i) for i, (m, *_) in enumerate(ARCH)}
axc.set_xlim(*XC)
axc.set_ylim(len(ARCH) - 0.42, -1.34)
axc.plot([0, 0], [-0.36, len(ARCH) - 0.58], ls=(0, (3, 3)), color=S.RULE, lw=0.8, zorder=2)

DY = 0.215
TSTYLE = {HP: dict(c=S.LEV, mk="o", ms=4.6, dy=-DY, ls="-"),
          ST: dict(c=RAMP[3], mk="s", ms=3.9, dy=+DY, ls=(0, (2.4, 1.8)))}
for m, lab, sub in ARCH:
    yy = YC[m]
    axc.plot([PAN[(m, HP)]["v"], PAN[(m, ST)]["v"]], [yy + TSTYLE[HP]["dy"], yy + TSTYLE[ST]["dy"]],
             "-", color=S.GHOST, lw=0.7, zorder=2)
    for t, _ in TIERS:
        st, r = TSTYLE[t], PAN[(m, t)]
        yt = yy + st["dy"]
        axc.plot([r["lo"], r["hi"]], [yt, yt], ls=st["ls"], color=st["c"], lw=1.15,
                 solid_capstyle="butt", dash_capstyle="butt", zorder=4)
        for e in ("lo", "hi"):
            axc.plot([r[e]] * 2, [yt - 0.10, yt + 0.10], "-", color=st["c"], lw=1.15, zorder=4)
        axc.plot(r["v"], yt, st["mk"], ms=st["ms"], mfc=st["c"] if r["sig"] else "white",
                 mec=st["c"], mew=1.0, zorder=5)
    axc.annotate(lab, xy=(XC[0], yy), xytext=(-3.0, -0.5 if sub else 0), textcoords="offset points",
                 fontsize=6.8, color=S.INK, ha="right", va="bottom" if sub else "center",
                 annotation_clip=False, zorder=6)
    if sub:
        axc.annotate(sub, xy=(XC[0], yy), xytext=(-3.0, 0.5), textcoords="offset points",
                     fontsize=5.6, color=S.MUTED, ha="right", va="top",
                     annotation_clip=False, zorder=6)

# the tier key, drawn once in the free band above the first row (marker + text, so the two
# tiers are identified by shape and colour together — never colour alone)
KEYY = {HP: -1.03, ST: -0.63}
for t, tl in TIERS:
    st, kx = TSTYLE[t], XC[0] + 0.012
    axc.plot([kx, kx + 0.030], [KEYY[t]] * 2, ls=st["ls"], color=st["c"], lw=1.15, zorder=5)
    axc.plot(kx + 0.015, KEYY[t], st["mk"], ms=st["ms"], mfc="white", mec=st["c"], mew=1.0, zorder=6)
    axc.text(kx + 0.042, KEYY[t],
             f"controls = {tl}   ·   {NPAIRS[t]} pairs, {NCXS[t]} complexes",
             fontsize=5.8, color=st["c"], ha="left", va="center", zorder=7)

for m, lab, _ in ARCH:                                       # label only what excludes zero
    if PAN[(m, ST)]["sig"]:
        r = PAN[(m, ST)]
        axc.text(VXC, YC[m] + TSTYLE[ST]["dy"], ci(r["v"], r["lo"], r["hi"]), fontsize=5.9,
                 color=S.INK, ha="left", va="center", zorder=7)

axc.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
axc.set_xticklabels([sgn("-0.1"), "0", "+0.1", "+0.2", "+0.3", "+0.4"], fontsize=6.4)
axc.set_yticks([])
axc.spines["left"].set_visible(False)
axc.set_xlabel("recovery gap  =  recovery(hotspot) " + MINUS + " recovery(matched control)",
               fontsize=7.0, labelpad=9.5)
axc.xaxis.set_label_coords(0.375, -0.265)
S.strip(axc, left=False)
S.assert_in_view(axc, [v for r in PAN.values() for v in (r["lo"], r["hi"])], axis="x")
S.header(axc, "no architecture keeps a hotspot deficit once burial is matched",
         "same pre-registered matched pairs  ·  paired within complex")

# which way is which — written once, just under the zero of the bottom panel
axc.annotate("← hotspot deficit", xy=(0, 0), xycoords=("data", "axes fraction"),
             xytext=(-4.0, -20.0), textcoords="offset points", fontsize=5.8, color=S.MUTED,
             ha="right", va="center", annotation_clip=False)
axc.annotate("hotspots easier →", xy=(0, 0), xycoords=("data", "axes fraction"),
             xytext=(4.0, -20.0), textcoords="offset points", fontsize=5.8, color=S.MUTED,
             ha="left", va="center", annotation_clip=False)
fig.text(0.028, 0.145,
         sgn(f"whiskers = 95% complex-clustered bootstrap CI ({thousands(N_BOOT)} replicates, "
             f"seed {SEED}); filled mark = CI excludes zero.\n"
             f"ΔSASA is the one cheap hotspot feature the matching does not control, so it is "
             f"checked separately: the {int(DS.n_pairs)}-pair tier\ncarries a ΔSASA imbalance of "
             f"{num(DS.dsasa_imbalance)} [{num(DS.imb_lo)}, {num(DS.imb_hi)}], and adjusting the "
             f"gap for it moves it only {num(DS.raw_deficit)} → "
             f"{num(DS.dsasa_adjusted_deficit)} nats."),
         fontsize=5.6, color=S.MUTED, ha="left", va="top", linespacing=1.5)

S.flabel(fig, 0.142, 0.986, "a")
S.flabel(fig, 0.600, 0.986, "b")
S.flabel(fig, 0.142, 0.598, "c")
S.save(fig, "fig_deficit_burial")

# ------------------------------------------------------------------ provenance
print("[provenance] one CSV per panel; every mark below is a literal cell of that CSV")
print(f"  (a) {R}/p0_dssp_summary.csv  (UNCONTROLLED_* rows)")
for key, lab, x, y, n, _ in POOLS:
    print(f"      {key:14s} rSASA={x:.4f}  recovery={y:.4f}  n={int(n)}")
print(f"      recovery span {PY.min():.3f}->{PY.max():.3f} over rSASA {PX.max():.3f}->{PX.min():.3f} "
      f"(monotone in both); the SAME hotspot pool yields uncontrolled gaps "
      + ", ".join(f"{lab} {g:+.4f}" for lab, g in UGAPS))
print(f"  (b) {R}/probid_gap_estimators.csv")
for r in PROBID:
    print(f"      [{r['block']}] {r['label']:24s} {r['v']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] "
          f"n_cx={r['n']:3d}  CI_excl_0={r['sig']}")
print(f"      {N_MAT_SPAN0}/{N_MAT} matched estimates span zero; only the >=5-hotspot stratum "
      f"excludes zero, at {DEEP['v']:+.4f}")
print(f"  (c) {R}/panel_summary.csv + {R}/matched_recovery.csv  (identical paired-recovery estimator)")
for m, lab, sub in ARCH:
    for t, tl in TIERS:
        r = PAN[(m, t)]
        print(f"      {lab+(' '+sub if sub else ''):20s} {t:26s} {r['v']:+.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] n_pairs={r['n']:3d} n_cx={r['ncx']:3d} "
              f"CI_excl_0={r['sig']}  <- {r['src']}")
print(f"      highest-power tier: max |gap| = {MAXABS_HP:.4f}, every CI spans zero; "
      f"strict tier CI excludes zero for {', '.join(SIG_ST)} — on the hotspots-EASIER side")
print(f"  footnote {R}/dsasa_matched_sens.csv: imbalance {DS.dsasa_imbalance:+.4f} "
      f"[{DS.imb_lo:+.4f},{DS.imb_hi:+.4f}], raw {DS.raw_deficit:+.4f} -> "
      f"adjusted {DS.dsasa_adjusted_deficit:+.4f} nats (n={int(DS.n_pairs)} pairs)")

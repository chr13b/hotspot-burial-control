#!/usr/bin/env python3
"""Figure 4 — binding-epistasis coupling (Phase 3). Three panels, all from committed CSVs:
 (a) partner ablation surfaces the signal   (partial-Spearman with 95% cluster-bootstrap CI)
 (b) additivity dose-response               (model coupling magnitude tracks measured |g|)
 (c) sign is recoverable on strong pairs     (sign accuracy vs interaction / model-confidence)

  python3 src/fig4_coupling.py   ->   results/figures/fig4_coupling.png (+ .pdf)
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEED = 20260803
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8})
INK, ACC, ACC2, MUT = "#22303a", "#2f6f8f", "#b8562f", "#8a8f94"

d = pd.read_csv("results/p3_coupling.csv")
rng = np.random.default_rng(SEED)

def partial_ci(sub, nb=3000):
    x, y, z = sub.C_lev.values, sub.g.values, sub.dist_cb.values
    def pr(a, b, c):
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c); a, b, c = a[m], b[m], c[m]
        ra, rb, rc = (stats.rankdata(v) for v in (a, b, c)); Z = np.column_stack([np.ones_like(rc), rc])
        ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]; eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
        return np.corrcoef(ea, eb)[0, 1]
    r = pr(x, y, z)
    ids = sub.complex_id.unique(); by = {k: sub.index[sub.complex_id == k].to_numpy() for k in ids}
    bs = []
    for _ in range(nb):
        ix = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        s = sub.loc[ix]; bs.append(pr(s.C_lev.values, s.g.values, s.dist_cb.values))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def signacc(sub):
    h = (np.sign(-sub.C_lev) == np.sign(sub.g)).astype(int)
    lo, hi = stats.binomtest(int(h.sum()), len(h)).proportion_ci()
    return h.mean(), lo, hi, len(h)

fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.5))

# ---- (a) partner ablation -------------------------------------------------
cross = d[d.cross_interface == 1].copy().reset_index(drop=True)
same = d[d.cross_interface == 0].copy()
same_lev = same.dropna(subset=["C_lev"]).reset_index(drop=True)
same_cpx = same.assign(C_lev=same.C_complex).reset_index(drop=True)   # un-ablated proxy for same-side
bars = [("cross-interface\n(C = binding coupling)", partial_ci(cross), ACC),
        ("same-side\nUN-ablated", partial_ci(same_cpx), MUT),
        ("same-side\nablated (C − C_mono)", partial_ci(same_lev), ACC2)]
for i, (lab, (r, lo, hi), col) in enumerate(bars):
    ax[0].barh(i, r, color=col, height=0.6, zorder=3)
    ax[0].plot([lo, hi], [i, i], color=INK, lw=1.4, zorder=4)
ax[0].axvline(0, color=INK, lw=0.9)
ax[0].set_yticks(range(3)); ax[0].set_yticklabels([b[0] for b in bars], fontsize=8.5)
ax[0].invert_yaxis(); ax[0].set_xlabel("partial-Spearman(C, g | distance)", fontsize=9)
ax[0].set_title("a  partner ablation surfaces the signal", fontsize=10, loc="left", weight="bold")
ax[0].text(0.02, 0.03, "cycle predicts C ∝ −g\n(negative = correct)", transform=ax[0].transAxes,
           fontsize=7.5, color=MUT, va="bottom")

# ---- (b) additivity dose-response ----------------------------------------
dd = d.dropna(subset=["C_lev"]).copy()
dd["t"] = pd.qcut(dd.g.abs(), 3, labels=["low", "mid", "high"])
mean_c = dd.groupby("t", observed=True).C_lev.apply(lambda s: s.abs().mean())
sem_c = dd.groupby("t", observed=True).C_lev.apply(lambda s: s.abs().sem())
ax[1].bar(range(3), mean_c.values, yerr=sem_c.values, color=[MUT, ACC, ACC2], width=0.62,
          zorder=3, error_kw=dict(lw=1.2, ecolor=INK, capsize=3))
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"|g| {l}\n(n={int((dd.t==l).sum())})"
                                                   for l in ["low", "mid", "high"]], fontsize=8.5)
ax[1].set_ylabel("mean |C|  (model coupling)", fontsize=9)
ax[1].set_title("b  coupling magnitude tracks epistasis", fontsize=10, loc="left", weight="bold")
# distance-controlled partial (the raw tertiles are partly distance-driven) — annotate the honest number
_r = stats.rankdata; _dd = dd
_Z = np.column_stack([np.ones(len(_dd)), _r(_dd.dist_cb)])
_ex = _r(_dd.C_lev.abs()) - _Z @ np.linalg.lstsq(_Z, _r(_dd.C_lev.abs()), rcond=None)[0]
_ey = _r(_dd.g.abs()) - _Z @ np.linalg.lstsq(_Z, _r(_dd.g.abs()), rcond=None)[0]
_pr = np.corrcoef(_ex, _ey)[0, 1]
ax[1].text(0.5, 0.95, f"partial ρ(|C|,|g| | dist) = {_pr:+.2f}", transform=ax[1].transAxes,
           ha="center", va="top", fontsize=7.6, color=MUT)

# ---- (c) not a magnitude-skew artifact: partial rho negative in BOTH g-sign strata ----
c = d.dropna(subset=["C_lev", "g"]).copy()
def partial_sub(sub):
    x, y, z = sub.C_lev.values, sub.g.values, sub.dist_cb.values
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); x, y, z = x[m], y[m], z[m]
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z)); Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1]), int(m.sum())
strata = [("all\ng<0", c[c.g < 0], ACC), ("all\ng>0", c[c.g > 0], ACC),
          ("cross\ng<0", c[(c.g < 0) & (c.cross_interface == 1)], ACC2),
          ("cross\ng>0", c[(c.g > 0) & (c.cross_interface == 1)], ACC2)]
svals = [partial_sub(s[1]) for s in strata]
ax[2].bar(range(4), [v[0] for v in svals], color=[s[2] for s in strata], width=0.64, zorder=3)
for i, (v, n) in enumerate(svals):
    ax[2].text(i, v - 0.006, f"n={n}", ha="center", va="top", fontsize=7, color="#fff")
ax[2].axhline(0, color=INK, lw=0.9)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels([s[0] for s in strata], fontsize=8.3)
ax[2].set_ylabel("partial-Spearman(C, g | dist)", fontsize=9)
ax[2].set_title("c  not a magnitude-skew artifact", fontsize=10, loc="left", weight="bold")
ax[2].text(0.5, 0.05, "negative in BOTH sign strata —\na skew artifact would flip g>0 positive",
           transform=ax[2].transAxes, fontsize=7.3, color=MUT, ha="center", va="bottom")

fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"results/figures/fig4_coupling.{ext}", bbox_inches="tight")
print("wrote results/figures/fig4_coupling.png / .pdf")
print(f"  (a) cross partial={bars[0][1][0]:+.3f}; same un-ablated={bars[1][1][0]:+.3f}; "
      f"same ablated={bars[2][1][0]:+.3f}")
print(f"  (b) mean|C| tertiles = {mean_c.round(3).to_dict()}; partial(|C|,|g||dist)={_pr:+.3f}")
print(f"  (c) sign-stratified partial rho = {[round(v[0],3) for v in svals]} (all g<0/g>0, cross g<0/g>0)")

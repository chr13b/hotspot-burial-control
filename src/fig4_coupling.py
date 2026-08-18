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

# ---- (c) sign recovery ----------------------------------------------------
c = d.dropna(subset=["C_lev", "g"]).copy()
gth = [0.0, 0.5, 1.0, 1.5, 2.0]
gpts = [signacc(c[c.g.abs() > t]) for t in gth]
cq = [0.0, 0.5, 0.75, 0.9]
cpts = [signacc(c[c.C_lev.abs() >= c.C_lev.abs().quantile(q)]) for q in cq]
xg = range(len(gth))
ax[2].errorbar(xg, [p[0] for p in gpts], yerr=[[p[0]-p[1] for p in gpts], [p[2]-p[0] for p in gpts]],
               marker="o", color=ACC, lw=1.6, capsize=3, label="by interaction |g|>t", zorder=3)
ax[2].errorbar(xg[:len(cq)], [p[0] for p in cpts],
               yerr=[[p[0]-p[1] for p in cpts], [p[2]-p[0] for p in cpts]],
               marker="s", color=ACC2, lw=1.6, capsize=3, ls="--", label="by model |C| quantile", zorder=3)
ax[2].axhline(0.5, color=MUT, lw=1.0, ls=":"); ax[2].text(0.05, 0.505, "chance", color=MUT, fontsize=7.5)
ax[2].set_xticks(list(xg)); ax[2].set_xticklabels(["all", "0.5", "1.0", "1.5", "2.0"], fontsize=8.5)
ax[2].set_xlabel("|g| threshold (kcal/mol)  /  |C| quantile", fontsize=8.5)
ax[2].set_ylabel("sign accuracy", fontsize=9); ax[2].set_ylim(0.45, 0.78)
ax[2].legend(fontsize=7.3, frameon=False, loc="upper left")
ax[2].set_title("c  the sign recovers on strong pairs", fontsize=10, loc="left", weight="bold")

fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"results/figures/fig4_coupling.{ext}", bbox_inches="tight")
print("wrote results/figures/fig4_coupling.png / .pdf")
print(f"  (a) cross partial={bars[0][1][0]:+.3f}; same un-ablated={bars[1][1][0]:+.3f}; "
      f"same ablated={bars[2][1][0]:+.3f}")
print(f"  (b) mean|C| tertiles = {mean_c.round(3).to_dict()}")
print(f"  (c) sign: all={gpts[0][0]:.3f}, |g|>1={gpts[2][0]:.3f}, |C|>p90={cpts[3][0]:.3f}")

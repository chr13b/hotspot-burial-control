#!/usr/bin/env python3
"""
EXPLORATORY (post-hoc, not pre-registered): is "which complexes are hard" predictable from structure?

Since the burial-matched predicted-backbone deficit reproduces ACROSS two independent predictors
(Exp D, per-complex rho=+0.57), the per-complex deficit is a stable quantity — so we ask what structural
feature predicts a large deficit. Per-complex deficit = mean(d_af2, d_of3) (d = logp(hot)-logp(ctl);
negative = hotspots recovered worse). Features aggregated over each complex's interface positions.

Headline (see results/deficit_predictors.csv): the deficit is modestly but significantly predictable from
interface BURIAL — more-buried interfaces have LARGER deficits (mean neighbour count rho=-0.21; mean rSASA
rho=+0.34), even though the deficit is already burial-MATCHED within complex. i.e. the predicted-backbone
deficit concentrates in deeply-buried interfaces, exactly where inverse folding is most confident and a
predicted backbone's small errors bite hardest. KL and hotspot count do NOT predict it. The frustration
proxy (mean d_bind_local) also correlates (rho=+0.34) but its sign is harder to interpret and is not leaned on.

  python3 src/deficit_predictors.py --out results/deficit_predictors.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260803


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percomplex", default="results/expD_af2_of3_corr_percomplex.csv")
    ap.add_argument("--joined", default="results/kl_detector_joined.csv")
    ap.add_argument("--frust", default="results/frustration_monomer_joined.csv")
    ap.add_argument("--out", default="results/deficit_predictors.csv")
    a = ap.parse_args()

    pc = pd.read_csv(a.percomplex)
    pc["deficit"] = pc[["d_af2", "d_of3"]].mean(axis=1)
    kl = pd.read_csv(a.joined); kl = kl[kl.is_interface == 1]
    agg = kl.groupby("complex_id").agg(
        mean_nbr=("nbr", "mean"), mean_rsasa=("rsasa_complex", "mean"), mean_kl=("kl", "mean"),
        mean_logp=("logp_native", "mean"), n_iface=("is_hot", "size"), n_hot=("is_hot", "sum")
    ).reset_index()
    agg["hot_frac"] = agg.n_hot / agg.n_iface
    try:
        fr = pd.read_csv(a.frust); fr = fr[fr.is_interface == 1]
        agg = agg.merge(fr.groupby("complex_id").d_bind_local.mean().rename("mean_dbind").reset_index(),
                        on="complex_id", how="left")
    except Exception:
        pass
    d = pc.merge(agg, on="complex_id", how="inner")

    rng = np.random.default_rng(SEED)
    feats = [c for c in ["mean_nbr", "mean_rsasa", "mean_kl", "mean_logp", "n_iface", "n_hot",
                         "hot_frac", "mean_dbind"] if c in d]
    rows = []
    for f in feats:
        sub = d[["deficit", f]].dropna(); x = sub[f].values; y = sub.deficit.values
        rho = stats.spearmanr(x, y).correlation
        bs = [stats.spearmanr(x[i], y[i]).correlation for i in (rng.choice(len(x), len(x), True) for _ in range(2000))]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(feature=f, spearman_vs_deficit=rho, lo=float(lo), hi=float(hi),
                         n=len(sub), sig=bool(lo > 0 or hi < 0)))
        print(f"  {f:12} rho={rho:+.3f} [{lo:+.3f},{hi:+.3f}] {'*' if (lo>0 or hi<0) else ' '}")
    out = pd.DataFrame(rows); out["deficit_mean"] = d.deficit.mean(); out["n_complexes"] = len(d)
    out["seed"] = SEED; out["command"] = f"python3 src/deficit_predictors.py --out {a.out}"
    out["note"] = "EXPLORATORY post-hoc; deficit=mean(d_af2,d_of3) negative=harder"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} (n={len(d)}). Headline: more-buried interfaces have larger deficits; KL/hot-count do not predict.")


if __name__ == "__main__":
    main()

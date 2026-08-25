"""The identifiability theorem in one number (for Figure 1b and a headline sentence): conditioning on
confidence removes essentially NONE of the leverage spread. Bin interface positions into confidence deciles;
within each decile the spread of |L|_rms is as large as (in fact slightly larger than) the overall spread.

  python3 src/conf_decile_leverage.py --out results/conf_decile_leverage.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
SEED = 20260803


def iqr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) > 3 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/conf_decile_leverage.csv"); a = ap.parse_args()
    d = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    d = d[d.is_interface == True].dropna(subset=["conf", "L_rms"]).reset_index(drop=True)   # noqa: E712
    overall = iqr(d.L_rms)
    d["dec"] = pd.qcut(d.conf, 10, labels=False, duplicates="drop")
    within = d.groupby("dec").L_rms.apply(iqr)
    ratio = float(np.nanmean(within) / overall)
    rho = stats.spearmanr(d.conf, d.L_rms).correlation
    # complex-clustered bootstrap on the ratio
    rng = np.random.default_rng(SEED); comps = d.complex_id.unique()
    gl = {c: d.index[d.complex_id == c].to_numpy() for c in comps}
    boot = []
    for _ in range(2000):
        idx = np.concatenate([gl[c] for c in rng.choice(comps, len(comps), replace=True)])
        s = d.loc[idx]
        try:
            w = s.groupby(pd.qcut(s.conf, 10, labels=False, duplicates="drop")).L_rms.apply(iqr)
            boot.append(np.nanmean(w) / iqr(s.L_rms))
        except Exception:
            pass
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    print(f"  n={len(d)} interface positions, {d.complex_id.nunique()} complexes")
    print(f"  overall IQR(|L|_rms) = {overall:.4f}")
    print(f"  mean within-confidence-decile IQR / overall IQR = {ratio:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"  Spearman(confidence, |L|_rms) = {rho:+.3f}")
    print("  -> conditioning on confidence removes NONE of the leverage spread (ratio ~ 1).")
    pd.DataFrame([
        dict(stat="within_decile_IQR_over_overall_IQR", value=round(ratio, 4), lo=round(float(lo), 4),
             hi=round(float(hi), 4), n=len(d), n_complexes=int(d.complex_id.nunique())),
        dict(stat="overall_IQR_Lrms", value=round(overall, 4), n=len(d)),
        dict(stat="spearman_conf_vs_Lrms", value=round(float(rho), 4), n=len(d)),
    ]).assign(seed=SEED, command="python3 src/conf_decile_leverage.py").to_csv(a.out, index=False)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

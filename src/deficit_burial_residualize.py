#!/usr/bin/env python3
"""T5 — does the cross-predictor deficit agreement (OF3 vs AF2, rho~0.57) survive BURIAL?

Kill-shot #5: OF3 and AF2 share PDB training + the MSA paradigm, so two predictors agreeing on which
complexes are 'hard' is what you'd expect if both simply do worse on deeply-buried interfaces — a burial
confound one meta-level up from the one we dissolved in Phase 0. Test: partial Spearman(d_of3, d_af2 |
interface burial [, nbr, interface size]); and does rho survive dropping the top-3 leverage complexes?

Per-complex deficits from expD_af2_of3_corr_percomplex.csv (d = logp(hot) - logp(ctl); negative = the
predictor is worse at hotspots). Per-complex burial/nbr/size from kl_detector_joined (interface rows).
Complex-level bootstrap, seed 20260803.
  python3 src/deficit_burial_residualize.py --out results/deficit_burial_residualize.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803
R = "results"


def rankz(v):
    r = pd.Series(v).rank().to_numpy()
    return (r - r.mean()) / (r.std() + 1e-12)


def partial_spearman(x, y, Z):
    """Partial Spearman of x,y controlling for columns of Z (rank-residualization)."""
    rx, ry = rankz(x), rankz(y)
    if Z is None or (hasattr(Z, "shape") and Z.shape[1] == 0):
        return float(np.corrcoef(rx, ry)[0, 1])
    A = np.column_stack([np.ones(len(rx))] + [rankz(Z[:, k]) for k in range(Z.shape[1])])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/deficit_burial_residualize.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    pc = pd.read_csv(f"{R}/expD_af2_of3_corr_percomplex.csv")
    j = pd.read_csv(f"{R}/kl_detector_joined.csv")
    ji = j[j.is_interface == 1]
    perc = ji.groupby("complex_id").agg(burial=("burial", "mean"), nbr=("nbr", "mean"),
                                        size=("burial", "size")).reset_index()
    df = pc.merge(perc, on="complex_id", how="inner").dropna(subset=["d_af2", "d_of3", "burial"])
    df = df.reset_index(drop=True)
    merge_frac = len(df) / len(pc)
    print(f"  merged {len(df)}/{len(pc)} complexes with interface burial (frac={merge_frac:.2f})")

    x, y = df.d_of3.to_numpy(), df.d_af2.to_numpy()
    bur = df.burial.to_numpy()
    Zb = df[["burial"]].to_numpy()
    Zm = df[["burial", "nbr", "size"]].to_numpy()

    raw = float(stats.spearmanr(x, y).correlation)
    par_b = partial_spearman(x, y, Zb)
    par_m = partial_spearman(x, y, Zm)
    # also: do OF3/AF2 deficits individually correlate with burial? (is there a confound to remove at all)
    rho_of3_bur = float(stats.spearmanr(x, bur).correlation)
    rho_af2_bur = float(stats.spearmanr(y, bur).correlation)

    # complex-bootstrap CIs
    def boot(fn, n=5000):
        out = []
        for _ in range(n):
            i = rng.choice(len(df), len(df), True)
            try:
                out.append(fn(i))
            except Exception:
                pass
        return np.percentile(out, [2.5, 97.5])
    raw_ci = boot(lambda i: stats.spearmanr(x[i], y[i]).correlation)
    parb_ci = boot(lambda i: partial_spearman(x[i], y[i], Zb[i]))
    parm_ci = boot(lambda i: partial_spearman(x[i], y[i], Zm[i]))

    # top-3 leverage jackknife: which single removals most INFLATE rho (rho_full - rho_without_i > 0)
    lev = np.array([raw - stats.spearmanr(np.delete(x, i), np.delete(y, i)).correlation for i in range(len(df))])
    top3 = np.argsort(lev)[::-1][:3]
    keep = np.setdiff1d(np.arange(len(df)), top3)
    raw_drop3 = float(stats.spearmanr(x[keep], y[keep]).correlation)
    parb_drop3 = partial_spearman(x[keep], y[keep], Zb[keep])
    print(f"  raw rho(d_of3,d_af2)          = {raw:+.3f} [{raw_ci[0]:+.3f},{raw_ci[1]:+.3f}]")
    print(f"  partial | burial              = {par_b:+.3f} [{parb_ci[0]:+.3f},{parb_ci[1]:+.3f}]")
    print(f"  partial | burial+nbr+size     = {par_m:+.3f} [{parm_ci[0]:+.3f},{parm_ci[1]:+.3f}]")
    print(f"  corr(d_of3,burial)={rho_of3_bur:+.3f}  corr(d_af2,burial)={rho_af2_bur:+.3f}  (confound present only if these are large)")
    print(f"  drop top-3 leverage (complexes {list(df.complex_id.iloc[top3])}): raw={raw_drop3:+.3f}  partial|burial={parb_drop3:+.3f}")

    survives = (parb_ci[0] > 0) and (raw_drop3 > 0.3)
    verdict = ("SURVIVES burial (agreement is NOT a recursive burial confound)" if survives
               else "does NOT survive — cross-predictor tax may be recursive burial")
    print(f"  VERDICT: {verdict}")

    rows = [dict(metric="raw_rho", value=round(raw, 4), lo=round(raw_ci[0], 4), hi=round(raw_ci[1], 4)),
            dict(metric="partial_rho_given_burial", value=round(par_b, 4), lo=round(parb_ci[0], 4), hi=round(parb_ci[1], 4)),
            dict(metric="partial_rho_given_burial_nbr_size", value=round(par_m, 4), lo=round(parm_ci[0], 4), hi=round(parm_ci[1], 4)),
            dict(metric="corr_dof3_burial", value=round(rho_of3_bur, 4), lo="", hi=""),
            dict(metric="corr_daf2_burial", value=round(rho_af2_bur, 4), lo="", hi=""),
            dict(metric="raw_rho_drop_top3", value=round(raw_drop3, 4), lo="", hi=""),
            dict(metric="partial_rho_burial_drop_top3", value=round(parb_drop3, 4), lo="", hi="")]
    out = pd.DataFrame(rows)
    out["n_complexes"] = len(df); out["merge_frac"] = round(merge_frac, 3); out["seed"] = SEED
    out["top3_dropped"] = ";".join(df.complex_id.iloc[top3]); out["verdict"] = verdict
    out["command"] = "python3 src/deficit_burial_residualize.py"
    out.to_csv(a.out, index=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()

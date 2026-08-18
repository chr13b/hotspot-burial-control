#!/usr/bin/env python3
"""KL audit, part 3 — the decisive test, using THIS PROJECT'S OWN pre-registered protocol.

Parts 1-2 showed the committed readout (an unfitted equal-weight z-sum dAUROC, compared against a
null of 0 when its true noise floor is -0.022) cannot decide the question. The clean, model-free,
combiner-free test is the one the project already pre-registered for Phase 0 (PREREG.md): WITHIN-
COMPLEX OPTIMAL 1:1 MATCHED PAIRS on geometry, extended with the third geometry feature ΔSASA.

  match hot -> null inside the same complex on   |Δburial| <= 0.05, |Δnbr| <= 1, |ΔΔSASA| <= 0.05
  then ask: is KL higher in the HOT member of the pair?

Matched-pair AUROC = P(KL_hot > KL_null) over pairs. 0.5 = KL carries nothing beyond the matched
geometry. No combiner, no weights, no regression, no stratum-resolution choice.
Negative control: logp_native (confidence) must stay at 0.5. Sanity control: ΔSASA must collapse
to ~0.5 because it is matched.

  python3 src/kl_readout_audit3.py --out results/kl_readout_audit3.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

SEED = 20260803
NBOOT = 2000
BIG = 1e6


def match_pairs(df, tol_bur=0.05, tol_nbr=1, tol_dsa=0.05):
    """Within-complex optimal 1:1 hot->null matching on burial, nbr, ΔSASA."""
    out = []
    for cid, gdf in df.groupby("cid"):
        H = gdf[gdf.y == 1]; N = gdf[gdf.y == 0]
        if len(H) == 0 or len(N) == 0:
            continue
        db = np.abs(H.burial.values[:, None] - N.burial.values[None, :])
        dn = np.abs(H.nbr.values[:, None] - N.nbr.values[None, :])
        dd = np.abs(H.dsasa.values[:, None] - N.dsasa.values[None, :])
        ok = (db <= tol_bur) & (dn <= tol_nbr) & (dd <= tol_dsa)
        cost = np.where(ok, db / tol_bur + dn / max(tol_nbr, 1e-9) + dd / tol_dsa, BIG)
        ri, ci = linear_sum_assignment(cost)
        for i, jj in zip(ri, ci):
            if cost[i, jj] < BIG:
                out.append(dict(cid=cid, hi=H.index[i], ni=N.index[jj],
                                dbur=db[i, jj], dnbr=dn[i, jj], ddsa=dd[i, jj]))
    return pd.DataFrame(out)


def run(name, df, feats, rows, tols):
    """tols are given in SD UNITS of each feature so the two fixtures are directly comparable.
    (Bennett's ΔSASA is absolute Å²; SKEMPI's p0 drsasa is relative — a fixed absolute tolerance
    is meaningless across both.) nbr keeps its native ±count scale."""
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    sd_b, sd_d = df.burial.std(), df.dsasa.std()
    for f_bur, tol_nbr, f_dsa in tols:
        tol_bur, tol_dsa = f_bur * sd_b, f_dsa * sd_d
        print(f"\n[{name}] tolerance {f_bur}SD burial = {tol_bur:.4f}, "
              f"{f_dsa}SD ΔSASA = {tol_dsa:.4f}, nbr ±{tol_nbr}")
        P = match_pairs(df, tol_bur, tol_nbr, tol_dsa)
        if len(P) == 0:
            continue
        cids = P.cid.unique()
        idx_by = {c: np.where(P.cid.values == c)[0] for c in cids}
        print(f"    {len(P)} matched pairs over {len(cids)} complexes")
        print(f"    residual imbalance: |Δburial|={P.dbur.mean():.4f}  |Δnbr|={P.dnbr.mean():.3f}  "
              f"|ΔΔSASA|={P.ddsa.mean():.4f}")
        for f in feats:
            if f not in df:
                continue
            vh = df[f].to_numpy(float)[P.hi.values]; vn = df[f].to_numpy(float)[P.ni.values]
            d = vh - vn
            m = np.isfinite(d)
            if m.sum() < 10:
                continue
            # matched-pair AUROC = P(hot > null) + 0.5 P(tie)
            mp = float((np.sign(d[m]) > 0).mean() + 0.5 * (d[m] == 0).mean())
            b = []
            for _ in range(NBOOT):
                t = np.concatenate([idx_by[c] for c in rng.choice(cids, len(cids), True)])
                dd = d[t]; dd = dd[np.isfinite(dd)]
                if len(dd):
                    b.append(float((np.sign(dd) > 0).mean() + 0.5 * (dd == 0).mean()))
            lo, hi = np.percentile(b, [2.5, 97.5])
            w = stats.wilcoxon(d[m]).pvalue if len(np.unique(d[m])) > 1 else np.nan
            vd = "ADDS beyond matched geometry" if lo > 0.5 else (
                 "ANTI" if hi < 0.5 else "null (nothing beyond geometry)")
            print(f"    {f:8s} matched-pair AUROC = {mp:.4f} [{lo:.4f},{hi:.4f}]  "
                  f"meanΔ={np.nanmean(d):+.4f}  Wilcoxon p={w:.2e}  {vd}")
            rows.append(dict(fixture=name, tol=f"bur{f_bur}SD_nbr{tol_nbr}_dsasa{f_dsa}SD",
                             quantity=f, matched_pair_auroc=round(mp, 4), lo=round(lo, 4),
                             hi=round(hi, 4), mean_delta=round(float(np.nanmean(d)), 5),
                             wilcoxon_p=w, verdict=vd, n_pairs=len(P), n_complexes=len(cids),
                             resid_dbur=round(float(P.dbur.mean()), 4),
                             resid_dnbr=round(float(P.dnbr.mean()), 3),
                             resid_ddsasa=round(float(P.ddsa.mean()), 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kl_readout_audit3.csv")
    a = ap.parse_args()
    rows = []
    j = pd.read_csv("results/kl_detector_joined.csv")
    j = j[j.is_interface == 1].copy(); j["icode"] = j.icode.fillna("").astype(str)
    pos = pd.read_csv("results/p0_positions.csv",
                      usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    pos["icode"] = pos.icode.fillna("").astype(str)
    j = j.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left")
    j = j.rename(columns={"drsasa": "dsasa", "is_hot": "y", "complex_id": "cid"}).dropna(
        subset=["dsasa", "kl", "burial", "nbr", "y"])
    j["conf"] = j.logp_native; j["negH"] = -j.H_complex
    run("SKEMPI_crystal", j, ["kl", "jsd", "dH", "negH", "conf", "dsasa", "burial"], rows,
        [(0.15, 1, 0.15), (0.30, 1, 0.30), (0.50, 2, 0.50)])

    b = pd.read_csv("results/bennett_kl_positions.csv")
    b = b[(b.native_match == 1) & (b.is_interface == 1)].copy()
    b["y"] = (b.restr >= 0.75).astype(int); b = b.rename(columns={"parent": "cid"})
    run("Bennett_denovo", b, ["kl", "dsasa", "burial"], rows,
        [(0.15, 1, 0.15), (0.30, 1, 0.30), (0.50, 2, 0.50)])

    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_boot"] = NBOOT
    out["note"] = ("decisive combiner-free test of the KL demotion: within-complex optimal 1:1 "
                   "matched pairs on burial+nbr+ΔSASA (the project's pre-registered Phase-0 protocol "
                   "extended with ΔSASA); matched-pair AUROC = P(KL_hot > KL_null)")
    out["command"] = "python3 src/kl_readout_audit3.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

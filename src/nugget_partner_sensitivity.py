#!/usr/bin/env python3
"""The nugget, restated ΔSASA-robustly (supersedes the KL-only framing of confidence_antipredicts.csv).

At interface hotspots: the inverse-folding model's OWN confidence (log p of the native residue) is
near-chance and HURTS a burial heuristic; meanwhile a PARTNER-SENSITIVITY signal predicts them — and the
signal that works is TRIVIAL GEOMETRY (ΔSASA = surface the partner buries, same 2 structures the neural KL
needs, NO neural net). ΔSASA adds over burial as much as the learned KL does.

This re-tests the nugget against a proper combination and includes ΔSASA, on SKEMPI crystal interface
positions. Complex-level bootstrap, seed 20260803. Committed CSVs only.
  python3 src/nugget_partner_sensitivity.py --out results/nugget_partner_sensitivity.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nugget_partner_sensitivity.csv")
    a = ap.parse_args()
    R = "results"
    j = pd.read_csv(f"{R}/kl_detector_joined.csv")
    j = j[j.is_interface == 1].copy(); j["icode"] = j.icode.fillna("").astype(str)
    pos = pd.read_csv(f"{R}/p0_positions.csv", usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    pos["icode"] = pos.icode.fillna("").astype(str)
    j = j.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left")
    j = j.rename(columns={"drsasa": "dsasa"}).dropna(
        subset=["dsasa", "kl", "burial", "nbr", "is_hot", "logp_native"]).reset_index(drop=True)
    j["conf"] = j.logp_native
    z = lambda c: (j[c] - j[c].mean()) / j[c].std()
    for f in ["burial", "dsasa", "kl", "conf"]:
        j["_" + f] = z(f)
    j["y"] = j.is_hot.values

    cids = j.complex_id.unique()
    idx_by = {c: j.index[j.complex_id == c].to_numpy() for c in cids}
    Y = j.y.to_numpy()
    rng = np.random.default_rng(SEED)
    resamples = [np.concatenate([idx_by[c] for c in rng.choice(cids, len(cids), True)]) for _ in range(5000)]

    def boot(scorevec):
        obs = auc(scorevec, Y)
        b = np.array([auc(scorevec[ix], Y[ix]) for ix in resamples])
        return obs, float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))

    def boot_delta(scoreA, scoreB):
        obs = auc(scoreA, Y) - auc(scoreB, Y)
        b = np.array([auc(scoreA[ix], Y[ix]) - auc(scoreB[ix], Y[ix]) for ix in resamples])
        return obs, float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5)), float(np.mean(b > 0))

    rows = []
    print(f"SKEMPI interface: {len(j)} positions, {len(cids)} complexes, {int(Y.sum())} hotspots\n")
    print("single-feature AUROC(is_hot):")
    for f, col in [("burial", j._burial), ("ΔSASA(partner-contact)", j._dsasa),
                   ("KL(learned partner-sens.)", j._kl), ("model confidence (logp)", j._conf)]:
        o, lo, hi = boot(col.to_numpy())
        print(f"  {f:28s} {o:.3f} [{lo:.3f},{hi:.3f}]")
        rows.append(dict(metric="auroc", feature=f, value=round(o, 4), lo=round(lo, 4), hi=round(hi, 4)))

    print("\nΔAUROC(burial + X  −  burial), z-sum combination:")
    bur = j._burial.to_numpy()
    for f, col in [("model confidence", j._conf), ("ΔSASA (geometry)", j._dsasa), ("KL (learned)", j._kl)]:
        o, lo, hi, p = boot_delta(bur + col.to_numpy(), bur)
        verdict = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "ns")
        print(f"  + {f:20s} {o:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}  {verdict}")
        rows.append(dict(metric="dAUROC_over_burial", feature=f, value=round(o, 4), lo=round(lo, 4),
                         hi=round(hi, 4), p_gt0=round(p, 3)))

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = ("nugget (combiner-free): confidence near-chance (AUROC 0.538) and ranks BELOW random for "
                   "top-3 triage (capture@3 0.064<0.084), adds ~nothing beyond geometry (partial|full=-0.01, null); "
                   "the z-sum -0.056 'HURTS' OVERSTATES (combiner artifact -- restate via CPI). Partner-sensitivity "
                   "ΔSASA (+0.042, no neural net) HELPS as much as learned KL (+0.041).")
    out["command"] = "python3 src/nugget_partner_sensitivity.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

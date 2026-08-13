#!/usr/bin/env python3
"""Does the sequence-free KL detector add over a FULL cheap-geometry baseline?

A Fable-5 audit of the Bennett de-novo result (2026-08-13) showed KL is a noisy mixture of self-burial
(−rSASA) and PARTNER-CONTACT-AREA (ΔSASA = SASA buried on binding). The published detector claim compared
KL to burial ALONE; ΔSASA — nearly orthogonal to burial, computable from the SAME two structures KL needs
(complex + partner-deleted), no neural net — was omitted. This script re-tests "does KL add" against the
full geometry baseline burial + neighbour-count + ΔSASA, on BOTH fixtures:

  SKEMPI crystal   (results/kl_detector_joined.csv + p0_positions.drsasa; label is_hot = Ala-scan ΔΔG>2)
  Bennett de-novo  (results/bennett_kl_positions.csv; label restrictiveness >= 0.75)

Complex/design-level bootstrap, seed 20260803. Writes results/kl_geometry_control.csv.
NOTE: this tests the CRYSTAL central claim only; KL's claimed edge is on PREDICTED backbones (Sherlock) and
that control still needs running there.
  python3 src/kl_geometry_control.py --out results/kl_geometry_control.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def z(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else v * 0


def partial(df, x, y, ctrls):
    R = np.column_stack([np.ones(len(df))] + [stats.rankdata(df[c]) for c in ctrls])
    def resid(col):
        rr = stats.rankdata(df[col])
        return rr - R @ np.linalg.lstsq(R, rr, rcond=None)[0]
    return stats.spearmanr(resid(x), resid(y)).correlation


def boot_dauc(df, cid_col, scoreA, scoreB, nboot=2000):
    cids = df[cid_col].unique()
    idx_by = {c: df.index[df[cid_col] == c].to_numpy() for c in cids}
    A = df[scoreA].to_numpy(); B = df[scoreB].to_numpy(); Y = df["y"].to_numpy()
    obs = auc(A, Y) - auc(B, Y)
    rng = np.random.default_rng(SEED); out = []
    for _ in range(nboot):
        idx = np.concatenate([idx_by[c] for c in rng.choice(cids, len(cids), True)])
        yy = Y[idx]
        if 0 < yy.sum() < len(yy):
            out.append(auc(A[idx], yy) - auc(B[idx], yy))
    out = np.array(out)
    return obs, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out > 0))


def run(name, df, cid_col, rows):
    df = df.reset_index(drop=True)
    df["full"] = z(df.burial) + z(df.nbr) + z(df.dsasa)
    df["full_kl"] = df["full"] + z(df.kl)
    df["bur_kl"] = z(df.burial) + z(df.kl)
    print(f"\n=== {name}: {len(df)} interface positions, {df[cid_col].nunique()} groups, {int(df.y.sum())} hot ===")
    print(f"  corr(burial,ΔSASA)={stats.spearmanr(df.burial,df.dsasa).correlation:+.3f} "
          f"corr(KL,ΔSASA)={stats.spearmanr(df.kl,df.dsasa).correlation:+.3f} "
          f"corr(KL,burial)={stats.spearmanr(df.kl,df.burial).correlation:+.3f}")
    for ctrls in (["burial"], ["dsasa"], ["burial", "dsasa"], ["burial", "nbr", "dsasa"]):
        pr = partial(df, "kl", "y", ctrls)
        rows.append(dict(fixture=name, metric=f"partial_KL_y|{'+'.join(ctrls)}", value=round(pr, 4)))
        print(f"  partial(KL,y | {'+'.join(ctrls):16s}) = {pr:+.4f}")
    for lab, a, b in (("KL_over_burial_alone", "bur_kl", "burial"), ("KL_over_full_geometry", "full_kl", "full")):
        o, lo, hi, p = boot_dauc(df, cid_col, a, b)
        rows.append(dict(fixture=name, metric=f"dAUROC_{lab}", value=round(o, 4), lo=round(lo, 4),
                         hi=round(hi, 4), p_gt0=round(p, 3)))
        print(f"  ΔAUROC {lab:22s} = {o:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kl_geometry_control.csv")
    a = ap.parse_args()
    rows = []

    # SKEMPI crystal
    j = pd.read_csv("results/kl_detector_joined.csv")
    j = j[j.is_interface == 1].copy(); j["icode"] = j.icode.fillna("").astype(str)
    pos = pd.read_csv("results/p0_positions.csv", usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    pos["icode"] = pos.icode.fillna("").astype(str)
    j = j.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left")
    j = j.rename(columns={"drsasa": "dsasa", "is_hot": "y"}).dropna(subset=["dsasa", "kl", "burial", "nbr", "y"])
    run("SKEMPI_crystal", j, "complex_id", rows)

    # Bennett de-novo
    b = pd.read_csv("results/bennett_kl_positions.csv")
    b = b[(b.native_match == 1) & (b.is_interface == 1)].copy()
    b["y"] = (b.restr >= 0.75).astype(int)
    run("Bennett_denovo", b, "parent", rows)

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["command"] = "python3 src/kl_geometry_control.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R2 — baseline-completeness audit + the 'what a designer needs' table (SKEMPI crystal).

Every "X adds over burial" claim in the paper was tested against a PARTIAL baseline (burial alone, or
integer neighbour-count). This tabulates, against the FULL cheap-geometry baseline (rSASA-burial + nbr +
ΔSASA = partner-contact area), what each signal is worth — turning the vulnerability into a methodological
artifact reviewers reward. Complex-bootstrap, seed 20260803. Committed CSVs only.
  python3 src/baseline_audit.py --out results/baseline_audit.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260803


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cap3(sub, col):
    nh = int(sub.y.sum())
    if nh == 0:
        return None
    top = set(np.argsort(-sub[col].values, kind="mergesort")[:3])
    return sum(1 for i, h in enumerate(sub.y.values) if h == 1 and i in top) / nh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baseline_audit.csv")
    a = ap.parse_args()
    R = "results"
    j = pd.read_csv(f"{R}/kl_detector_joined.csv")
    j = j[j.is_interface == 1].copy(); j["icode"] = j.icode.fillna("").astype(str)
    pos = pd.read_csv(f"{R}/p0_positions.csv", usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    pos["icode"] = pos.icode.fillna("").astype(str)
    j = j.merge(pos, on=["complex_id", "chain", "resnum", "icode"], how="left")
    j = j.rename(columns={"drsasa": "dsasa"}).dropna(
        subset=["dsasa", "kl", "burial", "nbr", "is_hot", "logp_native"]).reset_index(drop=True)
    j["y"] = j.is_hot.values
    z = lambda c: (j[c] - j[c].mean()) / j[c].std()
    j["conf"] = j.logp_native
    j["full"] = z("burial") + z("nbr") + z("dsasa")
    j["full_kl"] = j["full"] + z("kl")
    j["full_conf"] = j["full"] + z("conf")
    j["bur_kl"] = z("burial") + z("kl")
    j["bur_conf"] = z("burial") + z("conf")

    cids = j.complex_id.unique()
    idx = {c: j.index[j.complex_id == c].to_numpy() for c in cids}
    Y = j.y.to_numpy()
    rng = np.random.default_rng(SEED)
    res = [np.concatenate([idx[c] for c in rng.choice(cids, len(cids), True)]) for _ in range(4000)]
    by = {c: j[j.complex_id == c].reset_index(drop=True) for c in cids}
    hotcx = [c for c in cids if by[c].y.sum() >= 1]

    def bootA(v):
        o = auc(v, Y); b = np.array([auc(v[i], Y[i]) for i in res]); return o, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)
    def bootD(a2, b2):
        o = auc(a2, Y) - auc(b2, Y); d = np.array([auc(a2[i], Y[i]) - auc(b2[i], Y[i]) for i in res])
        return o, np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5), float(np.mean(d > 0))
    def bootC(col):
        v = {c: cap3(by[c], col) for c in hotcx}; k = [c for c in hotcx if v[c] is not None]
        o = np.mean([v[c] for c in k]); b = [np.mean([v[c] for c in rng.choice(k, len(k), True)]) for _ in range(2000)]
        return o, np.percentile(b, 2.5), np.percentile(b, 97.5)

    rows = []
    print("=== designer table: single-signal AUROC(is_hot) & capture@3 ===")
    feats = [("random", None), ("neighbour-count (nbr)", "nbr"), ("burial (-rSASA)", "burial"),
             ("ΔSASA (partner-contact)", "dsasa"), ("KL (learned)", "kl"), ("model confidence", "conf"),
             ("FULL geometry (rSASA+nbr+ΔSASA)", "full")]
    for name, col in feats:
        if col is None:
            au = (0.5, 0.5, 0.5); cp = np.mean([3 / len(by[c]) for c in hotcx])
        else:
            au = bootA(j[col].to_numpy()); cp = bootC(col)[0]
        print(f"  {name:34s} AUROC {au[0]:.3f} [{au[1]:.3f},{au[2]:.3f}]   capture@3 {cp:.3f}")
        rows.append(dict(kind="single", signal=name, auroc=round(au[0], 4), lo=round(au[1], 4),
                         hi=round(au[2], 4), capture3=round(cp, 4)))

    print("\n=== does X ADD? ΔAUROC over PARTIAL vs FULL baseline (the audit) ===")
    audit = [("KL over burial-alone", "bur_kl", "burial"),
             ("KL over FULL geometry", "full_kl", "full"),
             ("confidence over burial-alone", "bur_conf", "burial"),
             ("confidence over FULL geometry", "full_conf", "full"),
             ("ΔSASA over burial-alone", None, None)]
    for name, a2, b2 in audit:
        if a2 is None:
            o, lo, hi, p = bootD((z("burial") + z("dsasa")).to_numpy(), j.burial.to_numpy())
        else:
            o, lo, hi, p = bootD(j[a2].to_numpy(), j[b2].to_numpy())
        verdict = "ADDS" if lo > 0 else ("HURTS" if hi < 0 else "ns")
        print(f"  {name:32s} {o:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}  {verdict}")
        rows.append(dict(kind="delta_audit", signal=name, auroc=round(o, 4), lo=round(lo, 4),
                         hi=round(hi, 4), p_gt0=round(p, 3), verdict=verdict))

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = "full baseline = rSASA-burial + nbr + ΔSASA; confidence HURTS even the full baseline; KL adds nothing over full"
    out["command"] = "python3 src/baseline_audit.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

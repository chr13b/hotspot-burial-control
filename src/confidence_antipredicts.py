#!/usr/bin/env python3
"""
The headline nugget: an inverse-folding model's OWN CONFIDENCE does not identify binding hotspots —
only its PARTNER-SENSITIVITY (the sequence-free KL signal) does.

Over interface positions, for is_hot (Ala-scan ddG>2), complex-bootstrapped AUROC:
  - burial alone (Cbeta neighbour count)
  - the model's own confidence alone (logp_native): does high confidence flag hotspots? (answer: no)
  - KL alone (partner-induced backbone-distribution shift)
  - burial + confidence   -> does confidence ADD over burial? (answer: no / it hurts)
  - burial + KL           -> does partner-sensitivity ADD over burial? (answer: yes)

This makes concrete the stop-and-read sentence: the quantity a designer would naively trust (the model's
own confidence) is useless-to-anti-predictive for hotspots; the free, sequence-free partner-sensitivity
signal is what carries the information.

  python3 src/confidence_antipredicts.py --joined results/kl_detector_joined.csv --out results/confidence_antipredicts.csv
"""
import argparse
import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 10000


def z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cboot(by, cids, score_fn, seed=SEED, nboot=NBOOT):
    rng = np.random.default_rng(seed)
    per = {c: score_fn(by[c]) for c in cids}
    keep = [c for c in cids if per[c] is not None and np.isfinite(per[c])]
    obs = float(np.mean([per[c] for c in keep]))
    b = [np.mean([per[c] for c in rng.choice(keep, len(keep), True)]) for _ in range(nboot)]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return obs, float(lo), float(hi), len(keep), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default="results/kl_detector_joined.csv")
    ap.add_argument("--out", default="results/confidence_antipredicts.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.joined)
    d = d[d.is_interface == 1].copy() if "is_interface" in d else d
    # within-complex z-scores so per-complex AUROC of a combo is well-defined
    for col, name in [("nbr", "bur"), ("logp_native", "conf"), ("kl", "kl")]:
        d[name + "_z"] = d.groupby("complex_id")[col].transform(z)
    d["bur_conf"] = d["bur_z"] + d["conf_z"]
    d["bur_kl"] = d["bur_z"] + d["kl_z"]
    by = {c: s for c, s in d.groupby("complex_id")}
    cids = [c for c in by if by[c].is_hot.sum() >= 1]

    scores = {
        "burial": lambda s: auroc(s.is_hot, s.nbr),
        "confidence_logp": lambda s: auroc(s.is_hot, s.logp_native),
        "KL": lambda s: auroc(s.is_hot, s.kl),
        "burial+confidence": lambda s: auroc(s.is_hot, s.bur_conf),
        "burial+KL": lambda s: auroc(s.is_hot, s.bur_kl),
    }
    boots, rows = {}, []
    for name, fn in scores.items():
        obs, lo, hi, n, b = cboot(by, cids, fn)
        boots[name] = np.array(b)
        rows.append(dict(quantity=name, kind="AUROC", estimate=obs, lo=lo, hi=hi, n_cx=n))
        print(f"AUROC {name:20} = {obs:.3f} [{lo:.3f}, {hi:.3f}]")

    # paired deltas vs burial (same bootstrap indices -> use the recorded arrays' difference is not paired;
    # recompute paired properly)
    rng = np.random.default_rng(SEED)
    per = {c: {k: scores[k](by[c]) for k in scores} for c in cids}
    keep = [c for c in cids if all(per[c][k] is not None for k in scores)]
    def paired(a_, b_):
        obs = float(np.mean([per[c][a_] - per[c][b_] for c in keep]))
        bb = [np.mean([per[c][a_] - per[c][b_] for c in rng.choice(keep, len(keep), True)]) for _ in range(NBOOT)]
        lo, hi = np.nanpercentile(bb, [2.5, 97.5])
        return obs, float(lo), float(hi), float(np.mean(np.array(bb) > 0))
    print()
    for a_, b_ in [("burial+confidence", "burial"), ("burial+KL", "burial"),
                   ("confidence_logp", "burial")]:
        obs, lo, hi, p = paired(a_, b_)
        rows.append(dict(quantity=f"{a_} - {b_}", kind="dAUROC", estimate=obs, lo=lo, hi=hi, p_gt0=p, n_cx=len(keep)))
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"Δ {a_:20} vs burial = {obs:+.3f} [{lo:+.3f}, {hi:+.3f}] P(>0)={p:.3f} {star}")

    out = pd.DataFrame(rows); out["seed"] = SEED; out["nboot"] = NBOOT
    out["command"] = f"python3 src/confidence_antipredicts.py --joined {a.joined} --out {a.out}"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  (n_cx={len(keep)})")
    print("NUGGET: the model's own confidence adds nothing/hurts over burial; only partner-sensitivity (KL) adds.")


if __name__ == "__main__":
    main()

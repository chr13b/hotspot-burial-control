#!/usr/bin/env python3
"""Apples-to-apples strengthener (audit W2): run the leverage position-level CPI on the EXACT
5,742-position sample that nugget_cpi.csv used, so confidence/KL/leverage are on ONE per-observation
CPI scale and the '13,401 directly comparable to nugget' overstatement is retired with a real number.

The full leverage position frame is 13,401 interface positions / 343 complexes; nugget_cpi.py runs on
5,742 positions / 141 complexes (the subset that also carries a committed KL-detector value). CPI is a
per-observation mean log-loss, so the two are NOT numerically comparable. Restricting the leverage
features to nugget's own sample gives the honest side-by-side:
  confidence should reproduce nugget's +0.000 exactly; leverage L(->Ala) should be several x the scalar KL.

Reuses the verified machinery verbatim (leverage_decomposition.position_frame + .cpi + .zs); pure sample
restriction, no new statistics. seed 20260803.
  python3 src/leverage_nugget_match.py --out results/leverage_nugget_match.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leverage_decomposition as LD


def nugget_sample_keys():
    """The exact (complex_id, chain, resnum, icode) set nugget_cpi.py conditions on."""
    j = pd.read_csv("results/kl_detector_joined.csv")
    j = j[j.is_interface == 1].copy()
    j["icode"] = j.icode.fillna("").astype(str)
    p0 = pd.read_csv("results/p0_positions.csv",
                     usecols=["complex_id", "chain", "resnum", "icode", "drsasa"])
    p0["icode"] = p0.icode.fillna("").astype(str)
    j = j.merge(p0, on=["complex_id", "chain", "resnum", "icode"], how="left").rename(
        columns={"drsasa": "dsasa"})
    j = j.dropna(subset=["dsasa", "kl", "burial", "nbr", "is_hot", "logp_native"])
    return j[["complex_id", "chain", "resnum", "icode"]].drop_duplicates()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/leverage_nugget_match.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(LD.SEED)

    pos, _Lvec, _lP, _lQ = LD.position_frame()
    pos["icode"] = pos.icode.fillna("").astype(str)
    keys = nugget_sample_keys()
    m = pos.merge(keys, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    need = ["burial", "nbr", "drsasa", "L_ala", "klP", "conf"]
    m = m.dropna(subset=need).reset_index(drop=True)
    y = m.is_hot.to_numpy().astype(float)
    g = m.complex_id.to_numpy()
    print(f"[nugget-match] leverage frame {len(pos)} positions -> matched to nugget sample "
          f"{len(m)} positions / {m.complex_id.nunique()} complexes / {int(y.sum())} hotspots "
          f"({100*y.mean():.1f}% base rate)")

    for c in need:
        m[c + "z"] = LD.zs(m[c])
    Z = m[["burialz", "nbrz", "drsasaz"]].to_numpy()

    rows = []
    ref = {"leverage L(->Ala)": None, "scalar KL": None, "confidence": None}
    for f, lab in [("L_alaz", "leverage L(->Ala)"), ("klPz", "scalar KL"), ("confz", "confidence")]:
        c, lo, hi, p, _sZ, _ = LD.cpi(y, g, Z, m[f].to_numpy().copy(), rng)
        ref[lab] = c
        verdict = "ADDS (CI>0)" if lo > 0 else "conditionally INDEPENDENT (CI spans 0)"
        print(f"  CPI[{lab:20s} | burial+nbr+ΔSASA] = {c:+.5f} [{lo:+.5f},{hi:+.5f}] "
              f"P(>0)={p:.3f}  {verdict}")
        rows.append(dict(sample="nugget_5742", feature=lab, cpi=round(c, 5), lo=round(lo, 5),
                         hi=round(hi, 5), p_gt0=round(p, 3), verdict=verdict,
                         n=len(m), n_groups=int(m.complex_id.nunique()), n_hot=int(y.sum())))
    if ref["scalar KL"] and abs(ref["scalar KL"]) > 1e-9:
        print(f"  -> leverage L is {ref['leverage L(->Ala)']/ref['scalar KL']:.1f}x the scalar KL "
              f"on the matched sample; confidence adds {ref['confidence']:+.5f} (≈ geometry)")
    out = pd.DataFrame(rows)
    out["seed"] = LD.SEED
    out["note"] = ("audit W2 strengthener: leverage features on the EXACT nugget_cpi.csv sample so "
                   "confidence/KL/L are on one per-observation CPI scale")
    out["command"] = "python3 src/leverage_nugget_match.py"
    out.to_csv(a.out, index=False)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

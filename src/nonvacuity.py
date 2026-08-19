"""Precise non-vacuity of the blindness result (audit D): the fraction of leverage spread that survives
matching the bound distribution P, with the CORRECT reference (median |ΔL| between RANDOM position pairs),
not SD(L). If confidence (a functional of P) determined leverage, matched pairs would have |ΔL| -> 0.

  python3 src/nonvacuity.py --out results/nonvacuity.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
SEED = 20260803

def med_random(L, rng, npairs=200000):
    L = np.asarray(L, float); L = L[np.isfinite(L)]
    i = rng.integers(0, len(L), npairs); j = rng.integers(0, len(L), npairs)
    return float(np.median(np.abs(L[i] - L[j])))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/nonvacuity.csv"); a = ap.parse_args()
    rng = np.random.default_rng(SEED); rows = []
    for name, posf, decf, fix in [
        ("ProteinMPNN", "results/leverage_skempi_positions.csv", "results/leverage_decomposition.csv", "SKEMPI"),
        ("ESM-IF1", "results/leverage_esmif_positions.csv", "results/leverage_esmif.csv", None)]:
        lp = pd.read_csv(posf); lp = lp[lp.is_interface == True]                       # noqa: E712
        mr = med_random(lp.L_rms, rng)
        dec = pd.read_csv(decf); r = dec[dec.test == "theorem_confidence_blind_to_leverage"]
        if fix: r = r[r.fixture == fix]
        mm = float(r.stat.iloc[0])
        frac = mm / mr
        print(f"  {name:11s}: matched median|ΔL|={mm:.4f}, random-pair median|ΔL|={mr:.4f} "
              f"-> survives {100*frac:.0f}% of the random-pair leverage spread")
        rows.append(dict(model=name, matched_median_dL=round(mm, 4), random_median_dL=round(mr, 4),
                         survives_frac=round(frac, 3)))
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"[wrote] {a.out}")

if __name__ == "__main__":
    main()

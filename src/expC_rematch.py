"""Exp C exploratory: within-binder burial-matched pairs (gap-power salvage).

The committed SECONDARY_B pairs match each interface hotspot to a burial-matched control anywhere in the
complex; because the target is the larger partner, the control usually lands on the TARGET. Restricting
the pre-registered gap to pairs with BOTH ends on the diffused binder therefore leaves only ~7 pairs /
5 complexes — too few to read a dose-response. This script RE-MATCHES each binder interface hotspot to a
burial-matched control ON THE SAME (diffused) BINDER, using the IDENTICAL rule as SECONDARY_B
(rSASA_complex ±0.05, same SS class, |Δnbr| ≤ 1; control = any non-hot interface position; optimal 1:1),
but with the candidate pool restricted to the binder chains. It is fixed on the CRYSTAL, BEFORE any
generated backbone is scored, so no reading is moved after seeing a number.

EXPLORATORY: no pre-registered falsifier attaches. It is reported ALONGSIDE (never instead of) the
pre-registered reuse+restrict gap, whose retained pair count is reported honestly.

Usage:
  python3 src/expC_rematch.py --out results/p0_dssp_pairs_EXPC_within_binder.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p0_burial_matched import match_pairs  # identical matching logic, reused verbatim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--complexes", default="results/expC_complexes.csv")
    ap.add_argument("--out", default="results/p0_dssp_pairs_EXPC_within_binder.csv")
    a = ap.parse_args()

    pos = pd.read_csv(a.positions)
    cx = pd.read_csv(a.complexes)
    binder = {r.complex_id: set(str(r.binder_chains)) for r in cx.itertuples()}

    # keep only Exp C complexes, and within them only BINDER-chain positions (the diffused half)
    pos = pos[pos["complex_id"].isin(binder)].copy()
    keep = [str(r.chain) in binder.get(r.complex_id, set()) for r in pos.itertuples()]
    posb = pos[keep].reset_index(drop=True)

    # SECONDARY_B rule (control="any", strict=False, nbr_tol=1, rs_tol=0.05), pool = binder only
    pairs = match_pairs(posb, control="any", strict=False, nbr_tol=1)
    pairs.to_csv(a.out, index=False)
    n_cx = pairs["complex_id"].nunique() if not pairs.empty else 0
    print(f"[expC_rematch] {len(pairs)} within-binder pairs across {n_cx} complexes -> {a.out}")
    if not pairs.empty:
        print("  per-complex counts:",
              pairs.groupby("complex_id").size().sort_values(ascending=False).head(12).to_dict())


if __name__ == "__main__":
    main()

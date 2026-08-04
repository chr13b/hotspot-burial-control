"""Recompute secondary structure with pydssp and patch it into a positions CSV.

Secondary-structure class is one of the three matching constraints, so the matched
pairs depend on it. The original run used a self-implemented Kabsch-Sander assignment
that over-calls both helix and strand (see ftax_common.secondary_structure). This
recomputes SS with pydssp and rewrites the `ss` column, keeping the original as
`ss_kabsch`, so stage 2 can be re-run without redoing any model inference.

Usage:
  python3 src/patch_ss.py --positions results/p0_positions.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    a = ap.parse_args()

    pos = pd.read_csv(a.positions)
    pos["icode"] = pos["icode"].fillna("").astype(str)
    if "ss_kabsch" not in pos.columns:
        pos["ss_kabsch"] = pos["ss"]

    new_ss, n_ok, n_fail = {}, 0, 0
    for cid in sorted(pos["complex_id"].unique()):
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            n_fail += 1
            continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
            if cx is None:
                n_fail += 1
                continue
            ss = fc.secondary_structure(cx, prefer_pydssp=True)
            for i in range(cx.n):
                new_ss[(cid, cx.chains[i], int(cx.resnums[i]), cx.icodes[i])] = ss[i]
            n_ok += 1
        except Exception as e:
            print(f"  skip {cid}: {type(e).__name__}: {e}")
            n_fail += 1

    key = list(zip(pos["complex_id"], pos["chain"], pos["resnum"], pos["icode"]))
    pos["ss"] = [new_ss.get(k, np.nan) for k in key]
    n_missing = pos["ss"].isna().sum()
    pos["ss"] = pos["ss"].fillna(pos["ss_kabsch"])

    print(f"[patch_ss] {n_ok} complexes re-assigned, {n_fail} failed, "
          f"{n_missing} positions fell back to Kabsch-Sander")
    agree = (pos["ss"] == pos["ss_kabsch"]).mean()
    print(f"[patch_ss] pydssp agrees with Kabsch-Sander at {agree:.1%} of positions")
    for c in ("H", "E", "L"):
        print(f"    {c}: kabsch {(pos['ss_kabsch']==c).mean():.3f} -> "
              f"pydssp {(pos['ss']==c).mean():.3f}")
    pos.to_csv(a.positions, index=False)
    print(f"[patch_ss] wrote {a.positions}")


if __name__ == "__main__":
    main()

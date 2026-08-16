#!/usr/bin/env python3
"""7-complex drop bias check.

ESM-IF1 could not score 7 oversized complexes (>1200 residues), so the MPNN(344) vs ESM-IF1(337)
comparison is not strictly like-for-like. Control: re-run the ProteinMPNN CPIs on the SAME 337-complex
subset (drop the 7) and compare to the committed full-344 values. If MPNN-on-337 ≈ MPNN-on-344, the
drop is negligible and the MPNN-vs-ESM-IF1 gap is a real model effect, not sample selection.

Reuses the verified machinery (leverage_decomposition.position_frame/build_skempi/cpi). seed 20260803.
  python3 src/leverage_dropcheck.py --out results/leverage_dropcheck.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leverage_decomposition as LD

DROPPED = {"1KBH_A_B", "2NYY_DC_A", "2NZ9_DC_A", "3VR6_ABCDEF_GH",
           "4GXU_ABCDEF_MN", "4LRX_AB_CD", "4NM8_ABCDEF_HL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/leverage_dropcheck.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(LD.SEED)
    rows = []

    # ---- position level
    pos, _Lvec, _lP, _lQ = LD.position_frame()
    gi = pos[pos.complex_id.isin(DROPPED)]
    print(f"[drop] 7 complexes contribute {len(gi)} / {len(pos)} interface positions "
          f"({int(gi.is_hot.sum())} / {int(pos.is_hot.sum())} strict hotspots)")
    for name, d in [("full-344", pos), ("subset-337", pos[~pos.complex_id.isin(DROPPED)])]:
        dd = d.dropna(subset=["burial", "nbr", "drsasa", "L_ala", "conf", "klP"]).reset_index(drop=True)
        y = dd.is_hot.to_numpy().astype(float); g = dd.complex_id.to_numpy()
        for c in ["burial", "nbr", "drsasa", "L_ala", "conf", "klP"]:
            dd[c + "z"] = LD.zs(dd[c])
        Z = dd[["burialz", "nbrz", "drsasaz"]].to_numpy()
        cL, loL, hiL, _, _, _ = LD.cpi(y, g, Z, dd["L_alaz"].to_numpy().copy(), rng)
        cC, loC, hiC, _, _, _ = LD.cpi(y, g, Z, dd["confz"].to_numpy().copy(), rng)
        print(f"  POS {name:11s}: n={len(dd):5d} hot={int(y.sum()):3d} | "
              f"CPI(L_ala|geom)={cL:+.5f}[{loL:+.5f},{hiL:+.5f}]  "
              f"CPI(conf|geom)={cC:+.5f}[{loC:+.5f},{hiC:+.5f}]")
        rows.append(dict(level="position", sample=name, n=len(dd), n_hot=int(y.sum()),
                         cpi_L=round(cL, 5), cpi_L_lo=round(loL, 5), cpi_L_hi=round(hiL, 5),
                         cpi_conf=round(cC, 5)))

    # ---- mutation level
    mut = LD.build_skempi([])
    mut["destab"] = (mut.ddG >= LD.HOT_DDG).astype(int)
    gm = mut[mut.complex_id.isin(DROPPED)]
    print(f"[drop] 7 complexes contribute {len(gm)} / {len(mut)} single mutations")
    for name, d in [("full-344", mut), ("subset-337", mut[~mut.complex_id.isin(DROPPED)])]:
        dd = d.dropna(subset=["burial", "nbr", "drsasa", "L"])
        if "is_interface" in dd.columns:
            dd = dd[dd.is_interface == 1]
        dd = dd.reset_index(drop=True)
        y = dd.destab.to_numpy().astype(float); g = dd.complex_id.to_numpy()
        for c in ["burial", "nbr", "drsasa", "L"]:
            dd[c + "z"] = LD.zs(dd[c])
        Z = dd[["burialz", "nbrz", "drsasaz"]].to_numpy()
        cL, loL, hiL, _, _, _ = LD.cpi(y, g, Z, dd["Lz"].to_numpy().copy(), rng)
        sp = stats.spearmanr(dd.L, dd.ddG, nan_policy="omit").correlation
        print(f"  MUT {name:11s}: n={len(dd):5d} | CPI(L|geom)={cL:+.5f}[{loL:+.5f},{hiL:+.5f}]  "
              f"Spearman(L,ddG)={sp:+.4f}")
        rows.append(dict(level="mutation", sample=name, n=len(dd),
                         cpi_L=round(cL, 5), cpi_L_lo=round(loL, 5), cpi_L_hi=round(hiL, 5),
                         spearman_L_ddG=round(float(sp), 4)))

    out = pd.DataFrame(rows); out["seed"] = LD.SEED
    out["note"] = "MPNN full-344 vs subset-337 (the 7 ESM-IF1-dropped complexes) — drop-bias control"
    out["command"] = "python3 src/leverage_dropcheck.py"
    out.to_csv(a.out, index=False)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

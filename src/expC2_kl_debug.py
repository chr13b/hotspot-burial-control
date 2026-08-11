"""Why is the interface-formed C2-KL (kl_by_T_ifaceok) missing for T5-30 in expC2_dose.csv?
Replicates the analyze KL path on interface-formed backbones and prints, per level: interface-position
count, finite-KL count, strict-hotspot count, complex count, and the paired_dauroc result (or the reason
it returns None). Reads only the columns needed from the 122MB positions CSV."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from expC_analyze import paired_dauroc  # noqa: E402

S = os.path.expandvars("$SCRATCH/expC2")
pos = pd.read_csv(f"{S}/scored_positions.csv",
                  usecols=["backbone_id", "complex_id", "partial_T", "chain", "resnum", "kl", "nbr"])
pos["chain"] = pos["chain"].astype(str)
bb = pd.read_csv(f"{S}/scored_backbones.csv")[["backbone_id", "interface_ok"]]
okbb = set(bb[bb.interface_ok == 1]["backbone_id"])

idf = pd.read_csv("results/p0_dssp_interface_resid.csv",
                  usecols=lambda c: c in ("complex_id", "chain", "resnum", "is_interface"))
idf = idf[idf["is_interface"] == True]
iface = {(r.complex_id, str(r.chain), int(r.resnum)) for r in idf.itertuples()}
strict = pd.read_csv("results/p0_dssp_pairs_strict_hot2_null.csv")
hotpos = {(r.complex_id, str(r.hot_chain), int(r.hot_resnum)) for r in strict.itertuples()}

pos = pos[[(r.complex_id, r.chain, int(r.resnum)) in iface for r in pos.itertuples()]].copy()
pos["is_hot"] = [int((r.complex_id, r.chain, int(r.resnum)) in hotpos) for r in pos.itertuples()]
print(f"interface-position rows: {len(pos)};  finite kl overall: {np.isfinite(pos.kl).mean():.3f}")

sub_ok = pos[pos["backbone_id"].isin(okbb)]
print("\n=== interface-FORMED subset, per level ===")
for T, g in sub_ok.groupby("partial_T"):
    agg = g.groupby(["complex_id", "chain", "resnum"]).agg(
        kl=("kl", "mean"), nbr=("nbr", "mean"), is_hot=("is_hot", "max")).reset_index()
    fin = agg[np.isfinite(agg.kl) & np.isfinite(agg.nbr)]
    r = paired_dauroc(agg)
    res = (f"dAUROC={r['dauroc']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] n_cx={r['n_cx']}" if r else "None")
    print(f"  T={int(T):2d}: positions={len(agg):4d}  finite_kl={len(fin):4d}  "
          f"hot(finite)={int(fin.is_hot.sum()):3d}  n_cx(finite)={fin.complex_id.nunique():2d}  "
          f"n_cx_with_hot={fin[fin.is_hot == 1].complex_id.nunique():2d}  -> {res}")

print("\n=== KL finiteness on interface-formed NOISED backbones (why nan?) ===")
noised_ok = sub_ok[sub_ok.partial_T != 0]
print(f"  noised interface-formed interface positions: {len(noised_ok)}; finite kl: {np.isfinite(noised_ok.kl).mean():.3f}")
print(f"  finite nbr: {np.isfinite(noised_ok.nbr).mean():.3f}")

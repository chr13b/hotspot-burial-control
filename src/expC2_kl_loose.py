"""C2-KL completion: interface-formed ΔAUROC-over-burial per level for STRICT (pre-registered, ΔΔG>2)
and LOOSE (exploratory, ΔΔG>1) hotspot sets. The pre-registered strict reading is unpowered on the
naturally-docked subset (only ~2 strict hotspots / 1 complex among interface-formed noised backbones);
the loose set has many more positions and may be computable. Reports both regardless of outcome."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expC_analyze import paired_dauroc  # noqa: E402

S = os.path.expandvars("$SCRATCH/expC2")
pos = pd.read_csv(f"{S}/scored_positions.csv",
                  usecols=["backbone_id", "complex_id", "partial_T", "chain", "resnum", "kl", "nbr"])
pos["chain"] = pos["chain"].astype(str)
bb = pd.read_csv(f"{S}/scored_backbones.csv")[["backbone_id", "interface_ok"]]
okbb = set(bb[bb.interface_ok == 1]["backbone_id"])

idf = pd.read_csv("results/p0_dssp_interface_resid.csv",
                  usecols=lambda c: c in ("complex_id", "chain", "resnum", "is_interface", "label"))
idf = idf[idf["is_interface"] == True]
lab = {(r.complex_id, str(r.chain), int(r.resnum)): str(r.label) for r in idf.itertuples()}
iface = set(lab)

pos = pos[[(r.complex_id, r.chain, int(r.resnum)) in iface for r in pos.itertuples()]].copy()
key = list(zip(pos.complex_id, pos.chain, pos.resnum.astype(int)))
labels = np.array([lab.get(k, "") for k in key])
pos["is_strict"] = (labels == "hot_strict").astype(int)
pos["is_loose"] = np.isin(labels, ["hot_strict", "hot_loose"]).astype(int)

sub_ok = pos[pos["backbone_id"].isin(okbb)]
print("=== interface-FORMED ΔAUROC-over-burial per level (strict=pre-reg, loose=exploratory) ===")
rows = []
for T, g in sub_ok.groupby("partial_T"):
    agg = g.groupby(["complex_id", "chain", "resnum"]).agg(
        kl=("kl", "mean"), nbr=("nbr", "mean"),
        is_strict=("is_strict", "max"), is_loose=("is_loose", "max")).reset_index()
    out = {"partial_T": int(T)}
    line = f"  T={int(T):2d}: "
    for name, col in (("strict", "is_strict"), ("loose", "is_loose")):
        a = agg.assign(is_hot=agg[col])
        r = paired_dauroc(a)
        nhot = int(agg[col].sum()); ncxh = agg[agg[col] == 1].complex_id.nunique()
        if r:
            line += f"[{name}] ΔAUROC={r['dauroc']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] n_cx={r['n_cx']} nhot={nhot} | "
            out[f"{name}_dauroc"] = r["dauroc"]; out[f"{name}_lo"] = r["lo"]; out[f"{name}_hi"] = r["hi"]
            out[f"{name}_ncx"] = r["n_cx"]
        else:
            line += f"[{name}] None (nhot={nhot}, ncx_hot={ncxh}) | "
            out[f"{name}_dauroc"] = np.nan
        out[f"{name}_nhot"] = nhot
    print(line)
    rows.append(out)
pd.DataFrame(rows).to_csv("results/expC2_kl_hotpoor.csv", index=False)
print("[expC2_kl_loose] wrote results/expC2_kl_hotpoor.csv")

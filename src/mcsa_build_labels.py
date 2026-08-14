#!/usr/bin/env python3
"""Provenance: build ~/ftax/data/m-csa/mcsa_labels.csv + mcsa_pdblist.txt from M-CSA curated_data.csv.

Catalytic residue = M-CSA role type 'reactant' (the mechanistic residues; the role-group catalytic set is a
subset and makes no difference — audited). Selects enzymes with >=3 catalytic residues on <=2 chains, capped
at 130 for feasible download+scoring. Reproduces the labels used by catalytic_dissociation.py / catalytic_audit.py.
  python3 src/mcsa_build_labels.py
"""
import os
import numpy as np, pandas as pd
MCSA = os.path.expanduser("~/ftax/data/m-csa")


def main():
    d = pd.read_csv(f"{MCSA}/curated_data.csv", engine="python", on_bad_lines="skip")
    r = d[d["residue/reactant/product/cofactor"] == "residue"].rename(columns={
        "PDB": "pdb", "PDB code": "restype", "chain/kegg compound": "chain",
        "resid/chebi id": "resid", "role type": "roletype"})
    r["resid"] = pd.to_numeric(r["resid"], errors="coerce")
    r = r.dropna(subset=["resid"]); r["resid"] = r["resid"].astype(int)
    r["catalytic"] = (r.roletype == "reactant").astype(int)
    g = r.groupby(["pdb", "chain", "resid"]).agg(catalytic=("catalytic", "max"),
                                                 restype=("restype", "first")).reset_index()
    per = g[g.catalytic == 1].groupby("pdb").agg(n_cat=("resid", "size"), n_ch=("chain", "nunique")).reset_index()
    sel = per[(per.n_cat >= 3) & (per.n_ch <= 2)].pdb.tolist()[:130]
    g[g.pdb.isin(sel)].to_csv(f"{MCSA}/mcsa_labels.csv", index=False)
    open(f"{MCSA}/mcsa_pdblist.txt", "w").write("\n".join(sorted(set(sel))))
    lab = g[g.pdb.isin(sel)]
    print(f"enzymes {len(sel)}  labeled positions {len(lab)}  catalytic {int(lab.catalytic.sum())}; "
          f"wrote {MCSA}/mcsa_labels.csv + mcsa_pdblist.txt")


if __name__ == "__main__":
    main()

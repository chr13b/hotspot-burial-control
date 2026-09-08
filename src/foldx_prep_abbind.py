#!/usr/bin/env python3
"""Task 2 prep — build the FoldX Lane-A worklist for the AB-Bind antibody-antigen fixture, with a hard positive
control (CLAUDE.md rule 6) before any FoldX runs.

The AB-Bind raw source (~/ftax/data/ab-bind, holding the authoritative Partners(A_B) chain-group partition) was
purged from $SCRATCH; the committed L/ddG fixture (results/leverage_abbind_mutations.csv) survives. We re-fetched
the AUTHORITATIVE public AB-Bind release and stage it at ABDIR. This script:
  1. POSITIVE CONTROL: re-fetched single-mutation ddG must match the committed fixture's ddG (proves the recovered
     Partners(A_B) is the one L was computed against). -> results/_abbind_xcheck.csv
  2. WORKLIST: for each committed interface single mutant on a real crystal (HM_* homology models excluded, PDB
     present in ~/ftax/data/PDBs), emit a Lane-A row with complex_id = <pdb>_<g1>_<g2> so foldx_ddg.py can run
     AnalyseComplex(g1,g2). -> results/foldx_worklist_abbind.csv

  python3 src/foldx_prep_abbind.py
"""
import hashlib
import os
import re

import numpy as np
import pandas as pd

ABDIR = os.path.expanduser("~/ftax/data/ab-bind")
# AB-Bind-specific structures (from the AB-Bind release) live in ABDIR — same files L was computed on; their
# numbering matches the Mutation tokens, unlike the generic RCSB PDBs. FoldX must repair THESE.
PDBS = os.environ.get("FOLDX_PDBS", ABDIR)
AB_MUT_RE = re.compile(r"^([A-Za-z0-9]+):([A-Z])(-?\d+)([A-Z])$")   # same regex as leverage_decomposition.py


def main():
    src = f"{ABDIR}/AB-Bind_experimental_data.csv"
    ab = pd.read_csv(src, encoding="latin-1").rename(
        columns={"#PDB": "pdb", "Partners(A_B)": "partners", "ddG(kcal/mol)": "ddg"})
    ab["ddg"] = pd.to_numeric(ab["ddg"], errors="coerce")
    partners = ab.dropna(subset=["partners"]).groupby("pdb").partners.first().to_dict()

    # re-fetched single-mutation ddG lookup (positive-control target)
    single = ab[~ab["Mutation"].astype(str).str.contains(",")].copy()
    ref = {}
    for _, r in single.iterrows():
        m = AB_MUT_RE.match(str(r["Mutation"]).strip())
        if m and np.isfinite(r["ddg"]):
            ref[(r["pdb"], m.group(1), m.group(2), int(m.group(3)), m.group(4))] = float(r["ddg"])
    print(f"[refetch] {src}\n[refetch] sha256={hashlib.sha256(open(src,'rb').read()).hexdigest()}")
    print(f"[refetch] {len(partners)} complexes with partners; {len(ref)} single-mutation ddG entries")

    # committed L fixture (interface single mutants with L + experimental ddG)
    com = pd.read_csv("results/leverage_abbind_mutations.csv")
    com["icode"] = com.icode.fillna("").astype(str)
    com = com[com.is_interface == 1].copy()
    com = com.drop_duplicates(subset=["complex_id", "chain", "resnum", "icode", "wt", "mut"]).copy()  # exact-dup rows exist

    # 1) POSITIVE CONTROL: refetch ddG vs committed ddG on the same (pdb,chain,wt,resnum,mut)
    checks = []
    for r in com.itertuples():
        k = (r.complex_id, r.chain, r.wt, int(r.resnum), r.mut)
        rd = ref.get(k, np.nan)
        checks.append(dict(complex_id=r.complex_id, chain=r.chain, wt=r.wt, resnum=int(r.resnum), mut=r.mut,
                           ddg_committed=round(float(r.ddG), 4), ddg_refetch=(round(float(rd), 4) if np.isfinite(rd) else np.nan),
                           matched=bool(np.isfinite(rd)),
                           dabs=(round(abs(float(rd) - float(r.ddG)), 4) if np.isfinite(rd) else np.nan)))
    xc = pd.DataFrame(checks)
    xc.to_csv("results/_abbind_xcheck.csv", index=False)
    n = len(xc); nm = int(xc.matched.sum())
    mism = int((xc.dabs > 0.01).sum()); maxd = float(np.nanmax(xc.dabs.values)) if nm else float("nan")
    print(f"[xcheck] committed interface single-muts={n}; matched-in-refetch={nm} ({nm/n:.1%}); "
          f"ddG mismatch(|Δ|>0.01)={mism}; max|Δ|={maxd:.4f}  -> results/_abbind_xcheck.csv")

    # 2) WORKLIST for real-crystal complexes
    rows, skip = [], {}
    for r in com.itertuples():
        pdb = str(r.complex_id)
        if pdb.startswith("HM") or "_" in pdb:
            skip["homology_model"] = skip.get("homology_model", 0) + 1; continue
        if pdb not in partners:
            skip["no_partners"] = skip.get("no_partners", 0) + 1; continue
        if partners[pdb].count("_") != 1:
            skip["bad_partners"] = skip.get("bad_partners", 0) + 1; continue
        if not os.path.exists(f"{PDBS}/{pdb}.pdb"):
            skip["no_pdb"] = skip.get("no_pdb", 0) + 1; continue
        g1, g2 = partners[pdb].split("_")
        rows.append(dict(lane="A", complex_id=f"{pdb}_{g1}_{g2}", arm="abbind", k=0, chain=r.chain,
                         wt=r.wt, resnum=int(r.resnum), icode=r.icode, mut=r.mut))
    wl = pd.DataFrame(rows)
    wl.to_csv("results/foldx_worklist_abbind.csv", index=False)
    print(f"[worklist] {len(wl)} mutations over {wl.complex_id.nunique()} complexes "
          f"-> results/foldx_worklist_abbind.csv")
    print(f"[worklist] skipped: {skip}")
    print(f"[worklist] complexes: {sorted(wl.complex_id.unique())}")


if __name__ == "__main__":
    main()

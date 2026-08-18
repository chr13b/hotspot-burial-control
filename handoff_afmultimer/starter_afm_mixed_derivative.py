#!/usr/bin/env python3
"""Project #3 Tier-1 SKELETON: AlphaFold-Multimer confidence blindness vs the partner-ablation mixed
derivative, on the SAME 344 SKEMPI complexes as the main paper. Fill the two TODOs (AF runner + pLDDT/PAE
parse); the analysis is already wired to the main paper's CPI so the AF result is method-identical.

Run (Sherlock, after MSAs cached):  python3 starter_afm_mixed_derivative.py --stage predict   # GPU
                                    python3 starter_afm_mixed_derivative.py --stage analyse   # CPU

Pre-register the falsifiers in PREREG.md BEFORE running --predict (mirror ../CLAUDE.md + ../BRIEF.md §4).
"""
import argparse, os, sys
import numpy as np, pandas as pd

MAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(MAIN, "src"))
import leverage_decomposition as LD          # reuse cpi(), drop_influential(), within-stratum AUROC
SEED = 20260803
POS = os.path.join(MAIN, "results/leverage_skempi_positions.csv")   # chain,resnum,is_hot,rsasa,nbr,is_interface
COMPLEXES = os.path.join(os.path.dirname(__file__), "skempi_complexes.txt")
OUT = "afm_mixed_derivative.csv"


# ------------------------------------------------------------------ GPU stage (Sherlock)
def run_af(pdb, chains, out_dir):
    """TODO(kickoff): predict `chains` of `pdb` with ColabFold/AF-Multimer; return path to the ranked PDB +
    the PAE json. Cache MSAs. For the BOUND complex pass all chains of both groups; for the MONOMER pass one
    group's chains only (the partner-ablated reference)."""
    raise NotImplementedError("wire ColabFold/AF-Multimer here")


def parse_conf(pdb_path, pae_path):
    """TODO(kickoff): return DataFrame[chain,resnum,plddt,ifPAE] from an AF prediction. pLDDT is the b-factor
    column; ifPAE = mean predicted-aligned-error from this residue to the partner chain (from the PAE matrix)."""
    raise NotImplementedError("wire the pLDDT/PAE parser here")


def stage_predict(a):
    cids = [l.strip() for l in open(COMPLEXES) if l.strip()][: a.limit or None]
    rows = []
    for cid in cids:
        pdb, g1, g2 = cid.split("_")
        bound = run_af(pdb, g1 + g2, f"af/{cid}/bound")
        mono1 = run_af(pdb, g1, f"af/{cid}/mono_{g1}")
        mono2 = run_af(pdb, g2, f"af/{cid}/mono_{g2}") if g2 else (None, None)
        cb = parse_conf(*bound)
        cm = pd.concat([parse_conf(*mono1)] + ([parse_conf(*mono2)] if g2 else []), ignore_index=True)
        m = cb.merge(cm, on=["chain", "resnum"], suffixes=("_cplx", "_mono"))
        m["complex_id"] = cid
        m["dConf"] = m.plddt_cplx - m.plddt_mono          # partner-ablation mixed derivative (confidence)
        rows.append(m)
    pd.concat(rows).to_csv(OUT.replace(".csv", "_raw.csv"), index=False)
    print(f"[predict] wrote {OUT.replace('.csv','_raw.csv')}")


# ------------------------------------------------------------------ CPU analysis (method == main paper)
def stage_analyse(a):
    af = pd.read_csv(OUT.replace(".csv", "_raw.csv"))
    lab = pd.read_csv(POS)                                  # is_hot + geometry, same positions as the IF paper
    d = af.merge(lab, on=["complex_id", "chain", "resnum"]).query("is_interface == True")
    rng = np.random.default_rng(SEED)
    y = d.is_hot.astype(int).to_numpy(); g = d.complex_id.to_numpy()
    Z = d[["rsasa_complex", "nbr", "drsasa"]].to_numpy(float)     # the geometry controls the IF paper used
    rows = []
    for name, X in [("confidence pLDDT(complex)", d.plddt_cplx.to_numpy()),      # expect BLIND (CPI ~ 0)
                    ("interface PAE(complex)", d.ifPAE_cplx.to_numpy()),
                    ("LEVERAGE dConf (partner ablation)", d.dConf.to_numpy())]:   # the test object
        c, lo, hi, p, _, _ = LD.cpi(y, g, Z, X, rng)
        verdict = "ADDS" if lo > 0 else "blind/ns"
        print(f"  CPI({name} | geometry) = {c:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {verdict}")
        rows.append(dict(feature=name, cpi=round(c, 5), lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[analyse] wrote {OUT}  (prediction: confidence blind, partner-ablation dConf adds — or the bound)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["predict", "analyse"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    (stage_predict if a.stage == "predict" else stage_analyse)(a)

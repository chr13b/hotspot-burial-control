#!/usr/bin/env python3
"""Parse Boltz-2 outputs -> results/iptm_steer_boltz.csv (EXACT schema of parse_iptm.py's iptm_steer.csv),
so results/analyse_iptm.py runs on it unchanged. Phase 3 (pre-reg results/PREREG_boltz.md).

Interface metrics use the SAME crystal interface set (leverage_skempi_positions.csv) and the SAME g1-then-g2
residue order (fc.load_complex) as parse_iptm.py — the Boltz input chains were written A(=g1) then B(=g2), so
the folded residue order matches the crystal-complex order and the indices align.

Boltz layout (per fold dir <out>/<fid>/): boltz_results_<fid>/predictions/<fid>/ containing
  confidence_<fid>_model_0.json   (keys incl. iptm, ptm, complex_plddt),
  pae_<fid>_model_0.npz|json      (PAE matrix),
  plddt_<fid>_model_0.npz         (per-res pLDDT)  — else read from the model .pdb/.cif B-factor.

  python3 src/parse_boltz.py --out-dir $SCRATCH/ftax/iptm/boltz_out --subset results/iptm_subset.txt \
      --out results/iptm_steer_boltz.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc          # noqa: E402
import leverage_decomposition as LD  # noqa: E402

DATA = LD.DATA
COND = [("wt", -1), ("L", 0), ("L", 1), ("L", 2), ("random", 0), ("random", 1), ("random", 2)]


def fold_id(cid, d, k):
    return f"{cid}__wt" if d == "wt" else f"{cid}__{d}__k{k}"


def _find(pred_root, fid, prefix):
    hits = glob.glob(f"{pred_root}/**/{prefix}*{fid}*", recursive=True)
    return sorted(hits)


def load_boltz(out_dir, fid):
    """-> dict(iptm, ptm, plddt[n], pae[n,n]) or None."""
    root = f"{out_dir}/{fid}"
    cj = glob.glob(f"{root}/**/confidence_*{fid}*.json", recursive=True) or \
        glob.glob(f"{root}/**/confidence*.json", recursive=True)
    if not cj:
        return None
    d = json.load(open(cj[0]))
    iptm = d.get("iptm", d.get("protein_iptm"))
    ptm = d.get("ptm")
    # PAE
    pae = None
    pnpz = glob.glob(f"{root}/**/pae_*.npz", recursive=True)
    pjson = glob.glob(f"{root}/**/pae_*.json", recursive=True)
    if pnpz:
        z = np.load(pnpz[0])
        pae = z[z.files[0]]
    elif pjson:
        pj = json.load(open(pjson[0]))
        pae = np.asarray(pj.get("pae", pj.get("predicted_aligned_error")), float)
    # pLDDT: prefer npz, else read the structure B-factor
    plddt = None
    lnpz = glob.glob(f"{root}/**/plddt_*.npz", recursive=True)
    if lnpz:
        z = np.load(lnpz[0])
        plddt = np.asarray(z[z.files[0]], float)
    else:
        struct = (glob.glob(f"{root}/**/{fid}*model_0.pdb", recursive=True) or
                  glob.glob(f"{root}/**/*model_0.pdb", recursive=True) or
                  glob.glob(f"{root}/**/*.pdb", recursive=True))
        if struct:
            plddt = _plddt_from_pdb(struct[0])
    if plddt is not None and np.nanmax(plddt) <= 1.5:      # Boltz plddt is 0-1 -> scale to 0-100 (AF2 parity)
        plddt = plddt * 100.0
    return dict(iptm=iptm, ptm=ptm, plddt=plddt, pae=pae)


def _plddt_from_pdb(path):
    vals, seen = [], set()
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            key = (line[21], line[22:27])
            if key in seen:
                continue
            seen.add(key)
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                vals.append(np.nan)
    return np.asarray(vals, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ["SCRATCH"] + "/ftax/iptm/boltz_out")
    ap.add_argument("--subset", default="results/iptm_subset.txt")
    ap.add_argument("--positions", default="results/leverage_skempi_positions.csv")
    ap.add_argument("--out", default="results/iptm_steer_boltz.csv")
    a = ap.parse_args()

    pos = pd.read_csv(a.positions, low_memory=False)
    pos["icode"] = pos.icode.fillna("").astype(str)
    pos = pos[pos.is_interface == True]                                              # noqa: E712
    iface = {}
    for r in pos.itertuples():
        iface.setdefault(r.complex_id, set()).add((r.chain, int(r.resnum), r.icode))

    cids = [ln.strip() for ln in open(a.subset) if ln.strip()]
    rows, missing = [], 0
    for cid in cids:
        pdb, g1, g2 = cid.split("_")
        cx = fc.load_complex(f"{DATA}/PDBs/{pdb}.pdb", pdb, g1, g2)
        if cx is None:
            continue
        keys = [(cx.chains[j], int(cx.resnums[j]), cx.icodes[j]) for j in range(cx.n)]
        ik = iface.get(cid, set())
        g1i = np.array([j for j in range(cx.n) if keys[j] in ik and cx.group[j] == 1])
        g2i = np.array([j for j in range(cx.n) if keys[j] in ik and cx.group[j] == 2])
        alli = np.array([j for j in range(cx.n) if keys[j] in ik])
        for direction, k in COND:
            fid = fold_id(cid, direction, k)
            sc = load_boltz(a.out_dir, fid)
            if sc is None or sc["iptm"] is None:
                missing += 1
                continue
            ipae = np.nan
            if sc["pae"] is not None and len(g1i) and len(g2i) and sc["pae"].shape[0] == cx.n:
                P = sc["pae"]
                ipae = float(np.concatenate([P[np.ix_(g1i, g2i)].ravel(),
                                             P[np.ix_(g2i, g1i)].ravel()]).mean())
            iplddt = np.nan
            if sc["plddt"] is not None and sc["plddt"].size == cx.n and len(alli):
                iplddt = float(sc["plddt"][alli].mean())
            rows.append(dict(complex_id=cid, direction=direction, k=k,
                             iptm=sc["iptm"], ptm=sc["ptm"],
                             interface_pae=round(ipae, 4) if np.isfinite(ipae) else np.nan,
                             interface_plddt=round(iplddt, 4) if np.isfinite(iplddt) else np.nan,
                             n_iface=int(len(alli))))
    df = pd.DataFrame(rows)
    df["seed"] = 20260803
    df.to_csv(a.out, index=False)
    nfold = df.groupby("complex_id").size() if len(df) else pd.Series(dtype=int)
    print(f"[parse-boltz] {len(df)} folds over {df.complex_id.nunique() if len(df) else 0} complexes "
          f"({missing} missing); complexes with all 7 = {int((nfold == 7).sum())}")
    if len(df):
        print(f"[parse-boltz] wt interface ipTM median = {df[df.direction=='wt'].iptm.median():.3f} "
              f"(sanity ~0.6-0.9); wrote {a.out}")


if __name__ == "__main__":
    main()

"""Exp C2 precompute: the pre-registered 50/50 hotspot split + RFdiffusion ppi.hotspot_res.

PREREG_expC2 §1/§3. For each of the 55 complexes (results/expC_complexes.csv):
  1. Gather the labelled BINDER hotspots = distinct (chain,resnum) on the binder appearing as a `hot`
     position in any committed results/p0_dssp_pairs_*.csv (same union src/expC_setup.py labelled with).
  2. Split them 50/50 into {conditioned, heldout} with a complex-seeded RNG (seed 20260803 + sha1(cid)),
     ceil(n/2) to conditioned so single-hotspot complexes are still pinned. Fixed once, before any backbone.
  3. ppi.hotspot_res = the TARGET crystal residues within 5.0 A (virtual Cbeta) of any CONDITIONED-arm
     binder hotspot, mapped to RFdiffusion input-PDB chain space (held target's output letter + its
     1..Lt renumbering) via results/expC_outmap.json. Only the conditioned arm's contacts are passed.

Emit:
  results/expC2_hotspot_split.csv        complex_id, chain, resnum, aa, ddG_max, arm
  results/expC2_hotspot_res.json         {cid: "B30,B31,..."}  (inner list for ppi.hotspot_res=[...])
  results/expC2_hotspot_res_summary.csv  complex_id, n_hot, n_cond, n_held, n_hotspot_res, hotspot_res

Usage (under $SCRATCH/ftax/env.sh — needs biopython):
  python3 src/expC2_hotspot_res.py --data-dir $FTAX_DATA
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc  # noqa: E402

SPLIT_SEED = 20260803
CONTACT_A = 5.0  # default; overridable via --cutoff (see PREREG_expC2 §1 + the setup deviation note)


def complex_seed(cid):
    """Deterministic per-complex seed (Python's hash() is salted; sha1 is stable across runs)."""
    h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 1_000_000_000
    return SPLIT_SEED + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes-csv", default="results/expC_complexes.csv")
    ap.add_argument("--outmap", default="results/expC_outmap.json")
    ap.add_argument("--pairs-glob", default="results/p0_dssp_pairs_*.csv")
    ap.add_argument("--data-dir", default=os.environ.get("FTAX_DATA", os.path.expanduser("~/ftax/data")))
    ap.add_argument("--cutoff", type=float, default=CONTACT_A, help="target-contact Cbeta cutoff (A)")
    ap.add_argument("--out-split", default="results/expC2_hotspot_split.csv")
    ap.add_argument("--out-json", default="results/expC2_hotspot_res.json")
    ap.add_argument("--out-summary", default="results/expC2_hotspot_res_summary.csv")
    a = ap.parse_args()

    meta = pd.read_csv(a.complexes_csv)
    outmap = json.load(open(a.outmap))

    # labelled hotspots per complex, with the strongest ddG seen (union over all committed pair variants,
    # `hot` positions only). Keep chain/resnum/aa/ddG.
    hot = {}
    for pf in sorted(glob.glob(a.pairs_glob)):
        d = pd.read_csv(pf)
        for r in d.itertuples():
            key = (r.complex_id, str(r.hot_chain), int(r.hot_resnum))
            ddg = float(getattr(r, "hot_ddG")) if not pd.isna(getattr(r, "hot_ddG", np.nan)) else np.nan
            prev = hot.get(key)
            hot[key] = (str(r.hot_aa), max(ddg, prev[1]) if prev and np.isfinite(prev[1]) else ddg)

    split_rows, summary_rows, res_json = [], [], {}
    for r in meta.itertuples():
        cid, pdb = r.complex_id, r.pdb
        binder = set(str(r.binder_chains)); target = set(str(r.target_chains))
        g1, g2 = cid.split("_")[1], cid.split("_")[2]

        # binder hotspots for this complex (on a binder chain)
        cid_hot = sorted([(ch, rn, hot[(cid, ch, rn)][0], hot[(cid, ch, rn)][1])
                          for (c, ch, rn) in hot if c == cid and ch in binder],
                         key=lambda t: (t[0], t[1]))
        n = len(cid_hot)
        # 50/50 split, conditioned gets the ceil
        rng = np.random.default_rng(complex_seed(cid))
        perm = rng.permutation(n) if n else np.array([], int)
        n_cond = int(np.ceil(n / 2))
        cond_idx = set(perm[:n_cond].tolist())
        conditioned = [cid_hot[i] for i in range(n) if i in cond_idx]
        heldout = [cid_hot[i] for i in range(n) if i not in cond_idx]
        for arm, lst in (("conditioned", conditioned), ("heldout", heldout)):
            for (ch, rn, aa, ddg) in lst:
                split_rows.append(dict(complex_id=cid, chain=ch, resnum=rn, aa=aa, ddG_max=ddg, arm=arm))

        # target contacts (crystal) within 5 A Cbeta of any CONDITIONED binder hotspot
        tokens = []
        cx = fc.load_complex(os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb"), pdb, g1, g2)
        if cx is not None and conditioned:
            idx_by = {}
            for i in range(cx.n):
                idx_by.setdefault((str(cx.chains[i]), int(cx.resnums[i])), []).append(i)
            tgt_idx = [i for i in range(cx.n) if str(cx.chains[i]) in target]
            tgt_cb = cx.CB[tgt_idx] if tgt_idx else np.zeros((0, 3))
            # reverse map: crystal (chain,resnum) -> output token (out_letter + out_resnum), from outmap
            rev = {}
            for oc, rows in outmap.get(cid, {}).items():
                for j, entry in enumerate(rows):
                    rev.setdefault((str(entry[0]), int(entry[1])), f"{oc}{j+1}")
            contacts = set()
            for (ch, rn, aa, ddg) in conditioned:
                for i in idx_by.get((ch, rn), []):
                    if not tgt_idx:
                        continue
                    dist = np.linalg.norm(tgt_cb - cx.CB[i], axis=1)
                    for k in np.flatnonzero(dist < a.cutoff):
                        ti = tgt_idx[k]
                        contacts.add((str(cx.chains[ti]), int(cx.resnums[ti])))
            toks = sorted({rev[c] for c in contacts if c in rev},
                          key=lambda s: (s[0], int(s[1:])))
            tokens = toks
        res_json[cid] = ",".join(tokens)
        summary_rows.append(dict(complex_id=cid, n_hot=n, n_cond=len(conditioned), n_held=len(heldout),
                                 n_hotspot_res=len(tokens), hotspot_res=",".join(tokens)))

    pd.DataFrame(split_rows).to_csv(a.out_split, index=False)
    json.dump(res_json, open(a.out_json, "w"))
    sm = pd.DataFrame(summary_rows)
    sm.to_csv(a.out_summary, index=False)
    print(f"[expC2_hotspot_res] {len(sm)} complexes; "
          f"binder hotspots median {int(sm.n_hot.median())} (range {sm.n_hot.min()}-{sm.n_hot.max()}); "
          f"hotspot_res median {int(sm.n_hotspot_res.median())} (range {sm.n_hotspot_res.min()}-{sm.n_hotspot_res.max()}); "
          f"complexes with a held-out hotspot: {(sm.n_held > 0).sum()}; "
          f"complexes with 0 hotspot_res: {(sm.n_hotspot_res == 0).sum()}")


if __name__ == "__main__":
    main()

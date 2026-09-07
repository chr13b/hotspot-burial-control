#!/usr/bin/env python3
"""Judge the DUMPED (actually-folded) steered sequences by every available leverage cache — the uniform,
multi-judge extension of the judge matrix (adds MIF, cross-checks ProteinMPNN/ESM-IF1).

For each (complex, arm) and each judge j with a per-position leverage cache leverage_pq_skempi_<j>.csv, compute
the mean over the <=3 dumped interface residues of the judge's own leverage L_j(a) at the sampled residue.
Emits the same schema cfg_judge_matrix.py consumes (complex_id, direction, alpha, meanL_<j>...). Uniform method
across judges (unlike cfg_steer.csv's K=64 meanL, this is over the 3 folded samples), so it is directly
comparable across ProteinMPNN, ESM-IF1, and MIF.

  python3 src/judge_dumped.py --seqs results/cfg_steer_seqs.csv       --out results/cfg_judge_dumped_setA.csv
  python3 src/judge_dumped.py --seqs results/cfg_steer_esmif_seqs.csv --out results/cfg_judge_dumped_setB.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "models")):
    sys.path.insert(0, p)
import ftax_common as fc            # noqa: E402
import leverage_decomposition as LD  # noqa: E402
import cfg_steer as cs              # noqa: E402  (interface_set, load_L)

DATA, AA20, IDX = LD.DATA, LD.AA20, LD.IDX
JUDGE_CACHE = {"mpnn": "results/leverage_pq_skempi.csv",
               "esmif": "results/leverage_pq_skempi_esmif.csv",
               "mif": "results/leverage_pq_skempi_mif.csv"}


def chains_to_cxseq(cx, chains):
    """'A:SEQ|B:SEQ2' -> per-cx-position amino acid (cx order == the order seq_to_chains wrote)."""
    per = {}
    for part in chains.split("|"):
        c, seq = part.split(":", 1)
        per[c] = seq
    out = ["X"] * cx.n
    ctr = {}
    for j in range(cx.n):
        c = cx.chains[j]
        m = ctr.get(c, 0)
        ctr[c] = m + 1
        if c in per and m < len(per[c]):
            out[j] = per[c][m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--directions", default="", help="comma-sep; default=auto-detect all directions except wt")
    a = ap.parse_args()
    iface = cs.interface_set()
    judges = {k: cs.load_L(v) for k, v in JUDGE_CACHE.items() if os.path.exists(v)}
    print(f"[judge-dumped] judges available: {list(judges)}")
    df = pd.read_csv(a.seqs)
    df["k"] = df.k.astype(int)
    DIRS = [d.strip() for d in a.directions.split(",") if d.strip()] or \
           [d for d in df.direction.unique() if d != "wt"]
    print(f"[judge-dumped] directions: {DIRS}")
    rows = []
    for cid, g in df.groupby("complex_id"):
        if cid not in iface:
            continue
        pdb, g1, g2 = cid.split("_")
        path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None:
            continue
        keys = [(cx.chains[j], int(cx.resnums[j]), cx.icodes[j]) for j in range(cx.n)]
        ipos = [j for j in range(cx.n) if keys[j] in iface[cid]]
        if not ipos:
            continue
        for direction in DIRS:
            sub = g[(g.direction == direction) & (g.alpha == a.alpha)]
            if not len(sub):
                continue
            acc = {jk: [] for jk in judges}
            for _, r in sub.iterrows():
                aa = chains_to_cxseq(cx, r.chains)
                for j in ipos:
                    ai = IDX.get(aa[j], -1)
                    if ai < 0:
                        continue
                    for jk, Lc in judges.items():
                        Lm = Lc.get(cid, {}).get(keys[j])
                        if Lm is not None:
                            acc[jk].append(float(Lm[ai]))
            row = dict(complex_id=cid, direction=direction, alpha=a.alpha)
            for jk in judges:
                row[f"meanL_{jk}"] = round(float(np.mean(acc[jk])), 4) if acc[jk] else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"[judge-dumped] {len(out)} rows ({out.complex_id.nunique()} complexes) -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build Boltz-2 input FASTAs from cfg_steer_seqs.csv for the batch-1 subset (Phase 3).

One FASTA per (complex, arm, k), chains as separate protein entities; --use_msa_server computes the MSAs on the
SAME MMseqs2 server AF2/ColabFold used (MSA-source parity). Fold-id convention identical to the AF2 ipTM run
(<cid>__wt, <cid>__{L,random}__k{0,1,2}) so the same interface indexing applies.

  Boltz FASTA (one per fold):   >A|protein\\nSEQ_A\\n>B|protein\\nSEQ_B

  python3 src/build_boltz_inputs.py --seqs results/cfg_steer_seqs.csv --subset results/iptm_subset.txt \
      --indir $SCRATCH/ftax/iptm/boltz_in [--arms wt] [--limit N]
"""
import argparse
import os

import pandas as pd

COND = [("wt", -1), ("L", 0), ("L", 1), ("L", 2), ("random", 0), ("random", 1), ("random", 2)]


def fold_id(cid, d, k):
    return f"{cid}__wt" if d == "wt" else f"{cid}__{d}__k{k}"


def to_boltz_fasta(chains):
    """'A:SEQ1|B:SEQ2' -> Boltz multi-entity FASTA text (each chain a protein entity)."""
    out = []
    for part in chains.split("|"):
        c, seq = part.split(":", 1)
        out.append(f">{c}|protein")
        out.append(seq)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="results/cfg_steer_seqs.csv")
    ap.add_argument("--subset", default="results/iptm_subset.txt")
    ap.add_argument("--indir", default=os.environ["SCRATCH"] + "/ftax/iptm/boltz_in")
    ap.add_argument("--arms", default="all", choices=["all", "wt"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.indir, exist_ok=True)
    df = pd.read_csv(a.seqs)
    df["k"] = df.k.astype(int)
    subset = [l.strip() for l in open(a.subset) if l.strip()]
    if a.limit:
        subset = subset[:a.limit]
    conds = [("wt", -1)] if a.arms == "wt" else COND
    manifest = []
    for cid in subset:
        g = df[df.complex_id == cid]
        for d, k in conds:
            row = g[(g.direction == d) & (g.k == k)]
            if not len(row):
                continue
            fid = fold_id(cid, d, k)
            with open(f"{a.indir}/{fid}.fasta", "w") as f:
                f.write(to_boltz_fasta(row.iloc[0].chains))
            manifest.append(fid)
    with open(f"{a.indir}/manifest.txt", "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"[boltz-in] wrote {len(manifest)} FASTAs (+ manifest.txt) to {a.indir} "
          f"({len(subset)} complexes, arms={a.arms})")


if __name__ == "__main__":
    main()

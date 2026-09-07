#!/usr/bin/env python3
"""Build ColabFold FASTAs for the NAIVE arm only (Phase 4 of PREREG_naive.md), reusing the committed
wt/L/random folds. Reads results/cfg_steer_naive_seqs.csv (direction=naive, k in {0,1,2}) restricted to the
FROZEN 60-complex ipTM subset (results/iptm_subset.txt) and writes one multimer FASTA per (complex, naive, k),
fold_id '{cid}__naive__k{k}' -- the same naming/format as build_iptm_fastas.py so parse_iptm.py/analyse_iptm.py
consume them identically. Only ~180 new folds (60 x 3); wt/L/random are NOT re-folded (that is the compute saving).

  python3 src/build_naive_fastas.py --fadir $SCRATCH/ftax/iptm/fastas_naive
"""
import argparse
import os
import pandas as pd


def chains_to_fasta(chains):
    parts = [p.split(":", 1)[1] for p in chains.split("|")]
    return ":".join(parts), sum(len(p) for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="results/cfg_steer_naive_seqs.csv")
    ap.add_argument("--subset", default="results/iptm_subset.txt")
    ap.add_argument("--fadir", default=os.environ["SCRATCH"] + "/ftax/iptm/fastas_naive")
    ap.add_argument("--max-res", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(a.fadir, exist_ok=True)

    subset = [l.strip() for l in open(a.subset) if l.strip()]
    df = pd.read_csv(a.seqs)
    df["k"] = df.k.astype(int)
    df = df[(df.direction == "naive") & (df.complex_id.isin(subset))]

    manifest, nfa, skipped = [], 0, []
    for cid in subset:
        g = df[df.complex_id == cid]
        for k in (0, 1, 2):
            row = g[g.k == k]
            if not len(row):
                skipped.append((cid, k)); continue
            fa_seq, n = chains_to_fasta(row.iloc[0].chains)
            if n > a.max_res:
                skipped.append((cid, k)); continue
            fid = f"{cid}__naive__k{k}"
            with open(f"{a.fadir}/{fid}.fasta", "w") as f:
                f.write(f">{fid}\n{fa_seq}\n")
            manifest.append(fid); nfa += 1
    with open(f"{a.fadir}/manifest.txt", "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"[build-naive] wrote {nfa} naive FASTAs (+ manifest.txt) to {a.fadir}")
    if skipped:
        print(f"[build-naive] SKIPPED {len(skipped)} (missing seq or >{a.max_res} res): {skipped[:6]}"
              f"{' ...' if len(skipped) > 6 else ''}")


if __name__ == "__main__":
    main()

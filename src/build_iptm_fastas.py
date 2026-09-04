#!/usr/bin/env python3
"""Pre-register the ipTM fold subset and build ColabFold FASTAs from the CFG-steered sequences.

Reads results/cfg_steer_seqs.csv (cfg_steer.py --dump-seqs: per complex, direction in {wt,L,random},
k in {-1,0,1,2}, chains='A:SEQ|B:SEQ|...'). Selects a SEED-shuffled first-N subset of complexes that
have ALL 7 conditions (wt + L k0-2 + random k0-2) and <= max_res residues, writes the frozen subset to
results/iptm_subset.txt (committed BEFORE any fold — rule 1), and one multimer FASTA per (complex,
direction, k) to <fadir> (chains joined by ':' in g1-then-g2 order, ColabFold's multimer separator).

  python3 src/build_iptm_fastas.py --n 60 --max-res 600 \
      --seqs results/cfg_steer_seqs.csv --fadir $SCRATCH/ftax/iptm/fastas --subset-out results/iptm_subset.txt
"""
import argparse
import os
import numpy as np
import pandas as pd

SEED = 20260803
COND = [("wt", -1), ("L", 0), ("L", 1), ("L", 2), ("random", 0), ("random", 1), ("random", 2)]


def chains_to_fasta(chains):
    """'A:SEQ1|B:SEQ2' -> 'SEQ1:SEQ2' (ColabFold multimer), preserving order."""
    parts = [p.split(":", 1)[1] for p in chains.split("|")]
    return ":".join(parts), sum(len(p) for p in parts)


def fold_id(cid, direction, k):
    return f"{cid}__wt" if direction == "wt" else f"{cid}__{direction}__k{k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="results/cfg_steer_seqs.csv")
    ap.add_argument("--fadir", default=os.environ["SCRATCH"] + "/ftax/iptm/fastas")
    ap.add_argument("--subset-out", default="results/iptm_subset.txt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-res", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(a.fadir, exist_ok=True)

    df = pd.read_csv(a.seqs)
    df["k"] = df.k.astype(int)
    have = {}
    nres = {}
    for cid, g in df.groupby("complex_id"):
        present = set((r.direction, int(r.k)) for r in g.itertuples())
        have[cid] = all(c in present for c in COND)
        wt = g[g.direction == "wt"]
        if len(wt):
            _, n = chains_to_fasta(wt.iloc[0].chains)
            nres[cid] = n
    elig = sorted(c for c in have if have[c] and nres.get(c, 1e9) <= a.max_res)
    print(f"[build] {df.complex_id.nunique()} complexes in seqs; {len(elig)} eligible "
          f"(all 7 conditions, <= {a.max_res} res)")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(elig))
    subset = sorted(elig[i] for i in order[:a.n])
    with open(a.subset_out, "w") as f:
        f.write("\n".join(subset) + "\n")
    print(f"[build] FROZEN subset: {len(subset)} complexes -> {a.subset_out} (SEED={SEED})")

    manifest = []
    nfa = 0
    for cid in subset:
        g = df[df.complex_id == cid]
        for direction, k in COND:
            row = g[(g.direction == direction) & (g.k == k)]
            if not len(row):
                continue
            fa_seq, _ = chains_to_fasta(row.iloc[0].chains)
            fid = fold_id(cid, direction, k)
            with open(f"{a.fadir}/{fid}.fasta", "w") as f:
                f.write(f">{fid}\n{fa_seq}\n")
            manifest.append(fid)
            nfa += 1
    with open(f"{a.fadir}/manifest.txt", "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"[build] wrote {nfa} FASTAs (+ manifest.txt) to {a.fadir}  "
          f"(~{nfa} folds = {len(subset)} complexes x 7 conditions)")


if __name__ == "__main__":
    main()

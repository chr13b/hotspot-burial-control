"""Experiment D, Task 1 prep: build ColabFold (AF2-multimer) FASTA inputs from Exp A's queries.

Reuses the BYTE-IDENTICAL per-chain sequences OpenFold3 received in Exp A (results/expA_queries.json,
chains ordered list(g1)+list(g2)), so AF2-multimer and OF3 get the same inputs and the predicted<->
crystal residue mapping (results/expA_resmap.json) transfers unchanged. One FASTA per complex; chains
are joined by ':' (ColabFold's multimer separator, assigned chain IDs A,B,C,... in this order).

Usage:
  python3 src/expD_build_fastas.py --queries results/expA_queries.json --out-dir $SCRATCH/ftax/expD/fastas
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="results/expA_queries.json")
    ap.add_argument("--out-dir", default=os.path.expandvars("$SCRATCH/ftax/expD/fastas"))
    a = ap.parse_args()

    q = json.load(open(a.queries))["queries"]
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    manifest = []
    for cid, rec in q.items():
        seqs = [c["sequence"] for c in rec["chains"]]
        with open(os.path.join(a.out_dir, f"{cid}.fasta"), "w") as f:
            f.write(f">{cid}\n{':'.join(seqs)}\n")
        manifest.append(cid)
        n += 1
    with open(os.path.join(a.out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"[build_fastas] wrote {n} FASTAs to {a.out_dir} (+ manifest.txt)")


if __name__ == "__main__":
    main()

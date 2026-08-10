"""Split the 141-complex OpenFold3 query JSON into K balanced chunks for a Slurm array.

Round-robin assignment (items[i::k]) so large and small complexes are spread evenly across chunks
rather than clustered. Each chunk keeps the {"queries": {...}} schema OpenFold3 expects.

Usage:
  python3 src/expA_split_queries.py --in results/expA_queries.json \
      --out-dir $SCRATCH/ftax/expA/chunks --k 12
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/expA_queries.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()

    q = json.load(open(a.inp))["queries"]
    items = list(q.items())
    os.makedirs(a.out_dir, exist_ok=True)
    sizes = []
    for i in range(a.k):
        chunk = dict(items[i::a.k])
        with open(os.path.join(a.out_dir, f"chunk_{i}.json"), "w") as f:
            json.dump({"queries": chunk}, f, indent=1)
        sizes.append(len(chunk))
    print(f"[split] {len(items)} complexes -> {a.k} chunks, sizes {sizes} -> {a.out_dir}")


if __name__ == "__main__":
    main()

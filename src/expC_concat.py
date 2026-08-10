"""Concatenate sharded expC_score outputs into single positions/backbones CSVs."""
import glob
import os
import sys

import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "."
for kind in ("positions", "backbones"):
    fs = sorted(glob.glob(os.path.join(d, f"scored_sh*_{kind}.csv")))
    if not fs:
        print(f"[concat] no {kind} shards in {d}")
        continue
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out = os.path.join(d, f"scored_{kind}.csv")
    df.to_csv(out, index=False)
    print(f"[concat] {kind}: {len(df)} rows from {len(fs)} shards -> {out} "
          f"({df['backbone_id'].nunique()} backbones)")

#!/usr/bin/env python3
"""Parse AF2-multimer folds of the NAIVE arm only -> results/iptm_steer_naive.csv (naive rows; the committed
wt/L/random folds in iptm_steer.csv are reused, not re-folded). Reuses parse_iptm.py's helpers so the interface
indexing / metric extraction is byte-identical to the standing result.

  python3 src/parse_iptm_naive.py --out-dir $SCRATCH/ftax/iptm/naive_out --out results/iptm_steer_naive.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parse_iptm as pi   # noqa: E402  (load_scores, interface_indices, parse_fid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ["SCRATCH"] + "/ftax/iptm/naive_out")
    ap.add_argument("--subset", default="results/iptm_subset.txt")
    ap.add_argument("--positions", default="results/leverage_skempi_positions.csv")
    ap.add_argument("--out", default="results/iptm_steer_naive.csv")
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
        cx = pi.interface_indices(cid)
        if cx is None:
            continue
        keys = [(cx.chains[j], int(cx.resnums[j]), cx.icodes[j]) for j in range(cx.n)]
        ik = iface.get(cid, set())
        g1i = np.array([j for j in range(cx.n) if keys[j] in ik and cx.group[j] == 1])
        g2i = np.array([j for j in range(cx.n) if keys[j] in ik and cx.group[j] == 2])
        alli = np.array([j for j in range(cx.n) if keys[j] in ik])
        for k in (0, 1, 2):
            fid = f"{cid}__naive__k{k}"
            sc = pi.load_scores(a.out_dir, fid)
            if sc is None:
                missing += 1
                continue
            ipae = np.nan
            if sc["pae"] is not None and len(g1i) and len(g2i):
                P = sc["pae"]
                if P.shape[0] == cx.n:
                    ipae = float(np.concatenate([P[np.ix_(g1i, g2i)].ravel(),
                                                 P[np.ix_(g2i, g1i)].ravel()]).mean())
            iplddt = float(sc["plddt"][alli].mean()) if (sc["plddt"].size == cx.n and len(alli)) else np.nan
            rows.append(dict(complex_id=cid, direction="naive", k=k,
                             iptm=sc["iptm"], ptm=sc["ptm"],
                             interface_pae=round(ipae, 4) if np.isfinite(ipae) else np.nan,
                             interface_plddt=round(iplddt, 4) if np.isfinite(iplddt) else np.nan,
                             n_iface=int(len(alli))))
    df = pd.DataFrame(rows)
    df["seed"] = 20260803
    df.to_csv(a.out, index=False)
    print(f"[parse-naive] {len(df)} naive folds over {df.complex_id.nunique()} complexes "
          f"({missing} missing) -> {a.out}")
    if len(df):
        print(f"[parse-naive] naive interface ipTM: median={df.iptm.median():.3f}")


if __name__ == "__main__":
    main()

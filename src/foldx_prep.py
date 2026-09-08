#!/usr/bin/env python3
"""FoldX-INDEPENDENT prep for the two FoldX lanes (pre-reg: results/PREREG_foldx.md). Needs no FoldX binary:
it computes the Lane-A L-vs-experimental baseline (the comparator for FoldX) and emits the exact per-mutation
work-list FoldX will build for both lanes, verifying the Lane-B interface-mutation plumbing.

Lane A: SKEMPI single-mut interface fixture -> Spearman(L, experimental ddG); worklist rows (complex, chain, wt,
        resnum, icode, mut) for FoldX BuildModel.
Lane B: 60 ipTM complexes x arms {L, naive, random} x k -> interface mutations (arm seq vs wt) in FoldX terms.

  python3 src/foldx_prep.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "models"), os.path.join(HERE, "decoding")):
    sys.path.insert(0, p)
import ftax_common as fc            # noqa: E402
import leverage_decomposition as LD  # noqa: E402
import judge_dumped as jd           # noqa: E402  (chains_to_cxseq)
DATA, AA20, IDX = LD.DATA, LD.AA20, LD.IDX


def lane_a(worklist):
    d = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    d["icode"] = d.icode.fillna("").astype(str)
    m = d.dropna(subset=["ddG", "L"]).copy()
    m = m[m.is_interface == 1]
    if "fixture" in m.columns:
        print(f"[laneA] fixture breakdown (interface, ddG+L present): {m.fixture.value_counts().to_dict()}")
    rho, p = spearmanr(m.L.to_numpy(float), m.ddG.to_numpy(float))
    print(f"[laneA] Spearman(L, experimental ddG) = {rho:+.4f}  (p={p:.2e}, n={len(m)}, "
          f"{m.complex_id.nunique()} complexes)")
    for r in m.itertuples():
        worklist.append(dict(lane="A", complex_id=r.complex_id, arm="skempi", k=0, chain=r.chain,
                             wt=r.wt, resnum=int(r.resnum), icode=r.icode, mut=r.mut))
    pd.DataFrame([dict(metric="spearman_L_vs_expddG", rho=round(float(rho), 4), p=float(p),
                       n=int(len(m)), n_complex=int(m.complex_id.nunique()), seed=LD.SEED,
                       note="Lane-A comparator (FoldX side pending binary); interface single mutants, ddG+L")]
                 ).to_csv("results/foldx_laneA_Lbaseline.csv", index=False)


def lane_b(worklist, subset="results/iptm_subset.txt"):
    cids = [l.strip() for l in open(subset) if l.strip()]
    seqs = pd.concat([pd.read_csv("results/cfg_steer_seqs.csv"),
                      pd.read_csv("results/cfg_steer_naive_seqs.csv")], ignore_index=True)
    seqs["k"] = seqs.k.astype(int)
    counts, nonint = {}, 0
    for cid in cids:
        pdb, g1, g2 = cid.split("_"); path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None:
            continue
        wtrow = seqs[(seqs.complex_id == cid) & (seqs.direction == "wt")]
        if not len(wtrow):
            continue
        wt_aa = jd.chains_to_cxseq(cx, wtrow.iloc[0].chains)
        for arm in ("L", "naive", "random"):
            for k in (0, 1, 2):
                row = seqs[(seqs.complex_id == cid) & (seqs.direction == arm) & (seqs.k == k)]
                if not len(row):
                    continue
                aa = jd.chains_to_cxseq(cx, row.iloc[0].chains)
                nm = 0
                for j in range(cx.n):
                    if aa[j] != wt_aa[j] and aa[j] in IDX and wt_aa[j] in IDX:
                        worklist.append(dict(lane="B", complex_id=cid, arm=arm, k=k, chain=cx.chains[j],
                                             wt=wt_aa[j], resnum=int(cx.resnums[j]), icode=cx.icodes[j], mut=aa[j]))
                        nm += 1
                counts.setdefault(arm, []).append(nm)
    for arm, v in counts.items():
        print(f"[laneB] arm={arm:6s}: {len(v)} (complex,k) sets, mean {np.mean(v):.1f} mut/set, total {sum(v)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/foldx_worklist.csv")
    a = ap.parse_args()
    worklist = []
    lane_a(worklist)
    lane_b(worklist)
    wl = pd.DataFrame(worklist)
    wl.to_csv(a.out, index=False)
    print(f"[prep] wrote {len(wl)} mutation rows -> {a.out} "
          f"(A={int((wl.lane == 'A').sum())} single mutants, B={int((wl.lane == 'B').sum())} steered)")


if __name__ == "__main__":
    main()

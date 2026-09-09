#!/usr/bin/env python3
"""Compute JUDGE (ESM-IF1 / MIF) leverage on PREDICTED backbones (OF3 / AF2) over the interface positions, so the
predicted-backbone steering run (cfg_steer_predicted.py) can be judged ANTI-CIRCULARLY on the design-time structure.

Blocker this closes: leverage_predicted.py scores ProteinMPNN ONLY on predicted backbones; the anti-circular judges
have no predicted-backbone leverage cache. This reuses the EXACT scorers of leverage_esmif.py /
leverage_extra_models.py (a pure PDB-dir + keepset swap — same double difference, same alphabet controls), so the
predicted judge leverage is apples-to-apples with the crystal judge caches leverage_pq_skempi_{esmif,mif}.csv.

Interface positions are taken from the PREDICTED structure (leverage_predicted_<source>_positions.csv, is_interface)
— the honest design-time set the steering samples over. `crystal` source reproduces the committed crystal cache (a
positive control on this wrapper).

  python3 src/leverage_judge_predicted.py --model esmif --source of3 --device cuda   # GPU ~2 s/cx
  python3 src/leverage_judge_predicted.py --model mif   --source af2 --device cpu
  -> results/leverage_pq_predicted_<source>_<model>.csv  (schema: complex_id,chain,resnum,icode,aa,lP_*,lQ_*)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models"))
import leverage_decomposition as LD          # noqa: E402
import leverage_esmif as LE                   # noqa: E402
import leverage_extra_models as LEM           # noqa: E402

SEED = LD.SEED
SRC = {   # source -> (pdb dir, interface-positions file on the predicted structure)
    "crystal": (os.path.expanduser("~/ftax/data/PDBs"), "results/leverage_skempi_positions.csv"),
    "of3": (os.path.expandvars("$SCRATCH/ftax/predicted/PDBs"), "results/leverage_predicted_of3_positions.csv"),
    "af2": (os.path.expandvars("$SCRATCH/ftax/expD/PDBs"), "results/leverage_predicted_af2_positions.csv"),
}


def keepset(posfile):
    """complex_id -> {(chain,resnum,icode)} for interface positions on the (predicted) structure."""
    p = pd.read_csv(posfile, low_memory=False)
    p["icode"] = p.icode.fillna("").astype(str)
    mask = p.is_interface.isin([True, 1, "True", "1"])
    p = p[mask]
    keep = {}
    for r in p.itertuples():
        keep.setdefault(r.complex_id, set()).add((r.chain, int(r.resnum), r.icode))
    return keep


def make_lp(model_name, device, mif_seeds):
    """Return lp(cx) -> [cx.n, 21] MPNN-alphabet log-probs (the same scorers the crystal judge caches used)."""
    if model_name == "esmif":
        import ftax_common as fc
        import ftax_esmif as fe
        model, alphabet = fe.load_esmif(device=device)
        amap = fe.build_alphabet_map(alphabet)
        back = "".join(alphabet.get_tok(int(i)) for i in amap)
        assert back == fc.MPNN_ALPHABET, f"ALPHABET SLIP: {back!r}"   # +control: an ESM->MPNN slip drops rec to ~0.05
        print("[+control] ESM-IF1 alphabet map round-trips OK", flush=True)
        return lambda cx: LE.esmif_lp(model, alphabet, cx, device)
    return LEM.make_scorer(model_name, mif_seeds)      # mif: loaded on CPU inside make_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["esmif", "mif"])
    ap.add_argument("--source", required=True, choices=list(SRC))
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda' (esmif only; mif is cpu)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mif-seeds", type=int, default=6)
    ap.add_argument("--max-residues", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    pdbdir, posfile = SRC[a.source]
    keep = keepset(posfile)
    cx_ids = sorted(keep)
    if a.limit:
        cx_ids = cx_ids[:a.limit]
    out = a.out or f"results/leverage_pq_predicted_{a.source}_{a.model}.csv"
    lp = make_lp(a.model, a.device, a.mif_seeds)
    print(f"[judge-pred] model={a.model} source={a.source} pdbdir={pdbdir} {len(cx_ids)} complexes "
          f"({sum(len(keep[c]) for c in cx_ids)} interface positions) -> {out}", flush=True)

    done = set()
    if os.path.exists(out) and not a.limit:
        try:
            done = set(pd.read_csv(out, usecols=["complex_id"]).complex_id)
            print(f"[judge-pred] resuming, {len(done)} complexes already scored", flush=True)
        except Exception:
            done = set()
    fh = open(out, "a" if done else "w", newline="")
    writer, n, t0, recs = None, 0, time.time(), []
    for ci, cid in enumerate(cx_ids):
        if cid in done:
            continue
        pdb, g1, g2 = cid.split("_")
        path = f"{pdbdir}/{pdb}.pdb"
        if not os.path.exists(path):
            print(f"  skip {cid}: no pdb at {path}", flush=True)
            continue
        try:
            rows, rec = LEM._score_one(lp, path, pdb, g1, g2, keep[cid], a.max_residues)
        except Exception as e:
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
            continue
        if isinstance(rec, tuple):
            print(f"  drop {cid} (too large: {rec[1]} res)", flush=True)
            continue
        if rec is not None:
            recs.append(rec)
        for r in rows:
            r = dict(complex_id=cid, **r)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if not done:
                    writer.writeheader()
            writer.writerow(r)
            n += 1
        fh.flush()
        if (ci + 1) % 10 == 0 or ci == len(cx_ids) - 1:
            dt = time.time() - t0
            print(f"[judge-pred] {ci+1}/{len(cx_ids)} {cid} rec={rec:.3f} rows={len(rows)} "
                  f"({dt/max(1,ci+1):.1f}s/cx)", flush=True)
    fh.close()
    print(f"[judge-pred] wrote {out}: {n} rows; mean complex top-1 recovery "
          f"{np.mean(recs) if recs else float('nan'):.3f} (healthy ~0.35-0.55; ~0.05 = broken alphabet)", flush=True)


if __name__ == "__main__":
    main()

"""Replicate the burial-matched hotspot analysis across a panel of inverse-folding models.

Answers the single most common reviewer objection to the Phase 0 result: "you tested one
1.7M-parameter model."

The matched pairs are determined by STRUCTURE (rSASA, secondary structure, neighbour
count) and by experimental ddG labels - not by the model. So the pairs computed once in
p0_burial_matched.py are reused verbatim, and each model only has to re-score the
complexes those pairs live in (141 of 343). Identical pairs across models also means the
cross-model comparison is exactly matched.

Panel (see src/models/ftax_panel.py):
  mpnn_vanilla   1.7M   GNN, autoregressive      native multichain
  mpnn_soluble   1.7M   GNN, autoregressive      native multichain   (weight variant)
  esmif        141.7M   GVP-transformer, causal  native multichain
  pifold         6.6M   GNN, one-shot            single-chain-trained, bit-exact
  mif            3.4M   GNN, masked              single-chain-trained

pifold and mif have no chain representation, so `junction_flank` residues either side of
an artificial chain junction are dropped (their features are fabricated there).

Usage:
  python3 src/p0_multimodel.py --models pifold,mpnn_soluble --out results/panel
"""

import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models"))
import ftax_common as fc

N_BOOT, SEED = 10000, 20260803


def boot(df, col, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df.loc[df["complex_id"] == c, col].values for c in cids}
    m = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        m[b] = np.nanmean(np.concatenate([by[cids[i]] for i in pick]))
    return float(np.nanmean(df[col])), *np.nanpercentile(m, [2.5, 97.5])


def junction_positions(cx, flank):
    """Indices within `flank` residues of an artificial chain junction."""
    if flank <= 0:
        return set()
    bad = set()
    for i in range(1, cx.n):
        if cx.chains[i] != cx.chains[i - 1]:
            for j in range(max(0, i - flank), min(cx.n, i + flank)):
                bad.add(j)
    return bad


def score_model(name, cx_ids, data_dir, out_csv, resume=True):
    import ftax_panel as panel
    meta = panel.PANEL[name]
    flank = meta["junction_flank"]

    done = set()
    if resume and os.path.exists(out_csv):
        try:
            done = set(pd.read_csv(out_csv)["complex_id"])
            print(f"[{name}] resuming, {len(done)} complexes already scored")
        except Exception:
            done = set()

    handle = panel.load(name)
    fh = open(out_csv, "a" if done else "w", newline="")
    writer, n, t0 = None, 0, time.time()

    for ci, cid in enumerate(cx_ids):
        if cid in done:
            continue
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
            if cx is None:
                continue
            lp = panel.score(name, handle, cx)                 # [n_orders, L, 21]
            mean_lp = lp.mean(axis=0)
            mix = fc.order_mixture_logprobs(lp)[:, :20]
            mix = mix - np.log(np.exp(mix).sum(axis=1, keepdims=True))
            bad = junction_positions(cx, flank)
            nat = np.array([fc.MPNN_ALPHABET.index(a) for a in cx.seq])
            for i in range(cx.n):
                row = dict(model=name, complex_id=cid, chain=cx.chains[i],
                           resnum=int(cx.resnums[i]), icode=cx.icodes[i], aa=cx.seq[i],
                           logp_native=float(mean_lp[i, nat[i]]),
                           mode_aa=fc.MPNN_ALPHABET[int(mix[i, :20].argmax())],
                           near_junction=int(i in bad), n_orders=lp.shape[0])
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if not done:
                        writer.writeheader()
                writer.writerow(row)
                n += 1
            fh.flush()
            del lp, mean_lp, mix
            gc.collect()
        except Exception as e:
            print(f"  [{name}] skip {cid}: {type(e).__name__}: {e}", flush=True)
            continue
        if (ci + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[{name}] {ci+1}/{len(cx_ids)}  {el:.0f}s  {n} residues", flush=True)
    fh.close()
    return n


def analyse(name, pos_csv, pairs_prefix, drop_junction=True):
    pos = pd.read_csv(pos_csv)
    if drop_junction and "near_junction" in pos.columns:
        n0 = len(pos)
        pos = pos[pos["near_junction"] == 0]
        dropped = n0 - len(pos)
    else:
        dropped = 0
    k = pos.set_index(["complex_id", "chain", "resnum"])["logp_native"]
    hit = (pos["aa"] == pos["mode_aa"]).astype(float)
    kh = pos.assign(hit=hit).set_index(["complex_id", "chain", "resnum"])["hit"]

    out = []
    import glob
    for f in sorted(glob.glob(f"{pairs_prefix}_pairs_*.csv")):
        tag = os.path.basename(f).split("_pairs_")[1][:-4]
        pr = pd.read_csv(f)
        rows = []
        for _, r in pr.iterrows():
            try:
                h = float(k.loc[(r.complex_id, r.hot_chain, r.hot_resnum)])
                c = float(k.loc[(r.complex_id, r.ctl_chain, r.ctl_resnum)])
                hh = float(kh.loc[(r.complex_id, r.hot_chain, r.hot_resnum)])
                ch = float(kh.loc[(r.complex_id, r.ctl_chain, r.ctl_resnum)])
            except (KeyError, TypeError):
                continue
            rows.append(dict(complex_id=r.complex_id, d=h - c, d_rec=hh - ch))
        d = pd.DataFrame(rows)
        if len(d) < 10:
            continue
        m, lo, hi = boot(d, "d")
        mr, lor, hir = boot(d, "d_rec")
        out.append(dict(model=name, analysis=tag, n_pairs=len(d),
                        n_complexes=d["complex_id"].nunique(),
                        gap_logp=m, lo=lo, hi=hi,
                        gap_recovery=mr, rec_lo=lor, rec_hi=hir,
                        n_dropped_junction=dropped))
        print(f"  {name:14s} {tag:30s} n={len(d):4d} logp gap={m:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"   recovery gap={mr:+.4f} [{lor:+.4f},{hir:+.4f}]")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pifold")
    ap.add_argument("--out", default="results/panel")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--complexes", default="results/pair_complexes.txt")
    ap.add_argument("--pairs-prefix", default="results/p0_dssp")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch threads; lower = smaller peak RSS on a contended box")
    ap.add_argument("--max-batch", type=int, default=2,
                    help="cap decoding-order batch; 7.5GB box shared with other jobs")
    ap.add_argument("--max-len", type=int, default=100000)
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    import torch
    torch.set_num_threads(a.threads)
    os.environ["FTAX_MAX_BATCH"] = str(a.max_batch)
    os.environ["FTAX_MAX_LEN"] = str(a.max_len)
    cx_ids = [l.strip() for l in open(a.complexes) if l.strip()]
    allres = []
    for name in a.models.split(","):
        name = name.strip()
        pos_csv = f"{a.out}_{name}_positions.csv"
        if not a.analyse_only:
            print(f"\n=== scoring {name} over {len(cx_ids)} complexes ===", flush=True)
            t0 = time.time()
            n = score_model(name, cx_ids, a.data_dir, pos_csv)
            print(f"[{name}] {n} residues in {time.time()-t0:.0f}s -> {pos_csv}")
        if os.path.exists(pos_csv):
            print(f"\n=== matched-pair analysis: {name} ===")
            allres.append(analyse(name, pos_csv, a.pairs_prefix))
    if allres:
        df = pd.concat(allres, ignore_index=True).assign(command=cmd)
        outf = f"{a.out}_summary.csv"
        if os.path.exists(outf):
            df = pd.concat([pd.read_csv(outf), df], ignore_index=True) \
                   .drop_duplicates(subset=["model", "analysis"], keep="last")
        df.to_csv(outf, index=False)
        print(f"\n[done] wrote {outf}")


if __name__ == "__main__":
    main()

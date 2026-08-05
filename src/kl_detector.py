"""Residue-agnostic partner-sensitivity: a DESIGN-TIME hotspot detector.

`d_bind_local` (src/frustration_monomer.py) is the strongest hotspot signal this project
found, but it needs the native residue identity - it is log p(native | complex) minus
log p(native | monomer). That makes it a SCORING statistic for an existing complex
(in-silico alanine scanning, epitope mapping, ddG ranking), not something a designer can
use on a backbone before any sequence exists.

This computes the residue-agnostic version. Using ProteinMPNN's UNCONDITIONAL
(backbone-only, sequence-free) distribution at each position:

    P_i = p(. | bound complex backbone)        Q_i = p(. | own chain group backbone)

    KL_i  = KL(P_i || Q_i)          how much the partner's presence moves the
                                     distribution at position i - pure geometry,
                                     no residue identity anywhere
    JSD_i = Jensen-Shannon(P_i,Q_i)  symmetric, bounded variant
    dH_i  = H(Q_i) - H(P_i)          entropy reduction caused by the partner

If KL detects hotspots as well as d_bind_local does, the detector becomes usable at
design time and the claim gets much stronger.

Usage:
  python3 src/kl_detector.py --out results/kl_detector
"""

import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

N_BOOT, SEED = 10000, 20260803


def _dists(lp20):
    """[L,21] log-probs -> [L,20] normalised probabilities over the 20 standard AAs."""
    z = lp20[:, :20]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def boot_auc(df, score_col, label_col="is_hot", n_boot=2000, seed=SEED):
    """Complex-level bootstrap of AUROC."""
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df.loc[df["complex_id"] == c, [score_col, label_col]].values for c in cids}

    def auc(a):
        s, y = a[:, 0], a[:, 1]
        m = np.isfinite(s)
        s, y = s[m], y[m]
        if y.sum() == 0 or y.sum() == len(y):
            return np.nan
        r = stats.rankdata(s)
        n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    point = auc(np.concatenate([by[c] for c in cids]))
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        out[b] = auc(np.concatenate([by[cids[i]] for i in pick]))
    return float(point), *np.nanpercentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kl_detector")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--complexes", default="results/pair_complexes.txt")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--mpnn-weights",
                    default=os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)
    out_csv = f"{a.out}_positions.csv"

    if not a.analyse_only:
        import torch
        torch.set_num_threads(a.threads)
        model, _ = fc.load_mpnn(a.mpnn_weights)
        cx_ids = [l.strip() for l in open(a.complexes) if l.strip()]
        done = set()
        if os.path.exists(out_csv):
            try:
                done = set(pd.read_csv(out_csv)["complex_id"])
                print(f"[kl] resuming, {len(done)} complexes done")
            except Exception:
                done = set()
        fh = open(out_csv, "a" if done else "w", newline="")
        writer, n, t0 = None, 0, time.time()

        for ci, cid in enumerate(cx_ids):
            if cid in done:
                continue
            pdb, g1, g2 = cid.split("_")
            path = os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb")
            if not os.path.exists(path):
                continue
            try:
                cx = fc.load_complex(path, pdb, g1, g2)
                if cx is None:
                    continue
                P = _dists(fc.mpnn_unconditional_logprobs(model, cx))   # bound complex
                # monomer distributions, one chain group at a time
                Q = np.full_like(P, np.nan)
                for grp, chains in ((1, g1), (2, g2)):
                    mono = fc.load_complex(path, pdb, chains, "", require_both=False)
                    if mono is None or mono.n < 5:
                        continue
                    Qm = _dists(fc.mpnn_unconditional_logprobs(model, mono))
                    idx = {(c, int(r), i): k for k, (c, r, i)
                           in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
                    for j in range(cx.n):
                        k = idx.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
                        if k is not None:
                            Q[j] = Qm[k]
                    del Qm, mono
                ok = np.isfinite(Q).all(axis=1)
                eps = 1e-12
                kl = np.where(ok, (P * (np.log(P + eps) - np.log(Q + eps))).sum(axis=1), np.nan)
                M = 0.5 * (P + Q)
                jsd = np.where(ok, 0.5 * (P * (np.log(P + eps) - np.log(M + eps))).sum(axis=1)
                               + 0.5 * (Q * (np.log(Q + eps) - np.log(M + eps))).sum(axis=1), np.nan)
                hP = -(P * np.log(P + eps)).sum(axis=1)
                hQ = np.where(ok, -(Q * np.log(Q + eps)).sum(axis=1), np.nan)

                for j in range(cx.n):
                    row = dict(complex_id=cid, chain=cx.chains[j],
                               resnum=int(cx.resnums[j]), icode=cx.icodes[j],
                               aa=cx.seq[j], kl=float(kl[j]), jsd=float(jsd[j]),
                               dH=float(hQ[j] - hP[j]), H_complex=float(hP[j]))
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                        if not done:
                            writer.writeheader()
                    writer.writerow(row)
                    n += 1
                fh.flush()
                del P, Q, kl, jsd, cx
                gc.collect()
            except Exception as e:
                print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
                continue
            if (ci + 1) % 20 == 0:
                print(f"[kl] {ci+1}/{len(cx_ids)}  {time.time()-t0:.0f}s  {n} residues",
                      flush=True)
        fh.close()
        print(f"[kl] wrote {out_csv}: {n} residues")

    # ------------------------------------------------------------- analysis
    kl = pd.read_csv(out_csv)
    kl["icode"] = kl["icode"].fillna("").astype(str)
    pos = pd.read_csv(a.positions,
                      usecols=["complex_id", "chain", "resnum", "icode", "label",
                               "is_interface", "rsasa_complex", "logp_native", "nbr"])
    pos["icode"] = pos["icode"].fillna("").astype(str)
    pos["label"] = pos["label"].fillna("null")
    m = pos.merge(kl, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    m = m[m["is_interface"] & np.isfinite(m["kl"])]
    m["is_hot"] = (m["label"] == "hot_strict").astype(int)
    m["burial"] = -m["rsasa_complex"]
    m.to_csv(f"{a.out}_joined.csv", index=False)

    print(f"\n=== residue-agnostic detector: {len(m)} interface positions, "
          f"{m['complex_id'].nunique()} complexes, {int(m['is_hot'].sum())} strict hotspots ===")
    rows = []
    for name, col in [("burial (-rSASA) BASELINE", "burial"),
                      ("KL(complex || monomer)", "kl"),
                      ("JSD(complex, monomer)", "jsd"),
                      ("entropy drop dH", "dH"),
                      ("log p(native|complex)", "logp_native")]:
        pt, lo, hi = boot_auc(m, col)
        print(f"  {name:28s} AUROC = {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
        rows.append(dict(score=col, auroc=pt, lo=lo, hi=hi, n=len(m)))

    # simple combinations, via rank-averaging (no fitting -> no leakage)
    for name, cols in [("burial + KL", ["burial", "kl"]),
                       ("burial + JSD", ["burial", "jsd"]),
                       ("burial + KL + nbr", ["burial", "kl", "nbr"])]:
        z = np.mean([stats.rankdata(m[c]) / len(m) for c in cols], axis=0)
        m["_combo"] = z
        pt, lo, hi = boot_auc(m, "_combo")
        print(f"  {name:28s} AUROC = {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
        rows.append(dict(score=name, auroc=pt, lo=lo, hi=hi, n=len(m)))

    # burial-orthogonality
    print("\n  KL AUROC within burial quintiles (burial itself is ~0.50 within a stratum):")
    m["q"] = pd.qcut(m["rsasa_complex"], 5, labels=False, duplicates="drop")
    for q in sorted(m["q"].dropna().unique()):
        s = m[m["q"] == q]
        if s["is_hot"].sum() < 5:
            continue
        pt, lo, hi = boot_auc(s, "kl", n_boot=500)
        print(f"    quintile {int(q)+1}: AUROC = {pt:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"(n={len(s)}, {int(s['is_hot'].sum())} hot)")
        rows.append(dict(score=f"kl_burial_quintile{int(q)+1}", auroc=pt, lo=lo, hi=hi,
                         n=len(s)))

    pd.DataFrame(rows).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"\n[done] wrote {a.out}_summary.csv")


if __name__ == "__main__":
    main()

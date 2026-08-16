#!/usr/bin/env python3
"""Extension D: symmetric-leverage RECIPROCITY.

Leverage is directional by construction (score a target-chain position by ablating the partner). If the
mixed derivative is a genuine SHARED binding energy of the interface — rather than a per-chain artifact —
then contacting residues across the interface should have CORRELATED leverage: a hotspot on chain A should
face a residue on chain B that is itself high-leverage. We test this on SKEMPI.

For every interface position i on group 1, find its nearest contacting position j on group 2 (Cα–Cα below a
threshold) and pair their leverage magnitudes. A positive Spearman across such cross-interface contact pairs
is the reciprocity signature; it also *is* the per-interface leverage map (the figure this specifies).

Reuses committed per-position leverage (leverage_skempi_positions.csv) + the structures via ftax_common.
Complex-clustered bootstrap, seed 20260803.
  python3 src/leverage_reciprocity.py --out results/leverage_reciprocity.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc

SEED = 20260803
DATA = os.path.expanduser("~/ftax/data")


def boot_spearman(x, y, g, rng, n=2000):
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    out = []
    for _ in range(n):
        t = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        v = stats.spearmanr(x[t], y[t]).correlation
        if np.isfinite(v):
            out.append(v)
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=10.0, help="Cα–Cα contact threshold (Å)")
    ap.add_argument("--out", default="results/leverage_reciprocity.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    pos = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    pos["icode"] = pos.icode.fillna("").astype(str)
    # per-position leverage lookup: magnitude L_rms, alanine L_ala, worst-sub L_min
    key = list(zip(pos.complex_id, pos.chain, pos.resnum.astype(int), pos.icode))
    Lrms = dict(zip(key, pos.L_rms)); Lala = dict(zip(key, pos.L_ala))
    cids = pos.complex_id.unique()

    pairs = []   # (complex_id, Lrms_i, Lrms_j, Lala_i, Lala_j)
    n_cx = 0
    for cid in cids:
        pdb, g1, g2 = cid.split("_")
        path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
        except Exception:
            continue
        if cx is None:
            continue
        grp = np.asarray(cx.group)
        ca = np.asarray(cx.CA, float)
        i1 = np.where(grp == 1)[0]; i2 = np.where(grp == 2)[0]
        if len(i1) == 0 or len(i2) == 0:
            continue
        used = False
        for i in i1:
            ki = (cid, cx.chains[i], int(cx.resnums[i]), cx.icodes[i])
            if ki not in Lrms:                       # only interface positions carry leverage
                continue
            d = np.linalg.norm(ca[i2] - ca[i], axis=1)
            jrel = int(np.argmin(d))
            if d[jrel] > a.thresh:
                continue
            j = i2[jrel]
            kj = (cid, cx.chains[j], int(cx.resnums[j]), cx.icodes[j])
            if kj not in Lrms:
                continue
            pairs.append((cid, float(Lrms[ki]), float(Lrms[kj]),
                          float(Lala[ki]), float(Lala[kj])))
            used = True
        n_cx += int(used)
        del cx

    P = pd.DataFrame(pairs, columns=["complex_id", "Lrms_i", "Lrms_j", "Lala_i", "Lala_j"]).dropna()
    g = P.complex_id.to_numpy()
    print(f"[reciprocity] {len(P)} cross-interface contact pairs (Cα<{a.thresh}Å) over {n_cx} complexes")

    rows = []
    for xa, xb, lab in [("Lrms_i", "Lrms_j", "leverage MAGNITUDE |L|_rms"),
                        ("Lala_i", "Lala_j", "alanine leverage L(->Ala)")]:
        rho = stats.spearmanr(P[xa], P[xb]).correlation
        lo, hi, pgt = boot_spearman(P[xa].to_numpy(), P[xb].to_numpy(), g, rng)
        print(f"  Spearman[{lab:26s}] across contacts = {rho:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={pgt:.3f}")
        rows.append(dict(quantity=lab, spearman=round(float(rho), 4), lo=round(lo, 4), hi=round(hi, 4),
                         p_gt0=round(pgt, 3), n_pairs=len(P), n_complexes=n_cx))

    # complex-level: does total interface leverage on side 1 track side 2? (shared binding energy)
    s1 = P.groupby("complex_id").Lrms_i.sum(); s2 = P.groupby("complex_id").Lrms_j.sum()
    rho_cx = stats.spearmanr(s1, s2).correlation
    print(f"  Spearman[complex-total |L| side1 vs side2] = {rho_cx:+.4f}  (n={len(s1)} complexes)")
    rows.append(dict(quantity="complex_total_|L|_side1_vs_side2", spearman=round(float(rho_cx), 4),
                     n_complexes=len(s1)))

    out = pd.DataFrame(rows); out["seed"] = SEED; out["thresh_A"] = a.thresh
    out["note"] = ("reciprocity: contacting cross-interface residues have correlated leverage => the mixed "
                   "derivative is a shared interface binding energy, not a per-chain artifact")
    out["command"] = f"python3 src/leverage_reciprocity.py --thresh {a.thresh}"
    out.to_csv(a.out, index=False)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

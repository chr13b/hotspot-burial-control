#!/usr/bin/env python3
"""The DECISIVE geometry control on PREDICTED / non-native backbones.

Question
--------
On crystal SKEMPI the sequence-free KL detector adds essentially nothing over a
FULL cheap-geometry baseline (burial + neighbour-count + DeltaSASA partner-contact
area): most of KL's edge over burial-alone is just DeltaSASA, a trivial geometric
feature from the same two structures KL needs (crystal geometry control:
KL-over-burial-alone ~ +0.041, KL-over-full-geometry +0.007 [-0.004,+0.019] ns).

KL's *claimed* advantage is on PREDICTED / non-native backbones (Exp A OpenFold3,
Exp D AF2-multimer: KL DeltaAUROC-over-burial +0.062 / +0.054, stronger than
crystal). Off the native manifold, the model's learned distribution *might* encode
partner-sensitivity that raw contact-area misses. This run decides whether KL is a
real method or just recapitulates DeltaSASA everywhere.

Pre-registered reading (fixed before the numbers)
-------------------------------------------------
For each predicted-backbone class, if DeltaAUROC(full+KL - full) CI EXCLUDES 0
(equivalently partial Spearman(KL, is_hot | burial+nbr+DeltaSASA) > 0, CI excludes 0)
  -> KL SURVIVES the full-geometry control -> the method claim is rescued and
     sharpened: "KL adds specifically where the backbone is non-native."
If the CI CONTAINS 0
  -> demote KL-as-method; the paper leans on the (DeltaSASA-robust) nugget + the
     burial correction + cross-predictor reproducibility.
Either outcome is publishable and honest.

Method (identical control on every fixture)
-------------------------------------------
* DeltaSASA is computed ON EACH FIXTURE'S OWN BACKBONE (crystal / OF3 / AF2),
  never from the crystal, via the project's freesasa plumbing
  (ftax_common.residue_sasa). DeltaSASA(residue) = SASA(residue | its group alone)
  - SASA(residue | full complex) in A^2 = buried-by-partner area. Matches the
  crystal control's definition exactly.
* Features: burial = -rsasa_complex (committed column), nbr, dsasa, kl. Label
  y = is_hot. Interface positions only, restricted to rows with all four features
  present (a common set, so full vs full+KL is a strictly paired comparison).
* (1) partial Spearman(kl, y | burial) and (| burial, nbr, dsasa) via rank
      residualization (rank -> lstsq residualize -> Pearson of residuals).
* (2) z-sum AUROC: burial = z(burial); full = z(burial)+z(nbr)+z(dsasa);
      +KL = the composite + z(kl). z-standardization is GLOBAL/fixed (the composite
      is one fixed ranker); the complex bootstrap resamples complexes and
      re-evaluates AUROC -- the standard paired CI for a fixed ranker (adding a
      feature to a fixed composite). Both members of each delta use the SAME
      resample, so the CIs are paired.
* (3) complex-level bootstrap, seed 20260803, nrep default 10000, percentile CI, on
      DeltaAUROC(full+KL - full) and DeltaAUROC(burial+KL - burial), and on the two
      partial Spearman coefficients.
* Sanity: Spearman corr(kl, dsasa) and corr(burial, dsasa) per fixture.

This script ports the crystal geometry control (src/kl_geometry_control.py, parallel
session; crystal + Bennett de-novo) to the PREDICTED backbones. Faithfulness is
established by the POSITIVE CONTROL: the `crystal` fixture reproduces the committed
crystal numbers (results/kl_geometry_control.csv) -- its `relative` mode matches
dAUROC_KL_over_burial_alone 0.041 and dAUROC_KL_over_full_geometry 0.0074 to four
decimals. Only then are the predicted numbers trusted. (Written from the spec before
that file was committed; both now agree.)

Usage
-----
    python3 src/kl_geometry_control_predicted.py --out results/kl_geometry_control_predicted.csv \
        --fixtures crystal:results/kl_detector_joined.csv:$SCRATCH/ftax/data/PDBs \
                   OF3_expA:$SCRATCH/ftax/predicted/expA_kl_joined.csv:$SCRATCH/ftax/predicted/PDBs \
                   AF2_expD:$SCRATCH/ftax/expD/expD_kl_joined.csv:$SCRATCH/ftax/expD/PDBs

CPU only. Seed 20260803.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc  # noqa: E402

SEED = 20260803


# --------------------------------------------------------------- DeltaSASA

def parse_cid(cid):
    """'1JRH_LH_I' -> ('1JRH', ['L','H'], ['I'])."""
    p = cid.split("_")
    return p[0], list(p[1]), list(p[2])


def dsasa_for_complex(pdb_path, pdb_id, g1, g2):
    """Buried-by-partner area (A^2) per residue, keyed (chain, resnum, icode).

    Computed on THIS backbone only: SASA of each group in isolation minus SASA in
    the complex. >= 0; large = deeply buried by the partner.
    """
    sc = fc.residue_sasa(pdb_path, pdb_id, g1 + g2)   # in the complex
    s1 = fc.residue_sasa(pdb_path, pdb_id, g1)         # group 1 isolated
    s2 = fc.residue_sasa(pdb_path, pdb_id, g2)         # group 2 isolated
    g1s, g2s = set(g1), set(g2)
    out = {}
    for k, v in sc.items():
        ch = k[0]
        if ch in g1s:
            out[k] = s1.get(k, v) - v
        elif ch in g2s:
            out[k] = s2.get(k, v) - v
    return out


def norm_icode(ic):
    if ic is None:
        return ""
    if isinstance(ic, float) and np.isnan(ic):
        return ""
    s = str(ic).strip()
    return "" if s.lower() in ("nan", "none") else s


def load_fixture(joined_csv, pdb_dir):
    """Interface positions of a joined table + a DeltaSASA column from its backbones."""
    df = pd.read_csv(joined_csv)
    im = df["is_interface"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df = df[im].copy().reset_index(drop=True)
    df["icode_n"] = df["icode"].map(norm_icode)
    df["y"] = df["is_hot"].astype(float).round().astype(int)

    dsasa = np.full(len(df), np.nan)
    rows_by_cx = {}
    for pos, (cid, ch, rn, ic) in enumerate(
        zip(df["complex_id"], df["chain"], df["resnum"], df["icode_n"])
    ):
        rows_by_cx.setdefault(cid, []).append((pos, (str(ch), int(rn), ic)))

    missing_pdb = []
    for cid, rows in rows_by_cx.items():
        pdb, g1, g2 = parse_cid(cid)
        pp = os.path.join(pdb_dir, pdb + ".pdb")
        if not os.path.exists(pp):
            missing_pdb.append(cid)
            continue
        d = dsasa_for_complex(pp, pdb, g1, g2)
        for pos, key in rows:
            if key in d:
                dsasa[pos] = d[key]
    df["dsasa_abs"] = dsasa  # absolute buried area (A^2); relative variant derived in main()
    return df, missing_pdb


# --------------------------------------------------------------- statistics

def zc(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


def auroc(score, y):
    """Tie-aware AUROC via the Mann-Whitney rank statistic. Higher score = positive."""
    npos = int(y.sum())
    n = len(y)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(score)
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def partial_spearman(x, y, covars):
    """Partial Spearman corr(x, y | covars) by rank residualization."""
    n = len(x)
    rx = rankdata(x)
    ry = rankdata(y)
    R = np.column_stack([rankdata(c) for c in covars] + [np.ones(n)])
    bx, *_ = np.linalg.lstsq(R, rx, rcond=None)
    by, *_ = np.linalg.lstsq(R, ry, rcond=None)
    ex = rx - R @ bx
    ey = ry - R @ by
    sx, sy = ex.std(), ey.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((ex - ex.mean()) * (ey - ey.mean())).mean() / (sx * sy))


def pctci(a):
    a = a[~np.isnan(a)]
    if a.size == 0:
        return (np.nan, np.nan)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def analyze(df, label, nrep, seed):
    d = df.dropna(subset=["burial", "nbr", "dsasa", "kl"]).copy()
    n_all = len(df)
    n = len(d)
    y = d["y"].values.astype(int)
    burial = d["burial"].values.astype(float)
    nbr = d["nbr"].values.astype(float)
    dsasa = d["dsasa"].values.astype(float)
    kl = d["kl"].values.astype(float)

    zb, zn, zd, zk = zc(burial), zc(nbr), zc(dsasa), zc(kl)
    s_bur = zb
    s_burkl = zb + zk
    s_full = zb + zn + zd
    s_fullkl = s_full + zk

    a_bur, a_burkl = auroc(s_bur, y), auroc(s_burkl, y)
    a_full, a_fullkl = auroc(s_full, y), auroc(s_fullkl, y)
    d_burkl = a_burkl - a_bur
    d_fullkl = a_fullkl - a_full
    p_bur = partial_spearman(kl, y, [burial])
    p_full = partial_spearman(kl, y, [burial, nbr, dsasa])

    uniq = list(dict.fromkeys(d["complex_id"]))
    cxvals = d["complex_id"].values
    idx_by = [np.flatnonzero(cxvals == c) for c in uniq]
    ncx = len(uniq)
    rng = np.random.default_rng(seed)
    B_dburkl = np.full(nrep, np.nan)
    B_dfullkl = np.full(nrep, np.nan)
    B_pbur = np.full(nrep, np.nan)
    B_pfull = np.full(nrep, np.nan)
    for r in range(nrep):
        pick = rng.integers(0, ncx, ncx)
        sel = np.concatenate([idx_by[i] for i in pick])
        yy = y[sel]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        B_dburkl[r] = auroc(s_burkl[sel], yy) - auroc(s_bur[sel], yy)
        B_dfullkl[r] = auroc(s_fullkl[sel], yy) - auroc(s_full[sel], yy)
        B_pbur[r] = partial_spearman(kl[sel], yy, [burial[sel]])
        B_pfull[r] = partial_spearman(kl[sel], yy, [burial[sel], nbr[sel], dsasa[sel]])

    dbk_lo, dbk_hi = pctci(B_dburkl)
    dfk_lo, dfk_hi = pctci(B_dfullkl)
    pb_lo, pb_hi = pctci(B_pbur)
    pf_lo, pf_hi = pctci(B_pfull)

    sp_kl_d = float(spearmanr(kl, dsasa)[0])   # [0] = correlation, robust across scipy versions
    sp_bur_d = float(spearmanr(burial, dsasa)[0])

    survives = (not np.isnan(dfk_lo)) and (dfk_lo > 0)
    return dict(
        fixture=label, n_complexes=ncx, n_positions=n, n_positions_all_interface=n_all,
        n_hot=int(y.sum()),
        auroc_burial=a_bur, auroc_full=a_full, auroc_burial_kl=a_burkl, auroc_full_kl=a_fullkl,
        d_auroc_burkl_minus_bur=d_burkl, d_burkl_lo=dbk_lo, d_burkl_hi=dbk_hi,
        d_auroc_fullkl_minus_full=d_fullkl, d_fullkl_lo=dfk_lo, d_fullkl_hi=dfk_hi,
        partial_kl_given_burial=p_bur, pkb_lo=pb_lo, pkb_hi=pb_hi,
        partial_kl_given_full=p_full, pkf_lo=pf_lo, pkf_hi=pf_hi,
        spearman_kl_dsasa=sp_kl_d, spearman_burial_dsasa=sp_bur_d,
        kl_survives_full_geometry=bool(survives),
        nrep=nrep, seed=seed,
    )


def fmt(row):
    return (
        f"[{row['fixture']}] n_cx={row['n_complexes']} n_pos={row['n_positions']} "
        f"n_hot={row['n_hot']}\n"
        f"    AUROC  burial={row['auroc_burial']:.3f}  full={row['auroc_full']:.3f}  "
        f"burial+KL={row['auroc_burial_kl']:.3f}  full+KL={row['auroc_full_kl']:.3f}\n"
        f"    dAUROC(burial+KL - burial) = {row['d_auroc_burkl_minus_bur']:+.4f} "
        f"[{row['d_burkl_lo']:+.4f},{row['d_burkl_hi']:+.4f}]\n"
        f"    dAUROC(full+KL   - full)   = {row['d_auroc_fullkl_minus_full']:+.4f} "
        f"[{row['d_fullkl_lo']:+.4f},{row['d_fullkl_hi']:+.4f}]  "
        f"{'==> KL SURVIVES' if row['kl_survives_full_geometry'] else '==> KL does NOT survive (CI contains 0)'}\n"
        f"    partial(KL,is_hot | burial)              = {row['partial_kl_given_burial']:+.4f} "
        f"[{row['pkb_lo']:+.4f},{row['pkb_hi']:+.4f}]\n"
        f"    partial(KL,is_hot | burial+nbr+dSASA)    = {row['partial_kl_given_full']:+.4f} "
        f"[{row['pkf_lo']:+.4f},{row['pkf_hi']:+.4f}]\n"
        f"    corr(KL,dSASA)={row['spearman_kl_dsasa']:+.3f}  "
        f"corr(burial,dSASA)={row['spearman_burial_dsasa']:+.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True,
                    help="label:joined_csv:pdb_dir triples")
    ap.add_argument("--nrep", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--modes", nargs="+", default=["absolute"],
                    choices=["absolute", "relative"],
                    help="dSASA scaling: absolute A^2 (matches the user's snippet) and/or "
                         "relative (absolute / Tien max-ASA of the residue type). "
                         "'relative' is a robustness check on the one free choice.")
    args = ap.parse_args()

    rows = []
    for spec in args.fixtures:
        label, joined_csv, pdb_dir = spec.split(":", 2)
        joined_csv = os.path.expandvars(joined_csv)
        pdb_dir = os.path.expandvars(pdb_dir)
        print(f"\n=== {label}: {joined_csv} | {pdb_dir} ===", flush=True)
        df, missing = load_fixture(joined_csv, pdb_dir)
        merged = np.isfinite(df["dsasa_abs"].values).mean() if len(df) else float("nan")
        print(f"    interface rows={len(df)}  dSASA merge frac={merged:.4f}  "
              f"missing_pdb={len(missing)}"
              + (f" {missing[:8]}" if missing else ""), flush=True)
        for mode in args.modes:
            if mode == "relative":
                maxa = np.array([fc.MAXASA_TIEN.get(str(a), np.nan)
                                 for a in df["aa"].values])
                df["dsasa"] = df["dsasa_abs"].values / maxa
            else:
                df["dsasa"] = df["dsasa_abs"].values
            print(f"  --- dsasa_mode={mode} ---", flush=True)
            row = analyze(df, label, args.nrep, args.seed)
            row["fixture"] = label
            row["dsasa_mode"] = mode
            row["dsasa_merge_frac"] = round(float(merged), 4)
            row["n_missing_pdb"] = len(missing)
            rows.append(row)
            print(fmt(row), flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} fixtures, nrep={args.nrep}, seed={args.seed})",
          flush=True)


if __name__ == "__main__":
    main()

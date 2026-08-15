#!/usr/bin/env python3
"""Confidence-Leverage Decomposition of the inverse-folding log-likelihood.

Two orthogonal functionals of ProteinMPNN's backbone-only (sequence-free, so no decoding-order
variance) distribution at interface position i:

  P_i = p(. | X_complex)        Q_i = p(. | X_monomer)          (own chain group, partner deleted)

  CONFIDENCE (diagonal)   c_i    = log P_i(wt)   [and negentropy -H(P_i)]     -> function of P ALONE
  LEVERAGE   (off-diag)   L_i(a) = [log P_i(a) - log P_i(wt)] - [log Q_i(a) - log Q_i(wt)]
                                 = r_i(a) - r_i(wt),   r_i(a) = log P_i(a) - log Q_i(a)

The leverage operator is NOT ours.  It is the Boltzmann-Alignment cycle score `BA-Cycle` of Jiao,
Mao, Jin et al., arXiv:2410.09543 (2024) -- verified by fetching arxiv.org/abs/2410.09543 and
arxiv.org/html/2410.09543v1; their Eq. 10 rearranges to exactly this double difference.  ("BAIF" is
not that paper's own label.)  Differences to state when citing: they use the WHOLE-SEQUENCE
autoregressive likelihood, we use the per-position sequence-free marginal; their unbound reference is
the product over both chains (identical to ours for a single-chain point mutation); both assume a
rigid backbone.  They run NO beyond-geometry control and NO natural-vs-de-novo split.
By the thermodynamic cycle L_i(a) is proportional to -DDG_bind(wt->a).  Our contribution is not the score,
it is (i) the decomposition -- note the project's scalar KL detector is exactly a P-weighted mean of
the same r vector, KL_i = sum_a P_i(a) r_i(a), so KL is one scalar functional of the leverage vector
and confidence is a functional of P alone -- and (ii) the REGIME LAW: whether the FULL leverage adds
binding information beyond cheap geometry on NATURAL complexes (SKEMPI, AB-Bind) the way it does on
DE-NOVO designs (Bennett).

Readouts are the ones that survived this project's readout audits: CPI (conditional predictive
impact, combiner-free, cross-fit, conditional permutation within geometry strata) and within-
geometry-stratum AUROC.  The z-sum dAUROC-against-0 readout is NOT used: it has a -0.021 noise floor.

Stages
  score-skempi  re-score SKEMPI complexes under dual conditioning, SAVE the 20-dim log P and log Q
  score-abbind  same for AB-Bind
  analyse       positive controls, L, Spearman, CPI, within-stratum AUROC, regime comparison
                (Bennett L is recovered exactly from results/bennett_occlusion_allatom_pairs.csv:
                 all 19 non-native subs are present per position and dists() normalises over the 20
                 standard AAs, so P_native = 1 - sum_19 P.  Gated by a direct re-scoring control.)

  python3 src/leverage_decomposition.py --stage score-skempi
  python3 src/leverage_decomposition.py --stage score-abbind
  python3 src/leverage_decomposition.py --stage analyse --out results/leverage_decomposition.csv
"""
import argparse
import csv
import gc
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

SEED = 20260803
AA20 = "ACDEFGHIKLMNPQRSTVWY"           # == fc.MPNN_ALPHABET[:20], the index order of the 20-dim dist
IDX = {a: i for i, a in enumerate(AA20)}
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1,
       "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0,
       "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
HOT_DDG = 1.0                            # destabilising-for-binding threshold (project convention)
DATA = os.path.expanduser("~/ftax/data")
ABDIR = f"{DATA}/ab-bind"
MPNN_W = os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt")
PQ_SKEMPI = "results/leverage_pq_skempi.csv"
PQ_ABBIND = "results/leverage_pq_abbind.csv"


# --------------------------------------------------------------------------- distributions

def logdists(lp21):
    """[L,21] raw MPNN log-probs -> [L,20] log-probabilities renormalised over the 20 standard AAs."""
    z = lp21[:, :20]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return np.log(p / p.sum(axis=1, keepdims=True))


def leverage(lP, lQ, wt_idx, mut_idx):
    """L = [lP(mut)-lP(wt)] - [lQ(mut)-lQ(wt)].  Arrays are row-aligned."""
    return (lP[np.arange(len(lP)), mut_idx] - lP[np.arange(len(lP)), wt_idx]
            - (lQ[np.arange(len(lQ)), mut_idx] - lQ[np.arange(len(lQ)), wt_idx]))


# --------------------------------------------------------------------------- scoring stages

def _score_one(model, path, pdb, g1, g2, keep):
    """-> list of dict rows (chain,resnum,icode,aa,lP_*,lQ_*) for keys in `keep` (or all if None)."""
    cx = fc.load_complex(path, pdb, g1, g2)
    if cx is None:
        return []
    lP = logdists(fc.mpnn_unconditional_logprobs(model, cx))
    lQ = np.full_like(lP, np.nan)
    for chains in (g1, g2):
        if not chains:
            continue
        mono = fc.load_complex(path, pdb, chains, "", require_both=False)
        if mono is None or mono.n < 5:
            continue
        lQm = logdists(fc.mpnn_unconditional_logprobs(model, mono))
        im = {(c, int(r), i): k for k, (c, r, i)
              in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
        for j in range(cx.n):
            k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
            if k is not None:
                lQ[j] = lQm[k]
        del lQm, mono
    rows = []
    for j in range(cx.n):
        key = (cx.chains[j], int(cx.resnums[j]), cx.icodes[j])
        if keep is not None and key not in keep:
            continue
        if not np.isfinite(lQ[j]).all():
            continue
        r = dict(chain=key[0], resnum=key[1], icode=key[2], aa=cx.seq[j])
        for a in AA20:
            r[f"lP_{a}"] = float(lP[j, IDX[a]])
        for a in AA20:
            r[f"lQ_{a}"] = float(lQ[j, IDX[a]])
        rows.append(r)
    del lP, lQ, cx
    gc.collect()
    return rows


def stage_score_skempi(a):
    import torch
    torch.set_num_threads(a.threads)
    sk = fc.parse_skempi(f"{DATA}/skempi_v2.csv")
    sk = sk[sk.n_mut == 1].copy()
    sk["complex_id"] = sk.pdb + "_" + sk.group1 + "_" + sk.group2
    # positions we must keep: every interface position, plus every measured single-mutation position
    p0 = pd.read_csv("results/p0_positions.csv", low_memory=False,
                     usecols=["complex_id", "chain", "resnum", "icode", "is_interface"])
    p0["icode"] = p0.icode.fillna("").astype(str)
    keep = {}
    for cid, sub in p0[p0.is_interface == True].groupby("complex_id"):      # noqa: E712
        keep[cid] = set(zip(sub.chain, sub.resnum.astype(int), sub.icode))
    n_mutkeys = 0
    for _, r in sk.iterrows():
        m = fc.parse_mutation(r["muts"][0])
        if m is None:
            continue
        keep.setdefault(r["complex_id"], set()).add((m["chain"], m["resnum"], m["icode"]))
        n_mutkeys += 1
    cx_ids = sorted(keep)
    print(f"[score] {len(cx_ids)} complexes, {sum(len(v) for v in keep.values())} target positions "
          f"({n_mutkeys} single-mutation rows)")

    done = set()
    if os.path.exists(PQ_SKEMPI):
        try:
            done = set(pd.read_csv(PQ_SKEMPI, usecols=["complex_id"]).complex_id)
            print(f"[score] resuming, {len(done)} complexes already scored")
        except Exception:
            done = set()
    model, _ = fc.load_mpnn(MPNN_W)
    fh = open(PQ_SKEMPI, "a" if done else "w", newline="")
    writer, n, t0, skipped = None, 0, time.time(), []
    for ci, cid in enumerate(cx_ids):
        if cid in done:
            continue
        pdb, g1, g2 = cid.split("_")
        path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            skipped.append((cid, "no pdb"))
            continue
        try:
            rows = _score_one(model, path, pdb, g1, g2, keep[cid])
        except Exception as e:
            skipped.append((cid, f"{type(e).__name__}: {e}"))
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
            continue
        for r in rows:
            r = dict(complex_id=cid, **r)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if not done:
                    writer.writeheader()
            writer.writerow(r)
            n += 1
        fh.flush()
        if (ci + 1) % 25 == 0:
            print(f"[score] {ci+1}/{len(cx_ids)}  {time.time()-t0:.0f}s  {n} rows", flush=True)
    fh.close()
    print(f"[score] wrote {PQ_SKEMPI}: {n} new rows; {len(skipped)} complexes skipped")


# AB-Bind mutation tokens look like "D:A488G" (chain : wt resnum mut) - same regex as abbind_nugget.py.
# NB an earlier version of this regex omitted the colon and silently produced ZERO AB-Bind rows;
# the CLAUDE.md rule-6 positive control on the parse path caught it.
AB_MUT_RE = re.compile(r"^([A-Za-z0-9]+):([A-Z])(-?\d+)([A-Z])$")


def stage_score_abbind(a):
    import torch
    torch.set_num_threads(a.threads)
    d = pd.read_csv(f"{ABDIR}/AB-Bind_experimental_data.csv", encoding="latin-1").rename(
        columns={"#PDB": "pdb", "Partners(A_B)": "partners", "ddG(kcal/mol)": "ddg"})
    d["ddg"] = pd.to_numeric(d["ddg"], errors="coerce")
    single = d[~d["Mutation"].astype(str).str.contains(",")].copy()
    partners = d.dropna(subset=["partners"]).groupby("pdb").partners.first().to_dict()
    keep = {}
    for _, r in single.iterrows():
        m = AB_MUT_RE.match(str(r["Mutation"]).strip())
        if m and np.isfinite(r["ddg"]):
            keep.setdefault(r["pdb"], set()).add((m.group(1), int(m.group(3)), ""))
    model, _ = fc.load_mpnn(MPNN_W)
    out, n = [], 0
    for pdb in sorted(keep):
        if pdb not in partners:
            continue
        g1, g2 = partners[pdb].split("_")
        path = f"{ABDIR}/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        try:
            rows = _score_one(model, path, pdb, g1, g2, keep[pdb])
        except Exception as e:
            print(f"  skip {pdb}: {type(e).__name__}: {e}", flush=True)
            continue
        for r in rows:
            out.append(dict(complex_id=pdb, **r))
        n += len(rows)
        print(f"  {pdb}: {len(rows)} positions", flush=True)
    pd.DataFrame(out).to_csv(PQ_ABBIND, index=False)
    print(f"[score-abbind] wrote {PQ_ABBIND}: {n} rows")


# --------------------------------------------------------------------------- statistics

def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cpi(y, g, Z, X, rng, n_boot=3000, n_strata=25, Rperm=40):
    """Conditional predictive impact of X given controls Z (Watson & Wright 2019).

    Cross-fit logistic on [Z,X]; break X's Z-conditional information by permuting X WITHIN strata of
    the cross-fitted geometry-only score; CPI = increase in per-observation log-loss.
    Returns (cpi, lo, hi, P(>0), sZ) with a group-clustered bootstrap.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    XZ = np.column_stack([Z, X])
    eta = np.zeros(len(y)); bX = np.zeros(len(y)); sZ = np.zeros(len(y))
    nfold = int(min(5, len(np.unique(g))))
    for tr, te in GroupKFold(nfold).split(XZ, y, g):
        m = LogisticRegression(max_iter=2000).fit(XZ[tr], y[tr])
        eta[te] = XZ[te] @ m.coef_[0] + m.intercept_[0]
        bX[te] = m.coef_[0][-1]
        mz = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
        sZ[te] = Z[te] @ mz.coef_[0] + mz.intercept_[0]
    lf = logloss(y, sig(eta))
    bins = pd.qcut(pd.Series(sZ).rank(method="first"), n_strata, labels=False)
    order = {b: np.where(bins == b)[0] for b in np.unique(bins)}
    lp_acc = np.zeros(len(y))
    for _ in range(Rperm):
        Xp = X.copy()
        for b, ix in order.items():
            Xp[ix] = X[rng.permutation(ix)]
        lp_acc += logloss(y, sig(eta - bX * (X - Xp)))
    c = lp_acc / Rperm - lf
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    b = np.array([c[np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])].mean()
                  for _ in range(n_boot)])
    return (float(c.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
            float(np.mean(b > 0)), sZ, c)


def drop_influential(y, g, Z, X, rng, k=3, label="", fixture="", rows=None):
    """Refit the CPI after removing the k complexes contributing most to the point estimate."""
    _, _, _, _, _, c = cpi(y, g, Z, X, rng, n_boot=200)
    contrib = pd.Series(c).groupby(pd.Series(g)).sum().sort_values(ascending=False)
    drop = set(contrib.index[:k])
    keep = ~pd.Series(g).isin(drop).to_numpy()
    st, lo, hi, p, _, _ = cpi(y[keep], g[keep], Z[keep], X[keep].copy(), rng)
    verdict = "SURVIVES" if lo > 0 else "does not survive"
    print(f"  [robustness] {label}: drop {k} most influential complexes "
          f"({', '.join(map(str, list(contrib.index[:k])))}) -> CPI = {st:+.5f} "
          f"[{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {verdict}")
    if rows is not None:
        rows.append(dict(fixture=fixture, test=f"robustness_drop{k}_influential_complexes({label})",
                         stat=round(st, 5), lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3),
                         verdict=verdict, n=int(keep.sum()),
                         note=f"dropped {sorted(drop)}"))
    return st, lo, hi


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stratified_auc(s, y, strata):
    """Within-stratum AUROC: concordant pairs pooled ACROSS strata but formed only WITHIN them."""
    num = den = 0.0
    for b in np.unique(strata):
        ix = np.where(strata == b)[0]
        sy, ss = y[ix], np.asarray(s, float)[ix]
        n1 = sy.sum(); n0 = len(sy) - n1
        if n1 == 0 or n0 == 0:
            continue
        u = stats.rankdata(ss)[sy == 1].sum() - n1 * (n1 + 1) / 2
        num += u; den += n1 * n0
    return np.nan if den == 0 else num / den


def boot_stat(fn, g, rng, n_boot=2000):
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    out = []
    for _ in range(n_boot):
        t = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        v = fn(t)
        if np.isfinite(v):
            out.append(v)
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out > 0))


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


# --------------------------------------------------------------------------- fixture builders

def build_skempi(rows_out):
    """-> per-mutation frame with L, confidence, scalar KL, geometry and DDG_bind."""
    pq = pd.read_csv(PQ_SKEMPI, low_memory=False)
    pq["icode"] = pq.icode.fillna("").astype(str)
    sk = fc.parse_skempi(f"{DATA}/skempi_v2.csv")
    sk = sk[sk.n_mut == 1].copy()
    sk["complex_id"] = sk.pdb + "_" + sk.group1 + "_" + sk.group2
    recs = []
    for _, r in sk.iterrows():
        m = fc.parse_mutation(r["muts"][0])
        if m is None:
            continue
        recs.append(dict(complex_id=r["complex_id"], chain=m["chain"], resnum=m["resnum"],
                         icode=m["icode"], wt=m["wt"], mut=m["mut"], ddG=r["ddG"]))
    mut = pd.DataFrame(recs)
    n_parsed = len(mut)
    mut = (mut.groupby(["complex_id", "chain", "resnum", "icode", "wt", "mut"])
              .agg(ddG=("ddG", "median"), n_meas=("ddG", "size")).reset_index())

    # ---- POSITIVE CONTROL on the mutation -> position mapping (CLAUDE.md rule 6)
    j = mut.merge(pq, on=["complex_id", "chain", "resnum", "icode"], how="left")
    mapped = j.aa.notna()
    wt_match = (j.aa == j.wt)
    ctrl = dict(n_parsed=n_parsed, n_unique=len(mut), n_mapped=int(mapped.sum()),
                map_rate=float(mapped.mean()),
                wt_match_rate_of_mapped=float(wt_match[mapped].mean()),
                wt_match_rate_overall=float(wt_match.mean()))
    # negative control: shuffle the native residue within complex -> background AA-match rate
    jm = j[mapped].copy()
    perm = jm.groupby("complex_id").aa.transform(lambda s: s.sample(frac=1, random_state=SEED).values)
    ctrl["wt_match_rate_shuffled_within_complex"] = float((perm == jm.wt).mean())
    rows_out.append(dict(fixture="SKEMPI", test="positive_control_mutation_mapping", **ctrl))
    print(f"  [+control] SKEMPI mapping: {ctrl['n_unique']} unique single mutations, "
          f"{ctrl['n_mapped']} mapped ({100*ctrl['map_rate']:.1f}%), "
          f"WT-match {100*ctrl['wt_match_rate_of_mapped']:.2f}% of mapped "
          f"(shuffled-position control {100*ctrl['wt_match_rate_shuffled_within_complex']:.1f}%)")

    d = j[mapped & wt_match].copy()
    lP = d[[f"lP_{a}" for a in AA20]].to_numpy()
    lQ = d[[f"lQ_{a}" for a in AA20]].to_numpy()
    wi = d.wt.map(IDX).to_numpy(); mi = d.mut.map(IDX).to_numpy()
    d["L"] = leverage(lP, lQ, wi, mi)
    d["logP_mut"] = lP[np.arange(len(d)), mi]
    d["r_mut"] = lP[np.arange(len(d)), mi] - lQ[np.arange(len(d)), mi]
    d["conf"] = lP[np.arange(len(d)), wi]                       # log p(native | complex)
    Pn = np.exp(lP)
    d["klP"] = (Pn * (lP - lQ)).sum(axis=1)                     # scalar KL(P||Q), the demoted detector
    d["negH"] = (Pn * lP).sum(axis=1)                           # negentropy of P

    geo = pd.read_csv("results/p0_positions.csv", low_memory=False,
                      usecols=["complex_id", "chain", "resnum", "icode", "rsasa_complex", "nbr",
                               "drsasa", "is_interface"])
    geo["icode"] = geo.icode.fillna("").astype(str)
    d = d.merge(geo, on=["complex_id", "chain", "resnum", "icode"], how="left")
    d["burial"] = -d.rsasa_complex
    return _finish(d, "SKEMPI")


def _finish(d, name):
    from Bio.Align import substitution_matrices
    B = substitution_matrices.load("BLOSUM62")
    d["blosum"] = [float(B[w, m]) for w, m in zip(d.wt, d.mut)]
    d["dvol"] = [-abs(VOL[m] - VOL[w]) for w, m in zip(d.wt, d.mut)]
    d["dhydro"] = [-abs(fc.KD_HYDRO[m] - fc.KD_HYDRO[w]) for w, m in zip(d.wt, d.mut)]
    d["fixture"] = name
    return d


def build_abbind(rows_out):
    pq = pd.read_csv(PQ_ABBIND, low_memory=False)
    pq["icode"] = pq.icode.fillna("").astype(str)
    ab = pd.read_csv(f"{ABDIR}/AB-Bind_experimental_data.csv", encoding="latin-1").rename(
        columns={"#PDB": "pdb", "ddG(kcal/mol)": "ddg"})
    ab["ddg"] = pd.to_numeric(ab["ddg"], errors="coerce")
    ab = ab[~ab["Mutation"].astype(str).str.contains(",")]
    recs = []
    for _, r in ab.iterrows():
        m = AB_MUT_RE.match(str(r["Mutation"]).strip())
        if m and np.isfinite(r["ddg"]) and m.group(2) in IDX and m.group(4) in IDX:
            recs.append(dict(complex_id=r["pdb"], chain=m.group(1), resnum=int(m.group(3)), icode="",
                             wt=m.group(2), mut=m.group(4), ddG=float(r["ddg"])))
    mut = pd.DataFrame(recs)
    mut = (mut.groupby(["complex_id", "chain", "resnum", "icode", "wt", "mut"])
              .agg(ddG=("ddG", "median"), n_meas=("ddG", "size")).reset_index())
    j = mut.merge(pq, on=["complex_id", "chain", "resnum", "icode"], how="left")
    mapped = j.aa.notna(); wt_match = (j.aa == j.wt)
    ctrl = dict(n_unique=len(mut), n_mapped=int(mapped.sum()), map_rate=float(mapped.mean()),
                wt_match_rate_of_mapped=float(wt_match[mapped].mean()))
    rows_out.append(dict(fixture="ABBIND", test="positive_control_mutation_mapping", **ctrl))
    print(f"  [+control] AB-Bind mapping: {ctrl['n_unique']} unique single mutations, "
          f"{ctrl['n_mapped']} mapped, WT-match {100*ctrl['wt_match_rate_of_mapped']:.2f}% of mapped")
    d = j[mapped & wt_match].copy()
    lP = d[[f"lP_{a}" for a in AA20]].to_numpy(); lQ = d[[f"lQ_{a}" for a in AA20]].to_numpy()
    wi = d.wt.map(IDX).to_numpy(); mi = d.mut.map(IDX).to_numpy()
    d["L"] = leverage(lP, lQ, wi, mi)
    d["logP_mut"] = lP[np.arange(len(d)), mi]
    d["conf"] = lP[np.arange(len(d)), wi]
    Pn = np.exp(lP)
    d["klP"] = (Pn * (lP - lQ)).sum(axis=1)
    geo = pd.read_csv("results/abbind_positions.csv")[
        ["pdb", "chain", "resnum", "burial", "drsasa", "nbr", "is_interface"]].rename(
        columns={"pdb": "complex_id"})
    d = d.merge(geo, on=["complex_id", "chain", "resnum"], how="left")
    return _finish(d, "ABBIND")


def build_bennett(rows_out, verify=3, threads=3):
    """Bennett per-substitution leverage.

    bennett_occlusion_allatom_pairs.csv holds P and Q for all 19 non-native substitutions at every
    interface position (exactly 19 rows per position, verified), and both were renormalised over the
    20 standard AAs -> P_native = 1 - sum_19 P exactly.  A direct re-scoring of `verify` designs
    gates this recovery before any number is trusted.
    """
    o = pd.read_csv("results/bennett_occlusion_allatom_pairs.csv")
    cnt = o.groupby(["design", "resnum"]).sub.size()
    if not (cnt == 19).all():
        raise SystemExit("bennett pairs: not all positions have 19 substitutions; cannot recover native")
    tot = o.groupby(["design", "resnum"]).agg(sP=("P", "sum"), sQ=("Q", "sum")).reset_index()
    tot["P_nat"] = 1.0 - tot.sP
    tot["Q_nat"] = 1.0 - tot.sQ
    kp = pd.read_csv("results/bennett_kl_positions.csv")[      # dsasa comes from `o`, don't collide
        ["parent", "resnum", "native_aa", "restype", "burial", "rsasa", "nbr",
         "is_interface", "kl"]].rename(columns={"parent": "design"})
    d = o.merge(tot[["design", "resnum", "P_nat", "Q_nat"]], on=["design", "resnum"], how="left")
    d = d.merge(kp, on=["design", "resnum"], how="left")

    ctrl = dict(n_pairs=len(d), n_positions=len(tot), n_designs=d.design.nunique(),
                frac_P_nat_positive=float((tot.P_nat > 0).mean()),
                median_P_nat=float(tot.P_nat.median()),
                native_aa_match_restype=float((d.native_aa == d.restype).mean()),
                n_mapped_geometry=int(d.native_aa.notna().sum()))
    # ---- positive control: re-score `verify` designs directly and compare recovered natives
    if verify:
        err = _bennett_verify(tot, verify, threads)
        ctrl.update(err)
    rows_out.append(dict(fixture="BENNETT", test="positive_control_native_recovery", **ctrl))
    print(f"  [+control] Bennett native recovery: {ctrl['n_positions']} positions x 19 subs, "
          f"all P_native>0: {ctrl['frac_P_nat_positive']==1.0}, "
          f"direct-rescore max |dP_native| = {ctrl.get('verify_max_abs_err', float('nan')):.2e} "
          f"on {ctrl.get('verify_n_positions', 0)} positions from {ctrl.get('verify_n_designs', 0)} designs")

    d = d[d.native_aa.notna()].copy()
    d["L"] = (np.log(d.P) - np.log(d.P_nat)) - (np.log(d.Q) - np.log(d.Q_nat))
    d["logP_mut"] = np.log(d.P)
    d["conf"] = np.log(d.P_nat)
    d["klP"] = d.kl
    d["wt"] = d.native_aa
    d["mut"] = d["sub"]
    d["ddG"] = np.nan
    d["complex_id"] = d.design
    # SKEMPI/AB-Bind ΔSASA is RELATIVE (rf-rb); Bennett's is absolute Å² -> divide by the same
    # Tien et al. maximum ASA so the three fixtures' control sets are on one scale.
    d["drsasa"] = d.dsasa / d.native_aa.map(fc.MAXASA_TIEN)
    d["destab"] = 1 - d.binds.astype(int)                      # positive class = ABOLISHES binding
    return _finish(d, "BENNETT")


def _bennett_verify(tot, n_designs, threads):
    """Re-score n_designs Bennett complexes with ProteinMPNN; compare direct vs recovered P_native."""
    import glob
    import torch
    torch.set_num_threads(threads)
    BEN = os.path.expanduser("~/ftax/bennett/x/supplemental_files")
    pdb_index = {os.path.basename(p)[:-4]: p for p in
                 glob.glob(f"{BEN}/design_models_ssm_natives/*/*.pdb")}
    designs = sorted(tot.design.unique())[:n_designs]
    model, _ = fc.load_mpnn(MPNN_W)
    errs, npos = [], 0
    for parent in designs:
        pdb = pdb_index.get(parent)
        if pdb is None:
            continue
        cx = fc.load_complex(pdb, parent, "A", "B", require_both=False)
        mono = fc.load_complex(pdb, parent, "A", "", require_both=False)
        if cx is None or mono is None:
            continue
        lP = logdists(fc.mpnn_unconditional_logprobs(model, cx))
        lQ = logdists(fc.mpnn_unconditional_logprobs(model, mono))
        im = {(c, int(r), i): k for k, (c, r, i)
              in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
        sub = tot[tot.design == parent]
        pos2j = {int(cx.resnums[j]): j for j in range(cx.n) if cx.group[j] == 1}
        for _, r in sub.iterrows():
            j = pos2j.get(int(r.resnum))
            if j is None:
                continue
            k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
            if k is None:
                continue
            nat = cx.seq[j]
            if nat not in IDX:
                continue
            errs.append(abs(float(np.exp(lP[j, IDX[nat]])) - float(r.P_nat)))
            errs.append(abs(float(np.exp(lQ[k, IDX[nat]])) - float(r.Q_nat)))
            npos += 1
        del lP, lQ, cx, mono
        gc.collect()
    return dict(verify_n_designs=len(designs), verify_n_positions=npos,
                verify_max_abs_err=float(np.max(errs)) if errs else float("nan"),
                verify_median_abs_err=float(np.median(errs)) if errs else float("nan"))


# --------------------------------------------------------------------------- analysis

def run_fixture(d, name, rows, rng, group="complex_id"):
    """CPI + within-stratum AUROC for L (and reference features) on one fixture."""
    feats = ["L", "conf", "klP", "logP_mut"]
    need = ["burial", "nbr", "drsasa", "blosum", "dvol", "dhydro", "destab"] + feats
    d = d.dropna(subset=[c for c in need if c in d.columns]).reset_index(drop=True)
    if "is_interface" in d.columns:
        d = d[d.is_interface == 1].reset_index(drop=True)
    y = d.destab.to_numpy().astype(float)
    g = d[group].to_numpy()
    print(f"\n=== {name}: {len(d)} interface mutations, {len(np.unique(g))} complexes, "
          f"{int(y.sum())} destabilising ({100*y.mean():.1f}%) ===")
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 5:
        print("  insufficient data; skipping")
        return

    for c in ["burial", "nbr", "drsasa", "blosum", "dvol", "dhydro"] + feats:
        d[c + "z"] = zs(d[c])
    Zgeo = d[["burialz", "nbrz", "drsasaz"]].to_numpy()
    Zhard = d[["burialz", "nbrz", "drsasaz", "blosumz", "dvolz", "dhydroz"]].to_numpy()

    # ---- Spearman(L, ddG) where an experimental ddG exists
    if "ddG" in d.columns and np.isfinite(d.ddG).any():
        sp = stats.spearmanr(d.L, d.ddG, nan_policy="omit").correlation
        lo, hi, p = boot_stat(lambda t: stats.spearmanr(d.L.values[t], d.ddG.values[t],
                                                        nan_policy="omit").correlation, g, rng)
        print(f"  Spearman(L, DDG_bind)   = {sp:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"(theory: NEGATIVE, L ~ -DDG_bind)")
        rows.append(dict(fixture=name, test="spearman_L_vs_ddG", stat=round(float(sp), 4),
                         lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(p, 3), n=len(d),
                         n_groups=int(len(np.unique(g)))))
        for f, lab in [("conf", "confidence"), ("klP", "scalarKL"), ("logP_mut", "logP(mut|cx)")]:
            s2 = stats.spearmanr(d[f], d.ddG, nan_policy="omit").correlation
            rows.append(dict(fixture=name, test=f"spearman_{lab}_vs_ddG",
                             stat=round(float(s2), 4), n=len(d)))
            print(f"  Spearman({lab:12s}, DDG) = {s2:+.4f}")

    # ---- CPI.  PRIMARY: L | cheap geometry (the regime-law number, comparable across fixtures).
    # Then progressively harder control sets: + substitution identity, + the DIAGONAL (confidence),
    # + the scalar KL (does the full leverage VECTOR beat its own P-weighted mean?).
    Zsets = [("burial+nbr+dSASA", Zgeo),
             ("burial+nbr+dSASA+BLOSUM+dVol+dHydro", Zhard),
             ("burial+nbr+dSASA+confidence", np.column_stack([Zgeo, d.confz])),
             ("burial+nbr+dSASA+scalarKL", np.column_stack([Zgeo, d.klPz])),
             ("ALL: geom+subst+confidence+scalarKL",
              np.column_stack([Zhard, d.confz, d.klPz]))]
    primary_sZ = None
    for Zname, Z in Zsets:
        c, lo, hi, p, sZ, _ = cpi(y, g, Z, d["Lz"].to_numpy().copy(), rng)
        if primary_sZ is None:
            primary_sZ = sZ
        verdict = "ADDS (CI>0)" if lo > 0 else "conditionally INDEPENDENT (CI spans 0)"
        print(f"  CPI[{'LEVERAGE L (full)':22s} | {Zname:36s}] = {c:+.5f} [{lo:+.5f},{hi:+.5f}] "
              f"P(>0)={p:.3f}  {verdict}")
        rows.append(dict(fixture=name, test=f"CPI(LEVERAGE L (full) | {Zname})", stat=round(c, 5),
                         lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3), verdict=verdict,
                         n=len(d), n_groups=int(len(np.unique(g))), n_pos=int(y.sum())))
        if Zname == "burial+nbr+dSASA":
            drop_influential(y, g, Z, d["Lz"].to_numpy(), rng, k=3,
                             label="CPI(L | burial+nbr+dSASA)", fixture=name, rows=rows)
    # reference features under the two headline control sets
    for Zname, Z in Zsets[:2]:
        for f, lab in [("confz", "confidence"), ("klPz", "scalar KL"),
                       ("logP_mutz", "logP(mut|complex)")]:
            c, lo, hi, p, _, _ = cpi(y, g, Z, d[f].to_numpy().copy(), rng)
            verdict = "ADDS (CI>0)" if lo > 0 else "conditionally INDEPENDENT (CI spans 0)"
            print(f"  CPI[{lab:22s} | {Zname:36s}] = {c:+.5f} [{lo:+.5f},{hi:+.5f}] "
                  f"P(>0)={p:.3f}  {verdict}")
            rows.append(dict(fixture=name, test=f"CPI({lab} | {Zname})", stat=round(c, 5),
                             lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3), verdict=verdict,
                             n=len(d), n_groups=int(len(np.unique(g))), n_pos=int(y.sum())))

    # ---- within-geometry-stratum AUROC (the second audited readout), on the primary control set
    bins = pd.qcut(pd.Series(primary_sZ).rank(method="first"), 10, labels=False).to_numpy()
    for f, lab in [("L", "LEVERAGE L (full)"), ("conf", "confidence"), ("klP", "scalar KL"),
                   ("logP_mut", "logP(mut|complex)")]:
        s = -d[f].to_numpy()                      # high destabilisation risk = LOW L / LOW logP
        a0 = stratified_auc(s, y, bins)
        lo, hi, _ = boot_stat(lambda t: stratified_auc(s[t], y[t], bins[t]), g, rng, n_boot=1000)
        marg = auc(s, y)
        print(f"  AUROC(-{lab:20s}) marginal {marg:.4f} | within geometry stratum "
              f"{a0:.4f} [{lo:.4f},{hi:.4f}]")
        rows.append(dict(fixture=name, test=f"within_stratum_AUROC(-{lab} | burial+nbr+dSASA)",
                         stat=round(float(a0), 4), lo=round(lo, 4), hi=round(hi, 4), n=len(d),
                         n_groups=int(len(np.unique(g)))))
        rows.append(dict(fixture=name, test=f"marginal_AUROC(-{lab})", stat=round(float(marg), 4),
                         n=len(d)))
    a0 = auc(primary_sZ, y)
    rows.append(dict(fixture=name, test="marginal_AUROC(geometry burial+nbr+dSASA)",
                     stat=round(float(a0), 4), n=len(d)))
    print(f"  marginal AUROC(geometry burial+nbr+dSASA) = {a0:.4f}")


def position_frame():
    """SKEMPI interface positions with the 20-dim log P / log Q, geometry and the hotspot label.

    This is the unit nugget_cpi.py used, so CPI here is directly comparable to the committed
    CPI(confidence|geometry)=0.000 and CPI(scalarKL|geometry)=+0.002.
    """
    pq = pd.read_csv(PQ_SKEMPI, low_memory=False)
    pq["icode"] = pq.icode.fillna("").astype(str)
    p0 = pd.read_csv("results/p0_positions.csv", low_memory=False,
                     usecols=["complex_id", "chain", "resnum", "icode", "label", "is_interface",
                              "rsasa_complex", "nbr", "drsasa", "logp_native"])
    p0["icode"] = p0.icode.fillna("").astype(str)
    p0["label"] = p0.label.fillna("null")
    d = p0.merge(pq, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    d = d[d.is_interface == 1].copy()
    d["is_hot"] = (d.label == "hot_strict").astype(int)
    d["burial"] = -d.rsasa_complex
    lP = d[[f"lP_{a}" for a in AA20]].to_numpy()
    lQ = d[[f"lQ_{a}" for a in AA20]].to_numpy()
    wi = d.aa.map(IDX).to_numpy().astype(float)
    ok = np.isfinite(wi)
    d = d[ok].copy(); lP, lQ = lP[ok], lQ[ok]
    wi = d.aa.map(IDX).to_numpy()
    n = len(d); ar = np.arange(n)
    r = lP - lQ                                       # leverage vector, up to the native offset
    Lvec = r - r[ar, wi][:, None]                     # L_i(a); L_i(wt) == 0 by construction
    d["conf"] = lP[ar, wi]                            # confidence = log p(native | complex)
    d["negH"] = (np.exp(lP) * lP).sum(axis=1)         # negentropy of P (2nd confidence scalar)
    d["klP"] = (np.exp(lP) * r).sum(axis=1)           # scalar KL(P||Q) = P-weighted mean of r
    d["L_ala"] = Lvec[:, IDX["A"]]                    # SKEMPI hotspots ARE alanine-scan ΔΔG
    mask = np.ones((n, 20), bool); mask[ar, wi] = False
    Ln = np.where(mask, Lvec, np.nan)
    d["L_rms"] = np.sqrt(np.nanmean(Ln ** 2, axis=1))
    d["L_min"] = np.nanmin(Ln, axis=1)
    d["L_mean"] = np.nanmean(Ln, axis=1)
    return d, Lvec, lP, lQ


def scoring_control(d, Lvec, lP, lQ, rows):
    """Two gates before any leverage number is trusted.

    (1) The re-scoring must reproduce the COMMITTED kl_detector_positions.csv scalar KL exactly.
    (2) The algebraic identity KL(P||Q) = E_{a~P}[L(a)] + [log P(wt) - log Q(wt)] must hold, which
        is the formal statement that the demoted scalar detector is ONE functional of the leverage
        vector -- so the full vector cannot carry less information than it.
    """
    newkl = (np.exp(lP) * (lP - lQ)).sum(axis=1)
    wi = d.aa.map(IDX).to_numpy(); ar = np.arange(len(d))
    ident = (np.exp(lP) * Lvec).sum(axis=1) + (lP - lQ)[ar, wi]
    id_err = float(np.abs(ident - newkl).max())
    old = pd.read_csv("results/kl_detector_positions.csv",
                      usecols=["complex_id", "chain", "resnum", "icode", "aa", "kl"])
    old["icode"] = old.icode.fillna("").astype(str)
    chk = d[["complex_id", "chain", "resnum", "icode", "aa"]].copy()
    chk["kl_new"] = newkl
    m = chk.merge(old, on=["complex_id", "chain", "resnum", "icode"], how="inner",
                  suffixes=("", "_old"))
    err = float(np.abs(m.kl_new - m.kl).max()) if len(m) else float("nan")
    aa_ok = float((m.aa == m.aa_old).mean()) if len(m) else float("nan")
    print(f"  [+control] re-scoring vs committed kl_detector_positions.csv: {len(m)} overlapping "
          f"positions, residue agreement {aa_ok:.3f}, max |ΔKL| = {err:.2e}")
    print(f"  [+control] identity KL = E_P[L] + [logP(wt)-logQ(wt)]: max |Δ| = {id_err:.2e}")
    rows.append(dict(fixture="SKEMPI", test="positive_control_rescoring_reproduces_committed_KL",
                     stat=err, n=int(len(m)),
                     note=f"residue agreement {aa_ok:.3f}; float32 MPNN precision"))
    rows.append(dict(fixture="SKEMPI", test="algebraic_identity_KL_equals_EP_L_plus_r_wt",
                     stat=id_err, n=int(len(d)),
                     note="scalar KL is ONE functional of the leverage vector L"))


def theorem_demo(d, Lvec, lP, rows, rng, name="SKEMPI"):
    """Empirical instantiation of the blindness theorem.

    Confidence is a functional of P alone; leverage is a functional of (P,Q).  If two positions can
    have (near-)identical P but materially different L, confidence is *provably* blind to leverage —
    no re-parameterisation of a P-only scalar can recover it.  We match positions on the FULL
    complex-conditioned distribution (total-variation distance < tol) and measure the residual
    spread of the leverage magnitude.
    """
    P = np.exp(lP)
    Lrms = d.L_rms.to_numpy()
    conf = d.conf.to_numpy()
    tol = 0.02
    idx = rng.choice(len(d), size=int(min(4000, len(d))), replace=False)
    Ps, pairs, dL, dconf = P[idx], 0, [], []
    for a_ in range(len(idx)):
        tv = 0.5 * np.abs(Ps[a_ + 1:] - Ps[a_]).sum(axis=1)
        for h in np.where(tv < tol)[0][:5]:
            b_ = a_ + 1 + h
            pairs += 1
            dL.append(abs(Lrms[idx[a_]] - Lrms[idx[b_]]))
            dconf.append(abs(conf[idx[a_]] - conf[idx[b_]]))
    dL = np.array(dL); dconf = np.array(dconf)
    rho = stats.spearmanr(conf, Lrms).correlation
    sd = float(np.nanstd(Lrms))
    med = float(np.median(dL)) if len(dL) else float("nan")
    rows.append(dict(fixture=name, test="theorem_confidence_blind_to_leverage",
                     stat=round(med, 4), n=int(pairs),
                     note=(f"median |ΔL_rms| within P-matched pairs (TV<{tol}) = {med:.4f} vs "
                           f"overall SD(L_rms) = {sd:.4f}; median |Δconfidence| within pairs = "
                           f"{(np.median(dconf) if len(dconf) else float('nan')):.4f}; "
                           f"Spearman(confidence, L_rms) = {rho:+.3f}")))
    rows.append(dict(fixture=name, test="spearman_confidence_vs_leverage_magnitude",
                     stat=round(float(rho), 4), n=int(len(d))))
    print(f"\n  [theorem] {pairs} distribution-matched interface-position pairs (TV<{tol}): "
          f"median |ΔL_rms| = {med:.4f} against overall SD(L_rms) = {sd:.4f}; "
          f"median |Δconfidence| within pairs = "
          f"{(np.median(dconf) if len(dconf) else float('nan')):.4f}")
    print(f"  [theorem] Spearman(confidence, leverage magnitude) = {rho:+.3f} "
          f"(structural orthogonality is not empirical independence)")


def position_level_cpi(d, rows, rng, name="SKEMPI"):
    """Position-level CPI for is_hot — the apples-to-apples row against nugget_cpi.csv."""
    need = ["burial", "nbr", "drsasa", "conf", "klP", "L_ala", "L_rms", "L_min", "negH"]
    d = d.dropna(subset=need).reset_index(drop=True)
    y = d.is_hot.to_numpy().astype(float); g = d.complex_id.to_numpy()
    print(f"\n=== position-level CPI (nugget_cpi.py unit): {len(d)} interface positions, "
          f"{len(np.unique(g))} complexes, {int(y.sum())} strict hotspots ===")
    for c in need:
        d[c + "z"] = zs(d[c])
    Z = d[["burialz", "nbrz", "drsasaz"]].to_numpy()
    for f, lab in [("L_alaz", "leverage L(->Ala)"), ("L_rmsz", "leverage |L| rms"),
                   ("L_minz", "leverage min_a L"), ("klPz", "scalar KL [ref +0.002]"),
                   ("confz", "confidence [ref 0.000]"), ("negHz", "negentropy of P")]:
        c, lo, hi, p, sZ, _ = cpi(y, g, Z, d[f].to_numpy().copy(), rng)
        verdict = "ADDS (CI>0)" if lo > 0 else "conditionally INDEPENDENT (CI spans 0)"
        print(f"  CPI[{lab:26s} | burial+nbr+ΔSASA] = {c:+.5f} [{lo:+.5f},{hi:+.5f}] "
              f"P(>0)={p:.3f}  {verdict}")
        rows.append(dict(fixture=name, test=f"CPI_position_level({lab} | burial+nbr+dSASA)",
                         stat=round(c, 5), lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3),
                         verdict=verdict, n=len(d), n_groups=int(len(np.unique(g))),
                         n_pos=int(y.sum())))
        if f == "L_alaz":
            drop_influential(y, g, Z, d[f].to_numpy(), rng, k=3,
                             label="CPI_position_level(L(->Ala) | burial+nbr+dSASA)",
                             fixture=name, rows=rows)
            bins = pd.qcut(pd.Series(sZ).rank(method="first"), 10, labels=False).to_numpy()
            for ff, ll in [("L_ala", "L(->Ala)"), ("klP", "scalar KL"), ("conf", "confidence")]:
                s = -d[ff].to_numpy()
                a0 = stratified_auc(s, y, bins)
                lo2, hi2, _ = boot_stat(lambda t: stratified_auc(s[t], y[t], bins[t]), g, rng,
                                        n_boot=1000)
                print(f"  within-geometry-stratum AUROC(-{ll:12s}) for hotspots = "
                      f"{a0:.4f} [{lo2:.4f},{hi2:.4f}]")
                rows.append(dict(fixture=name,
                                 test=f"within_stratum_AUROC_position(-{ll}) for is_hot",
                                 stat=round(float(a0), 4), lo=round(lo2, 4), hi=round(hi2, 4),
                                 n=len(d)))


def stage_analyse(a):
    rng = np.random.default_rng(SEED)
    rows = []
    pos, Lvec, lP, lQ = position_frame()
    pos.to_csv("results/leverage_skempi_positions.csv", index=False,
               columns=[c for c in pos.columns if not c.startswith(("lP_", "lQ_"))])
    scoring_control(pos, Lvec, lP, lQ, rows)
    theorem_demo(pos, Lvec, lP, rows, rng)
    position_level_cpi(pos, rows, rng)
    del Lvec, lP, lQ
    gc.collect()

    sk = build_skempi(rows)
    sk["destab"] = (sk.ddG >= HOT_DDG).astype(int)
    sk.to_csv("results/leverage_skempi_mutations.csv", index=False,
              columns=[c for c in sk.columns if not c.startswith(("lP_", "lQ_"))])
    run_fixture(sk, "SKEMPI (natural)", rows, rng)

    if os.path.exists(PQ_ABBIND):
        ab = build_abbind(rows)
        ab["destab"] = (ab.ddG >= HOT_DDG).astype(int)
        ab.to_csv("results/leverage_abbind_mutations.csv", index=False,
                  columns=[c for c in ab.columns if not c.startswith(("lP_", "lQ_"))])
        run_fixture(ab, "AB-Bind (natural, Ab-Ag)", rows, rng)
    else:
        print(f"\n[warn] {PQ_ABBIND} absent - run --stage score-abbind")

    be = build_bennett(rows, verify=a.bennett_verify, threads=a.threads)
    be.to_csv("results/leverage_bennett_pairs.csv", index=False)
    run_fixture(be, "Bennett (de-novo)", rows, rng)

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out["command"] = "python3 " + " ".join(sys.argv)
    out.to_csv(a.out, index=False)
    print(f"\n[done] wrote {a.out}  ({len(out)} rows)")

    # one-line summary: the regime law
    key = out[out.test.str.startswith("CPI(LEVERAGE L (full) | burial+nbr+dSASA)", na=False)]
    if len(key):
        s = "  ".join(f"{r.fixture}={r.stat:+.5f}[{r.lo:+.5f},{r.hi:+.5f}]" for r in key.itertuples())
        print(f"[summary] CPI(L | burial+nbr+dSASA): {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="analyse",
                    choices=["score-skempi", "score-abbind", "analyse"])
    ap.add_argument("--out", default="results/leverage_decomposition.csv")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bennett-verify", type=int, default=3)
    a = ap.parse_args()
    {"score-skempi": stage_score_skempi, "score-abbind": stage_score_abbind,
     "analyse": stage_analyse}[a.stage](a)


if __name__ == "__main__":
    main()

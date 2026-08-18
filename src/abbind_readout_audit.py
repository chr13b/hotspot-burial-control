#!/usr/bin/env python3
"""AUDIT of results/abbind_bigidea1.csv — the "de-novo-specific" demotion.

The demotion rests on dAUROC(logP over burial+ΔSASA+BLOSUM+volume) = +0.008 [-0.014,+0.026]
(src/abbind_bigidea1.py:128-142). Three problems, the same three the catalytic audit found:

1. WRONG QUANTITY. `logP` = log p(mut | complex backbone) is confounded with the MUTANT AMINO
   ACID'S IDENTITY by construction (p(Trp|.) is low wherever Trp is the mutant), exactly as
   log p(native) was in src/catalytic_audit.py. Worse here: the baseline it is asked to beat
   ALREADY CONTAINS two identity encodings (BLOSUM62 and volume change), so the identity part of
   logP is subtracted twice. Neither the position ENTROPY (determinacy) nor an identity-normalised
   form (the mutant's RANK within the 20-way distribution) was ever computed.

2. COMPRESSIVE READOUT. dAUROC over a 0.660 baseline with n=420 / 27 complexes. Calibrated here
   with a synthetic positive control and with the estimator's own noise floor.

3. The nested test never used `logodds` (P/Q), though the script computes it (line 89) — only
   Z("logP") enters the baseline (line 129).

Correct readouts: within-mutant-identity stratified AUROC, identity-normalised quantities,
noise-floor-calibrated nested dAUROC, CPI. Caches the per-mutation table (the original saved only
summary rows, so nothing could be re-analysed without re-running ProteinMPNN).

  python3 src/abbind_readout_audit.py --out results/abbind_readout_audit.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ftax_common as fc
import abbind_nugget as AB

SEED = 20260803
NBOOT = 2000
MPNN20 = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(MPNN20)}
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1,
       "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0,
       "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
CACHE = "results/abbind_readout_audit_positions.csv"


def dists(lp):
    z_ = lp[:, :20]; z_ = z_ - z_.max(1, keepdims=True); p = np.exp(z_)
    return np.log(p / p.sum(1, keepdims=True))


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sauc(val, y, k):
    """Pooled WITHIN-STRATUM AUROC — verbatim from src/catalytic_audit.py:101."""
    val = np.asarray(val, float); ok = np.isfinite(val)
    val, y, k = val[ok], np.asarray(y, float)[ok], np.asarray(k)[ok]
    order = np.lexsort((val, k)); ks, vs, ys = k[order], val[order], y[order]; n = len(ks)
    if n == 0:
        return np.nan
    gs = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    gid = np.zeros(n, np.int64); gid[gs[1:]] = 1; gid = np.cumsum(gid)
    r = np.arange(n) - gs[gid] + 1.0
    nb = np.r_[True, (ks[1:] != ks[:-1]) | (vs[1:] != vs[:-1])]
    bs = np.flatnonzero(nb); bz = np.diff(np.r_[bs, n])
    r = np.repeat((r[bs] + (r[bs] + bz - 1)) / 2.0, bz)
    ng = int(gid[-1]) + 1
    n1 = np.bincount(gid, weights=ys, minlength=ng)
    n0 = np.bincount(gid, minlength=ng).astype(float) - n1
    U = np.bincount(gid, weights=r * ys, minlength=ng) - n1 * (n1 + 1) / 2
    ok2 = (n1 > 0) & (n0 > 0); den = (n1[ok2] * n0[ok2]).sum()
    return U[ok2].sum() / den if den else np.nan


def z(v):
    v = np.asarray(v, float); s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 1e-12 else v * 0


def build_cache(out_csv):
    from Bio.Align import substitution_matrices
    BL = substitution_matrices.load("BLOSUM62")
    d = pd.read_csv(f"{AB.ABDIR}/AB-Bind_experimental_data.csv", encoding="latin-1").rename(
        columns={"#PDB": "pdb", "Partners(A_B)": "partners", "ddG(kcal/mol)": "ddg"})
    d["ddg"] = pd.to_numeric(d["ddg"], errors="coerce")
    single = d[~d["Mutation"].astype(str).str.contains(",")].copy()
    muts = []
    for _, r in single.iterrows():
        m = AB.MUT_RE.match(str(r["Mutation"]).strip())
        if m and np.isfinite(r["ddg"]):
            muts.append(dict(pdb=r["pdb"], chain=m.group(1), wt=m.group(2), resnum=int(m.group(3)),
                             mut=m.group(4), ddg=float(r["ddg"])))
    muts = pd.DataFrame(muts)
    partners = d.dropna(subset=["partners"]).groupby("pdb").partners.first().to_dict()
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    recs = []
    for pdb, part in sorted(partners.items()):
        g1, g2 = part.split("_"); path = f"{AB.ABDIR}/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
            if cx is None or cx.n > 2500:
                continue
            P = dists(fc.mpnn_unconditional_logprobs(model, cx))
            Qmap = {}
            for grp in (g1, g2):
                mono = fc.load_complex(path, pdb, grp, "", require_both=False)
                if mono is None:
                    continue
                Qm = dists(fc.mpnn_unconditional_logprobs(model, mono))
                for k in range(mono.n):
                    Qmap[(mono.chains[k], int(mono.resnums[k]))] = Qm[k]
            pos = {(cx.chains[j], int(cx.resnums[j])): j for j in range(cx.n)}
            pp = np.exp(P); HP = -(pp * P).sum(1)                       # position entropy (complex)
            for _, r in muts[muts.pdb == pdb].iterrows():
                j = pos.get((r.chain, r.resnum))
                if j is None or r.mut not in IDX or r.wt not in IDX:
                    continue
                q = Qmap.get((r.chain, r.resnum))
                row = dict(pdb=pdb, chain=r.chain, resnum=r.resnum, wt=r.wt, mut=r.mut, ddg=r.ddg,
                           logP=float(P[j, IDX[r.mut]]), logP_wt=float(P[j, IDX[r.wt]]),
                           H_complex=float(HP[j]),
                           # identity-normalised forms of the SAME distribution:
                           rank_mut=float(stats.rankdata(P[j])[IDX[r.mut]]),      # 1..20, high = model likes it
                           logP_minus_wt=float(P[j, IDX[r.mut]] - P[j, IDX[r.wt]]),
                           blosum=float(BL[r.wt, r.mut]), dvol=-abs(VOL[r.mut] - VOL[r.wt]))
                if q is not None:
                    qq = np.exp(q)
                    row.update(logQ=float(q[IDX[r.mut]]),
                               logodds=float(P[j, IDX[r.mut]] - q[IDX[r.mut]]),
                               H_mono=float(-(qq * q).sum()),
                               rank_mut_Q=float(stats.rankdata(q)[IDX[r.mut]]))
                recs.append(row)
            print(f"  {pdb} done", flush=True)
        except Exception as e:
            print(f"  skip {pdb}: {type(e).__name__}: {e}", flush=True)
    e = pd.DataFrame(recs)
    geo = pd.read_csv("results/abbind_positions.csv")[
        ["pdb", "chain", "resnum", "burial", "drsasa", "is_interface", "nbr"]]
    e = e.merge(geo, on=["pdb", "chain", "resnum"], how="left")
    e.to_csv(out_csv, index=False)
    print(f"cached {out_csv}: {len(e)} mutations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/abbind_readout_audit.csv")
    a = ap.parse_args()
    if not os.path.exists(CACHE):
        print("building per-mutation cache (ProteinMPNN) ...", flush=True)
        build_cache(CACHE)
    e = pd.read_csv(CACHE)
    ei = e[e.is_interface == 1].dropna(subset=["logP", "burial", "drsasa"]).reset_index(drop=True)
    ei["destab"] = (ei.ddg >= 1.0).astype(int)
    ei["dH"] = ei.H_mono - ei.H_complex
    ei["negH"] = -ei.H_complex
    y = ei.destab.to_numpy(float); g = ei.pdb.to_numpy()
    ids = np.unique(g); idx_by = {c: np.where(g == c)[0] for c in ids}
    rng = np.random.default_rng(SEED)
    print(f"\ninterface mutations {len(ei)}  complexes {len(ids)}  destabilising {int(y.sum())}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    BASE = ["burial", "drsasa", "blosum", "dvol"]
    Xb = np.column_stack([z(ei[c]) for c in BASE])

    def cv(X):
        o = np.zeros(len(y))
        for tr, te in GroupKFold(min(5, len(ids))).split(X, y, g):
            m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            o[te] = X[te] @ m.coef_[0] + m.intercept_[0]
        return o

    sb = cv(Xb); base_auc = auc(sb, y)
    u0 = Xb.sum(1)                       # the unfitted equal-weight analogue
    print(f"baseline AUROC: fitted-CV {base_auc:.4f}   unfitted z-sum {auc(u0, y):.4f}")

    # strata: mutant identity (removes the by-construction identity confound in logP exactly)
    k_mut = pd.factorize(ei["mut"])[0].astype(np.int64)
    k_pair = pd.factorize(ei["wt"] + ">" + ei["mut"])[0].astype(np.int64)
    # mutant identity x burial tertile
    bt = pd.qcut(ei.burial.rank(method="first"), 3, labels=False).astype(str)
    k_mb = pd.factorize(ei["mut"] + "|" + bt)[0].astype(np.int64)
    for nm, k in [("mut", k_mut), ("wt>mut", k_pair), ("mut x burial-tertile", k_mb)]:
        u = pd.DataFrame({"k": k, "y": y}).groupby("k").y.agg(["sum", "count"])
        print(f"  strata[{nm}]: {len(u)} cells, {int(((u['sum']>0)&(u['sum']<u['count'])).sum())} informative")

    rows = []
    FE = ["logP", "logodds", "rank_mut", "logP_minus_wt", "negH", "dH", "logQ", "burial", "drsasa"]
    for f in FE:
        if f not in ei:
            continue
        v = ei[f].to_numpy(float)
        # sign: destabilising should be LOW model preference -> negate the preference-like ones
        sgn = -1.0 if f in ("logP", "logodds", "rank_mut", "logP_minus_wt", "logQ") else 1.0
        v = sgn * v
        raw = auc(v, y)
        s_mut, s_pair, s_mb = sauc(v, y, k_mut), sauc(v, y, k_pair), sauc(v, y, k_mb)
        bm, bp = [], []
        for _ in range(NBOOT):
            t = np.concatenate([idx_by[c] for c in rng.choice(ids, len(ids), True)])
            a1, a2 = sauc(v[t], y[t], k_mut[t]), sauc(v[t], y[t], k_mb[t])
            if np.isfinite(a1): bm.append(a1)
            if np.isfinite(a2): bp.append(a2)
        lm, hm = np.percentile(bm, [2.5, 97.5]); lb, hb = np.percentile(bp, [2.5, 97.5])
        vn = np.where(np.isfinite(v), v, np.nanmean(v))
        sf = cv(np.column_stack([Xb, z(vn)])); d_fit = auc(sf, y) - base_auc
        bd = []
        for _ in range(NBOOT):
            t = np.concatenate([idx_by[c] for c in rng.choice(ids, len(ids), True)]); yy = y[t]
            if 0 < yy.sum() < len(yy):
                bd.append(auc(sf[t], yy) - auc(sb[t], yy))
        ld, hd = np.percentile(bd, [2.5, 97.5])
        vd = "ADDS" if lm > 0.5 else ("ANTI" if hm < 0.5 else "chance")
        print(f"  {f:14s} raw={raw:.3f} | within-MUT sAUROC={s_mut:.4f} [{lm:.4f},{hm:.4f}] {vd:6s} "
              f"| within-(wt>mut)={s_pair:.4f} | within-(mut x burial)={s_mb:.4f} [{lb:.4f},{hb:.4f}] "
              f"| dAUROC_fitted={d_fit:+.4f} [{ld:+.4f},{hd:+.4f}]")
        rows.append(dict(quantity=f, raw_auroc=round(raw, 4),
                         within_mut_sauroc=round(s_mut, 4), wm_lo=round(lm, 4), wm_hi=round(hm, 4),
                         wm_verdict=vd, within_wtmut_sauroc=round(s_pair, 4),
                         within_mut_burial_sauroc=round(s_mb, 4), wmb_lo=round(lb, 4), wmb_hi=round(hb, 4),
                         dauroc_fitted=round(d_fit, 4), dfit_lo=round(ld, 4), dfit_hi=round(hd, 4),
                         dfit_p=round(float(np.mean(np.array(bd) > 0)), 4),
                         n=len(ei), n_destab=int(y.sum()), n_complex=len(ids),
                         base_auroc_fitted=round(base_auc, 4)))

    # ---- power calibration: what within-MUT sAUROC does the committed readout need to fire? ----
    print("\n  --- power calibration of dAUROC over the geometry+similarity baseline ---")
    for delta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        ss, dfv, duv = [], [], []
        for _ in range(12):
            X = rng.normal(size=len(y)) + delta * y
            ss.append(sauc(X, y, k_mut))
            dfv.append(auc(cv(np.column_stack([Xb, z(X)])), y) - base_auc)
            duv.append(auc(u0 + z(X), y) - auc(u0, y))
        print(f"    within-MUT sAUROC={np.mean(ss):.3f} -> dAUROC fitted={np.mean(dfv):+.4f}  "
              f"unfitted={np.mean(duv):+.4f}")
        rows.append(dict(quantity=f"POSCONTROL_delta={delta}", within_mut_sauroc=round(float(np.mean(ss)), 4),
                         dauroc_fitted=round(float(np.mean(dfv)), 4), n=len(ei), n_destab=int(y.sum())))

    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_boot"] = NBOOT
    out["note"] = ("audit of abbind_bigidea1.csv: within-mutant-identity stratified AUROC replaces "
                   "dAUROC over a baseline that already encodes substitution identity; adds entropy "
                   "and identity-normalised (rank) forms of the same distribution")
    out["command"] = "python3 src/abbind_readout_audit.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

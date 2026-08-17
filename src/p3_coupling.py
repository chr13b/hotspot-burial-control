#!/usr/bin/env python3
"""Phase 3 — the coupling extension: does the model know binding EPISTASIS, not just single effects?

The paper's single-mutant object is the partner-ablated log-odds (leverage)
  L_i(a) = [logP_i(a) - logP_i(wt)] - [logQ_i(a) - logQ_i(wt)]   ~  -DDG_bind(wt->a),
a FIRST mixed derivative (partner ablation x one mutation).  StaB-ddG (appendix B) claims, untested,
that such models also carry pairwise BINDING epistasis.  The natural object is the SECOND mixed
derivative -- the categorical Jacobian of Zhang & Ovchinnikov (PNAS 2024), partner-ablated:

  C_ij(a,b) = L_ij(a,b) - L_i(a) - L_j(b)
            = [ how much setting j->b shifts the (a-vs-wt) log-odds at i ]  (partner-ablated).

Operationally, in the CONDITIONAL (autoregressive, teacher-forced) ProteinMPNN, C is read by flipping
the input residue at j to its mutant and measuring the shift in the conditional log-odds at i (for
orders where j is decoded before i), symmetrised over the two directions and averaged over decode
orders.  Because C is a difference of log-ODDS at a fixed position, the per-position normalisation
cancels -- raw conditional logits suffice.

Partner ablation:
 * CROSS-INTERFACE pair (i in group1, j in group2): the two residues couple ONLY through binding, so
   no single monomer contains both -> C_monomer = 0 and C_lev = C_complex.  This is the clean set.
 * SAME-SIDE pair (both in one group): C_monomer is the intra-fold coupling; C_lev = C_cplx - C_mono.

Validation target (experimental epistasis, from SKEMPI double mutants with BOTH singles measured):
  g_ij = DDG_ab - DDG_a - DDG_b   (kcal/mol).   Cycle:  L ~ -DDG  =>  C_lev ~ -g,  so we EXPECT
  Spearman(C_lev, g) < 0.  Headline is a partial correlation controlling inter-residue distance
  (contacts trivially couple), with a complex-clustered bootstrap; a binary CPI(|C| | distance) using
  the project's own estimator is the method-consistent robustness cut.

Positive controls (rule 6): additive pairs (g~0) must give C~0; the two directions C_{i->j}, C_{j->i}
must agree; every triangle's wt must match the crystal residue or it is dropped and counted.

  python3 src/p3_coupling.py --limit 4                                   # smoke
  python3 src/p3_coupling.py --stage score  --out results/p3_coupling.csv
  python3 src/p3_coupling.py --stage analyse --out results/p3_coupling.csv
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc
import leverage_decomposition as LD

SEED = 20260803
DATA = LD.DATA
MPNN_W = LD.MPNN_W
ALPHA = fc.MPNN_ALPHABET                      # 21-letter, raw-logit index order
LIDX = {a: i for i, a in enumerate(ALPHA)}
OUT_DEFAULT = "results/p3_coupling.csv"


# --------------------------------------------------------------------------- triangles

def build_triangles():
    """SKEMPI -> one row per (double mutant with BOTH singles measured).  g = DDG_ab-DDG_a-DDG_b."""
    sk = fc.parse_skempi(f"{DATA}/skempi_v2.csv")
    sk["complex_id"] = sk.pdb + "_" + sk.group1 + "_" + sk.group2
    sk["mutkey"] = sk["Mutation(s)_cleaned"].fillna("")
    grp = (sk.groupby(["complex_id", "pdb", "group1", "group2", "mutkey", "n_mut"], as_index=False)
             .agg(ddG=("ddG", "median"), n_meas=("ddG", "size")))
    singles = {(r.complex_id, r.mutkey): r.ddG for r in grp[grp.n_mut == 1].itertuples()}
    tri = []
    for r in grp[grp.n_mut == 2].itertuples():
        parts = [p.strip() for p in r.mutkey.split(",")]
        if len(parts) != 2:
            continue
        a, b = parts
        ga, gb = singles.get((r.complex_id, a)), singles.get((r.complex_id, b))
        if ga is None or gb is None:
            continue
        tri.append(dict(complex_id=r.complex_id, pdb=r.pdb, group1=r.group1, group2=r.group2,
                        m1=a, m2=b, ddG_ab=r.ddG, ddG_a=ga, ddG_b=gb, g=r.ddG - ga - gb))
    return pd.DataFrame(tri)


# --------------------------------------------------------------------------- model coupling

def make_orders(n, seeds):
    """Reproduce mpnn_conditional_logprobs' decode orders EXACTLY, and their per-position ranks.

    Returns rank[K,n]: rank[k,p] = decoding step of position p in order k (smaller = earlier).
    """
    import torch
    P = []
    for sd in seeds:
        g = torch.Generator(device="cpu").manual_seed(int(sd))
        randn = torch.randn(1, n, generator=g)
        P.append(torch.argsort((torch.ones(1, n) + 0.0001) * torch.abs(randn))[0].numpy())
    P = np.stack(P)                                  # [K,n]: P[k,t] = position decoded at step t
    return np.argsort(P, axis=1)                     # inverse perm: rank[k,pos] = step


def clone_with_mut(cx, idx, aa):
    c2 = fc.ComplexStruct()
    for s in cx.__slots__:
        try:
            setattr(c2, s, getattr(cx, s))
        except AttributeError:
            pass
    s2 = cx.seq.copy()
    s2[idx] = aa
    c2.seq = s2
    return c2


def _directed(cond_pass, base, rank, jp, ip, mut_i, wt_i):
    """Order-averaged shift in (mut_i-vs-wt_i) log-odds at position ip caused by conditioning jp->mut,
    using only orders where jp is decoded before ip.  cond_pass/base are [K,n,21] raw logits."""
    Lm, Lw = LIDX[mut_i], LIDX[wt_i]
    active = rank[:, jp] < rank[:, ip]
    if not active.any():
        return np.nan
    d = ((cond_pass[active, ip, Lm] - cond_pass[active, ip, Lw])
         - (base[active, ip, Lm] - base[active, ip, Lw]))
    return float(d.mean())


def coupling_for_struct(model, cx, pairs, seeds):
    """pairs: list of (j1,wt1,m1, j2,wt2,m2).  Returns {idx: (C_sym, C_12, C_21)} keyed by pair order.

    j1/j2 are position indices into cx.  Runs one teacher-forced base pass + one perturbed pass per
    distinct (position, mutant) conditioner; reads every pair off the cached passes."""
    rank = make_orders(cx.n, seeds)
    base = fc.mpnn_conditional_logprobs(model, cx, seeds)              # [K,n,21]
    conds = {}
    for (j1, w1, m1, j2, w2, m2) in pairs:
        conds.setdefault((j1, m1), None)
        conds.setdefault((j2, m2), None)
    for key in list(conds):
        jp, aa = key
        conds[key] = fc.mpnn_conditional_logprobs(model, clone_with_mut(cx, jp, aa), seeds)
    out = {}
    for idx, (j1, w1, m1, j2, w2, m2) in enumerate(pairs):
        c12 = _directed(conds[(j1, m1)], base, rank, j1, j2, m2, w2)   # condition j1, read j2
        c21 = _directed(conds[(j2, m2)], base, rank, j2, j1, m1, w1)   # condition j2, read j1
        vals = [v for v in (c12, c21) if np.isfinite(v)]
        csym = float(np.mean(vals)) if vals else np.nan
        out[idx] = (csym, c12, c21)
    del base, conds
    gc.collect()
    return out


# --------------------------------------------------------------------------- scoring stage

def stage_score(a):
    import torch
    os.environ["FTAX_MAX_BATCH"] = str(a.order_batch)   # cap peak activation memory (OOM guard)
    torch.set_num_threads(a.threads)
    seeds = list(range(a.seeds))
    tri = build_triangles()
    print(f"[score] {len(tri)} triangles over {tri.complex_id.nunique()} complexes "
          f"(seeds={a.seeds}, threads={a.threads})", flush=True)

    done = set()
    if os.path.exists(a.out) and not a.overwrite:
        try:
            done = set(pd.read_csv(a.out, usecols=["complex_id"]).complex_id)
            print(f"[score] resuming, {len(done)} complexes already scored", flush=True)
        except Exception:
            done = set()
    model, _ = fc.load_mpnn(MPNN_W)

    cols = ["complex_id", "pdb", "m1", "m2", "wt1", "wt2", "chain1", "chain2", "resnum1", "resnum2",
            "cross_interface", "dist_cb", "contact", "seqsep", "ddG_ab", "ddG_a", "ddG_b", "g",
            "C_complex", "C12", "C21", "C_monomer", "C_lev"]
    fh = open(a.out, "a" if (done and not a.overwrite) else "w", newline="")
    if not (done and not a.overwrite):
        fh.write(",".join(cols) + "\n"); fh.flush()

    cids = sorted(tri.complex_id.unique())
    if a.limit:
        cids = cids[:a.limit]
    n_mismatch = n_unmapped = n_rows = 0
    dropped_big = []
    t0 = time.time()
    for ci, cid in enumerate(cids):
        if cid in done:
            continue
        sub = tri[tri.complex_id == cid]
        pdb, g1, g2 = cid.split("_")
        path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            print(f"  skip {cid}: no pdb", flush=True); continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
            if cx is None:
                print(f"  skip {cid}: load None", flush=True); continue
            if cx.n > a.max_residues:               # OOM guard: drop-and-log oversized complexes
                dropped_big.append((cid, cx.n, len(sub)))
                print(f"  drop {cid}: n={cx.n} > max_residues={a.max_residues} "
                      f"({len(sub)} triangles)", flush=True)
                del cx; gc.collect(); continue
            cxmap = {(cx.chains[j], int(cx.resnums[j]), str(cx.icodes[j])): j for j in range(cx.n)}
            g1set, g2set = set(g1), set(g2)

            # map triangles -> position indices; positive control on wt identity
            recs, cpx_pairs = [], []
            for t in sub.itertuples():
                p1, p2 = fc.parse_mutation(t.m1), fc.parse_mutation(t.m2)
                if not p1 or not p2:
                    n_unmapped += 1; continue
                j1 = cxmap.get((p1["chain"], p1["resnum"], p1["icode"]))
                j2 = cxmap.get((p2["chain"], p2["resnum"], p2["icode"]))
                if j1 is None or j2 is None:
                    n_unmapped += 1; continue
                if str(cx.seq[j1]) != p1["wt"] or str(cx.seq[j2]) != p2["wt"]:
                    n_mismatch += 1; continue
                side1 = 1 if p1["chain"] in g1set else (2 if p1["chain"] in g2set else 0)
                side2 = 1 if p2["chain"] in g1set else (2 if p2["chain"] in g2set else 0)
                if side1 == 0 or side2 == 0:
                    n_unmapped += 1; continue
                cross = int(side1 != side2)
                d = float(np.linalg.norm(cx.CB[j1] - cx.CB[j2]))
                seqsep = abs(p1["resnum"] - p2["resnum"]) if p1["chain"] == p2["chain"] else -1
                recs.append(dict(t=t, p1=p1, p2=p2, j1=j1, j2=j2, side1=side1, side2=side2,
                                 cross=cross, dist=d, seqsep=seqsep))
                cpx_pairs.append((j1, p1["wt"], p1["mut"], j2, p2["wt"], p2["mut"]))
            if not recs:
                continue

            Ccpx = coupling_for_struct(model, cx, cpx_pairs, seeds)

            # monomer couplings only for same-side pairs, grouped by their side's group string
            need_mono = {}
            for k, r in enumerate(recs):
                if not r["cross"]:
                    grp = g1 if r["side1"] == 1 else g2
                    need_mono.setdefault(grp, []).append(k)
            Cmono = {}
            for grp, kidx in need_mono.items():
                mono = fc.load_complex(path, pdb, grp, "", require_both=False)
                if mono is None or mono.n < 5:
                    continue
                mm = {(mono.chains[j], int(mono.resnums[j]), str(mono.icodes[j])): j
                      for j in range(mono.n)}
                mpairs, backmap = [], []
                for k in kidx:
                    r = recs[k]
                    mj1 = mm.get((r["p1"]["chain"], r["p1"]["resnum"], r["p1"]["icode"]))
                    mj2 = mm.get((r["p2"]["chain"], r["p2"]["resnum"], r["p2"]["icode"]))
                    if mj1 is None or mj2 is None:
                        continue
                    mpairs.append((mj1, r["p1"]["wt"], r["p1"]["mut"], mj2, r["p2"]["wt"], r["p2"]["mut"]))
                    backmap.append(k)
                if mpairs:
                    res = coupling_for_struct(model, mono, mpairs, seeds)
                    for local, k in enumerate(backmap):
                        Cmono[k] = res[local][0]
                del mono
                gc.collect()

            for k, r in enumerate(recs):
                t, p1, p2 = r["t"], r["p1"], r["p2"]
                csym, c12, c21 = Ccpx[k]
                if r["cross"]:                      # no monomer contains both -> C_lev = C_complex
                    cmono, clev = 0.0, csym
                else:                               # same side: subtract intra-fold coupling, else drop
                    cmono = Cmono.get(k, np.nan)
                    clev = (csym - cmono) if np.isfinite(cmono) else np.nan
                row = [cid, pdb, t.m1, t.m2, p1["wt"], p2["wt"], p1["chain"], p2["chain"],
                       p1["resnum"], p2["resnum"], r["cross"], round(r["dist"], 3),
                       int(r["dist"] < 8.0), r["seqsep"], t.ddG_ab, t.ddG_a, t.ddG_b, t.g,
                       csym, c12, c21, cmono, clev]
                fh.write(",".join(str(x) for x in row) + "\n"); n_rows += 1
            fh.flush()
            del cx
            gc.collect()
            rate = (ci + 1) / max(time.time() - t0, 1e-9)
            print(f"  [{ci+1}/{len(cids)}] {cid}: {len(recs)} pairs "
                  f"({sum(r['cross'] for r in recs)} cross)  {rate:.2f} cplx/s", flush=True)
        except Exception as e:
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
            continue
    fh.close()
    print(f"[score] wrote {n_rows} rows -> {a.out}   "
          f"(wt-mismatch dropped={n_mismatch}, unmapped dropped={n_unmapped})", flush=True)
    if dropped_big:
        tot = sum(t for _, _, t in dropped_big)
        print(f"[score] OOM-guard dropped {len(dropped_big)} complexes >{a.max_residues} res "
              f"({tot} triangles): {[c for c, _, _ in dropped_big]}", flush=True)


# --------------------------------------------------------------------------- analysis stage

def partial_spearman_boot(x, y, z, groups, rng, n_boot=3000):
    """Partial Spearman(x,y | z): rank all three, residualise ranks of x and y on rank(z), correlate
    residuals.  Complex-clustered bootstrap CI.  Returns (rho, lo, hi, P(<0), n)."""
    x, y, z, groups = map(np.asarray, (x, y, z, groups))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z, groups = x[m], y[m], z[m], groups[m]

    def prho(xi, yi, zi):
        rx, ry, rz = (stats.rankdata(v) for v in (xi, yi, zi))
        rz1 = np.column_stack([np.ones_like(rz), rz])
        ex = rx - rz1 @ np.linalg.lstsq(rz1, rx, rcond=None)[0]
        ey = ry - rz1 @ np.linalg.lstsq(rz1, ry, rcond=None)[0]
        if ex.std() < 1e-12 or ey.std() < 1e-12:
            return np.nan
        return float(np.corrcoef(ex, ey)[0, 1])

    rho = prho(x, y, z)
    ids = np.unique(groups)
    by = {k: np.where(groups == k)[0] for k in ids}
    bs = []
    for _ in range(n_boot):
        ix = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        bs.append(prho(x[ix], y[ix], z[ix]))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return (rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
            float(np.mean(bs < 0)), int(len(x)))


def _boot_spearman(x, y, groups, rng, n_boot=3000):
    x, y, groups = map(np.asarray, (x, y, groups))
    m = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[m], y[m], groups[m]
    rho = stats.spearmanr(x, y).correlation
    ids = np.unique(groups); by = {k: np.where(groups == k)[0] for k in ids}
    bs = []
    for _ in range(n_boot):
        ix = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        r = stats.spearmanr(x[ix], y[ix]).correlation
        if np.isfinite(r):
            bs.append(r)
    bs = np.array(bs)
    return rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.mean(bs < 0)), int(len(x))


def stage_analyse(a):
    df = pd.read_csv(a.out)
    rng = np.random.default_rng(SEED)
    rows = []

    def block(name, d, feat):
        if len(d) < 10 or d[feat].notna().sum() < 10:
            print(f"  [{name}] n={len(d)} too small / no {feat}"); return
        rho, lo, hi, p, n = _boot_spearman(d[feat], d.g, d.complex_id, rng)
        prho, plo, phi, pp, pn = partial_spearman_boot(d[feat], d.g, d.dist_cb, d.complex_id, rng)
        sign = float(np.mean(np.sign(-d[feat]) == np.sign(d.g)))     # C ~ -g  => -C and g share sign
        print(f"  [{name}] {feat}: n={n} cplx={d.complex_id.nunique()}  "
              f"Spearman(C,g)={rho:+.3f} [{lo:+.3f},{hi:+.3f}] P(<0)={p:.3f} | "
              f"partial|dist={prho:+.3f} [{plo:+.3f},{phi:+.3f}] P(<0)={pp:.3f} | sign(-C,g)={sign:.3f}")
        rows.append(dict(set=name, feature=feat, n=n, n_complex=int(d.complex_id.nunique()),
                         spearman=round(rho, 4), sp_lo=round(lo, 4), sp_hi=round(hi, 4), sp_P_lt0=round(p, 3),
                         partial_dist=round(prho, 4), pd_lo=round(plo, 4), pd_hi=round(phi, 4),
                         pd_P_lt0=round(pp, 3), sign_agree=round(sign, 3)))

    print(f"[analyse] {len(df)} triangles, {df.complex_id.nunique()} complexes "
          f"({int(df.cross_interface.sum())} cross-interface)")
    print("  expected sign: C ~ -g  =>  Spearman(C,g) < 0")
    block("all/C_lev", df, "C_lev")
    block("cross/C_lev", df[df.cross_interface == 1], "C_lev")
    block("same/C_lev", df[df.cross_interface == 0], "C_lev")
    block("cross/C_complex", df[df.cross_interface == 1], "C_complex")   # cross: C_complex==C_lev
    block("same/C_complex", df[df.cross_interface == 0], "C_complex")    # un-ablated control

    # additivity positive control: |C| should grow with |g|
    d = df[df.C_lev.notna()].copy()
    d["g"] = pd.to_numeric(d.g, errors="coerce")
    d = d[d.g.notna()]
    if len(d) >= 6:
        print("  [control] mean |C_lev| by |g| tertile (expect increasing):")
        d["gbin"] = pd.qcut(d.g.abs(), 3, labels=["low|g|", "mid|g|", "high|g|"], duplicates="drop")
        for b, gg in d.groupby("gbin", observed=True):
            print(f"      {b}: mean|C_lev|={gg.C_lev.abs().mean():.4f}  n={len(gg)}")

    # direction symmetry sanity: C12 vs C21 should correlate positively
    m = df.C12.notna() & df.C21.notna()
    if m.sum() > 10:
        r = stats.spearmanr(df.C12[m], df.C21[m]).correlation
        print(f"  [control] direction symmetry Spearman(C12,C21)={r:+.3f} (n={int(m.sum())}, expect >0)")

    # method-consistent robustness: binary CPI(|C_lev| | distance) with the project estimator
    d = df[df.C_lev.notna() & df.dist_cb.notna()].copy()
    if d.complex_id.nunique() >= 5 and len(d) >= 30:
        y = (d.g.abs() > 0.5).astype(int).to_numpy()
        if 0 < y.sum() < len(y):
            Z = d[["dist_cb"]].to_numpy(float)
            X = d.C_lev.abs().to_numpy()
            g = d.complex_id.to_numpy()
            c, lo, hi, pp, _, _ = LD.cpi(y, g, Z, X, np.random.default_rng(SEED))
            print(f"  [CPI] |C_lev| beyond distance, outcome |g|>0.5: "
                  f"CPI={c:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={pp:.3f}")
            rows.append(dict(set="cpi|C_lev|>dist", feature="binary|g|>0.5", n=len(d),
                             n_complex=int(d.complex_id.nunique()), spearman=round(c, 5),
                             sp_lo=round(lo, 5), sp_hi=round(hi, 5), sp_P_lt0=round(1 - pp, 3),
                             partial_dist=np.nan, pd_lo=np.nan, pd_hi=np.nan, pd_P_lt0=np.nan,
                             sign_agree=np.nan))

    # robustness 1: cross-interface CPI + drop-3-most-influential-complexes survival
    cc = df[(df.cross_interface == 1) & df.C_lev.notna() & df.dist_cb.notna()].copy()
    if cc.complex_id.nunique() >= 6 and len(cc) >= 30:
        y = (cc.g.abs() > 0.5).astype(int).to_numpy()
        if 0 < y.sum() < len(y):
            Z = cc[["dist_cb"]].to_numpy(float); X = cc.C_lev.abs().to_numpy(); g = cc.complex_id.to_numpy()
            c, lo, hi, pp, _, cvec = LD.cpi(y, g, Z, X, np.random.default_rng(SEED), n_boot=2000)
            contrib = pd.Series(cvec).groupby(pd.Series(g)).sum().sort_values(ascending=False)
            drop = set(contrib.index[:3]); keep = ~pd.Series(g).isin(drop).to_numpy()
            c2, lo2, hi2, pp2, _, _ = LD.cpi(y[keep], g[keep], Z[keep], X[keep].copy(),
                                             np.random.default_rng(SEED), n_boot=2000)
            surv = "SURVIVES" if lo2 > 0 else "does not survive"
            print(f"  [CPI cross] |C_lev|>dist, |g|>0.5: CPI={c:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={pp:.3f}  "
                  f"| drop3 {sorted(drop)}: {c2:+.5f} [{lo2:+.5f},{hi2:+.5f}] {surv}")
            rows.append(dict(set="cpi_cross|C_lev|>dist", feature="binary|g|>0.5", n=len(cc),
                             n_complex=int(cc.complex_id.nunique()), spearman=round(c, 5),
                             sp_lo=round(lo, 5), sp_hi=round(hi, 5), sp_P_lt0=round(1 - pp, 3),
                             partial_dist=round(c2, 5), pd_lo=round(lo2, 5), pd_hi=round(hi2, 5),
                             pd_P_lt0=round(1 - pp2, 3), sign_agree=np.nan))

    # robustness 2: contact-split (cross-interface) -- signal must not be purely the contact boundary
    for lab, sub in [("cross/contact<8A", cc[cc.contact == 1]),
                     ("cross/noncontact>=8A", cc[cc.contact == 0])]:
        if len(sub) >= 10 and sub.complex_id.nunique() >= 3:
            prho, plo, phi, pp, pn = partial_spearman_boot(sub.C_lev, sub.g, sub.dist_cb,
                                                           sub.complex_id, rng)
            print(f"  [contact-split] {lab}: partial|dist={prho:+.3f} [{plo:+.3f},{phi:+.3f}] "
                  f"P(<0)={pp:.3f}  n={pn} cplx={sub.complex_id.nunique()}")
            rows.append(dict(set=lab, feature="C_lev", n=pn, n_complex=int(sub.complex_id.nunique()),
                             spearman=np.nan, sp_lo=np.nan, sp_hi=np.nan, sp_P_lt0=np.nan,
                             partial_dist=round(prho, 4), pd_lo=round(plo, 4), pd_hi=round(phi, 4),
                             pd_P_lt0=round(pp, 3), sign_agree=np.nan))

    summ = a.out.replace(".csv", "_summary.csv")
    pd.DataFrame(rows).to_csv(summ, index=False)
    print(f"[analyse] wrote {summ}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyse", "both"], default="both")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--seeds", type=int, default=8, help="number of decode orders")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--order-batch", dest="order_batch", type=int, default=1,
                    help="decode orders per model forward (1=lowest memory; OOM guard)")
    ap.add_argument("--max-residues", dest="max_residues", type=int, default=800,
                    help="drop-and-log complexes larger than this (OOM guard)")
    ap.add_argument("--limit", type=int, default=0, help="first N complexes (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.stage in ("score", "both"):
        stage_score(a)
    if a.stage in ("analyse", "both"):
        stage_analyse(a)


if __name__ == "__main__":
    main()

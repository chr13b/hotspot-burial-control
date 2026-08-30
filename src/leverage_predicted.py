#!/usr/bin/env python3
"""Does the mixed-derivative binding signal SURVIVE on PREDICTED backbones? (§6 design-regime anchor)

The crystal leverage result (leverage_decomposition.py) and the crystal noise ladder
(leverage_noise_ladder.py) both show CPI(L | geometry) > 0 on native/accurate backbones and a collapse
by ~1 A of jitter. Designers, however, condition on backbones from a FOLDING MODEL, not crystals. This
script re-runs the EXACT committed leverage pipeline on OpenFold3 and AF2-multimer predicted backbones
for the complexes shared with the SKEMPI DDG fixture, and asks whether CPI(L | geometry) and the
geometry+|L| hotspot ranker still hold.

ONE-VARIABLE MANIPULATION (stated, not hidden). The SKEMPI interface-mutation fixture is held FIXED
from the committed crystal analysis -- same complexes, same positions, same hotspot labels, same DDG.
Only two things are swapped, and they are exactly what a designer working off a predicted backbone gets:
  (1) leverage L is recomputed by the IDENTICAL scorer (leverage_decomposition._score_one -> ProteinMPNN
      sequence-free unconditional marginal, complex vs partner-deleted monomer) reading the PREDICTED
      complex PDB;
  (2) geometry burial/nbr/DSASA is taken FROM THE PREDICTED structure (the expA/expD p0 positions files,
      built by p0_burial_matched.py on the same predicted PDBs) -- the honest baseline a designer has,
      not the crystal geometry.
Interface membership and the alanine-scan hotspot label are properties of the DDG data, not the
backbone, so they are inherited from the committed crystal fixture; this keeps the sample identical
across crystal / OF3 / AF2 so the ONLY thing changing is the backbone the derivative is read from.

POSITIVE CONTROL (rule 6, gates everything). `--source crystal` runs the same code on the crystal PDBs
restricted to the shared set. It must (i) reproduce the committed leverage_pq_skempi.csv lP/lQ to
float32 precision, and (ii) land near the committed crystal CPI(L | geom) (approx +0.059 mutation-level,
+0.0048 position-level). If it does not, the predicted number is uninterpretable -- stop and debug.

  # score (writes results/leverage_pq_predicted_{source}.csv, gitignored)
  python3 src/leverage_predicted.py --stage score --source crystal
  python3 src/leverage_predicted.py --stage score --source of3
  python3 src/leverage_predicted.py --stage score --source af2
  # analyse all scored sources + pooled -> results/leverage_predicted.csv (+ _ranker.csv)
  python3 src/leverage_predicted.py --stage analyse --out results/leverage_predicted.csv
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc
import leverage_decomposition as LD

SEED = LD.SEED
AA20 = LD.AA20
IDX = LD.IDX
SCRATCH = os.environ["SCRATCH"]

# Per-source: predicted complex PDB dir + the p0 positions file carrying that source's geometry.
# crystal geometry lives in results/p0_positions.csv (the committed fixture geometry).
SOURCES = {
    "crystal": dict(pdbdir=os.path.expanduser("~/ftax/data/PDBs"),
                    geom="results/p0_positions.csv", predicted=False),
    "of3":     dict(pdbdir=f"{SCRATCH}/ftax/predicted/PDBs",
                    geom=f"{SCRATCH}/ftax/predicted/expA_p0_positions.csv", predicted=True),
    "af2":     dict(pdbdir=f"{SCRATCH}/ftax/expD/PDBs",
                    geom=f"{SCRATCH}/ftax/expD/expD_p0_positions.csv", predicted=True),
}
PQ = "results/leverage_pq_predicted_{source}.csv"
GEOCOLS = ["complex_id", "chain", "resnum", "icode", "rsasa_complex", "nbr", "drsasa"]


# --------------------------------------------------------------------------- shared set

def shared_complexes():
    """Complexes present in ALL of {OF3 geom, AF2 geom, SKEMPI interface-mut fixture}."""
    sk = pd.read_csv("results/leverage_skempi_mutations.csv", usecols=["complex_id", "is_interface"],
                     low_memory=False)
    sk = set(sk[sk.is_interface == 1].complex_id.unique())
    of3 = set(pd.read_csv(SOURCES["of3"]["geom"], usecols=["complex_id"], low_memory=False).complex_id)
    af2 = set(pd.read_csv(SOURCES["af2"]["geom"], usecols=["complex_id"], low_memory=False).complex_id)
    return sorted(sk & of3 & af2)


def keepset():
    """Exactly LD.stage_score_skempi's keep-set: interface positions + measured single-mut positions."""
    sk = fc.parse_skempi(f"{LD.DATA}/skempi_v2.csv")
    sk = sk[sk.n_mut == 1].copy()
    sk["complex_id"] = sk.pdb + "_" + sk.group1 + "_" + sk.group2
    p0 = pd.read_csv("results/p0_positions.csv", low_memory=False,
                     usecols=["complex_id", "chain", "resnum", "icode", "is_interface"])
    p0["icode"] = p0.icode.fillna("").astype(str)
    keep = {}
    for cid, sub in p0[p0.is_interface == True].groupby("complex_id"):      # noqa: E712
        keep[cid] = set(zip(sub.chain, sub.resnum.astype(int), sub.icode))
    for _, r in sk.iterrows():
        m = fc.parse_mutation(r["muts"][0])
        if m is None:
            continue
        keep.setdefault(r["complex_id"], set()).add((m["chain"], m["resnum"], m["icode"]))
    return keep


# --------------------------------------------------------------------------- score stage

def stage_score(a):
    import torch
    torch.set_num_threads(a.threads)
    src = SOURCES[a.source]
    keep = keepset()
    shared = set(shared_complexes())
    cids = [c for c in sorted(keep) if c in shared]
    if a.limit:
        cids = cids[:a.limit]
    print(f"[score:{a.source}] {len(cids)} shared complexes, "
          f"{sum(len(keep[c]) for c in cids)} target positions; pdbdir={src['pdbdir']}", flush=True)
    model, _ = fc.load_mpnn(LD.MPNN_W)
    out = PQ.format(source=a.source)
    fh = open(out, "w", newline="")
    writer, n, t0, skipped = None, 0, time.time(), []
    for ci, cid in enumerate(cids):
        pdb, g1, g2 = cid.split("_")
        path = f"{src['pdbdir']}/{pdb}.pdb"
        if not os.path.exists(path):
            skipped.append((cid, "no pdb"))
            continue
        try:
            rows = LD._score_one(model, path, pdb, g1, g2, keep[cid])
        except Exception as e:
            skipped.append((cid, f"{type(e).__name__}: {e}"))
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
            continue
        for r in rows:
            r = dict(complex_id=cid, **r)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                writer.writeheader()
            writer.writerow(r)
            n += 1
        fh.flush()
        if (ci + 1) % 25 == 0:
            print(f"[score:{a.source}] {ci+1}/{len(cids)}  {time.time()-t0:.0f}s  {n} rows", flush=True)
    fh.close()
    print(f"[score:{a.source}] wrote {out}: {n} rows over {len(cids)-len(skipped)} complexes; "
          f"{len(skipped)} skipped {skipped[:5]}", flush=True)


# --------------------------------------------------------------------------- frame builders

def _pq(source):
    pq = pd.read_csv(PQ.format(source=source), low_memory=False)
    pq["icode"] = pq.icode.fillna("").astype(str)
    return pq


def _geom(source):
    g = pd.read_csv(SOURCES[source]["geom"], usecols=GEOCOLS, low_memory=False)
    g["icode"] = g.icode.fillna("").astype(str)
    g["burial"] = -g.rsasa_complex
    return g[["complex_id", "chain", "resnum", "icode", "burial", "nbr", "drsasa"]]


def build_mut_frame(source, shared):
    """Mutation-level fixture, geometry+L swapped to `source`. Mirrors LD.build_skempi arithmetic."""
    fix = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    fix["icode"] = fix.icode.fillna("").astype(str)
    fix = fix[fix.complex_id.isin(shared)].copy()
    keepcols = ["complex_id", "chain", "resnum", "icode", "wt", "mut", "ddG", "n_meas",
                "is_interface", "destab"]
    fix = fix[keepcols]

    pq = _pq(source)
    d = fix.merge(pq, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    # positive control: WT identity must match the residue in the scored (predicted) structure
    mapped_rate = len(d) / len(fix)
    wt_match = (d.aa == d.wt)
    d = d[wt_match].copy()
    lP = d[[f"lP_{x}" for x in AA20]].to_numpy()
    lQ = d[[f"lQ_{x}" for x in AA20]].to_numpy()
    wi = d.wt.map(IDX).to_numpy(); mi = d.mut.map(IDX).to_numpy()
    d["L"] = LD.leverage(lP, lQ, wi, mi)
    d["logP_mut"] = lP[np.arange(len(d)), mi]
    d["conf"] = lP[np.arange(len(d)), wi]
    Pn = np.exp(lP)
    d["klP"] = (Pn * (lP - lQ)).sum(axis=1)
    d = d.merge(_geom(source), on=["complex_id", "chain", "resnum", "icode"], how="left")
    d = LD._finish(d, f"predicted_{source}")
    d["destab"] = (d.ddG >= LD.HOT_DDG).astype(int)
    return d, dict(map_rate=round(mapped_rate, 4), wt_match=round(float(wt_match.mean()), 4),
                   n=len(d), n_cx=int(d.complex_id.nunique()))


def build_pos_frame(source, shared):
    """Position-level frame: predicted lP/lQ + predicted geometry + crystal is_hot/is_interface.
    Mirrors LD.position_frame arithmetic (Lvec, conf, klP, negH, L_ala, L_rms, L_min)."""
    lab = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False,
                      usecols=["complex_id", "chain", "resnum", "icode", "is_interface", "is_hot"])
    lab["icode"] = lab.icode.fillna("").astype(str)
    lab = lab[lab.complex_id.isin(shared)]

    pq = _pq(source)
    d = lab.merge(pq, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    d = d[d.is_interface == 1].copy()
    lP = d[[f"lP_{x}" for x in AA20]].to_numpy()
    lQ = d[[f"lQ_{x}" for x in AA20]].to_numpy()
    wi = d.aa.map(IDX).to_numpy().astype(float)
    ok = np.isfinite(wi)
    d = d[ok].reset_index(drop=True); lP, lQ = lP[ok], lQ[ok]
    wi = d.aa.map(IDX).to_numpy()
    n = len(d); ar = np.arange(n)
    r = lP - lQ
    Lvec = r - r[ar, wi][:, None]
    d["conf"] = lP[ar, wi]
    d["negH"] = (np.exp(lP) * lP).sum(axis=1)
    d["klP"] = (np.exp(lP) * r).sum(axis=1)
    d["L_ala"] = Lvec[:, IDX["A"]]
    mask = np.ones((n, 20), bool); mask[ar, wi] = False
    Ln = np.where(mask, Lvec, np.nan)
    d["L_rms"] = np.sqrt(np.nanmean(Ln ** 2, axis=1))
    d["L_min"] = np.nanmin(Ln, axis=1)
    d = d.merge(_geom(source), on=["complex_id", "chain", "resnum", "icode"], how="left")
    return d


# --------------------------------------------------------------------------- combined ranker (w4)

def combined_ranker(d, rng, name, rows):
    """OOF cross-fit logistic AUROC of is_hot on geometry vs geometry+|L|_rms; paired complex-boot.
    Identical metric to src/w4_combined_ranker.py, on the predicted-source position frame."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    d = d[(d.is_interface == 1) & d.L_rms.notna() & d.L_ala.notna()
          & d.burial.notna() & d.nbr.notna() & d.drsasa.notna()].copy()
    y = d.is_hot.astype(int).to_numpy(); g = d.complex_id.to_numpy()

    def z(c):
        v = d[c].to_numpy(float); return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)

    GEO = np.column_stack([z("burial"), z("nbr"), z("drsasa")])
    feats = {"geometry (burial+nbr+dSASA)": GEO,
             "geometry + |L|_rms": np.column_stack([GEO, z("L_rms")])}

    def oof(X):
        eta = np.zeros(len(y)); nf = int(min(5, len(np.unique(g))))
        for tr, te in GroupKFold(nf).split(X, y, g):
            m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            eta[te] = X[te] @ m.coef_[0] + m.intercept_[0]
        return eta

    sc = {k: oof(X) for k, X in feats.items()}
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    base = sc["geometry (burial+nbr+dSASA)"]
    a_geo = LD.auc(base, y)
    a_comb = LD.auc(sc["geometry + |L|_rms"], y)
    boot = []
    for _ in range(3000):
        ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids), True)])
        boot.append(LD.auc(sc["geometry + |L|_rms"][ix], y[ix]) - LD.auc(base[ix], y[ix]))
    boot = np.array(boot); lo, hi, p = (float(np.percentile(boot, 2.5)),
                                        float(np.percentile(boot, 97.5)), float(np.mean(boot > 0)))
    # marginal |L|_rms alone, for context
    a_lrms = LD.auc(d.L_rms.to_numpy(), y)
    print(f"  [ranker {name}] AUROC geom={a_geo:.4f}  geom+|L|={a_comb:.4f}  "
          f"Δ={a_comb-a_geo:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}  (|L|_rms alone {a_lrms:.4f})")
    rows.append(dict(source=name, metric="ranker_AUROC_geometry", stat=round(a_geo, 4), n=len(d),
                     n_cx=int(len(ids)), n_hot=int(y.sum())))
    rows.append(dict(source=name, metric="ranker_AUROC_geometry+|L|rms", stat=round(a_comb, 4),
                     n=len(d), n_cx=int(len(ids)), n_hot=int(y.sum())))
    rows.append(dict(source=name, metric="ranker_dAUROC_addL_vs_geometry", stat=round(a_comb-a_geo, 4),
                     lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(p, 3), n=len(d),
                     n_cx=int(len(ids)), n_hot=int(y.sum())))
    rows.append(dict(source=name, metric="ranker_AUROC_|L|rms_alone", stat=round(a_lrms, 4), n=len(d)))


# --------------------------------------------------------------------------- CPI readouts

def cpi_mut(d, rng, name, rows):
    """Primary mutation-level CPI(L | geometry) + the +confidence and drop-3 robustness controls."""
    need = ["burial", "nbr", "drsasa", "L", "conf", "destab"]
    d = d.dropna(subset=need).reset_index(drop=True)
    d = d[d.is_interface == 1].reset_index(drop=True)
    y = d.destab.to_numpy().astype(float); g = d.complex_id.to_numpy()
    for c in ["burial", "nbr", "drsasa", "L", "conf"]:
        d[c + "z"] = LD.zs(d[c])
    Zgeo = d[["burialz", "nbrz", "drsasaz"]].to_numpy()
    sp = stats.spearmanr(d.L, d.ddG, nan_policy="omit").correlation
    Zsets = [("burial+nbr+dSASA", Zgeo),
             ("burial+nbr+dSASA+confidence", np.column_stack([Zgeo, d.confz]))]
    prim = None
    for zn, Z in Zsets:
        c, lo, hi, p, _, _ = LD.cpi(y, g, Z, d["Lz"].to_numpy().copy(), rng)
        if prim is None:
            prim = (c, lo, hi, p)
        v = "ADDS (CI>0)" if lo > 0 else "conditionally INDEPENDENT (CI spans 0)"
        print(f"  [CPI-mut {name}] L | {zn:30s} = {c:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {v}")
        rows.append(dict(source=name, metric=f"CPI_mut(L | {zn})", stat=round(c, 5), lo=round(lo, 5),
                         hi=round(hi, 5), p_gt0=round(p, 3), n=len(d), n_cx=int(d.complex_id.nunique()),
                         n_pos=int(y.sum()), spearman_L_ddG=round(float(sp), 4)))
        if zn == "burial+nbr+dSASA":
            st, l2, h2 = LD.drop_influential(y, g, Z, d["Lz"].to_numpy(), rng, k=3,
                                             label=f"CPI_mut(L|geom) [{name}]", fixture=name, rows=rows)
    print(f"  [CPI-mut {name}] Spearman(L, ddG) = {sp:+.4f}")
    return prim


def cpi_pos(d, rng, name, rows):
    """Position-level CPI(L(->Ala) | geometry) + L_rms, the apples-to-apples +0.0048 row."""
    need = ["burial", "nbr", "drsasa", "L_ala", "L_rms", "is_hot"]
    d = d.dropna(subset=need).reset_index(drop=True)
    y = d.is_hot.to_numpy().astype(float); g = d.complex_id.to_numpy()
    for c in ["burial", "nbr", "drsasa", "L_ala", "L_rms"]:
        d[c + "z"] = LD.zs(d[c])
    Z = d[["burialz", "nbrz", "drsasaz"]].to_numpy()
    prim = None
    for f, lab in [("L_alaz", "L(->Ala)"), ("L_rmsz", "|L|_rms")]:
        c, lo, hi, p, _, _ = LD.cpi(y, g, Z, d[f].to_numpy().copy(), rng)
        if prim is None:
            prim = (c, lo, hi, p)
        v = "ADDS (CI>0)" if lo > 0 else "conditionally INDEPENDENT (CI spans 0)"
        print(f"  [CPI-pos {name}] {lab:10s} | geom = {c:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {v}")
        rows.append(dict(source=name, metric=f"CPI_pos({lab} | burial+nbr+dSASA)", stat=round(c, 5),
                         lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3), n=len(d),
                         n_cx=int(d.complex_id.nunique()), n_pos=int(y.sum())))
    return prim


# --------------------------------------------------------------------------- crystal PQ gate

def crystal_gate(dm, dp, rows):
    """End-to-end gate: the crystal re-scoring must reproduce the COMMITTED leverage L, per mutation
    (leverage_skempi_mutations.csv) and per position (leverage_skempi_positions.csv). These committed
    CSVs are the ground truth the whole crystal result rests on; the PQ cache that produced them is
    gitignored, so we gate on the L values themselves -- a stronger, end-to-end check."""
    key = ["complex_id", "chain", "resnum", "icode", "wt", "mut"]
    om = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    om["icode"] = om.icode.fillna("").astype(str)
    m = dm.merge(om[key + ["L"]], on=key, suffixes=("_new", "_committed"), how="inner")
    em = float(np.abs(m.L_new - m.L_committed).max()) if len(m) else float("nan")
    op = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    op["icode"] = op.icode.fillna("").astype(str)
    pk = ["complex_id", "chain", "resnum", "icode"]
    p = dp.merge(op[pk + ["L_ala", "L_rms"]], on=pk, suffixes=("_new", "_committed"), how="inner")
    ep = float(np.nanmax([np.abs(p.L_ala_new - p.L_ala_committed).max(),
                          np.abs(p.L_rms_new - p.L_rms_committed).max()])) if len(p) else float("nan")
    print(f"  [+control] crystal re-score reproduces committed L: mutation-level max|ΔL|={em:.2e} "
          f"({len(m)} muts), position-level max|Δ(L_ala,L_rms)|={ep:.2e} ({len(p)} pos)  "
          f"(float32 MPNN precision ~1e-5)", flush=True)
    rows.append(dict(source="crystal", metric="positive_control_reproduces_committed_L_mut",
                     stat=em, n=int(len(m))))
    rows.append(dict(source="crystal", metric="positive_control_reproduces_committed_L_pos",
                     stat=ep, n=int(len(p))))


# --------------------------------------------------------------------------- analyse stage

def stage_analyse(a):
    rng = np.random.default_rng(SEED)
    shared = shared_complexes()
    print(f"[analyse] shared set = {len(shared)} complexes\n", flush=True)
    rows, ranker_rows = [], []
    have = [s for s in ("crystal", "of3", "af2") if os.path.exists(PQ.format(source=s))]
    print(f"[analyse] scored sources present: {have}\n", flush=True)

    mut_frames, pos_frames = {}, {}
    for s in have:
        print(f"\n===== source = {s} =====", flush=True)
        dm, ctrl = build_mut_frame(s, shared)
        print(f"  [map] {s}: mut rows map_rate={ctrl['map_rate']} wt_match={ctrl['wt_match']} "
              f"-> {ctrl['n']} rows / {ctrl['n_cx']} complexes", flush=True)
        rows.append(dict(source=s, metric="mapping_control", stat=ctrl["wt_match"],
                         n=ctrl["n"], n_cx=ctrl["n_cx"], note=f"map_rate={ctrl['map_rate']}"))
        dp = build_pos_frame(s, shared)
        mut_frames[s], pos_frames[s] = dm, dp
        if s == "crystal":
            crystal_gate(dm, dp, rows)
        cpi_mut(dm, rng, s, rows)
        cpi_pos(dp, rng, s, rows)
        combined_ranker(dp, rng, s, ranker_rows)
        dm.to_csv(f"results/leverage_predicted_{s}_mutations.csv", index=False,
                  columns=[c for c in dm.columns if not c.startswith(("lP_", "lQ_"))])
        dp.to_csv(f"results/leverage_predicted_{s}_positions.csv", index=False,
                  columns=[c for c in dp.columns if not c.startswith(("lP_", "lQ_"))])
        gc.collect()

    # pooled predicted (OF3 + AF2 stacked; bootstrap groups on complex so a complex's two backbones
    # cluster together -- respects that the two predictors of one complex are not independent).
    pred = [s for s in ("of3", "af2") if s in have]
    if len(pred) == 2:
        print(f"\n===== source = pooled ({'+'.join(pred)}) =====", flush=True)
        dm = pd.concat([mut_frames[s].assign(complex_id=mut_frames[s].complex_id)
                        for s in pred], ignore_index=True)
        dp = pd.concat([pos_frames[s] for s in pred], ignore_index=True)
        cpi_mut(dm, rng, "pooled_of3_af2", rows)
        cpi_pos(dp, rng, "pooled_of3_af2", rows)
        combined_ranker(dp, rng, "pooled_of3_af2", ranker_rows)

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out["shared_n_cx"] = len(shared)
    out["command"] = "python3 " + " ".join(sys.argv)
    out.to_csv(a.out, index=False)
    rk = pd.DataFrame(ranker_rows); rk["seed"] = SEED
    rk["command"] = "python3 " + " ".join(sys.argv)
    rk.to_csv(a.out.replace(".csv", "_ranker.csv"), index=False)
    print(f"\n[done] wrote {a.out} ({len(out)} rows) and {a.out.replace('.csv','_ranker.csv')} "
          f"({len(rk)} rows)")

    # headline
    key = out[out.metric == "CPI_mut(L | burial+nbr+dSASA)"]
    print("\n[summary] CPI_mut(L | geometry):")
    for r in key.itertuples():
        print(f"    {r.source:16s} = {r.stat:+.5f} [{r.lo:+.5f},{r.hi:+.5f}] "
              f"P(>0)={r.p_gt0:.3f}  Spearman(L,ddG)={r.spearman_L_ddG:+.4f}  (n={r.n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["score", "analyse"])
    ap.add_argument("--source", choices=list(SOURCES), help="required for --stage score")
    ap.add_argument("--out", default="results/leverage_predicted.csv")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="score only first N complexes (smoke test)")
    a = ap.parse_args()
    if a.stage == "score":
        if not a.source:
            ap.error("--stage score requires --source")
        stage_score(a)
    else:
        stage_analyse(a)


if __name__ == "__main__":
    main()

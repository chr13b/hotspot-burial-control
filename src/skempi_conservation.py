#!/usr/bin/env python3
"""Ceiling-raiser (audit W3): does the mixed derivative (leverage L) add binding signal beyond EVOLUTIONARY
CONSERVATION + geometry — the STANDARD hotspot-predictor feature set — not just beyond cheap geometry?

Score every SKEMPI interface position's sequence conservation with ESM-2 (150M) negentropy (reusing the
catalytic pipeline), join to the committed leverage positions, and run the project CPI estimator:
  CPI(conservation | geometry)              -- does conservation itself find hotspots? (the reviewer's baseline)
  CPI(L | geometry)                          -- the paper's number (reference)
  CPI(L | geometry + conservation)           -- THE headline: does L survive conservation?
  CPI(conservation | geometry + L)           -- the reverse

  python3 src/skempi_conservation.py --stage score    --out results/skempi_conservation_positions.csv
  python3 src/skempi_conservation.py --stage analyse  --out results/skempi_conservation_positions.csv
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ftax_common as fc
import leverage_decomposition as LD
from catalytic_dissociation import esm2_entropy          # reuse the verified per-position negentropy
AA20 = LD.AA20
SEED = 20260803
DATA = LD.DATA
POS = "results/leverage_skempi_positions.csv"


def stage_score(a):
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    import esm
    model_e, alphabet = esm.pretrained.esm2_t30_150M_UR50D(); model_e.eval()
    bc = alphabet.get_batch_converter(); aa_idx = [alphabet.get_idx(x) for x in AA20]
    print("loaded ESM-2 (150M)", flush=True)

    cids = sorted(pd.read_csv(POS, usecols=["complex_id"]).complex_id.unique())   # committed (pq is gitignored)
    if a.limit: cids = cids[:a.limit]
    done = set()
    if os.path.exists(a.out) and not a.overwrite:
        try: done = set(pd.read_csv(a.out, usecols=["complex_id"]).complex_id)
        except Exception: done = set()
    fh = open(a.out, "a" if (done and not a.overwrite) else "w", newline="")
    cols = ["complex_id", "chain", "resnum", "icode", "aa", "esm_negent", "esm_logp_native"]
    if not (done and not a.overwrite): fh.write(",".join(cols) + "\n"); fh.flush()
    t0, nrow, nskip = time.time(), 0, 0
    for ci, cid in enumerate(cids):
        if cid in done: continue
        pdb, g1, g2 = cid.split("_"); path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path): continue
        try:
            for chain in sorted(set(g1 + g2)):
                cx = fc.load_complex(path, pdb, chain, "", require_both=False)
                if cx is None or cx.n < 4 or cx.n > 1022:       # recover short peptides (MHC/inhibitors); ESM-2 context limit
                    nskip += 1; continue
                negent, lpn = esm2_entropy(model_e, alphabet, bc, "".join(cx.seq), aa_idx)
                for i in range(cx.n):
                    fh.write(",".join(str(x) for x in [cid, cx.chains[i], int(cx.resnums[i]),
                             str(cx.icodes[i]), cx.seq[i], round(float(negent[i]), 5),
                             round(float(lpn[i]), 5)]) + "\n"); nrow += 1
            fh.flush()
            if (ci + 1) % 20 == 0:
                print(f"  [{ci+1}/{len(cids)}] {nrow} rows  {(ci+1)/max(time.time()-t0,1e-9):.2f} cplx/s", flush=True)
        except Exception as e:
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
    fh.close(); print(f"[score] wrote {nrow} rows ({nskip} chains skipped) -> {a.out}", flush=True)


def stage_analyse(a):
    cons = pd.read_csv(a.out)
    cons["icode"] = cons.icode.fillna("").astype(str)
    pos = pd.read_csv(POS); pos["icode"] = pos.icode.fillna("").astype(str)
    d = pos.merge(cons[["complex_id", "chain", "resnum", "icode", "esm_negent"]],
                  on=["complex_id", "chain", "resnum", "icode"], how="inner")
    d = d[d.is_interface == True].copy()                          # noqa: E712
    d = d[d.esm_negent.notna() & d.L_ala.notna()]
    # Exclude positions on chains too short for a meaningful PLM conservation estimate (MHC peptides, short
    # inhibitors: their ESM-2 negentropy is a systematic outlier, mean -1.64 vs -0.85, because the model has
    # no context to use). This is a PRE-OUTCOME structural exclusion (chain length); those positions keep
    # valid ProteinMPNN leverage, reported separately below so the conservation conclusion is not sensitive to it.
    clen = cons.groupby(["complex_id", "chain"]).size().rename("clen").reset_index()
    d = d.merge(clen, on=["complex_id", "chain"], how="left")
    short = d[d.clen < 10]
    d = d[d.clen >= 10].copy()
    print(f"[analyse] {len(d)} interface positions matched, {d.complex_id.nunique()} complexes, "
          f"{int(d.is_hot.sum())} hotspots  (excluded {len(short)} positions / {int(short.is_hot.sum())} "
          f"hotspots on <10-res chains from the conservation set)")
    rng = np.random.default_rng(SEED)
    y = d.is_hot.astype(int).to_numpy(); g = d.complex_id.to_numpy()
    def zc(v): v = np.asarray(v, float); return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    GEO = np.column_stack([zc(d.burial), zc(d.nbr), zc(d.drsasa)])   # z-scored to match position_level_cpi (cpi() is not scale-invariant)
    cons_x = zc(d.esm_negent)
    L_x = zc(-d.L_ala)                                            # -L(->Ala): higher = more hotspot-like
    rows = []
    def run(name, Z, X):
        c, lo, hi, p, _, _ = LD.cpi(y, g, Z, X.copy(), rng)
        v = "ADDS" if lo > 0 else "conditionally independent"
        print(f"  CPI({name}) = {c:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {v}")
        rows.append(dict(test=f"CPI({name})", stat=round(c, 5), lo=round(lo, 5), hi=round(hi, 5),
                         p_gt0=round(p, 3), n=len(y), n_complex=int(d.complex_id.nunique())))
    run("conservation | geometry", GEO, cons_x)                              # reviewer's baseline: does it predict?
    run("leverage -L | geometry", GEO, L_x)                                  # the paper's reference
    run("leverage -L | geometry + conservation", np.column_stack([GEO, cons_x]), L_x)  # THE headline
    run("conservation | geometry + leverage", np.column_stack([GEO, L_x]), cons_x)     # the reverse
    # drop-3-influential robustness on the headline
    Zh = np.column_stack([GEO, cons_x])
    c, lo, hi, p, _, cvec = LD.cpi(y, g, Zh, L_x.copy(), np.random.default_rng(SEED))
    contrib = pd.Series(cvec).groupby(pd.Series(g)).sum().sort_values(ascending=False)
    drop = set(contrib.index[:3]); keep = ~pd.Series(g).isin(drop).to_numpy()
    c2, lo2, hi2, p2, _, _ = LD.cpi(y[keep], g[keep], Zh[keep], L_x[keep].copy(), np.random.default_rng(SEED))
    surv = "SURVIVES" if lo2 > 0 else "does not survive"
    print(f"  [robustness] headline drop-3 {sorted(drop)}: {c2:+.5f} [{lo2:+.5f},{hi2:+.5f}]  {surv}")
    rows.append(dict(test="CPI(leverage -L | geometry + conservation) drop-3", stat=round(c2, 5),
                     lo=round(lo2, 5), hi=round(hi2, 5), p_gt0=round(p2, 3), n=int(keep.sum()),
                     n_complex=int(pd.Series(g[keep]).nunique())))
    from scipy import stats as st
    print(f"  Spearman(conservation, -L) = {st.spearmanr(cons_x, L_x).correlation:+.3f} "
          f"(are they measuring the same thing?)")
    # concentration disclosure (audit G): how much of the headline rests on a few complexes?
    tot = contrib.sum()
    print(f"  [concentration] headline CPI: top-1 complex = {100*contrib.iloc[0]/tot:.0f}%, "
          f"top-3 = {100*contrib.iloc[:3].sum()/tot:.0f}%, top-10 = {100*contrib.iloc[:10].sum()/tot:.0f}% "
          f"({int((contrib<0).sum())}/{len(contrib)} complexes contribute negatively)")

    # ACTIONABLE ranker (audit fruit #1): does |L| add on top of geometry + CONSERVATION (the standard set)?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    def oof(X):
        eta = np.zeros(len(y)); nf = int(min(5, len(np.unique(g))))
        for tr, te in GroupKFold(nf).split(X, y, g):
            m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]); eta[te] = X[te] @ m.coef_[0] + m.intercept_[0]
        return eta
    def auroc(s, yy):
        r = st.rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1
        return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    Lrms = zc(d.L_rms)
    base = np.column_stack([GEO, cons_x])                        # geometry + conservation = the standard feature set
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    sb = oof(base); print(f"  [ranker] geometry+conservation AUROC = {auroc(sb, y):.4f}")
    for lab, add in [("+|L|_rms", Lrms), ("+(-L_ala)", L_x)]:
        s = oof(np.column_stack([base, add])); dpt = auroc(s, y) - auroc(sb, y)
        boot = []
        for _ in range(3000):
            ix = np.concatenate([by[c] for c in rng.choice(ids, len(ids), True)])
            boot.append(auroc(s[ix], y[ix]) - auroc(sb[ix], y[ix]))
        boot = np.array(boot)
        lo3, hi3, p3 = np.percentile(boot, 2.5), np.percentile(boot, 97.5), float(np.mean(boot > 0))
        print(f"  [ranker] geometry+conservation {lab}: AUROC={auroc(s, y):.4f}  Δ={dpt:+.4f} [{lo3:+.4f},{hi3:+.4f}] P(>0)={p3:.3f}")
        rows.append(dict(test=f"ranker geom+conservation {lab} vs geom+conservation", stat=round(dpt, 4),
                         lo=round(lo3, 4), hi=round(hi3, 4), p_gt0=round(p3, 3), n=len(y),
                         n_complex=int(d.complex_id.nunique())))

    pd.DataFrame(rows).to_csv(a.out.replace("_positions", ""), index=False)
    print(f"[analyse] wrote {a.out.replace('_positions','')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyse"], required=True)
    ap.add_argument("--out", default="results/skempi_conservation_positions.csv")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    (stage_score if a.stage == "score" else stage_analyse)(a)


if __name__ == "__main__":
    main()

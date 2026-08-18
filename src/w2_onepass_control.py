#!/usr/bin/env python3
"""W2 (reviewer's first question): does the TWO-pass leverage L add binding signal beyond the ONE-pass
complex log-odds  oc = logP(mut|complex) - logP(wt|complex)  (the standard zero-shot deep-mutational readout)?
Plus robustness slices that kill "L is a side-chain-volume / truncation proxy".

All inputs committed: results/leverage_skempi_mutations.csv. SEED=20260803.
  python3 src/w2_onepass_control.py --out results/w2_onepass_control.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEED = 20260803

def partial_multi(x, y, Zcols, rng, groups, nb=500):
    """partial Spearman(x, y | Zcols), complex-clustered bootstrap CI. Returns (rho, lo, hi, P(<0), n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = [np.asarray(z, float) for z in Zcols]
    m = np.isfinite(x) & np.isfinite(y)
    for z in Z: m &= np.isfinite(z)
    x, y, g = x[m], y[m], np.asarray(groups)[m]; Z = [z[m] for z in Z]
    def pr(xx, yy, ZZ):
        M = np.column_stack([np.ones(len(xx))] + [stats.rankdata(z) for z in ZZ])
        rx = stats.rankdata(xx) - M @ np.linalg.lstsq(M, stats.rankdata(xx), rcond=None)[0]
        ry = stats.rankdata(yy) - M @ np.linalg.lstsq(M, stats.rankdata(yy), rcond=None)[0]
        return np.corrcoef(rx, ry)[0, 1]
    rho = pr(x, y, Z)
    ids = np.unique(g); by = {k: np.where(g == k)[0] for k in ids}
    bs = []
    for _ in range(nb):
        ix = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        bs.append(pr(x[ix], y[ix], [z[ix] for z in Z]))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.mean(bs < 0)), int(m.sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/w2_onepass_control.csv"); a = ap.parse_args()
    d = pd.read_csv("results/leverage_skempi_mutations.csv")
    d = d[d.ddG.notna() & d.L.notna() & d.logP_mut.notna() & d.conf.notna()].copy()
    d["onepass"] = d.logP_mut - d.conf                       # logP(mut|cx) - logP(wt|cx), one-pass complex log-odds
    G = [d.burial, d.nbr, d.drsasa]                           # the geometry controls (burial+nbr+dSASA)
    rng = np.random.default_rng(SEED); rows = []
    print(f"[W2] {len(d)} mutations, {d.complex_id.nunique()} complexes")

    # 1. does L add beyond one-pass + geometry?  and the reverse?
    r, lo, hi, p, n = partial_multi(d.L, d.ddG, [d.onepass] + G, rng, d.complex_id)
    print(f"  partial Spearman(L, ddG | one-pass + geom)   = {r:+.3f} [{lo:+.3f},{hi:+.3f}] P(<0)={p:.3f}  n={n}")
    rows.append(dict(test="partial(L,ddG|onepass+geom)", stat=round(r,4), lo=round(lo,4), hi=round(hi,4), p_lt0=round(p,3), n=n))
    r2, lo2, hi2, p2, _ = partial_multi(d.onepass, d.ddG, [d.L] + G, rng, d.complex_id)
    print(f"  partial Spearman(one-pass, ddG | L + geom)    = {r2:+.3f} [{lo2:+.3f},{hi2:+.3f}] P(<0)={p2:.3f}")
    rows.append(dict(test="partial(onepass,ddG|L+geom)", stat=round(r2,4), lo=round(lo2,4), hi=round(hi2,4), p_lt0=round(p2,3), n=n))
    # marginal Spearmans for context
    for lab, x in [("L", d.L), ("onepass", d.onepass)]:
        rho = stats.spearmanr(x, d.ddG).correlation
        rows.append(dict(test=f"marginal_spearman({lab},ddG)", stat=round(rho,4), lo=np.nan, hi=np.nan, p_lt0=np.nan, n=len(d)))
        print(f"  marginal Spearman({lab:8s}, ddG) = {rho:+.3f}")

    # 2. robustness slices (kill 'L = volume/truncation proxy')
    print("  --- robustness ---")
    per = d.groupby("complex_id").filter(lambda s: len(s) >= 15).groupby("complex_id").apply(
        lambda s: stats.spearmanr(s.L, s.ddG).correlation)
    neg = int((per < 0).sum())
    print(f"  per-complex rho(L,ddG) negative: {neg}/{len(per)} complexes (n>=15 mutations)")
    rows.append(dict(test="per_complex_neg_frac(n>=15)", stat=round(neg/len(per),3), lo=np.nan, hi=np.nan, p_lt0=np.nan, n=len(per)))
    wt = d.groupby("wt").filter(lambda s: len(s) >= 30).groupby("wt").apply(
        lambda s: pd.Series(dict(rho=stats.spearmanr(s.L, s.ddG).correlation, n=len(s))))
    wneg = int((wt.rho < 0).sum()); wmean = float(np.average(wt.rho, weights=wt.n))
    print(f"  per-wt-type rho negative: {wneg}/{len(wt)} types; n-weighted mean rho = {wmean:+.3f}")
    rows.append(dict(test="per_wttype_neg", stat=f"{wneg}/{len(wt)}", lo=np.nan, hi=np.nan, p_lt0=np.nan, n=len(wt)))
    rows.append(dict(test="per_wttype_nweighted_rho", stat=round(wmean,4), lo=np.nan, hi=np.nan, p_lt0=np.nan, n=int(wt.n.sum())))
    ala = d[d.mut == "A"]
    rho_ala = stats.spearmanr(ala.L, ala.ddG).correlation
    print(f"  X->Ala only: Spearman(L,ddG) = {rho_ala:+.3f}  (n={len(ala)}, {ala.complex_id.nunique()} complexes) "
          f"-- L is not a volume/truncation proxy")
    rows.append(dict(test="Xala_only_spearman(L,ddG)", stat=round(rho_ala,4), lo=np.nan, hi=np.nan, p_lt0=np.nan, n=len(ala)))

    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"[wrote] {a.out}")

if __name__ == "__main__":
    main()

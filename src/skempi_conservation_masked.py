#!/usr/bin/env python3
"""Ceiling-raiser #3 (bulletproof the conservation control): the FIELD-STANDARD PLM conservation estimator is
the MASKED marginal (mask a position, read the model's distribution there), not the unmasked forward we used.
Score ESM-2 (150M) masked-marginal negentropy at every SKEMPI interface position, then re-run the conservation
CPI: does leverage still add beyond geometry + *masked* conservation?

  python3 src/skempi_conservation_masked.py --stage score    --out results/skempi_conservation_masked.csv
  python3 src/skempi_conservation_masked.py --stage analyse  --out results/skempi_conservation_masked.csv
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ftax_common as fc
import leverage_decomposition as LD
AA20 = LD.AA20
SEED = 20260803
DATA = LD.DATA
POS = "results/leverage_skempi_positions.csv"


def stage_score(a):
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    import esm
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D(); model.eval()
    bc = alphabet.get_batch_converter(); aa_idx = [alphabet.get_idx(x) for x in AA20]
    mask_i = alphabet.mask_idx
    print("loaded ESM-2 (150M) for masked marginals", flush=True)

    pos = pd.read_csv(POS); pos["icode"] = pos.icode.fillna("").astype(str)
    pos = pos[pos.is_interface == True]                                    # noqa: E712  only interface positions needed
    cids = sorted(pos.complex_id.unique())
    if a.limit: cids = cids[:a.limit]
    done = set()
    if os.path.exists(a.out) and not a.overwrite:
        try: done = set(pd.read_csv(a.out).complex_id.unique())
        except Exception: done = set()
    fh = open(a.out, "a" if (done and not a.overwrite) else "w", newline="")
    if not (done and not a.overwrite): fh.write("complex_id,chain,resnum,icode,masked_negent\n"); fh.flush()
    t0, nrow = time.time(), 0
    for ci, cid in enumerate(cids):
        if cid in done: continue
        pdb, g1, g2 = cid.split("_"); path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path): continue
        sub = pos[pos.complex_id == cid]
        try:
            for chain in sorted(sub.chain.unique()):
                cx = fc.load_complex(path, pdb, chain, "", require_both=False)
                if cx is None or cx.n < 10 or cx.n > 1022:                 # meaningful-conservation set (matches unmasked analyse)
                    continue
                seq = "".join(cx.seq)
                cmap = {(int(cx.resnums[j]), str(cx.icodes[j])): j for j in range(cx.n)}
                want = [(int(r.resnum), r.icode, cmap.get((int(r.resnum), r.icode)))
                        for r in sub[sub.chain == chain].itertuples()]
                want = [(rn, ic, j) for (rn, ic, j) in want if j is not None]
                if not want:
                    continue
                _, _, toks = bc([("s", seq)])                              # [1, L+2]
                base = toks[0].clone()
                # one masked forward per interface position, in mini-batches
                for s in range(0, len(want), a.batch):
                    chunk = want[s:s + a.batch]
                    batch = base.repeat(len(chunk), 1).clone()
                    for bi, (_, _, j) in enumerate(chunk):
                        batch[bi, j + 1] = mask_i                          # +1 for BOS
                    with torch.no_grad():
                        lg = model(batch)["logits"]                        # [b, L+2, vocab]
                    for bi, (rn, ic, j) in enumerate(chunk):
                        row = torch.log_softmax(lg[bi, j + 1, aa_idx], dim=0).numpy()
                        negent = float((np.exp(row) * row).sum())          # -H over the 20 aa (higher = more conserved)
                        fh.write(f"{cid},{chain},{rn},{ic},{round(negent,5)}\n"); nrow += 1
            fh.flush()
            if (ci + 1) % 20 == 0:
                print(f"  [{ci+1}/{len(cids)}] {nrow} positions  {(ci+1)/max(time.time()-t0,1e-9):.2f} cplx/s", flush=True)
        except Exception as e:
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
    fh.close(); print(f"[score] wrote {nrow} masked-marginal positions -> {a.out}", flush=True)


def stage_analyse(a):
    m = pd.read_csv(a.out); m["icode"] = m.icode.fillna("").astype(str)
    pos = pd.read_csv(POS); pos["icode"] = pos.icode.fillna("").astype(str)
    d = pos.merge(m, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    d = d[(d.is_interface == True) & d.masked_negent.notna() & d.L_ala.notna()].copy()   # noqa: E712
    print(f"[analyse] {len(d)} interface positions, {d.complex_id.nunique()} complexes, {int(d.is_hot.sum())} hotspots")
    rng = np.random.default_rng(SEED)
    y = d.is_hot.astype(int).to_numpy(); g = d.complex_id.to_numpy()
    def zc(v): v = np.asarray(v, float); return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    GEO = np.column_stack([zc(d.burial), zc(d.nbr), zc(d.drsasa)])
    cons = zc(d.masked_negent); L_x = zc(-d.L_ala)
    rows = []
    def run(name, Z, X):
        c, lo, hi, p, _, _ = LD.cpi(y, g, Z, X.copy(), rng)
        print(f"  CPI({name}) = {c:+.5f} [{lo:+.5f},{hi:+.5f}] P(>0)={p:.3f}  {'ADDS' if lo>0 else 'cond. indep.'}")
        rows.append(dict(test=f"CPI({name})", stat=round(c, 5), lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3),
                         n=len(y), n_complex=int(d.complex_id.nunique())))
    run("masked-conservation | geometry", GEO, cons)
    run("leverage -L | geometry", GEO, L_x)
    run("leverage -L | geometry + masked-conservation", np.column_stack([GEO, cons]), L_x)   # THE headline
    run("masked-conservation | geometry + leverage", np.column_stack([GEO, L_x]), cons)
    # drop-3-influential robustness + top-3 concentration on the masked headline (matches the unmasked template)
    Zh = np.column_stack([GEO, cons])
    _c, _lo, _hi, _p, _, cvec = LD.cpi(y, g, Zh, L_x.copy(), np.random.default_rng(SEED))
    contrib = pd.Series(cvec).groupby(pd.Series(g)).sum().sort_values(ascending=False)
    drop = set(contrib.index[:3]); keep = ~pd.Series(g).isin(drop).to_numpy()
    c2, lo2, hi2, p2, _, _ = LD.cpi(y[keep], g[keep], Zh[keep], L_x[keep].copy(), np.random.default_rng(SEED))
    tot = contrib.sum(); top3 = 100 * contrib.iloc[:3].sum() / tot
    print(f"  [robustness] masked headline drop-3 {sorted(drop)}: {c2:+.5f} [{lo2:+.5f},{hi2:+.5f}] "
          f"{'SURVIVES' if lo2 > 0 else 'does not survive'}; top-3 complexes = {top3:.0f}%")
    rows.append(dict(test="CPI(leverage -L | geometry + masked-conservation) drop-3", stat=round(c2, 5),
                     lo=round(lo2, 5), hi=round(hi2, 5), p_gt0=round(p2, 3), n=int(keep.sum()),
                     n_complex=int(pd.Series(g[keep]).nunique())))
    rows.append(dict(test="masked headline top-3 complex concentration pct", stat=round(top3, 1),
                     lo=None, hi=None, p_gt0=None, n=len(y), n_complex=int(d.complex_id.nunique())))
    from scipy import stats as st
    # is masked a better conservation signal than unmasked? correlate with the unmasked column if present
    u = pd.read_csv("results/skempi_conservation_positions.csv")[["complex_id", "chain", "resnum", "icode", "esm_negent"]]
    u["icode"] = u.icode.fillna("").astype(str)
    dd = d.merge(u, on=["complex_id", "chain", "resnum", "icode"], how="left")
    print(f"  Spearman(masked, unmasked) = {st.spearmanr(dd.masked_negent, dd.esm_negent, nan_policy='omit').correlation:+.3f}")
    print(f"  Spearman(masked-conservation, -L) = {st.spearmanr(cons, L_x).correlation:+.3f}")
    pd.DataFrame(rows).to_csv(a.out.replace(".csv", "_cpi.csv"), index=False)
    print(f"[analyse] wrote {a.out.replace('.csv','_cpi.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyse"], required=True)
    ap.add_argument("--out", default="results/skempi_conservation_masked.csv")
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    (stage_score if a.stage == "score" else stage_analyse)(a)


if __name__ == "__main__":
    main()

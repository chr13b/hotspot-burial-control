#!/usr/bin/env python3
"""R3 — ΔSASA-adjusted matched-pair sensitivity (pre-empt "you matched on nothing partner-dependent").

The pre-registered matched-pair design controls self-burial (rSASA), secondary structure, neighbour count
— all SELF-geometry. It does NOT match on ΔSASA (partner-contact area), the one partner-dependent axis the
paper's thesis is about. Hotspots carry more partner-contact than their matched controls; this quantifies
that imbalance and reports the ΔSASA-ADJUSTED deficit (regress the paired log-prob difference on the paired
ΔSASA difference, take the intercept). On CRYSTAL the null must survive; the same check on PREDICTED
backbones is R1 (Sherlock), where it is load-bearing.

Complex-level bootstrap, seed 20260803. Committed CSVs only.
  python3 src/dsasa_matched_sens.py --out results/dsasa_matched_sens.csv
"""
import argparse, glob, os
import numpy as np, pandas as pd

SEED = 20260803


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dsasa_matched_sens.csv")
    a = ap.parse_args()
    R = "results"
    pos = pd.read_csv(f"{R}/p0_positions.csv", usecols=["complex_id", "chain", "resnum", "drsasa"])
    dl = {(r.complex_id, str(r.chain), str(r.resnum)): r.drsasa for r in pos.itertuples()}

    files = ["p0_dssp_pairs_SECONDARY_B_any_interface.csv", "p0_dssp_pairs_AAMATCHED_any_interface.csv",
             "p0_dssp_pairs_PRIMARY_loose_null.csv"]
    rows = []
    for fn in files:
        path = f"{R}/{fn}"
        if not os.path.exists(path):
            continue
        pr = pd.read_csv(path)
        if "d_logp" not in pr.columns:
            continue
        pr["pdb"] = pr.complex_id.str.split("_").str[0]
        d_dsasa = []
        for r in pr.itertuples():
            h = dl.get((r.complex_id, str(r.hot_chain), str(r.hot_resnum)))
            c = dl.get((r.complex_id, str(r.ctl_chain), str(r.ctl_resnum)))
            d_dsasa.append(h - c if (h is not None and c is not None) else np.nan)
        pr["d_dsasa"] = d_dsasa
        pr = pr.dropna(subset=["d_logp", "d_dsasa"])
        if len(pr) < 20:
            continue

        rng = np.random.default_rng(SEED)
        cids = pr.complex_id.unique()
        by = {cc: pr[pr.complex_id == cc] for cc in cids}

        def stat(sub):
            x = sub.d_dsasa.values; ylp = sub.d_logp.values
            raw = ylp.mean()
            # OLS intercept of d_logp ~ d_dsasa (deficit adjusted to d_dsasa = 0)
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, ylp, rcond=None)[0]
            return raw, beta[0], x.mean()

        raw0, adj0, imb0 = stat(pr)
        raws, adjs, imbs = [], [], []
        for _ in range(5000):
            sub = pd.concat([by[c] for c in rng.choice(cids, len(cids), True)])
            r_, a_, i_ = stat(sub); raws.append(r_); adjs.append(a_); imbs.append(i_)
        def ci(v): return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        rlo, rhi = ci(raws); alo, ahi = ci(adjs); ilo, ihi = ci(imbs)
        print(f"[{fn.replace('p0_dssp_pairs_','').replace('.csv','')}] n={len(pr)} pairs, {len(cids)} cx")
        print(f"   ΔSASA imbalance (hot−ctl)  = {imb0:+.3f} [{ilo:+.3f},{ihi:+.3f}]  (matched d_rsasa≈0 by design)")
        print(f"   raw matched deficit d_logp = {raw0:+.3f} [{rlo:+.3f},{rhi:+.3f}]")
        print(f"   ΔSASA-ADJUSTED deficit     = {adj0:+.3f} [{alo:+.3f},{ahi:+.3f}]  <- survives if CI spans 0")
        rows.append(dict(pairs_file=fn, n_pairs=len(pr), n_cx=len(cids),
                         dsasa_imbalance=round(imb0, 4), imb_lo=round(ilo, 4), imb_hi=round(ihi, 4),
                         raw_deficit=round(raw0, 4), raw_lo=round(rlo, 4), raw_hi=round(rhi, 4),
                         dsasa_adjusted_deficit=round(adj0, 4), adj_lo=round(alo, 4), adj_hi=round(ahi, 4)))

    out = pd.DataFrame(rows); out["seed"] = SEED
    out["command"] = "python3 src/dsasa_matched_sens.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

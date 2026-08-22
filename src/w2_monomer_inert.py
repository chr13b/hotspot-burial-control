"""W2 strengthener, now with a committed source (audit rule 4). The two-pass argument's 'cleanest statement':
the MONOMER pass on its own is inert (Spearman(monomer log-odds, ddG) ~ 0), and the one-pass complex readout
and the full leverage are genuinely distinct quantities (their correlation, and how much variance the monomer
subtraction removes). Computed per interface mutation from the committed P/Q matrices.

  oc = logP(mut|complex) - logP(wt|complex)   (one-pass complex log-odds, the standard zero-shot readout)
  om = logP(mut|monomer) - logP(wt|monomer)   (monomer log-odds)
  L  = oc - om                                (leverage; monomer pass subtracted)

  python3 src/w2_monomer_inert.py --out results/w2_monomer_inert.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
AA20 = "ACDEFGHIKLMNPQRSTVWY"; IDX = {a: i for i, a in enumerate(AA20)}; SEED = 20260803


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/w2_monomer_inert.csv"); a = ap.parse_args()
    pq = pd.read_csv("results/leverage_pq_skempi.csv", low_memory=False)
    pq["icode"] = pq.icode.fillna("").astype(str)
    mut = pd.read_csv("results/leverage_skempi_mutations.csv", low_memory=False)
    if "is_interface" in mut.columns:
        mut = mut[mut.is_interface == 1]
    mut = mut.dropna(subset=["ddG", "wt", "mut"]).copy()
    mut["icode"] = mut.icode.fillna("").astype(str)
    key = ["complex_id", "chain", "resnum", "icode"]
    lP = {a_: pq.set_index(key)[f"lP_{a_}"] for a_ in AA20}
    lQ = {a_: pq.set_index(key)[f"lQ_{a_}"] for a_ in AA20}
    idxdf = pq.set_index(key)
    rows_oc, rows_om, rows_L, rows_g = [], [], [], []
    for r in mut.itertuples():
        k = (r.complex_id, r.chain, int(r.resnum), r.icode)
        if k not in idxdf.index or r.wt not in IDX or r.mut not in IDX:
            continue
        try:
            lp = idxdf.loc[k];
        except Exception:
            continue
        if isinstance(lp, pd.DataFrame):
            lp = lp.iloc[0]
        oc = lp[f"lP_{r.mut}"] - lp[f"lP_{r.wt}"]
        om = lp[f"lQ_{r.mut}"] - lp[f"lQ_{r.wt}"]
        if not (np.isfinite(oc) and np.isfinite(om)):
            continue
        rows_oc.append(oc); rows_om.append(om); rows_L.append(oc - om); rows_g.append(r.ddG)
    oc = np.array(rows_oc); om = np.array(rows_om); L = np.array(rows_L); g = np.array(rows_g)
    n = len(g)
    sp_om = stats.spearmanr(om, g).correlation
    sp_oc = stats.spearmanr(oc, g).correlation
    sp_L = stats.spearmanr(L, g).correlation
    corr_oc_L = float(np.corrcoef(oc, L)[0, 1])          # Pearson corr(one-pass, leverage)
    var_removed = 1.0 - np.var(L) / np.var(oc)           # fraction of one-pass variance removed by monomer term
    print(f"n={n}")
    print(f"  Spearman(monomer log-odds om, ddG) = {sp_om:+.3f}   <- 'monomer pass inert'")
    print(f"  Spearman(one-pass oc, ddG)         = {sp_oc:+.3f}")
    print(f"  Spearman(leverage L, ddG)          = {sp_L:+.3f}")
    print(f"  corr(one-pass oc, leverage L)      = {corr_oc_L:+.3f}")
    print(f"  var removed by monomer subtraction = {var_removed:+.3f}  (1 - var(L)/var(oc))")
    out = pd.DataFrame([
        dict(stat="spearman_monomer_om_vs_ddG", value=round(float(sp_om), 4), n=n),
        dict(stat="spearman_onepass_oc_vs_ddG", value=round(float(sp_oc), 4), n=n),
        dict(stat="spearman_leverage_L_vs_ddG", value=round(float(sp_L), 4), n=n),
        dict(stat="pearson_corr_onepass_leverage", value=round(corr_oc_L, 4), n=n),
        dict(stat="frac_onepass_variance_removed_by_monomer", value=round(float(var_removed), 4), n=n),
    ]); out["seed"] = SEED; out["command"] = "python3 src/w2_monomer_inert.py"
    out.to_csv(a.out, index=False); print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

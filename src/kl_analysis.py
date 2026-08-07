"""The full C5 analysis: is the residue-agnostic KL detector a real, design-relevant signal?

Consumes results/kl_detector_joined.csv (KL/JSD/dH per interface position, sequence-free),
results/frustration_monomer_joined.csv (d_bind_local, residue-AWARE), and computes
inter-chain contact counts from structure, then answers four questions with PAIRED
complex-level bootstraps (same resamples for both scores, so the delta CI is honest):

  Q1  does KL ADD to the burial baseline?                  (burial+KL vs burial)
  Q2  is KL just a contact count?                          (KL vs inter-chain contacts)
  Q3  does removing the sequence cost anything?            (burial+KL vs burial+d_bind_local)
  Q4  the metric a designer uses: within-complex ranking   (per-complex AUROC + precision@k)

and a fully backbone-only variant (neighbour count instead of rSASA) for the design-time claim.

Usage:
  python3 src/kl_analysis.py --out results/kl_analysis
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

SEED = 20260803
N_BOOT = 2000


def auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    r = stats.rankdata(s)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def rankavg(df, cols):
    return np.mean([stats.rankdata(df[c]) / len(df) for c in cols], axis=0)


def paired_auc(df, score_a, score_b, label="is_hot", n_boot=N_BOOT, seed=SEED):
    """Point AUROCs and paired-difference CI, resampling complexes."""
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df[df["complex_id"] == c] for c in cids}
    pa = auc(df[score_a].values, df[label].values)
    pb = auc(df[score_b].values, df[label].values)
    da, db, dd = [], [], []
    for _ in range(n_boot):
        d = pd.concat([by[cids[i]] for i in rng.choice(len(cids), len(cids), True)],
                      ignore_index=True)
        a, b = auc(d[score_a].values, d[label].values), auc(d[score_b].values, d[label].values)
        da.append(a); db.append(b); dd.append(a - b)
    q = lambda x: np.nanpercentile(x, [2.5, 97.5])
    return dict(a=pa, a_ci=q(da), b=pb, b_ci=q(db),
                delta=pa - pb, delta_ci=q(dd), p_gt0=float(np.mean(np.array(dd) > 0)))


def contact_counts(data_dir, complexes, cutoff=10.0):
    """Inter-chain CB contacts within `cutoff` and min cross-chain distance, per position."""
    out = {}
    for cid in complexes:
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None:
            continue
        g = cx.group
        d = np.linalg.norm(cx.CB[:, None, :] - cx.CB[None, :, :], axis=-1)
        cross = g[:, None] != g[None, :]
        for i in range(cx.n):
            dc = d[i][cross[i]]
            out[(cid, cx.chains[i], int(cx.resnums[i]))] = (
                int((dc < cutoff).sum()), float(dc.min()) if len(dc) else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/kl_analysis")
    ap.add_argument("--kl", default="results/kl_detector_joined.csv")
    ap.add_argument("--dbind", default="results/frustration_monomer_joined.csv")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    m = pd.read_csv(a.kl)
    m["icode"] = m["icode"].fillna("").astype(str)
    m["burial"] = -m["rsasa_complex"]

    # attach d_bind_local (residue-aware) on the same positions
    db = pd.read_csv(a.dbind)
    db["icode"] = db["icode"].fillna("").astype(str)
    db["d_bind_local"] = db["logp_native"] - db["logp_native_monomer"]
    m = m.merge(db[["complex_id", "chain", "resnum", "icode", "d_bind_local"]],
                on=["complex_id", "chain", "resnum", "icode"], how="left")

    # inter-chain contacts
    cc = contact_counts(a.data_dir, m["complex_id"].unique())
    m["xc10"] = [cc.get((r.complex_id, r.chain, int(r.resnum)), (np.nan, np.nan))[0]
                 for r in m.itertuples()]
    m = m[np.isfinite(m["kl"]) & np.isfinite(m["burial"])].copy()

    m["burial_KL"] = rankavg(m, ["burial", "kl"])
    m["nbr_KL"] = rankavg(m, ["nbr", "kl"])
    have_db = m["d_bind_local"].notna()
    md = m[have_db].copy()
    md["burial_dbind"] = rankavg(md, ["burial", "d_bind_local"])
    md["burial_KL_"] = rankavg(md, ["burial", "kl"])

    print(f"positions: {len(m)}  complexes: {m['complex_id'].nunique()}  "
          f"strict hotspots: {int(m['is_hot'].sum())}")
    rows = []

    print("\n=== Q1  does KL ADD to burial? (paired) ===")
    r = paired_auc(m, "burial_KL", "burial")
    print(f"  burial+KL {r['a']:.4f}  vs burial {r['b']:.4f}  "
          f"delta +{r['delta']:.4f} [{r['delta_ci'][0]:+.4f},{r['delta_ci'][1]:+.4f}]  P(>0)={r['p_gt0']:.3f}")
    rows.append(dict(q="Q1_KL_adds_to_burial", **{k: (r[k] if not isinstance(r[k], np.ndarray)
                                                      else list(r[k])) for k in r}))

    print("\n=== Q2  is KL just a contact count? (paired) ===")
    m["xc10r"] = stats.rankdata(m["xc10"].fillna(m["xc10"].median())) / len(m)
    r = paired_auc(m, "kl", "xc10r")
    print(f"  KL {r['a']:.4f}  vs inter-chain-contacts {r['b']:.4f}  "
          f"delta +{r['delta']:.4f} [{r['delta_ci'][0]:+.4f},{r['delta_ci'][1]:+.4f}]  P(>0)={r['p_gt0']:.3f}")
    rows.append(dict(q="Q2_KL_vs_contacts", **{k: (r[k] if not isinstance(r[k], np.ndarray)
                                                    else list(r[k])) for k in r}))
    m["burial_xc"] = rankavg(m, ["burial", "xc10r"])
    m["burial_KL_xc"] = rankavg(m, ["burial", "kl", "xc10r"])
    r2 = paired_auc(m, "burial_KL_xc", "burial_xc")
    print(f"  burial+contacts+KL {r2['a']:.4f} vs burial+contacts {r2['b']:.4f}  "
          f"delta +{r2['delta']:.4f} [{r2['delta_ci'][0]:+.4f},{r2['delta_ci'][1]:+.4f}]")
    rows.append(dict(q="Q2b_KL_adds_over_burial_contacts", **{k: (r2[k] if not isinstance(r2[k], np.ndarray) else list(r2[k])) for k in r2}))

    print("\n=== Q3  does removing the sequence cost anything? (paired) ===")
    r = paired_auc(md, "burial_KL_", "burial_dbind")
    print(f"  burial+KL(no seq) {r['a']:.4f}  vs burial+d_bind_local(needs seq) {r['b']:.4f}  "
          f"delta {r['delta']:+.4f} [{r['delta_ci'][0]:+.4f},{r['delta_ci'][1]:+.4f}]")
    print("  -> if the CI straddles 0, removing the sequence costs nothing")
    rows.append(dict(q="Q3_seqfree_vs_seqaware", **{k: (r[k] if not isinstance(r[k], np.ndarray)
                                                        else list(r[k])) for k in r}))

    print("\n=== Q4  within-complex ranking (what a designer uses) ===")
    rng = np.random.default_rng(SEED)
    per = []
    for cid, d in m.groupby("complex_id"):
        nh = int(d["is_hot"].sum())
        if nh == 0 or len(d) < 8:
            continue
        row = dict(complex_id=cid, nh=nh)
        for nm, sc in [("burial", d["burial"].values), ("KL", d["kl"].values),
                       ("burial_KL", d["burial_KL"].values), ("xc", d["xc10r"].values)]:
            row[f"auc_{nm}"] = auc(sc, d["is_hot"].values)
            top = np.argsort(-sc)[:max(1, nh)]
            row[f"p_{nm}"] = d["is_hot"].values[top].mean()
        per.append(row)
    per = pd.DataFrame(per)
    per.to_csv(f"{a.out}_percomplex.csv", index=False)
    for nm in ["burial", "KL", "burial_KL"]:
        print(f"  {nm:12s} within-cx AUROC {per[f'auc_{nm}'].mean():.4f}   "
              f"precision@k {per[f'p_{nm}'].mean():.4f}")
    for base in ["burial", "xc"]:
        dd = (per["auc_burial_KL"] - per[f"auc_{base}"]).dropna()
        bb = np.array([np.mean(rng.choice(dd.values, len(dd), True)) for _ in range(5000)])
        pp = (per["p_burial_KL"] - per[f"p_{base}"]).dropna()
        pbb = np.array([np.mean(rng.choice(pp.values, len(pp), True)) for _ in range(5000)])
        print(f"  burial+KL − {base:6s}: AUROC {dd.mean():+.4f} [{np.percentile(bb,2.5):+.4f},{np.percentile(bb,97.5):+.4f}]"
              f"   p@k {pp.mean():+.4f} [{np.percentile(pbb,2.5):+.4f},{np.percentile(pbb,97.5):+.4f}]")
        rows.append(dict(q=f"Q4_within_cx_burialKL_minus_{base}",
                         auroc_delta=float(dd.mean()),
                         auroc_ci=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                         pk_delta=float(pp.mean()),
                         pk_ci=[float(np.percentile(pbb, 2.5)), float(np.percentile(pbb, 97.5))]))

    pd.DataFrame(rows).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"\n[done] wrote {a.out}_summary.csv and {a.out}_percomplex.csv")


if __name__ == "__main__":
    main()

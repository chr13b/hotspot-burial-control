#!/usr/bin/env python3
"""Analyse the CFG-steered ipTM folds (results/iptm_steer.csv). Pre-reg results/PREREG_iptm.md.

Aggregate k=0..2 per (complex, direction) [mean; best-of-k secondary], then the PAIRED L-random contrast
(complex-clustered bootstrap 95% CI, P>0) for four metrics — ipTM, interface pAE (lower better -> sign
flipped), interface pLDDT, global pTM (localization control) — plus the pre-registered robust COMPOSITE
(z-mean of ipTM, -interface pAE, interface pLDDT across all folds). H1 = composite(L)>random AND
ipTM(L)>random (both CI>0). H2 = L-wt not strongly negative. H3 = |global pTM shift| << composite shift.

  python3 src/analyse_iptm.py --in results/iptm_steer.csv --out results/iptm_summary.csv
"""
import argparse
import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 5000
# (column, higher_is_better) ; interface metrics for the composite = the first three
METRICS = [("iptm", True), ("interface_pae", False), ("interface_plddt", True), ("ptm", True)]
COMPOSITE_COLS = [("iptm", True), ("interface_pae", False), ("interface_plddt", True)]


def zorient(df, col, higher_better):
    v = df[col].to_numpy(float)
    z = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    return z if higher_better else -z


def paired_boot(pivot, ca, cb, rng, nboot=NBOOT):
    """pivot indexed by complex_id with columns per direction; paired (ca-cb) cluster bootstrap."""
    s = pivot[[ca, cb]].dropna()
    d = (s[ca] - s[cb]).to_numpy()
    ids = np.arange(len(d))
    pt = float(np.mean(d))
    b = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(nboot)]
    b = np.array(b)
    return pt, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(np.mean(b > 0)), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/iptm_steer.csv")
    ap.add_argument("--out", default="results/iptm_summary.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(a.inp)

    # composite per fold (z across ALL folds)
    df["composite"] = np.mean([zorient(df, c, hb) for c, hb in COMPOSITE_COLS], axis=0)

    # keep only complexes with wt + >=1 L + >=1 random (paired-analyzable)
    ok = []
    for cid, g in df.groupby("complex_id"):
        d = set(g.direction)
        if {"wt", "L", "random"} <= d:
            ok.append(cid)
    df = df[df.complex_id.isin(ok)].copy()
    n_cx = len(ok)

    rows = []
    allcols = [("iptm", True), ("interface_pae", False), ("interface_plddt", True),
               ("ptm", True), ("composite", True)]
    for agg, aggname in [("mean", "mean_over_k"), ("max_oriented", "best_of_k")]:
        for col, hb in allcols:
            # aggregate k per (complex, direction)
            def agf(s):
                if agg == "mean":
                    return s.mean()
                return s.max() if hb else s.min()          # best-of-k: max if higher-better else min
            piv = (df.groupby(["complex_id", "direction"])[col].apply(agf)
                     .unstack("direction"))
            for ca, cb, lab in [("L", "random", "L_minus_random"), ("L", "wt", "L_minus_wt")]:
                if ca not in piv or cb not in piv:
                    continue
                pt, lo, hi, pgt, n = paired_boot(piv, ca, cb, rng)
                rows.append(dict(agg=aggname, metric=col, contrast=lab, higher_better=hb,
                                 delta=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4),
                                 p_gt0=round(pgt, 4), n=n))

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out.to_csv(a.out, index=False)

    def get(agg, metric, contrast):
        r = out[(out.agg == agg) & (out.metric == metric) & (out.contrast == contrast)]
        return r.iloc[0] if len(r) else None

    print(f"[iptm-analyse] {n_cx} complexes with wt+L+random\n")
    print("=== PAIRED L - random (mean over k), complex-clustered 95% CI ===")
    for col, _ in allcols:
        r = get("mean_over_k", col, "L_minus_random")
        if r is not None:
            note = "  <-- localization control (expect small)" if col == "ptm" else ""
            print(f"  {col:16s} Δ={r.delta:+.4f} [{r.lo:+.4f},{r.hi:+.4f}] P(>0)={r.p_gt0:.3f} n={int(r.n)}{note}")
    ic = get("mean_over_k", "iptm", "L_minus_random")
    cc = get("mean_over_k", "composite", "L_minus_random")
    pc = get("mean_over_k", "ptm", "L_minus_random")
    h1 = (cc is not None and ic is not None and cc.lo > 0 and ic.lo > 0)
    print(f"\n  H1 (composite>0 AND ipTM>0, both CI>0): {'PASS' if h1 else 'FAIL'}")
    if cc is not None and pc is not None:
        print(f"  H3 (localization): |ΔpTM|={abs(pc.delta):.4f} vs Δcomposite={cc.delta:.4f} -> "
              f"{'pTM shifts less' if abs(pc.delta) < abs(cc.delta) else 'pTM shifts AS MUCH (disclose)'}")
    print("\n=== L - wt (mean over k) — H2 (no collapse) ===")
    for col in ("iptm", "composite"):
        r = get("mean_over_k", col, "L_minus_wt")
        if r is not None:
            print(f"  {col:16s} Δ={r.delta:+.4f} [{r.lo:+.4f},{r.hi:+.4f}]")
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()

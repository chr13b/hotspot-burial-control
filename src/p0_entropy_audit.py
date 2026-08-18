#!/usr/bin/env python3
"""AUDIT of the Phase-0 burial-matched null — read out as ENTROPY, the matched quantity.

The Phase-0 verdict is computed on ONE quantity: d_logp = logp_native(hot) - logp_native(ctl)
(src/p0_burial_matched.py:352). logp_native is confounded with amino-acid identity BY
CONSTRUCTION, which src/catalytic_audit.py established is the wrong quantity for a determinacy
question; the matched-pair control pools are 4x enriched in G/P/C (the three most determinate
residue types), so the confound runs straight through the estimator.

The full 20-way order-averaged distribution is already on disk as lp_A..lp_Y
(p0_burial_matched.py:244-245) — the SAME lp_mean that produced logp_native — so entropy costs
one line and was never taken. This script takes it, on the identical committed pairs, with the
identical complex-level bootstrap.

POSITIVE CONTROL (gates everything): recompute d_logp from the join and require it to reproduce
the committed d_logp column. If the join is wrong, nothing downstream is trustworthy.

  python3 src/p0_entropy_audit.py --out results/p0_entropy_audit.csv
"""
import argparse, glob, os
import numpy as np, pandas as pd

SEED = 20260803
NBOOT = 10000
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def load_positions():
    cols = ["complex_id", "chain", "resnum", "icode", "aa", "logp_native"] + [f"lp_{a}" for a in AA20]
    p = pd.read_csv("results/p0_positions.csv", usecols=cols)
    L = p[[f"lp_{a}" for a in AA20]].to_numpy(float)
    L = L - L.max(1, keepdims=True)
    P = np.exp(L); P = P / P.sum(1, keepdims=True)
    p["H"] = -(P * np.log(P + 1e-12)).sum(1)
    p["p_max"] = P.max(1)
    p["negH"] = -p["H"]
    return p


def boot(vals, cx, rng, nboot=NBOOT):
    ids = np.unique(cx); idx = {c: np.where(cx == c)[0] for c in ids}
    out = []
    for _ in range(nboot):
        t = np.concatenate([idx[c] for c in rng.choice(ids, len(ids), True)])
        out.append(np.nanmean(vals[t]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(np.array(out) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p0_entropy_audit.csv")
    a = ap.parse_args()
    pos = load_positions()
    key = ["complex_id", "chain", "resnum"]
    dup = pos.duplicated(key).sum()
    print(f"positions {len(pos)}, duplicate (complex_id,chain,resnum) keys: {dup} "
          f"(dropped for the join)")
    pj = pos.drop_duplicates(key).set_index(key)

    rows = []
    for f in sorted(glob.glob("results/p0_dssp_pairs_*.csv")):
        tier = os.path.basename(f)[len("p0_dssp_pairs_"):-4]
        d = pd.read_csv(f)
        h = pj.reindex(pd.MultiIndex.from_arrays(
            [d.complex_id, d.hot_chain, d.hot_resnum]))
        c = pj.reindex(pd.MultiIndex.from_arrays(
            [d.complex_id, d.ctl_chain, d.ctl_resnum]))
        ok = h.H.notna().to_numpy() & c.H.notna().to_numpy()
        d = d[ok].reset_index(drop=True); h = h[ok]; c = c[ok]
        if len(d) < 5:
            continue
        # ---- POSITIVE CONTROL: reproduce the committed d_logp from the join ----
        d_logp_rec = (h.logp_native.to_numpy() - c.logp_native.to_numpy())
        agree = np.nanmax(np.abs(d_logp_rec - d.d_logp.to_numpy()))
        gate = "JOIN OK" if agree < 1e-6 else f"JOIN MISMATCH (max|Δ|={agree:.2e})"
        cx = d.complex_id.to_numpy()
        rng = np.random.default_rng(SEED)
        res = {}
        for nm, v in [("d_logp", d_logp_rec),
                      ("dH", h.H.to_numpy() - c.H.to_numpy()),
                      ("d_p_max", h.p_max.to_numpy() - c.p_max.to_numpy())]:
            lo, hi, p = boot(v, cx, np.random.default_rng(SEED))
            res[nm] = (float(np.nanmean(v)), lo, hi, p, float(np.nanstd(v)))
        gpc_h = h.aa.isin(list("GPC")).mean(); gpc_c = c.aa.isin(list("GPC")).mean()
        same_aa = (h.aa.to_numpy() == c.aa.to_numpy()).mean()
        print(f"\n{tier}  n={len(d)} pairs, {d.complex_id.nunique()} complexes   [{gate}]")
        print(f"   aa identical in {same_aa:.1%} of pairs | G/P/C: hot {gpc_h:.1%} vs ctl {gpc_c:.1%}")
        for nm in ("d_logp", "dH", "d_p_max"):
            m, lo, hi, p, sd = res[nm]
            star = " *" if (lo > 0 or hi < 0) else ""
            print(f"   {nm:8s} = {m:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f} sd={sd:.3f}{star}")
        rows.append(dict(tier=tier, n_pairs=len(d), n_complexes=d.complex_id.nunique(),
                         join_check=gate, frac_same_aa=round(float(same_aa), 4),
                         gpc_hot=round(float(gpc_h), 4), gpc_ctl=round(float(gpc_c), 4),
                         **{f"{k}_{s}": round(res[k][i], 5)
                            for k in ("d_logp", "dH", "d_p_max")
                            for i, s in enumerate(["mean", "lo", "hi", "p_gt0", "sd"])}))
    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_boot"] = NBOOT
    out["note"] = ("Phase-0 committed matched pairs re-read as ENTROPY (and p_max) from the same "
                   "order-averaged lp_A..lp_Y that produced logp_native; join gated by exact "
                   "reproduction of the committed d_logp column")
    out["command"] = "python3 src/p0_entropy_audit.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

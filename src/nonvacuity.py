"""Precise non-vacuity of the blindness result (audit D) — CORRECTED after an adversarial referee pass.

The question: holding the bound distribution `P` fixed, how much of the leverage-magnitude spread
survives? If confidence (a functional of `P`) determined leverage, distribution-matched positions
(TV(P_i,P_j)<0.02) would have |ΔL| -> 0.

**The bug this fixes.** The first version divided the matched-pair median |ΔL| by a random-pair median
drawn from *all* interface positions. But TV<0.02 matching selects a NON-REPRESENTATIVE, high-variance
subpopulation (ESM-IF1: pool SD(L_rms)=2.58 vs global 1.75; median 1.74 vs 1.44), so the global
denominator was too small and inflated the ratio — ESM-IF1 read a spurious 121% ("no more similar than
random"). A ratio >1 is in fact near-impossible without this selection bias, since Spearman(conf,L_rms)>0
and within-pair |Δconf| is tiny, so conditioning on P *must* reduce E|ΔL| against a same-population baseline.

**The fix.** Draw the denominator from the SAME population the matched pairs come from, two independent ways:
  (B) within-pool     — random pairs among the P-matchable positions themselves;
  (C) anchor-matched  — for each matched anchor, its P-matched partner vs K uniformly-random partners
                        (same anchor, so the anchor-selection effect cancels exactly).
Both isolate "effect of P-matching" from "the subpopulation is high-variance". Complex-clustered bootstrap
CI on the surviving fraction (ground rule 4). The old global-denominator ratio is retained in the CSV,
labelled biased, for auditability. SEED=20260803.

  python3 src/nonvacuity.py --out results/nonvacuity.csv
"""
import argparse
import numpy as np, pandas as pd
from collections import defaultdict

AA20 = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA20)}
SEED = 20260803


def frame(pqf):
    """Rebuild L_rms and P from the committed per-aa log-prob matrices (lP_*, lQ_*)."""
    pq = pd.read_csv(pqf, low_memory=False)
    pq["icode"] = pq.icode.fillna("").astype(str)
    p0 = pd.read_csv("results/p0_positions.csv", low_memory=False,
                     usecols=["complex_id", "chain", "resnum", "icode", "is_interface"])
    p0["icode"] = p0.icode.fillna("").astype(str)
    d = p0.merge(pq, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    d = d[d.is_interface == 1].copy()
    lP = d[[f"lP_{a}" for a in AA20]].to_numpy()
    lQ = d[[f"lQ_{a}" for a in AA20]].to_numpy()
    ok = np.isfinite(d.aa.map(IDX).to_numpy().astype(float))
    d, lP, lQ = d[ok].copy().reset_index(drop=True), lP[ok], lQ[ok]
    wi = d.aa.map(IDX).to_numpy(); ar = np.arange(len(d))
    r = lP - lQ
    Lvec = r - r[ar, wi][:, None]
    mask = np.ones((len(d), 20), bool); mask[ar, wi] = False
    d["L_rms"] = np.sqrt(np.nanmean(np.where(mask, Lvec, np.nan) ** 2, axis=1))
    return d, np.exp(lP)


def collect(d, P, rng, tol=0.02, K=20):
    """Matched pairs, then an anchor-matched random control drawn from the SAME P-matchable pool.

    Two passes: pass 1 finds the matched pairs and the pool of P-matchable positions; pass 2 gives each
    anchor K random partners drawn *from that pool* (not the global population — that was the bug), so the
    only thing that differs between numerator and denominator is whether the partner was P-matched."""
    Lrms = d.L_rms.to_numpy(); cids = d.complex_id.to_numpy(); n = len(d)
    idx = rng.choice(n, size=int(min(4000, n)), replace=False)
    Ps = P[idx]
    matched, anchors, members = [], [], []
    for a_ in range(len(idx)):
        tv = 0.5 * np.abs(Ps[a_ + 1:] - Ps[a_]).sum(axis=1)
        hits = np.where(tv < tol)[0][:5]
        if len(hits) == 0:
            continue
        ia = idx[a_]
        for h in hits:
            ib = idx[a_ + 1 + h]
            matched.append((abs(Lrms[ia] - Lrms[ib]), cids[ia]))
            members += [ia, ib]
        anchors.append((ia, len(hits)))
    pool = np.unique(members)
    anchor_rand = []
    for ia, m in anchors:                                  # same anchor, random partner FROM THE POOL
        for c in rng.choice(pool, K * m, replace=True):
            if c != ia:
                anchor_rand.append((abs(Lrms[ia] - Lrms[c]), cids[ia]))
    return matched, anchor_rand, np.array(members), Lrms


def ratio_ci(matched, denom, rng, nboot=2000):
    """Ratio of medians (matched / denom), complex-clustered bootstrap over the union of anchor complexes."""
    mm, dd = defaultdict(list), defaultdict(list)
    for v, c in matched: mm[c].append(v)
    for v, c in denom:   dd[c].append(v)
    comps = sorted(set(mm) | set(dd))

    def stat(cs):
        M = [v for c in cs for v in mm.get(c, [])]
        D = [v for c in cs for v in dd.get(c, [])]
        return (np.median(M) / np.median(D)) if M and D else np.nan

    point = stat(comps)
    boot = [stat(rng.choice(comps, len(comps), replace=True)) for _ in range(nboot)]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi), len(comps)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/nonvacuity.csv"); a = ap.parse_args()
    rows = []
    for name, pqf in [("ProteinMPNN", "results/leverage_pq_skempi.csv"),
                      ("ESM-IF1", "results/leverage_pq_skempi_esmif.csv")]:
        rng = np.random.default_rng(SEED)
        d, P = frame(pqf)
        matched, anchor_rand, members, Lrms = collect(d, P, rng)
        mdL = np.array([v for v, _ in matched])
        matched_med = float(np.median(mdL))

        # (C) anchor-matched random control — primary, with clustered CI
        frac_c, lo_c, hi_c, ncomp = ratio_ci(matched, anchor_rand, rng)
        # (B) within-pool (unique positions) — independent cross-check, no P used
        uniq = np.unique(members); pu = Lrms[uniq]
        r2 = np.random.default_rng(SEED + 3)
        i, j = r2.integers(0, len(pu), 200000), r2.integers(0, len(pu), 200000)
        within_u = float(np.median(np.abs(pu[i] - pu[j]))); frac_b = matched_med / within_u
        # old biased global denominator, kept for the record
        r1 = np.random.default_rng(SEED + 1)
        gi, gj = r1.integers(0, len(Lrms), 200000), r1.integers(0, len(Lrms), 200000)
        glob = float(np.median(np.abs(Lrms[gi] - Lrms[gj]))); frac_glob = matched_med / glob

        print(f"  {name:11s}: matched median|ΔL|={matched_med:.4f}; "
              f"survives_frac (C) anchor-matched = {frac_c:.3f} [{lo_c:.3f},{hi_c:.3f}], "
              f"(B) within-pool = {frac_b:.3f}; biased-global (old) = {frac_glob:.3f}")
        rows.append(dict(model=name, matched_median_dL=round(matched_med, 4),
                         survives_frac_anchor=round(frac_c, 3), lo=round(lo_c, 3), hi=round(hi_c, 3),
                         survives_frac_withinpool=round(frac_b, 3),
                         survives_frac_global_BIASED=round(frac_glob, 3),
                         n_pairs=len(mdL), n_complexes=ncomp, seed=SEED,
                         command="python3 src/nonvacuity.py"))
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

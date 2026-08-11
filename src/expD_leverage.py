"""Experiment D, Task 2 (CPU): the SYMMETRIC leverage / jackknife robustness check.

Motivation (adversarial review): the project ACCEPTS Exp A's predicted-backbone deficit
(SECONDARY_B d_pred = -0.191) but DEMOTES Exp C2's pre-registered slope-fire as a near-crystal
leverage artifact -- and the leverage diagnostic (src/expC2_slope_diag.py) was applied only to C2.
That is a pre-registration asymmetry. This script applies the IDENTICAL leave-k-complexes-out
scrutiny to Exp A's deficit, and (for visible symmetry) to Exp C2's within-binder gap.

Exp A block: per-pair predicted gap d = logp_native(hot | OF3) - logp_native(ctl | OF3), reusing the
  committed SECONDARY_B matched pairs and the OF3-predicted positions table on $SCRATCH.
Exp C2 block: per-backbone within-binder gap d, interface-formed generative backbones
  (interface_ok==1 & partial_T>0), from results/expC2_gap_perbackbone.csv.

For each block: pooled complex-level bootstrap (seed 20260803, 10,000 reps -- matches
expA_gap_reuse_pairs), leave-one-complex-out signed influence (full pooled mean minus the mean with
that complex removed), and the estimate + 95% CI after DROPPING the top-3 / top-5 complexes that
most SUPPORT the effect (those whose removal moves the estimate toward zero most). "Survives" = the
sign is preserved and the 95% CI still excludes zero after the drop.

READING (D-LEVERAGE, pre-registered in PREREG_expD.md 7): if Exp A's deficit survives
leave-3/5-out (CI still excludes zero) it is NOT a leverage artifact -> the asymmetry charge is
answered and -0.191 is reported as robust. If it does not, say so and demote it.

Usage (in the ftax env):
  python3 src/expD_leverage.py \
    --expA-pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
    --of3-positions $SCRATCH/ftax/predicted/expA_p0_positions.csv \
    --c2-perbackbone results/expC2_gap_perbackbone.csv \
    --out results/expD_leverage.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 10000


def pos_lookup(csv, col="logp_native"):
    """(complex_id, chain, resnum) -> logp; blank-icode row wins on collision (as expA_gap_reuse_pairs)."""
    df = pd.read_csv(csv, usecols=lambda c: c in ("complex_id", "chain", "resnum", "icode", col))
    df["icode"] = df["icode"].fillna("").astype(str)
    df = df.sort_values("icode").drop_duplicates(subset=["complex_id", "chain", "resnum"], keep="first")
    return {(r.complex_id, r.chain, int(r.resnum)): getattr(r, col) for r in df.itertuples()}


def expA_records(pairs_csv, pos_csv):
    lp = pos_lookup(pos_csv)
    pairs = pd.read_csv(pairs_csv)
    recs = []
    for r in pairs.itertuples():
        h = (r.complex_id, r.hot_chain, int(r.hot_resnum))
        c = (r.complex_id, r.ctl_chain, int(r.ctl_resnum))
        d = lp.get(h, np.nan) - lp.get(c, np.nan)
        if np.isfinite(d):
            recs.append((r.complex_id, float(d)))
    return recs


def c2_records(csv):
    df = pd.read_csv(csv)
    df = df[(df["interface_ok"] == 1) & (df["partial_T"] > 0)]
    return [(r.complex_id, float(r.d)) for r in df.itertuples() if np.isfinite(r.d)]


def cboot(recs, seed=SEED, nboot=NBOOT):
    """Pooled (pair/backbone-weighted) mean with a complex-resampling bootstrap; vectorized."""
    cids = sorted({c for c, _ in recs})
    by = {c: [] for c in cids}
    for c, d in recs:
        by[c].append(d)
    sums = np.array([np.nansum(by[c]) for c in cids], float)
    cnts = np.array([np.sum(np.isfinite(by[c])) for c in cids], float)
    obs = sums.sum() / cnts.sum()
    rng = np.random.default_rng(seed)
    ncx = len(cids)
    idx = rng.integers(0, ncx, size=(nboot, ncx))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(estimate=float(obs), lo=float(lo), hi=float(hi),
                p_gt0=float((bs > 0).mean()), p_lt0=float((bs < 0).mean()),
                n_units=int(cnts.sum()), n_cx=ncx,
                cids=cids, sums=sums, cnts=cnts)


def influence(res):
    """Signed leave-one-complex-out influence: full pooled mean minus mean with complex removed."""
    cids, sums, cnts, obs = res["cids"], res["sums"], res["cnts"], res["estimate"]
    ts, tc = sums.sum(), cnts.sum()
    return {c: float(obs - (ts - sums[i]) / (tc - cnts[i])) for i, c in enumerate(cids)}


def drop_topk(recs, infl, effect_sign, k, seed=SEED, mode="support"):
    """Drop k complexes. mode='support': most effect-supporting (pre-registered criterion, PREREG 7);
    mode='abs': largest |influence| (literal 'most-influential', supplementary robustness)."""
    key = (lambda c: infl[c] * effect_sign) if mode == "support" else (lambda c: abs(infl[c]))
    order = sorted(infl, key=key, reverse=True)
    drop = set(order[:k])
    kept = [(c, d) for c, d in recs if c not in drop]
    r = cboot(kept, seed=seed)
    return list(order[:k]), r


def analyze(name, recs):
    full = cboot(recs)
    infl = influence(full)
    sign = 1.0 if full["estimate"] >= 0 else -1.0
    iv = np.array(list(infl.values()))
    excl0 = (full["hi"] < 0) or (full["lo"] > 0)
    print(f"\n=== {name} ===")
    print(f"  full: {full['estimate']:+.4f} [{full['lo']:+.4f},{full['hi']:+.4f}]  "
          f"n_units={full['n_units']} n_cx={full['n_cx']}  P(>0)={full['p_gt0']:.3f} "
          f"CI-excludes-0={excl0}")
    print(f"  influence: min {iv.min():+.4f} max {iv.max():+.4f} median {np.median(iv):+.4f} "
          f"(effect sign {'+' if sign > 0 else '-'})")
    rows = [dict(block=name, subset="full", drop_k=0, dropped="",
                 estimate=full["estimate"], lo95=full["lo"], hi95=full["hi"],
                 p_gt0=full["p_gt0"], p_lt0=full["p_lt0"],
                 n_units=full["n_units"], n_cx=full["n_cx"], ci_excludes_0=excl0,
                 infl_min=float(iv.min()), infl_max=float(iv.max()), infl_median=float(np.median(iv)))]
    for mode, tag in (("support", "supporters"), ("abs", "absinfluence")):
        for k in (3, 5):
            dropped, r = drop_topk(recs, infl, sign, k, mode=mode)
            ex = (r["hi"] < 0) or (r["lo"] > 0)
            surv = ex and (np.sign(r["estimate"]) == sign)
            flag = "  [PRE-REGISTERED]" if mode == "support" else "  [supplementary]"
            print(f"  drop top-{k} {tag} {dropped}: {r['estimate']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}]  n_cx={r['n_cx']}  CI-excludes-0={ex}  survives={surv}{flag}")
            rows.append(dict(block=name, subset=f"drop_top{k}_{tag}", drop_k=k, drop_mode=mode,
                             dropped="|".join(dropped), estimate=r["estimate"], lo95=r["lo"], hi95=r["hi"],
                             p_gt0=r["p_gt0"], p_lt0=r["p_lt0"], n_units=r["n_units"], n_cx=r["n_cx"],
                             ci_excludes_0=ex, survives=surv))
    return rows, infl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expA-pairs", default="results/p0_dssp_pairs_SECONDARY_B_any_interface.csv")
    ap.add_argument("--of3-positions",
                    default=os.path.expandvars("$SCRATCH/ftax/predicted/expA_p0_positions.csv"))
    ap.add_argument("--af2-positions", default=None,
                    help="optional AF2 (Exp D) positions CSV -> add a symmetric leverage block")
    ap.add_argument("--c2-perbackbone", default="results/expC2_gap_perbackbone.csv")
    ap.add_argument("--out", default="results/expD_leverage.csv")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    blocks = [("expA_of3_SECONDARY_B_dpred", expA_records(a.expA_pairs, a.of3_positions))]
    if a.af2_positions:
        blocks.append(("expD_af2_SECONDARY_B_dpred", expA_records(a.expA_pairs, a.af2_positions)))
    blocks.append(("expC2_within_binder_iface_formed", c2_records(a.c2_perbackbone)))

    all_rows, infl_rows = [], []
    for name, recs in blocks:
        rows, infl = analyze(name, recs)
        all_rows += rows
        infl_rows += [dict(block=name, complex_id=c, influence=v) for c, v in
                      sorted(infl.items(), key=lambda kv: kv[1])]

    pd.DataFrame(all_rows).assign(seed=SEED, command=cmd).to_csv(a.out, index=False)
    infl_path = a.out.replace(".csv", "_influence.csv")
    pd.DataFrame(infl_rows).assign(command=cmd).to_csv(infl_path, index=False)
    print(f"\n[done] wrote {a.out} and {infl_path}")

    ea = [r for r in all_rows if r["block"] == "expA_of3_SECONDARY_B_dpred"]
    d5 = next(r for r in ea if r["subset"] == "drop_top5_supporters")
    verdict = ("NOT a leverage artifact (survives leave-5-out)" if d5.get("survives")
               else "CARRIED BY A FEW complexes (does not survive leave-5-out) -> demote")
    print(f"D-LEVERAGE reading (Exp A -0.19): {verdict}")


if __name__ == "__main__":
    main()

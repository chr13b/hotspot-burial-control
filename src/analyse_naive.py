#!/usr/bin/env python3
"""Paired specificity contrasts for the naive-guidance arm (pre-reg: results/PREREG_naive.md).

Combines the committed L / random judged rows with the new naive judged rows and computes, per judge, the paired
per-complex contrasts with the SAME complex-clustered bootstrap as cfg_judge_matrix.py::paired:
    naive - random   (expect ~0: a confident-but-not-binding tilt behaves like the random control)
    L - naive        (expect >0: the binding direction still wins)
    L - random       (the standing headline, for context)
Anti-circular judges (ProteinMPNN is the steered model -> self/circular, flagged): ESM-IF1, MIF.

Falsifier (pre-registered): under an anti-circular judge, naive - random >= L - random -> direction-specificity
is bounded (reported verbatim). Also fires if L - naive CI includes/<=0.

  python3 src/analyse_naive.py --out results/cfg_naive_summary.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 20260803
NBOOT = 5000
ALPHA = 2.0
JUDGE_NAME = {"mpnn": "ProteinMPNN", "esmif": "ESM-IF1", "pifold": "PiFold", "mif": "MIF"}
SELF_KEY = "mpnn"                       # ProteinMPNN is the steered model
CONTRASTS = [("naive", "random"), ("L", "naive"), ("L", "random")]


def paired(df, col, arm_a, arm_b, rng, nboot=NBOOT):
    piv = df.pivot_table(index="complex_id", columns="direction", values=col)
    if arm_a not in piv.columns or arm_b not in piv.columns:
        return None
    piv = piv.dropna(subset=[arm_a, arm_b])
    if len(piv) < 2:
        return None
    d = (piv[arm_a] - piv[arm_b]).to_numpy()
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return dict(n_cx=len(piv), mean_a=round(float(piv[arm_a].mean()), 4),
                mean_b=round(float(piv[arm_b].mean()), 4), delta=round(float(d.mean()), 4),
                lo=round(float(np.percentile(b, 2.5)), 4), hi=round(float(np.percentile(b, 97.5)), 4),
                p_gt0=round(float((b > 0).mean()), 4))


def load_method(method, L_random_csv, naive_csv):
    if not (os.path.exists(L_random_csv) and os.path.exists(naive_csv)):
        print(f"[skip] {method}: missing {L_random_csv} or {naive_csv}")
        return None
    lr = pd.read_csv(L_random_csv); nv = pd.read_csv(naive_csv)
    df = pd.concat([lr, nv], ignore_index=True)
    naive_cx = set(df[df.direction == "naive"].complex_id)     # fair: all contrasts on the naive complex set
    df = df[df.complex_id.isin(naive_cx)]
    df = df[np.isclose(df.alpha, ALPHA)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cfg_naive_summary.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    # (method, L/random source, naive source) -- dumped3 is primary (uniform, incl. MIF); K64 is higher-power (esmif)
    METHODS = [("dumped3", "results/cfg_judge_dumped_setA.csv", "results/cfg_naive_judge.csv"),
               ("K64", "results/cfg_steer.csv", "results/cfg_steer_naive.csv")]
    rows = []
    print("=== NAIVE-GUIDANCE SPECIFICITY (paired, complex-clustered 95% CI, alpha=2) ===")
    for method, lrc, nvc in METHODS:
        df = load_method(method, lrc, nvc)
        if df is None:
            continue
        judge_cols = [c for c in df.columns if c.startswith("meanL_")]
        print(f"\n-- method={method}  (judges: {[c.replace('meanL_','') for c in judge_cols]}) --")
        for jc in judge_cols:
            jkey = jc.replace("meanL_", ""); judge = JUDGE_NAME.get(jkey, jkey)
            is_self = int(jkey == SELF_KEY)
            for arm_a, arm_b in CONTRASTS:
                res = paired(df, jc, arm_a, arm_b, rng)
                if res is None:
                    continue
                rows.append(dict(method=method, judge=judge, is_self=is_self,
                                 contrast=f"{arm_a}-{arm_b}", **res))
                tag = "SELF/circ" if is_self else "anti-circ"
                print(f"  judge={judge:11s}[{tag}] {arm_a:6s}-{arm_b:6s} = {res['delta']:+.4f} "
                      f"[{res['lo']:+.4f},{res['hi']:+.4f}] P(>0)={res['p_gt0']:.3f} "
                      f"({arm_a}={res['mean_a']:+.3f} {arm_b}={res['mean_b']:+.3f} n={res['n_cx']})")
    out = pd.DataFrame(rows); out["seed"] = SEED
    out.to_csv(a.out, index=False)
    print(f"\n[analyse-naive] {len(out)} rows -> {a.out}")

    # ---- pre-registered falsifier read-out (anti-circular judges, dumped3 primary) ----
    print("\n=== FALSIFIER READ-OUT (anti-circular judges) ===")
    fired = []
    for method in ("dumped3", "K64"):
        sub = out[(out.method == method) & (out.is_self == 0)]
        for judge in sub.judge.unique():
            nr = sub[(sub.judge == judge) & (sub.contrast == "naive-random")]
            ln = sub[(sub.judge == judge) & (sub.contrast == "L-naive")]
            lr = sub[(sub.judge == judge) & (sub.contrast == "L-random")]
            if not len(nr) or not len(ln) or not len(lr):
                continue
            nr, ln, lr = nr.iloc[0], ln.iloc[0], lr.iloc[0]
            bounded = (nr.delta >= lr.delta) or (ln.hi <= 0) or (ln.lo <= 0)
            verdict = "BOUNDED (falsifier fires)" if bounded else "specificity holds"
            print(f"  [{method:7s}] {judge:11s}: naive-random={nr.delta:+.4f}  L-naive={ln.delta:+.4f}"
                  f"[{ln.lo:+.4f},{ln.hi:+.4f}]  L-random={lr.delta:+.4f}  -> {verdict}")
            if bounded:
                fired.append((method, judge))
    if fired:
        print(f"\nFALSIFIER FIRES for {fired} -> direction-specificity is BOUNDED; report verbatim.")
    else:
        print("\nAll anti-circular judges: naive-random < L-random AND L-naive CI>0 "
              "-> specificity holds (the binding direction is required; a confident tilt is not enough).")


if __name__ == "__main__":
    main()

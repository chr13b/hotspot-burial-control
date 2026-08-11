#!/usr/bin/env python3
"""
Post-hoc (EXPLORATORY, not pre-registered) robustness check on Exp C's C-PRIMARY.

Question raised in review: the binned dose table in FINDINGS_expC.md §4 shows every intermediate
iRMSD bin containing zero, but a *continuous* slope of the burial-matched gap d vs log10(interface
RMSD) might have more power and reveal a monotone dose-response the bins mask. Does it?

Answer (see results/expC_slope_check.csv): the continuous slope IS significantly negative over ALL
interface-formed backbones (-0.179, P(slope>0)=0.007) — BUT that significance is carried entirely by
the crystal anchor (iRMSD~=0, gap positive) at one end and the DISSOLVED >10A tail at the other.
Restricted to the physically-meaningful generated backbones (interface-formed AND iRMSD<=8 AND NOT the
partial_T=0 crystal), the slope is -0.009 [P=0.52] — FLAT. So the apparent continuous dose-response is
the same crystal-vs-dissolved confound the pre-registration (PREREG_expC §5 CONFOUND) was written to
distrust; it is NOT a physical-regime gradient. This SHARPENS the FINDINGS "suggestive, not decisive"
verdict and pre-empts a reviewer who runs the naive slope. It also motivates Exp C2: the generated-
physical regime is genuinely unresolved (point estimate ~0 at n_cx=13), needing either a powered
positive or a pre-registered TOST null.

Command:
  python3 src/expC_slope_check.py --gap results/expC_gap_perbackbone.csv --out results/expC_slope_check.csv
"""
import argparse, csv
import numpy as np
from scipy import stats

SEED = 20260803


def cboot(recs, fn, seed=SEED, nboot=10000):
    rng = np.random.default_rng(seed)
    cids = sorted({r["complex_id"] for r in recs})
    by = {c: [r for r in recs if r["complex_id"] == c] for c in cids}
    obs = fn(recs)
    boots = []
    for _ in range(nboot):
        samp = []
        for c in rng.choice(cids, len(cids), True):
            samp += by[c]
        v = fn(samp)
        if v is not None and np.isfinite(v):
            boots.append(v)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi), float(np.mean(np.array(boots) > 0)), len(cids)


def slope(recs):
    x = np.log10(np.clip([r["irmsd"] for r in recs], 1e-3, None))
    y = np.array([r["d"] for r in recs])
    return float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else None


def spear(recs):
    x = [r["irmsd"] for r in recs]
    y = [r["d"] for r in recs]
    return float(stats.spearmanr(x, y).correlation) if len(set(x)) > 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", default="results/expC_gap_perbackbone.csv")
    ap.add_argument("--out", default="results/expC_slope_check.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.gap)))
    for r in rows:
        r["d"] = float(r["d"]); r["irmsd"] = float(r["irmsd"])
        r["ok"] = int(r["interface_ok"]); r["T"] = int(r["partial_T"])
    ok = [r for r in rows if r["ok"] == 1]

    subsets = [
        ("iface_formed_all",            ok),
        ("iface_formed_irmsd_le8",      [r for r in ok if r["irmsd"] <= 8]),
        ("iface_formed_irmsd_le8_gen",  [r for r in ok if r["irmsd"] <= 8 and r["T"] > 0]),
        ("iface_formed_gen_all",        [r for r in ok if r["T"] > 0]),
    ]
    out = []
    for name, sub in subsets:
        for metric, fn in [("slope_d_vs_log10irmsd", slope), ("spearman_d_vs_irmsd", spear)]:
            obs, lo, hi, pgt0, ncx = cboot(sub, fn)
            out.append(dict(subset=name, metric=metric, n_bb=len(sub), n_cx=ncx,
                            estimate=obs, lo=lo, hi=hi, p_gt0=pgt0, seed=SEED,
                            command="python3 src/expC_slope_check.py --gap %s --out %s" % (a.gap, a.out)))
            print(f"{name:<30} {metric:<24} n_bb={len(sub):>3} n_cx={ncx:>2}  "
                  f"{obs:+.3f} [{lo:+.3f},{hi:+.3f}] P(>0)={pgt0:.3f}")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nEXPLORATORY (post-hoc, not pre-registered). Wrote {a.out}")
    print("VERDICT: continuous slope significant over ALL iface-formed backbones is a crystal-vs-"
          "dissolved artifact; generated-physical slope is flat (-0.009, P=0.52). Confirms 'suggestive"
          ", not decisive'.")


if __name__ == "__main__":
    main()

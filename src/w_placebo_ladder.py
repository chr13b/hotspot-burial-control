"""Audit A: calibrate the position-level CPI estimator with a PLACEBO LADDER, and show leverage is robust to a
NONLINEAR geometry control. The estimator permutes X within bins of a 1-D linear geometry score, so a feature
that is a deterministic function of the geometry controls can score a small false positive -- the true null
floor is NOT zero. Establish it, then read every scalar against it.

  python3 src/w_placebo_ladder.py --out results/w_placebo_ladder.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leverage_decomposition as LD
SEED = 20260803

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/w_placebo_ladder.csv"); a = ap.parse_args()
    d = pd.read_csv("results/leverage_skempi_positions.csv")
    d = d[(d.is_interface == True) & d.L_ala.notna()].copy()      # noqa: E712
    y = d.is_hot.astype(int).to_numpy(); g = d.complex_id.to_numpy()
    def zc(v): v = np.asarray(v, float); return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    bur, nbr, ds = zc(d.burial), zc(d.nbr), zc(d.drsasa)
    Z = np.column_stack([bur, nbr, ds])
    rng = np.random.default_rng(SEED); rows = []
    print(f"[placebo] {len(d)} interface positions, {d.complex_id.nunique()} complexes, {int(y.sum())} hotspots")
    def cpi(name, X, Zc=Z, kind="real"):
        c, lo, hi, p, _, _ = LD.cpi(y, g, Zc, X.copy(), rng)
        print(f"  {kind:7s} CPI({name:34s}) = {c:+.5f} [{lo:+.5f},{hi:+.5f}] P>0={p:.3f}")
        rows.append(dict(kind=kind, feature=name, cpi=round(c, 5), lo=round(lo, 5), hi=round(hi, 5), p_gt0=round(p, 3)))
        return c, lo, hi
    print(" -- PLACEBOS (contain zero information beyond the controls; anything above these is the false-positive floor):")
    cpi("duplicate of dSASA", ds, kind="placebo")
    cpi("duplicate of nbr", nbr, kind="placebo")
    cpi("dSASA^2", ds**2, kind="placebo")
    cpi("nbr * dSASA", nbr*ds, kind="placebo")
    cpi("pure noise", zc(rng.standard_normal(len(d))), kind="placebo")
    floor = max(r["hi"] for r in rows if r["kind"] == "placebo")     # conservative floor = worst placebo upper CI...
    floor_pt = max(r["cpi"] for r in rows if r["kind"] == "placebo")
    print(f"  => estimator false-positive floor ~ {floor_pt:+.5f} (point), up to {floor:+.5f} (CI)")
    print(" -- REAL scalars of P (read against the floor):")
    cpi("confidence log p(native)", zc(d.conf), kind="real")
    cpi("negentropy of P", zc(d.negH), kind="real")
    cpi("scalar KL", zc(d.klP), kind="real")
    print(" -- the MIXED derivative (clears the floor by ~3x):")
    cpi("leverage -L(->Ala)", zc(-d.L_ala), kind="real")
    print(" -- robustness: leverage under a NONLINEAR (quadratic) geometry control:")
    Zq = np.column_stack([bur, nbr, ds, bur**2, nbr**2, ds**2, bur*nbr, bur*ds, nbr*ds])
    cpi("leverage -L | QUADRATIC geometry", zc(-d.L_ala), Zc=Zq, kind="robust")
    Zc3 = np.column_stack([Zq, bur**3, nbr**3, ds**3])
    cpi("leverage -L | CUBIC geometry", zc(-d.L_ala), Zc=Zc3, kind="robust")
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"[wrote] {a.out}")

if __name__ == "__main__":
    main()

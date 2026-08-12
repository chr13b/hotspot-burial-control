#!/usr/bin/env python3
"""
#4-full: is ProBID-Net's OWN hotspot recovery gap a burial artifact? (the 6th-model rebuttal)

We ran the RELEASED ProBID-Net model (figshare modeloutput0.hdf5) on our SKEMPI crystal complexes
(src that produced results/probid_positions.csv). Now, at ProBID-Net's own hotspot definition
(Ala-scan ddG>2 = is_hot in kl_detector_joined):
  (1) POSITIVE CONTROL — the UNCONTROLLED gap: hotspot recovery vs non-hotspot interface recovery
      (ProBID-Net reported 0.334 / 0.472). Must reproduce the direction before trusting anything.
  (2) The BURIAL-MATCHED gap: recovery(hotspot) - recovery(burial-matched control) using the committed
      pydssp matched pairs. If it collapses toward zero, ProBID-Net's gap is a burial artifact TOO —
      shown on the originating model, the sharpest rebuttal.
Complex-level bootstrap; report coverage (some ProBID chains may be missing due to OOM on huge chains).

  python3 src/probid_gap.py --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv --out results/probid_gap.csv
"""
import argparse, glob
import numpy as np
import pandas as pd

SEED = 20260803


def cboot(vals_by_cx, nboot=10000):
    rng = np.random.default_rng(SEED)
    cids = list(vals_by_cx)
    obs = float(np.mean([vals_by_cx[c] for c in cids]))
    b = [np.mean([vals_by_cx[c] for c in rng.choice(cids, len(cids), True)]) for _ in range(nboot)]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return obs, float(lo), float(hi), float(np.mean(np.array(b) > 0)), len(cids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="results/probid_positions.csv")
    ap.add_argument("--joined", default="results/kl_detector_joined.csv")
    ap.add_argument("--pairs", default="results/p0_dssp_pairs_SECONDARY_B_any_interface.csv")
    ap.add_argument("--out", default="results/probid_gap.csv")
    a = ap.parse_args()

    pos = pd.read_csv(a.positions)
    pos["resnum"] = pos["resnum"].astype(str)
    rec = {(r.pdb, r.chain, r.resnum): r.recovered for r in pos.itertuples()}

    # ---- (1) uncontrolled: hotspot vs non-hotspot interface recovery ----
    kl = pd.read_csv(a.joined); kl = kl[kl.is_interface == 1].copy()
    kl["pdb"] = kl.complex_id.str.split("_").str[0]; kl["resnum"] = kl.resnum.astype(str)
    kl["rec"] = [rec.get((p, c, r)) for p, c, r in zip(kl.pdb, kl.chain, kl.resnum)]
    cov = kl.rec.notna().mean()
    kk = kl.dropna(subset=["rec"])
    rows = []
    # per-complex mean recovery in each class, then bootstrap the gap
    hot = {c: g[g.is_hot == 1].rec.mean() for c, g in kk.groupby("complex_id") if (g.is_hot == 1).any()}
    non = {c: g[g.is_hot == 0].rec.mean() for c, g in kk.groupby("complex_id") if (g.is_hot == 0).any()}
    both = [c for c in hot if c in non]
    gap_unc = {c: hot[c] - non[c] for c in both}
    rh = float(np.mean([hot[c] for c in both])); rn = float(np.mean([non[c] for c in both]))
    g, lo, hi, p, n = cboot(gap_unc)
    rows.append(dict(analysis="uncontrolled_hot_minus_nonhot", recov_hot=rh, recov_nonhot=rn,
                     gap=g, lo=lo, hi=hi, p_gt0=p, n_cx=n))
    print(f"coverage: {cov:.2f} of interface positions have ProBID recovery ({len(kk)} pts)")
    print(f"[UNCONTROLLED] hotspot {rh:.3f} vs non-hotspot {rn:.3f}  gap={g:+.3f} [{lo:+.3f},{hi:+.3f}] (n_cx={n})")
    print(f"               ProBID-Net reported 0.334 / 0.472 (gap -0.138)")

    # ---- (2) burial-matched: recovery(hot) - recovery(ctl) over committed pairs ----
    for pf in sorted(glob.glob(a.pairs)):
        pr = pd.read_csv(pf)
        pr["pdb"] = pr.complex_id.str.split("_").str[0]
        d = {}
        kept = dropped = 0
        for r in pr.itertuples():
            rh_ = rec.get((r.pdb, str(r.hot_chain), str(r.hot_resnum)))
            rc_ = rec.get((r.pdb, str(r.ctl_chain), str(r.ctl_resnum)))
            if rh_ is None or rc_ is None:
                dropped += 1; continue
            d.setdefault(r.complex_id, []).append(rh_ - rc_); kept += 1
        if not d:
            print(f"[MATCHED {pf}] no covered pairs"); continue
        percx = {c: float(np.mean(v)) for c, v in d.items()}
        g, lo, hi, p, n = cboot(percx)
        rows.append(dict(analysis=f"burial_matched::{pf.split('/')[-1]}", gap=g, lo=lo, hi=hi,
                         p_gt0=p, n_cx=n, pairs_kept=kept, pairs_dropped=dropped))
        print(f"[MATCHED {pf.split('/')[-1]}] recov(hot)-recov(ctl) = {g:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"(n_cx={n}, pairs {kept} kept / {dropped} dropped-uncovered)")

    out = pd.DataFrame(rows); out["coverage_interface"] = cov; out["seed"] = SEED
    out["command"] = f"python3 src/probid_gap.py --pairs {a.pairs} --out {a.out}"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

"""Ceiling-raiser: does the MIXED DERIVATIVE add in the GENUINE de-novo design regime? The R2 result shows
leverage survives on predicted backbones of *natural* (likely-memorized) complexes. This is the stronger test:
73 de-novo RFdiffusion+ProteinMPNN binders (Bennett 2023) with wet-lab SSM binding labels, L/Q/geometry already
scored by the committed pipeline (`results/leverage_bennett_pairs.csv`).

Pre-registration (the project's standing leverage falsifier applied to a new fixture): predict CPI(L|geometry)
> 0 on de-novo binders, AND -- the sharp point -- leverage adds where the scalar KL did NOT (FINDINGS_bennett:
KL ~ geometry on de-novo). If CPI(L|geom) <= 0 or <= CPI(KL|geom), the mixed derivative does not add in the
de-novo regime; report it honestly.

  python3 src/leverage_bennett_denovo.py --out results/leverage_bennett_denovo.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "models"))
import leverage_decomposition as LD
SEED = LD.SEED


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/leverage_bennett_denovo.csv"); a = ap.parse_args()
    d = pd.read_csv("results/leverage_bennett_pairs.csv", low_memory=False)
    if "is_interface" in d.columns:
        d = d[d.is_interface == 1]
    need = ["L", "burial", "nbr", "drsasa", "destab", "conf", "klP", "design"]
    d = d.dropna(subset=[c for c in need if c in d.columns]).reset_index(drop=True)
    y = d.destab.to_numpy().astype(float)
    grp = d.design.to_numpy() if "design" in d.columns else d.complex_id.to_numpy()
    for c in ["burial", "nbr", "drsasa", "L", "conf", "klP"]:
        d[c + "z"] = LD.zs(d[c])
    Z = d[["burialz", "nbrz", "drsasaz"]].to_numpy()
    rng = np.random.default_rng(SEED); rows = []
    print(f"[denovo-leverage] {len(d)} interface mutations, {len(np.unique(grp))} designs, "
          f"{int(y.sum())} destabilising ({100*y.mean():.1f}%)", flush=True)
    for name, col in [("leverage L", "Lz"), ("confidence", "confz"), ("scalar KL", "klPz")]:
        v, lo, hi, p, _, _ = LD.cpi(y, grp, Z, d[col].to_numpy().copy(), rng)
        sp = stats.spearmanr(d[col.replace("z", "")], d.ddG, nan_policy="omit").correlation if "ddG" in d.columns else np.nan
        print(f"  CPI({name:11s} | geometry) = {v:+.5f} [{lo:+.5f}, {hi:+.5f}]  P(>0)={p:.3f}   "
              f"Spearman(.,ddG)={sp:+.3f}", flush=True)
        rows.append(dict(feature=name, cpi_over_geom=round(v, 5), lo=round(lo, 5), hi=round(hi, 5),
                         p_gt0=round(p, 3), spearman_vs_ddG=round(float(sp), 4) if np.isfinite(sp) else None,
                         n_mut=len(d), n_designs=int(len(np.unique(grp))), n_destab=int(y.sum())))
    # two-pass SPECIFICITY (the honest claim): does L add beyond the ONE-PASS complex log-odds logP_mut,
    # not merely beyond position-level geometry? The one-pass readout is P-only (no monomer pass).
    if "logP_mut" in d.columns:
        d["logP_mutz"] = LD.zs(d.logP_mut)
        Z2 = d[["burialz", "nbrz", "drsasaz", "logP_mutz"]].to_numpy()
        vo, loo, hio, _, _, _ = LD.cpi(y, grp, Z, d.logP_mutz.to_numpy().copy(), rng)
        v2, lo2, hi2, p2, _, _ = LD.cpi(y, grp, Z2, d.Lz.to_numpy().copy(), rng)
        print(f"  CPI(one-pass logP(mut|cx) | geometry)    = {vo:+.5f} [{loo:+.5f}, {hio:+.5f}]")
        print(f"  CPI(leverage L | geometry + one-pass)    = {v2:+.5f} [{lo2:+.5f}, {hi2:+.5f}]  <- TWO-PASS increment")
        rows.append(dict(feature="one-pass logP(mut|complex)", cpi_over_geom=round(vo, 5), lo=round(loo, 5),
                         hi=round(hio, 5), p_gt0=1.0, n_mut=len(d), n_designs=int(len(np.unique(grp))), n_destab=int(y.sum())))
        rows.append(dict(feature="leverage L | geometry + one-pass (two-pass increment)", cpi_over_geom=round(v2, 5),
                         lo=round(lo2, 5), hi=round(hi2, 5), p_gt0=round(p2, 3), n_mut=len(d),
                         n_designs=int(len(np.unique(grp))), n_destab=int(y.sum())))
        # apples-to-apples with §4's SKEMPI passage: control for substitution identity too (one-pass's de-novo
        # strength is largely an amino-acid prior that collapses here, while L barely moves).
        if all(c in d.columns for c in ["blosum", "dvol", "dhydro"]):
            for c in ["blosum", "dvol", "dhydro"]:
                d[c + "z"] = LD.zs(d[c])
            base = d[["burialz", "nbrz", "drsasaz", "blosumz", "dvolz", "dhydroz", "logP_mutz"]].to_numpy()
            v3, lo3, hi3, p3, _, _ = LD.cpi(y, grp, base, d.Lz.to_numpy().copy(), rng)
            baseL = d[["burialz", "nbrz", "drsasaz", "blosumz", "dvolz", "dhydroz", "Lz"]].to_numpy()
            vr, lor, hir, pr, _, _ = LD.cpi(y, grp, baseL, d.logP_mutz.to_numpy().copy(), rng)
            print(f"  CPI(L | geom+subst + one-pass)           = {v3:+.5f} [{lo3:+.5f}, {hi3:+.5f}]  <- two-pass, subst-controlled")
            print(f"  CPI(one-pass | geom+subst + L) [reverse] = {vr:+.5f} [{lor:+.5f}, {hir:+.5f}]  <- neither subsumes the other")
            rows.append(dict(feature="leverage L | geom+subst + one-pass", cpi_over_geom=round(v3, 5), lo=round(lo3, 5),
                             hi=round(hi3, 5), p_gt0=round(p3, 3), n_mut=len(d), n_designs=int(len(np.unique(grp))), n_destab=int(y.sum())))
            rows.append(dict(feature="one-pass | geom+subst + L (reverse)", cpi_over_geom=round(vr, 5), lo=round(lor, 5),
                             hi=round(hir, 5), p_gt0=round(pr, 3), n_mut=len(d), n_designs=int(len(np.unique(grp))), n_destab=int(y.sum())))
    pd.DataFrame(rows).assign(seed=SEED, fixture="Bennett de-novo binders",
                              command="python3 src/leverage_bennett_denovo.py").to_csv(a.out, index=False)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

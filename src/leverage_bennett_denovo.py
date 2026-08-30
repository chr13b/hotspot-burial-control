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
    pd.DataFrame(rows).assign(seed=SEED, fixture="Bennett de-novo binders",
                              command="python3 src/leverage_bennett_denovo.py").to_csv(a.out, index=False)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()

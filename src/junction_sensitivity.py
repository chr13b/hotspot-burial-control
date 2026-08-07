"""Junction-radius sensitivity for the single-chain-trained panel models.

PiFold and MIF have no chain representation; a SKEMPI complex is fed to them as one
concatenated pseudo-chain. The +-2 junction mask in p0_multimodel removes the residues
whose geometric features are fabricated at the artificial junction, but message passing
propagates that perturbation several residues further (MIF's own docstring measures a 0.13
nat residual). This recomputes every panel model's matched-pair gaps while excluding any
pair with EITHER member within `flank` residues (in sequence, same chain) of a chain
junction, for flank in {2, 6, 11}, so the single-chain-model negativity can be shown to
attenuate (or not) with exclusion radius.

Usage:
  python3 src/junction_sensitivity.py --out results/junction_sensitivity
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

N_BOOT, SEED = 10000, 20260803


def boot(df, col):
    rng = np.random.default_rng(SEED)
    cids = df["complex_id"].unique()
    by = {c: df.loc[df["complex_id"] == c, col].values for c in cids}
    m = np.array([np.nanmean(np.concatenate(
        [by[cids[i]] for i in rng.choice(len(cids), len(cids), True)]))
        for _ in range(N_BOOT)])
    return float(np.nanmean(df[col])), *np.nanpercentile(m, [2.5, 97.5])


def junction_distance(data_dir, complexes):
    """For every (complex, chain, resnum): sequence distance to nearest same-chain junction."""
    out = {}
    for cid in complexes:
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None:
            continue
        # junctions are positions where the concatenated chain label changes
        chg = np.flatnonzero(cx.chains[1:] != cx.chains[:-1]) + 1  # first index of each new chain
        bounds = np.concatenate([[0], chg, [cx.n]])
        for b in range(len(bounds) - 1):
            lo, hi = bounds[b], bounds[b + 1]
            for j in range(lo, hi):
                d = min(j - lo, hi - 1 - j)  # distance to this segment's ends
                out[(cid, cx.chains[j], int(cx.resnums[j]))] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/junction_sensitivity")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--pairs-prefix", default="results/p0_dssp")
    ap.add_argument("--panel-prefix", default="results/panel")
    ap.add_argument("--complexes", default="results/pair_complexes.txt")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    complexes = [l.strip() for l in open(a.complexes) if l.strip()]
    print("computing junction distances ...", flush=True)
    jdist = junction_distance(a.data_dir, complexes)

    # vanilla lives in p0_dssp_positions; the rest in panel_<model>_positions
    models = {"mpnn_vanilla": "results/p0_positions.csv"}
    for f in glob.glob(f"{a.panel_prefix}_*_positions.csv"):
        name = os.path.basename(f)[len("panel_"):-len("_positions.csv")]
        if name == "esmif": continue  # still scoring; add after it finishes
        models[name] = f

    tiers = ["SECONDARY_B_any_interface", "SENS_nbr_tol2_any_interface", "PRIMARY_loose_null"]
    rows = []
    for name, pos_csv in sorted(models.items()):
        if not os.path.exists(pos_csv):
            continue
        pos = pd.read_csv(pos_csv)
        key = pos.set_index(["complex_id", "chain", "resnum"])["logp_native"]
        for tier in tiers:
            pf = f"{a.pairs_prefix}_pairs_{tier}.csv"
            if not os.path.exists(pf):
                continue
            pr = pd.read_csv(pf)
            recs = []
            for _, r in pr.iterrows():
                try:
                    h = float(key.loc[(r.complex_id, r.hot_chain, r.hot_resnum)])
                    c = float(key.loc[(r.complex_id, r.ctl_chain, r.ctl_resnum)])
                except (KeyError, TypeError):
                    continue
                dh = jdist.get((r.complex_id, r.hot_chain, int(r.hot_resnum)), 9999)
                dc = jdist.get((r.complex_id, r.ctl_chain, int(r.ctl_resnum)), 9999)
                recs.append(dict(complex_id=r.complex_id, d=h - c, jmin=min(dh, dc)))
            d = pd.DataFrame(recs)
            for flank in (2, 6, 11):
                s = d[d["jmin"] >= flank]
                if len(s) < 10:
                    continue
                m, lo, hi = boot(s, "d")
                rows.append(dict(model=name, tier=tier, flank=flank, n=len(s),
                                 gap=m, lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    out.assign(command=cmd).to_csv(f"{a.out}.csv", index=False)

    print("\n=== panel gap vs junction-exclusion radius (single-chain models: pifold, mif) ===")
    for tier in tiers:
        print(f"\n{tier}")
        sub = out[out.tier == tier]
        for name in sorted(sub.model.unique()):
            cells = []
            for flank in (2, 6, 11):
                r = sub[(sub.model == name) & (sub.flank == flank)]
                if len(r):
                    r = r.iloc[0]
                    star = "*" if (r.lo > 0 or r.hi < 0) else " "
                    cells.append(f"±{flank}:{r.gap:+.3f}[{r.lo:+.3f},{r.hi:+.3f}]{star}(n={int(r.n)})")
            print(f"  {name:14s} " + "  ".join(cells))
    print(f"\n[done] wrote {a.out}.csv   (* = CI excludes zero)")


if __name__ == "__main__":
    main()

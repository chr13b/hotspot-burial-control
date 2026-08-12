"""Build a kl_triage joined table for Exp C2 generative backbones (interface-formed), for the
non-native validation of the KL-triage method (Task B / issue #12).

Per-position aggregate (mean over interface-formed generative backbones) of kl, logp_native, nbr from
the C2 backbone scoring, merged with the BACKBONE-INDEPENDENT experimental labels (is_interface, is_hot)
from the crystal kl_detector_joined table, keyed by (complex_id, chain, resnum, icode). Output columns
are exactly kl_triage's inputs: complex_id, chain, resnum, icode, is_interface, is_hot, nbr, kl, logp_native.

  python3 src/expD_build_triage_c2.py --scored $SCRATCH/expC2/scored_positions.csv \
    --perbackbone results/expC2_gap_perbackbone.csv --crystal results/kl_detector_joined.csv \
    --out $SCRATCH/ftax/expD/expC2_triage_joined.csv
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--perbackbone", required=True)
    ap.add_argument("--crystal", default="results/kl_detector_joined.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pb = pd.read_csv(a.perbackbone)
    keep_bb = set(pb.loc[(pb["interface_ok"] == 1) & (pb["partial_T"] > 0), "backbone_id"])
    print(f"[c2] interface-formed generative backbones: {len(keep_bb)}")

    sc = pd.read_csv(a.scored, usecols=lambda c: c in
                     ("backbone_id", "complex_id", "chain", "resnum", "icode", "nbr", "logp_native", "kl"))
    sc = sc[sc["backbone_id"].isin(keep_bb)].copy()
    sc["icode"] = sc["icode"].fillna("").astype(str)
    print(f"[c2] scored rows on those backbones: {len(sc)}")

    agg = (sc.groupby(["complex_id", "chain", "resnum", "icode"], as_index=False)
             .agg(nbr=("nbr", "mean"), logp_native=("logp_native", "mean"), kl=("kl", "mean"),
                  n_bb=("backbone_id", "nunique")))

    cr = pd.read_csv(a.crystal, usecols=lambda c: c in
                     ("complex_id", "chain", "resnum", "icode", "is_interface", "is_hot"))
    cr["icode"] = cr["icode"].fillna("").astype(str)
    cr = cr.drop_duplicates(["complex_id", "chain", "resnum", "icode"])

    m = agg.merge(cr, on=["complex_id", "chain", "resnum", "icode"], how="inner")
    m = m[np.isfinite(m["kl"]) & np.isfinite(m["nbr"]) & np.isfinite(m["logp_native"])]
    n_if = int((m["is_interface"] == 1).sum())
    n_hot = int(((m["is_interface"] == 1) & (m["is_hot"] == 1)).sum())
    n_hot_cx = m[(m["is_interface"] == 1) & (m["is_hot"] == 1)]["complex_id"].nunique()
    print(f"[c2] merged positions: {len(m)}  interface: {n_if}  interface-hot: {n_hot}  "
          f"complexes w/>=1 interface hotspot: {n_hot_cx}")
    m.to_csv(a.out, index=False)
    print(f"[c2] wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T4/R2 — cross-model confidence nugget: is 'confidence is not competence' a property of INVERSE
FOLDING, not just ProteinMPNN?

For each IF model we already ran on SKEMPI, take its per-residue confidence (log p(native | backbone),
i.e. how sure the model is of the native residue) and ask: does it rank interface HOTSPOTS above chance?
The nugget says no. We report single-feature confidence-AUROC per model (combiner-free), with burial as
the same-positions reference, complex-clustered bootstrap, seed 20260803.

Models: ProteinMPNN (kl_detector_joined), ESM-IF1 / PiFold / MIF (panel_*_positions), ProBID (top_prob,
a peakedness proxy — flagged, not log p_native). Labels (is_hot / is_interface / burial) come from
kl_detector_joined and are joined onto each panel by (complex_id, chain, resnum, icode).

  python3 src/xmodel_confidence.py --out results/xmodel_confidence.csv
"""
import argparse
import numpy as np, pandas as pd

SEED = 20260803
R = "results"


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_auc(df, score_col, rng, nboot=5000):
    cids = df.complex_id.unique()
    idx = {c: np.where(df.complex_id.values == c)[0] for c in cids}
    y = df.is_hot.values; s = df[score_col].values
    out = []
    for _ in range(nboot):
        take = np.concatenate([idx[c] for c in rng.choice(cids, len(cids), True)])
        a = auc(y[take], s[take])
        if not np.isnan(a):
            out.append(a)
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/xmodel_confidence.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    j = pd.read_csv(f"{R}/kl_detector_joined.csv")
    j["icode"] = j.icode.fillna("").astype(str)
    labels = j[["complex_id", "chain", "resnum", "icode", "is_hot", "is_interface", "burial"]].copy()

    def prep(df, conf_col):
        df = df.copy()
        df["icode"] = df.icode.fillna("").astype(str) if "icode" in df else ""
        m = df.merge(labels, on=["complex_id", "chain", "resnum", "icode"], how="inner")
        m = m[m.is_interface == 1].dropna(subset=[conf_col, "is_hot", "burial"])
        m = m.rename(columns={conf_col: "conf"})
        return m.reset_index(drop=True)

    sources = []
    # ProteinMPNN — logp_native already in the joined table
    pm = j[j.is_interface == 1].dropna(subset=["logp_native", "is_hot", "burial"]).copy()
    pm = pm.rename(columns={"logp_native": "conf"}).reset_index(drop=True)
    sources.append(("ProteinMPNN", "logp_native", pm))
    # ESM-IF1 / PiFold / MIF panels — logp_native per position
    for name, f in [("ESM-IF1", "panel_esmif_positions.csv"),
                    ("PiFold", "panel_pifold_positions.csv"),
                    ("MIF", "panel_mif_positions.csv")]:
        try:
            sources.append((name, "logp_native", prep(pd.read_csv(f"{R}/{f}"), "logp_native")))
        except FileNotFoundError:
            print(f"  [skip] {name}: {f} not found")
    # ProBID — only top_prob (peakedness of the argmax), NOT logp_native. Different quantity; flagged.
    # ProBID keys by bare 4-char PDB (1A22) while joined complex_id is 1A22_A_B → join on the base code.
    try:
        pb = pd.read_csv(f"{R}/probid_positions.csv")
        pb["pdb_base"] = pb.pdb.astype(str).str.upper()
        lab_pb = labels.copy()
        lab_pb["pdb_base"] = lab_pb.complex_id.astype(str).str.replace(r"[_.].*", "", regex=True).str.upper()
        # drop (pdb_base,chain,resnum) keys with >1 icode → avoid mislabeling across insertion codes
        lab_pb = lab_pb.drop_duplicates(subset=["pdb_base", "chain", "resnum"], keep=False)
        pbm = pb.merge(lab_pb, on=["pdb_base", "chain", "resnum"], how="inner")
        pbm = pbm[pbm.is_interface == 1].dropna(subset=["top_prob", "is_hot", "burial"])
        pbm = pbm.rename(columns={"top_prob": "conf"}).reset_index(drop=True)
        print(f"  [ProBID join] {pb.pdb.nunique()} pdbs → {pbm.complex_id.nunique()} matched complexes, {len(pbm)} interface rows")
        sources.append(("ProBID(top_prob)", "top_prob", pbm))
    except FileNotFoundError:
        print("  [skip] ProBID: probid_positions.csv not found")

    rows = []
    for name, conf_col, m in sources:
        n, nh = len(m), int(m.is_hot.sum())
        if n == 0 or nh == 0:
            print(f"  [warn] {name}: n={n} n_hot={nh} — join produced no usable rows (positive-control FAIL)")
            rows.append(dict(model=name, conf_feature=conf_col, n=n, n_hot=nh,
                             auroc_conf=np.nan, conf_lo=np.nan, conf_hi=np.nan,
                             auroc_burial=np.nan, d_conf_minus_burial=np.nan, note="EMPTY JOIN"))
            continue
        ac = auc(m.is_hot, m.conf); clo, chi = boot_auc(m, "conf", rng)
        ab = auc(m.is_hot, m.burial)
        verdict = "confidence ~ CHANCE" if (clo <= 0.5 <= chi) else ("ABOVE chance" if clo > 0.5 else "BELOW chance")
        print(f"  {name:16s} n={n:5d} hot={nh:4d}  conf-AUROC={ac:.3f} [{clo:.3f},{chi:.3f}]  "
              f"burial-AUROC={ab:.3f}  Δ(conf-bur)={ac-ab:+.3f}  {verdict}")
        rows.append(dict(model=name, conf_feature=conf_col, n=n, n_hot=nh,
                         auroc_conf=round(ac, 4), conf_lo=round(clo, 4), conf_hi=round(chi, 4),
                         auroc_burial=round(ab, 4), d_conf_minus_burial=round(ac - ab, 4), note=verdict))

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out["command"] = "python3 src/xmodel_confidence.py"
    out.to_csv(a.out, index=False)
    n_chance = int((out.note == "confidence ~ CHANCE").sum())
    print(f"\n{n_chance}/{len(out)} models: interface-hotspot confidence-AUROC spans chance. wrote {a.out}")


if __name__ == "__main__":
    main()

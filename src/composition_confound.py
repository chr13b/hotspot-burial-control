#!/usr/bin/env python3
"""Q1-Extension-1: is the AMINO-ACID-COMPOSITION confound GENERAL across inverse-folding architectures?

ProBID-Net's hotspot deficit is largely explained by hotspots being enriched in residue types its voxel-CNN
recovers worst. Test whether the SAME holds for the panel (ESM-IF1, MIF, ProteinMPNN-soluble, PiFold): do
all models recover the hotspot-enriched residue types worse, and does residue-type composition ALONE
predict each model's uncontrolled interface hotspot deficit? If yes -> composition is a SECOND general
confound alongside burial (strengthens "the gap is a confound, not hotspot chemistry").

Recovery = (argmax mode_aa == native aa). Interface + hotspot labels from p0_positions (hot = hot_strict,
Ala-scan ΔΔG>2). Committed CSVs only. Writes results/composition_confound.csv.
  python3 src/composition_confound.py --out results/composition_confound.csv
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy import stats

AAs = list("ACDEFGHIKLMNPQRSTVWY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/composition_confound.csv")
    a = ap.parse_args()
    R = "results"
    lab = pd.read_csv(f"{R}/p0_positions.csv", usecols=["complex_id", "chain", "resnum", "icode",
                                                        "is_interface", "label"])
    lab["icode"] = lab.icode.fillna("").astype(str)
    lab["is_hot"] = (lab.label == "hot_strict").astype(int)

    recall_by_model = {}
    rows = []
    for pf in sorted(glob.glob(f"{R}/panel_*_positions.csv")):
        model = os.path.basename(pf).split("_")[1]
        d = pd.read_csv(pf)
        if "mode_aa" not in d or "aa" not in d:
            continue
        d["icode"] = d.icode.fillna("").astype(str)
        d["rec"] = (d.mode_aa == d.aa).astype(int)
        d = d.merge(lab, on=["complex_id", "chain", "resnum", "icode"], how="inner")
        d = d[d.is_interface == 1]
        hot = d[d.is_hot == 1]; non = d[d.is_hot == 0]
        if len(hot) < 20:
            continue
        # per-residue-type recall (this model)
        recall = d.groupby("aa").rec.mean()
        recall_by_model[model] = recall.reindex(AAs)
        # composition of hot vs non-hot interface positions
        comp_hot = hot.aa.value_counts(normalize=True).reindex(AAs).fillna(0)
        comp_non = non.aa.value_counts(normalize=True).reindex(AAs).fillna(0)
        # actual uncontrolled deficit vs composition-predicted deficit
        actual = hot.rec.mean() - non.rec.mean()
        pred = float((comp_hot * recall.reindex(AAs).fillna(recall.mean())).sum()
                     - (comp_non * recall.reindex(AAs).fillna(recall.mean())).sum())
        # hard types this model recovers worst, and hotspot enrichment there
        hard = recall.reindex(AAs).nsmallest(6).index.tolist()
        f_hot_hard = comp_hot[hard].sum(); f_non_hard = comp_non[hard].sum()
        print(f"[{model:12s}] actual deficit {actual:+.3f}  | composition-predicted {pred:+.3f}  "
              f"| hardest6={''.join(hard)}  hot in hard {f_hot_hard:.2f} vs non {f_non_hard:.2f}")
        rows.append(dict(model=model, actual_deficit=round(actual, 4), composition_predicted=round(pred, 4),
                         frac_explained=round(pred / actual, 2) if abs(actual) > 1e-6 else np.nan,
                         hardest6=''.join(hard), hot_frac_in_hard=round(f_hot_hard, 3),
                         non_frac_in_hard=round(f_non_hard, 3), n_hot=len(hot), n_interface=len(d)))

    # cross-model consistency of per-residue-type recall (are the same types hard everywhere?)
    print("\ncross-model Spearman of per-residue-type recall (are hard types consistent?):")
    mods = list(recall_by_model)
    for i in range(len(mods)):
        for k in range(i + 1, len(mods)):
            r = stats.spearmanr(recall_by_model[mods[i]], recall_by_model[mods[k]], nan_policy="omit")
            print(f"  {mods[i]:12s} vs {mods[k]:12s}: ρ={r.correlation:+.3f}")
            rows.append(dict(model=f"recall_corr::{mods[i]}_vs_{mods[k]}", actual_deficit=round(r.correlation, 3)))

    out = pd.DataFrame(rows); out["command"] = "python3 src/composition_confound.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

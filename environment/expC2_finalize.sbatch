#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:30:00
#SBATCH -J expC2_final
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_finalize.out
# Concat score shards, then the full C2 analysis. Submit after the score array:
#   sbatch --dependency=afterok:<score_arrayid> expC2_finalize.sbatch
set -e
source "$SCRATCH/ftax/env.sh"
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"
S="$SCRATCH/expC2"

echo "=== concat score shards ==="
python3 src/expC_concat.py "$S"

echo "=== C2-PRIMARY gap + C2-KL (pre-registered SECONDARY_B within-binder) ==="
python3 src/expC_analyze.py --positions "$S/scored_positions.csv" --backbones "$S/scored_backbones.csv" \
  --binder-map results/expC_complexes.csv --variant SECONDARY_B_any_interface --out results/expC2

echo "=== robustness: within-binder RE-MATCH gap ==="
python3 src/expC_analyze.py --positions "$S/scored_positions.csv" --backbones "$S/scored_backbones.csv" \
  --binder-map results/expC_complexes.csv --variant EXPC_within_binder --out results/expC2_rematch

echo "=== C2-PRIMARY continuous slope (physical-generated) + naive contrast ==="
python3 src/expC2_slope_check.py --gap-perbackbone results/expC2_gap_perbackbone.csv \
  --label expC2 --out results/expC2_slope_check.csv

echo "=== KILL C2b: conditioning leakage (conditioned vs held-out arm) ==="
python3 src/expC2_leakage.py --positions "$S/scored_positions.csv" --backbones "$S/scored_backbones.csv" \
  --split results/expC2_hotspot_split.csv --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
  --binder-map results/expC_complexes.csv --out results/expC2_leakage.csv

echo "=== SECONDARY: ddG_bind vs log-odds | burial, vs iRMSD ==="
python3 src/expC_secondary.py --positions "$S/scored_positions.csv" --backbones "$S/scored_backbones.csv" \
  --out results/expC2_secondary.csv

echo "=== interface QC: interface-formed fraction per level (the fix's headline) ==="
python3 - <<'PY'
import pandas as pd, os
S=os.path.expandvars("$SCRATCH/expC2")
bb=pd.read_csv(f"{S}/scored_backbones.csv")
q=bb.groupby("partial_T").agg(n=("backbone_id","nunique"), iface_ok=("interface_ok","sum"),
    med_irmsd=("irmsd","median"), med_tgtrmsd=("tgt_rmsd","median")).reset_index()
q["iface_frac"]=(q["iface_ok"]/q["n"]).round(3)
print(q.to_string(index=False)); q.to_csv("results/expC2_interface_qc.csv", index=False)
PY
echo "=== finalize done $(date) ==="

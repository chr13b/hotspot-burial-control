#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J expC_final
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC_finalize.out
# Concat score shards, then dose-response analysis (pre-reg gap + re-match gap + KL) + SECONDARY.
# Submit after the score array: sbatch --dependency=afterok:<score_arrayid> expC_finalize.sbatch
set -e
source "$SCRATCH/ftax/env.sh"
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"
S="$SCRATCH/expC"

echo "=== concat score shards ==="
python3 src/expC_concat.py "$S"

echo "=== C-PRIMARY gap + C-KL (pre-registered SECONDARY_B within-binder) ==="
python3 src/expC_analyze.py --positions "$S/scored_positions.csv" --backbones "$S/scored_backbones.csv" \
  --binder-map results/expC_complexes.csv --variant SECONDARY_B_any_interface --out results/expC

echo "=== robustness: within-binder RE-MATCH gap (exploratory) ==="
python3 src/expC_analyze.py --positions "$S/scored_positions.csv" --backbones "$S/scored_backbones.csv" \
  --binder-map results/expC_complexes.csv --variant EXPC_within_binder --out results/expC_rematch

echo "=== SECONDARY: ddG_bind vs log-odds | burial, vs iRMSD ==="
python3 src/expC_secondary.py --positions "$S/scored_positions.csv" --backbones "$S/scored_backbones.csv" \
  --out results/expC_secondary.csv

echo "=== interface QC: backbones dropped (interface dissolved) per level ==="
python3 - <<'PY'
import pandas as pd, os
S=os.path.expandvars("$SCRATCH/expC")
bb=pd.read_csv(f"{S}/scored_backbones.csv")
q=bb.groupby("partial_T").agg(n=("backbone_id","nunique"),
    iface_ok=("interface_ok","sum"),
    med_irmsd=("irmsd","median"), med_tgtrmsd=("tgt_rmsd","median")).reset_index()
print(q.to_string(index=False))
q.to_csv("results/expC_interface_qc.csv", index=False)
PY
echo "=== finalize done $(date) ==="

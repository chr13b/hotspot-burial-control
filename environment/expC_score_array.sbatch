#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 3:00:00
#SBATCH -J expC_score
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC_score_%A_%a.out
# Parallel scoring of all backbones (T0 controls + T5-40 ladder). Submit: sbatch --array=0-7%8 expC_score_array.sbatch
set -e
source "$SCRATCH/ftax/env.sh"
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"
N=8
python3 src/expC_score.py \
  --backbones-dir "$SCRATCH/expC/backbones" \
  --outmap results/expC_outmap.json \
  --crystal-dir "$FTAX_DATA/PDBs" \
  --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
  --mpnn-weights "$SCRATCH/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt" \
  --out "$SCRATCH/expC/scored_sh${SLURM_ARRAY_TASK_ID}" \
  --shard "${SLURM_ARRAY_TASK_ID}/${N}" --threads 4
echo "shard ${SLURM_ARRAY_TASK_ID}/${N} done $(date)"

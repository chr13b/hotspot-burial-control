#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 4:00:00
#SBATCH -J expC2_score
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_score_%A_%a.out
# Parallel scoring of all C2 backbones (T0 controls + noised ladder). Submit: sbatch --array=0-7%8 expC2_score_array.sbatch
# Reuses src/expC_score.py verbatim (same outmap, same pairs, same MPNN weights); only the backbones dir + out prefix change.
set -e
source "$SCRATCH/ftax/env.sh"
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"
N=8
python3 src/expC_score.py \
  --backbones-dir "$SCRATCH/expC2/backbones" \
  --outmap results/expC_outmap.json \
  --crystal-dir "$FTAX_DATA/PDBs" \
  --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
  --mpnn-weights "$SCRATCH/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt" \
  --out "$SCRATCH/expC2/scored_sh${SLURM_ARRAY_TASK_ID}" \
  --shard "${SLURM_ARRAY_TASK_ID}/${N}" --threads 4
echo "shard ${SLURM_ARRAY_TASK_ID}/${N} done $(date)"

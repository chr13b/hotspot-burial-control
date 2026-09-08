#!/bin/bash
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -J foldx
#SBATCH -o /scratch/users/cbertsch/ftax/foldx/logs/foldx_%A_%a.log
# Parameterized FoldX driver array (src/foldx_ddg.py). shard = SLURM_ARRAY_TASK_ID.
#   RepairPDB:  sbatch --array=0-29%30   --export=ALL,MODE=repair,NSHARDS=30 jobs_foldx_array.sbatch
#   Lane B ddg: sbatch --array=0-59%40   --export=ALL,MODE=ddg,LANE=B,NSHARDS=60,RUNS=5 jobs_foldx_array.sbatch
#   Lane A ddg: sbatch --array=0-99%50   --export=ALL,MODE=ddg,LANE=A,NSHARDS=100,RUNS=5 jobs_foldx_array.sbatch
set -uo pipefail
source "$SCRATCH/ftax/env.sh"
export FOLDX_BIN="$SCRATCH/ftax/foldx/v5_0/foldx_20261231"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
mkdir -p "$SCRATCH/ftax/foldx/logs"
MODE=${MODE:?set MODE}; NSHARDS=${NSHARDS:?set NSHARDS}; LANE=${LANE:-B}; RUNS=${RUNS:-5}
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID MODE=$MODE LANE=$LANE NSHARDS=$NSHARDS RUNS=$RUNS start=$(date)"
if [ "$MODE" = "repair" ]; then
  python3 src/foldx_ddg.py --mode repair --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
else
  OUT="results/_foldx_${LANE}_s${SLURM_ARRAY_TASK_ID}.csv"
  python3 src/foldx_ddg.py --mode ddg --lane "$LANE" --shard "$SLURM_ARRAY_TASK_ID" \
      --nshards "$NSHARDS" --runs "$RUNS" --out "$OUT"
fi
echo "task $SLURM_ARRAY_TASK_ID DONE $(date)"

#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -J of3_predict
#SBATCH -o /scratch/users/cbertsch/ftax/logs/of3_predict_%A_%a.out
# --array is set on the command line: sbatch --array=0-N%W of3_predict_array.sbatch
set -e
source "$SCRATCH/ftax/env_of3.sh"
# CHUNK can be overridden via --export=ALL,CHUNK=...; else use the array index.
CHUNK="${CHUNK:-${CHUNKDIR:-$SCRATCH/ftax/expA/chunks}/chunk_${SLURM_ARRAY_TASK_ID}.json}"
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK start=$(date)"; nvidia-smi -L
run_openfold predict \
  --query-json "$CHUNK" \
  --inference-ckpt-path "$SCRATCH/.openfold3/of3-p2-155k.pt" \
  --use-msa-server true \
  --use-templates false \
  --output-dir "$SCRATCH/ftax/predicted/of3_out" \
  --num-diffusion-samples 5 \
  --num-model-seeds 1
echo "task $SLURM_ARRAY_TASK_ID end=$(date)"

#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J of3_1A22
#SBATCH -o /scratch/users/cbertsch/ftax/logs/of3_1A22_%j.out
set -e
source "$SCRATCH/ftax/env_of3.sh"
echo "node=$(hostname) start=$(date)"; nvidia-smi -L
run_openfold predict \
  --query-json "$SCRATCH/ftax/expA/queries_1A22.json" \
  --inference-ckpt-path "$SCRATCH/.openfold3/of3-p2-155k.pt" \
  --use-msa-server true \
  --use-templates false \
  --output-dir "$SCRATCH/ftax/expA/pred_test" \
  --num-diffusion-samples 5 \
  --num-model-seeds 1
echo "end=$(date)"
echo "=== output tree ==="
find "$SCRATCH/ftax/expA/pred_test" -maxdepth 3 | head -40

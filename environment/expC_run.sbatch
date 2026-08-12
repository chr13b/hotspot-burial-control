#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_CC:7.0
# ^ V100 only: torch 1.9.1/cu111 in SE3nv compiles kernels up to sm_80; Sherlock's 24/48/80 GB
#   cards are all sm_86+ (RTX_3090 8.6 / L40S 8.9 / H100 9.0) and would fail "no kernel image".
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH -J expC_rfdiff
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC_rfdiff_%A_%a.out
# Submit as: sbatch --array=0-9%10 expC_run.sbatch   (round-robin over the 55 complexes)
set -e
source "$SCRATCH/ftax/env_se3.sh"
NTASKS="${NTASKS:-10}"
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
MAN="$REPO/results/expC_inputs.csv"
OUT="$SCRATCH/expC/backbones"; mkdir -p "$OUT"
CKPT="$RFDIFFUSION_DIR/models/Complex_base_ckpt.pt"
cd "$RFDIFFUSION_DIR"
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID start=$(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

i=0
tail -n +2 "$MAN" | while IFS=, read -r cid inp contig rest; do
  if [ $(( i % NTASKS )) -eq "$SLURM_ARRAY_TASK_ID" ]; then
    for T in 5 10 20 40; do
      # resume: skip if the 3rd design already exists
      if [ -f "$OUT/${cid}_T${T}_2.pdb" ]; then echo "skip $cid T$T (done)"; continue; fi
      echo "=== $cid  partial_T=$T  $(date) ==="
      python scripts/run_inference.py \
        inference.input_pdb="$inp" \
        "contigmap.contigs=[$contig]" \
        diffuser.partial_T="$T" inference.num_designs=3 \
        inference.write_trajectory=False \
        inference.output_prefix="$OUT/${cid}_T${T}" \
        inference.ckpt_override_path="$CKPT" 2>&1 | tail -2
    done
  fi
  i=$((i+1))
done
echo "task $SLURM_ARRAY_TASK_ID done $(date)"

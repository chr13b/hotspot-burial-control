#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -J boltz_fold
#SBATCH -o /scratch/users/cbertsch/ftax/logs/boltz_%A_%a.out
# Boltz-2 cross-folder (Phase 3). Boltz-2 is memory-hungry -> >=32GB GPUs (11GB OOM'd in the gate).
# Same MMseqs2 server MSA as AF2 (--use_msa_server) for MSA parity. Idempotent skip on the confidence JSON.
#   sbatch --array=0-M%6 --export=ALL,STRIDE=42,INDIR=...,OUT=... boltz_fold_array.sbatch
set -uo pipefail
source $SCRATCH/ftax/env_boltz.sh 2>/dev/null
INDIR=${INDIR:-$SCRATCH/ftax/iptm/boltz_in}
OUT=${OUT:-$SCRATCH/ftax/iptm/boltz_out}
mkdir -p "$OUT" $SCRATCH/ftax/logs
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID INDIR=$INDIR OUT=$OUT $(date)"; nvidia-smi -L || true
mapfile -t ALL < "$INDIR/manifest.txt"
N=${#ALL[@]}; STRIDE=${STRIDE:-42}
start=$(( SLURM_ARRAY_TASK_ID * STRIDE ))
echo "folds [$start,$((start+STRIDE))) of $N"
for (( i=start; i<start+STRIDE && i<N; i++ )); do
  fid=${ALL[$i]}
  if [ -n "$(find "$OUT/$fid" -name 'confidence_*_model_0.json' 2>/dev/null | head -1)" ]; then
    echo "skip $fid (done)"; continue
  fi
  echo "=== [$i/$N] $fid $(date +%H:%M:%S) ==="
  boltz predict "$INDIR/$fid.fasta" --out_dir "$OUT/$fid" --cache "$BOLTZ_CACHE" \
     --devices 1 --accelerator gpu --output_format pdb --use_msa_server --override 2>&1 | tail -3 \
     || echo "FAILED $fid rc=$?"
done
echo "boltz task $SLURM_ARRAY_TASK_ID DONE $(date)"

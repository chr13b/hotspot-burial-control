#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 24:00:00
#SBATCH -J iptm_fold_pred
#SBATCH -o /scratch/users/cbertsch/ftax/logs/iptm_fold_pred_%A_%a.out
# PHASE 2 — fold the PREDICTED-backbone (OF3) steered sequences, AF2-multimer (ColabFold container), TEMPLATES OFF,
# ONE model (identical to the crystal ipTM run; only the input FASTAs + output dir differ). Idempotent (skips done).
# submit: sbatch --array=0-18%6 --export=ALL,STRIDE=8 jobs_iptm_fold_pred.sbatch   (147 folds = 21 cx x 7 cond)
set -uo pipefail
SIF=$SCRATCH/ftax/colabfold.sif
PARAMS=$SCRATCH/ftax/colabfold_data
FADIR=$SCRATCH/ftax/iptm/fastas_pred
OUT=$SCRATCH/ftax/iptm/af2_out_pred
mkdir -p "$OUT" $SCRATCH/ftax/logs $SCRATCH/ftax/tmp
export APPTAINER_TMPDIR=$SCRATCH/ftax/tmp TMPDIR=$SCRATCH/ftax/tmp
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID start=$(date)"; nvidia-smi -L || true
[ -f "$SIF" ] || { echo "NO SIF at $SIF"; exit 1; }
mapfile -t ALL < "$FADIR/manifest.txt"
N=${#ALL[@]}; STRIDE=${STRIDE:-8}
start=$(( SLURM_ARRAY_TASK_ID * STRIDE ))
echo "folds [$start,$((start+STRIDE))) of $N"
for (( i=start; i<start+STRIDE && i<N; i++ )); do
  fid=${ALL[$i]}; fa=$FADIR/$fid.fasta
  if ls "$OUT/$fid/${fid}_unrelaxed_rank_001_"*.pdb >/dev/null 2>&1; then echo "skip $fid (done)"; continue; fi
  mkdir -p "$OUT/$fid"
  echo "=== [$i/$N] $fid $(date) ==="
  # timeout: the ColabFold mmseqs2 MSA server intermittently strands a request in PENDING forever (seen at 2h+);
  # cap each fold at 30 min so a stuck MSA is killed and the loop continues — a later (idempotent) resubmit retries it.
  timeout 1800 apptainer exec --nv -B $SCRATCH:$SCRATCH \
     --env TF_FORCE_UNIFIED_MEMORY=1 --env XLA_PYTHON_CLIENT_MEM_FRACTION=4.0 \
     "$SIF" colabfold_batch "$fa" "$OUT/$fid" \
        --data "$PARAMS" --model-type alphafold2_multimer_v3 \
        --msa-mode mmseqs2_uniref_env --num-models 1 --num-recycle 3 --num-seeds 1 --rank auto \
     || echo "FAILED/timeout $fid rc=$? $(date)"
done
echo "iptm_fold_pred task $SLURM_ARRAY_TASK_ID DONE $(date)"

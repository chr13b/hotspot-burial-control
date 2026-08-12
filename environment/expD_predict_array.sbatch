#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 24:00:00
#SBATCH -J expD_af2
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_af2_%A_%a.out
# AF2-multimer (ColabFold container) prediction of the 141 pair complexes, TEMPLATES OFF.
# --array set on the command line: sbatch --array=0-17%6 expD_predict_array.sbatch
set -uo pipefail
SIF=$SCRATCH/ftax/colabfold.sif
PARAMS=$SCRATCH/ftax/colabfold_data
FADIR=$SCRATCH/ftax/expD/fastas
OUT=$SCRATCH/ftax/expD/af2_out
mkdir -p "$OUT" "$PARAMS" $SCRATCH/ftax/logs $SCRATCH/ftax/tmp
export APPTAINER_TMPDIR=$SCRATCH/ftax/tmp TMPDIR=$SCRATCH/ftax/tmp

echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID start=$(date)"; nvidia-smi -L || true
[ -f "$SIF" ] || { echo "NO SIF at $SIF"; exit 1; }
mapfile -t ALL < "$FADIR/manifest.txt"
N=${#ALL[@]}
STRIDE=${STRIDE:-8}
start=$(( SLURM_ARRAY_TASK_ID * STRIDE ))
echo "processing complexes [$start, $((start+STRIDE))) of $N"
for (( i=start; i<start+STRIDE && i<N; i++ )); do
  cid=${ALL[$i]}
  fa=$FADIR/$cid.fasta
  if ls "$OUT/$cid/${cid}_unrelaxed_rank_001_"*.pdb >/dev/null 2>&1; then
    echo "skip $cid (rank_001 exists)"; continue
  fi
  mkdir -p "$OUT/$cid"
  echo "=== [$i/$N] $cid $(date) ==="
  apptainer exec --nv -B $SCRATCH:$SCRATCH \
     --env TF_FORCE_UNIFIED_MEMORY=1 --env XLA_PYTHON_CLIENT_MEM_FRACTION=4.0 \
     "$SIF" colabfold_batch "$fa" "$OUT/$cid" \
        --data "$PARAMS" \
        --model-type alphafold2_multimer_v3 \
        --msa-mode mmseqs2_uniref_env \
        --num-models 5 --num-recycle 3 --num-seeds 1 --rank auto \
     || echo "FAILED $cid rc=$?"
done
echo "expD_af2 task $SLURM_ARRAY_TASK_ID DONE $(date)"

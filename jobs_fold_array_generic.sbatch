#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 24:00:00
#SBATCH -J fold_arr
#SBATCH -o /scratch/users/cbertsch/ftax/logs/fold_%A_%a.out
# Parameterized AF2-multimer (ColabFold) fold array. Reuse for any manifest by overriding FADIR/OUT:
#   sbatch --array=0-M%6 --export=ALL,STRIDE=8,FADIR=$SCRATCH/ftax/iptm/fastas_batch2,OUT=$SCRATCH/ftax/iptm/af2_out fold_array_generic.sbatch
# Identical settings to the committed iptm_fold_array.sbatch (templates OFF, one model, mmseqs2 MSA, idempotent skip).
set -uo pipefail
SIF=$SCRATCH/ftax/colabfold.sif
PARAMS=$SCRATCH/ftax/colabfold_data
FADIR=${FADIR:?set FADIR}
OUT=${OUT:?set OUT}
mkdir -p "$OUT" $SCRATCH/ftax/logs $SCRATCH/ftax/tmp
export APPTAINER_TMPDIR=$SCRATCH/ftax/tmp TMPDIR=$SCRATCH/ftax/tmp
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID FADIR=$FADIR OUT=$OUT start=$(date)"; nvidia-smi -L || true
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
  apptainer exec --nv -B $SCRATCH:$SCRATCH \
     --env TF_FORCE_UNIFIED_MEMORY=1 --env XLA_PYTHON_CLIENT_MEM_FRACTION=4.0 \
     "$SIF" colabfold_batch "$fa" "$OUT/$fid" \
        --data "$PARAMS" --model-type alphafold2_multimer_v3 \
        --msa-mode mmseqs2_uniref_env --num-models 1 --num-recycle 3 --num-seeds 1 --rank auto \
     || echo "FAILED $fid rc=$?"
done
echo "fold task $SLURM_ARRAY_TASK_ID DONE $(date)"

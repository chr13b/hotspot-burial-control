#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 4:00:00
#SBATCH -J expD_af2_val
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_af2_validate.out
# Validate AF2-multimer (ColabFold container) on ONE complex: confirm it runs on the GPU, templates
# OFF, expected output layout, AND pre-download the AF2 params to $SCRATCH so the array doesn't race.
set -uo pipefail
SIF=$SCRATCH/ftax/colabfold.sif
PARAMS=$SCRATCH/ftax/colabfold_data
FADIR=$SCRATCH/ftax/expD/fastas
OUT=$SCRATCH/ftax/expD/af2_out
mkdir -p "$OUT" "$PARAMS" $SCRATCH/ftax/tmp
export APPTAINER_TMPDIR=$SCRATCH/ftax/tmp TMPDIR=$SCRATCH/ftax/tmp

cid=${CID:-1A22_A_B}
echo "node=$(hostname) $(date)"; nvidia-smi -L
[ -f "$SIF" ] || { echo "NO SIF at $SIF"; exit 1; }
mkdir -p "$OUT/$cid"
echo "=== predicting $cid (templates OFF) via container ==="
time apptainer exec --nv -B $SCRATCH:$SCRATCH \
   --env TF_FORCE_UNIFIED_MEMORY=1 --env XLA_PYTHON_CLIENT_MEM_FRACTION=4.0 \
   "$SIF" colabfold_batch "$FADIR/$cid.fasta" "$OUT/$cid" \
     --data "$PARAMS" \
     --model-type alphafold2_multimer_v3 \
     --msa-mode mmseqs2_uniref_env \
     --num-models 5 --num-recycle 3 --num-seeds 1 --rank auto \
   || { echo "VALIDATE_PREDICT_FAILED rc=$?"; exit 1; }
echo "=== outputs ==="; ls -la "$OUT/$cid" | head -40
echo "=== rank_001 pdb present? ==="; ls "$OUT/$cid/${cid}_unrelaxed_rank_001_"*.pdb 2>/dev/null && echo OK || echo "NO rank_001 pdb"
echo "=== templates flag (expect off/false) ==="; grep -iE 'template' "$OUT/$cid"/config.json 2>/dev/null | head
echo "expD_af2_validate DONE $(date)"

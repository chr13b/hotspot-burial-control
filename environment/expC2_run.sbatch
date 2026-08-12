#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=GPU_CC:7.0
# ^ V100 only: torch 1.9.1/cu111 in SE3nv compiles kernels up to sm_80; Sherlock's 24/48/80 GB cards
#   are sm_86+ (RTX_3090/L40S/H100) and fail "no kernel image".
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH -J expC2_rfdiff
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_rfdiff_%A_%a.out
# C2 interface-pinned partial diffusion. Submit full ladder: sbatch --array=0-9%10 expC2_run.sbatch
# Validation pass (subset): OUTDIR=... ONLY_CIDS="1A22_A_B 1BP3_A_B" TLEVELS="5 20" NDES=2 NTASKS=1 \
#                           sbatch --array=0-0 expC2_run.sbatch
# Option-B fallback: CKPT_OVERRIDE=$RFDIFFUSION_DIR/models/Base_ckpt.pt USE_HOTSPOTS=0 ...
set -e
source "$SCRATCH/ftax/env_se3.sh"
NTASKS="${NTASKS:-10}"
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
MAN="$REPO/results/expC_inputs.csv"
HRES_JSON="$REPO/results/expC2_hotspot_res.json"
OUT="${OUTDIR:-$SCRATCH/expC2/backbones}"; mkdir -p "$OUT"
CKPT="${CKPT_OVERRIDE:-$RFDIFFUSION_DIR/models/Complex_base_ckpt.pt}"
TLEVELS="${TLEVELS:-5 10 15 20 30}"
NDES="${NDES:-6}"
LASTIDX=$((NDES-1))

# cid -> hotspot_res inner string (comma-joined target tokens in input-PDB chain space), from committed json
declare -A HRES
while IFS=$'\t' read -r cid inner; do HRES["$cid"]="$inner"; done < <(python -c "
import json
d=json.load(open('$HRES_JSON'))
for k,v in d.items(): print(k+'\t'+v)
")

cd "$RFDIFFUSION_DIR"
echo "node=$(hostname) task=$SLURM_ARRAY_TASK_ID start=$(date)"
echo "ckpt=$CKPT  Tlevels=[$TLEVELS]  ndes=$NDES  use_hotspots=${USE_HOTSPOTS:-1}  out=$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader

i=0
tail -n +2 "$MAN" | while IFS=, read -r cid inp contig rest; do
  if [ -n "$ONLY_CIDS" ] && [[ " $ONLY_CIDS " != *" $cid "* ]]; then i=$((i+1)); continue; fi
  if [ $(( i % NTASKS )) -eq "$SLURM_ARRAY_TASK_ID" ]; then
    hr="${HRES[$cid]}"
    HRES_ARG=""
    if [ -n "$hr" ] && [ "${USE_HOTSPOTS:-1}" != "0" ]; then HRES_ARG="ppi.hotspot_res=[$hr]"; fi
    for T in $TLEVELS; do
      if [ -f "$OUT/${cid}_T${T}_${LASTIDX}.pdb" ]; then echo "skip $cid T$T (done)"; continue; fi
      echo "=== $cid partial_T=$T $([ -n "$HRES_ARG" ] && echo "hotspot_res=[$hr]" || echo "no-hotspot") $(date) ==="
      python scripts/run_inference.py \
        inference.input_pdb="$inp" \
        "contigmap.contigs=[$contig]" \
        $HRES_ARG \
        diffuser.partial_T="$T" inference.num_designs=$NDES \
        inference.write_trajectory=False \
        inference.output_prefix="$OUT/${cid}_T${T}" \
        inference.ckpt_override_path="$CKPT" 2>&1 | tail -3
    done
  fi
  i=$((i+1))
done
echo "task $SLURM_ARRAY_TASK_ID done $(date)"

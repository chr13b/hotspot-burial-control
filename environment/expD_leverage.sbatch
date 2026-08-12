#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:20:00
#SBATCH -J expD_lev
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_leverage.out
# Task 2 (CPU, predictor-independent): symmetric leverage jackknife + build AF2 FASTAs (prep).
set -euo pipefail
source $SCRATCH/ftax/env.sh
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"

echo "=== Task 2: symmetric leverage / jackknife ==="
python3 src/expD_leverage.py \
  --expA-pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
  --of3-positions $SCRATCH/ftax/predicted/expA_p0_positions.csv \
  --c2-perbackbone results/expC2_gap_perbackbone.csv \
  --out results/expD_leverage.csv

echo "=== prep: build AF2-multimer FASTA inputs (byte-identical to Exp A) ==="
python3 src/expD_build_fastas.py --queries results/expA_queries.json --out-dir $SCRATCH/ftax/expD/fastas
echo "expD_leverage.sbatch DONE $(date)"

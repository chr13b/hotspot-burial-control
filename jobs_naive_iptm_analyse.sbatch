#!/bin/bash
#SBATCH -p normal
#SBATCH -t 00:30:00
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH -J naive_iptm
#SBATCH -o /scratch/users/cbertsch/ftax/jobs/naive_iptm_%j.log
# Phase 4 parse+analyse (runs as an afterany dependency on the naive fold array): parse the naive AF2 folds
# and compute paired naive-random / L-naive / L-random on ipTM/composite, reusing the committed wt/L/random folds.
set -euo pipefail
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
echo "=== parse naive folds ==="
python3 src/parse_iptm_naive.py --out-dir "$SCRATCH/ftax/iptm/naive_out" --out results/iptm_steer_naive.csv
echo "=== paired ipTM specificity contrasts ==="
python3 src/analyse_iptm_naive.py --committed results/iptm_steer.csv --naive results/iptm_steer_naive.csv \
    --out results/iptm_summary_naive.csv
echo "=== DONE ==="

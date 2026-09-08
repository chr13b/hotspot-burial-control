#!/bin/bash
#SBATCH -p normal
#SBATCH -t 00:30:00
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -J foldx_an
#SBATCH -o /scratch/users/cbertsch/ftax/foldx/logs/foldx_analyse_%j.log
# Runs as an afterany dependency on both lane arrays: concat shard CSVs -> Lane A Spearmans + Lane B contrasts.
set -uo pipefail
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
echo "shards present: A=$(ls results/_foldx_A_s*.csv 2>/dev/null | wc -l) B=$(ls results/_foldx_B_s*.csv 2>/dev/null | wc -l)"
python3 src/foldx_analyse.py --a-out results/foldx_detection.csv --b-out results/foldx_steer.csv
echo "=== DONE ==="

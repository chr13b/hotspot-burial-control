#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH -J expC2_slopediag
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_slopediag.out
set -e
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
python3 src/expC2_slope_diag.py
echo "=== slopediag done $(date) ==="

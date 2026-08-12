#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0:20:00
#SBATCH -J expC2_klloose
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_klloose.out
set -e
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
python3 src/expC2_kl_loose.py
echo "=== klloose done $(date) ==="

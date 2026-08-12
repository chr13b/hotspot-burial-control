#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0:20:00
#SBATCH -J expC2_kldebug
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_kldebug.out
set -e
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
python3 src/expC2_kl_debug.py
echo "=== kldebug done $(date) ==="

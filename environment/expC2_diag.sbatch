#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH -J expC2_diag
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_diag.out
set -e
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
python3 src/expC2_contact_diag.py --data-dir "$FTAX_DATA"
echo "=== diag done $(date) ==="

#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:15:00
#SBATCH -J expC2_hres10
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_hres10.out
set -e
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
python3 src/expC2_hotspot_res.py --data-dir "$FTAX_DATA" --cutoff 10.0
echo "=== hres10 regen done $(date) ==="

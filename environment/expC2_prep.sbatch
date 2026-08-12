#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:20:00
#SBATCH -J expC2_prep
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_prep.out
# CPU precompute: (1) pre-registered 50/50 hotspot split + ppi.hotspot_res, and (2) the positive-control
# reproduction of the Exp C continuous slope with the new src/expC_slope_check.py.
set -e
source "$SCRATCH/ftax/env.sh"
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"
echo "=== (1) precompute hotspot_res + 50/50 split (PREREG_expC2 §1/§3) ==="
python3 src/expC2_hotspot_res.py --data-dir "$FTAX_DATA"
echo
echo "=== (2) POSITIVE CONTROL: reproduce Exp C slope on committed expC_gap_perbackbone.csv ==="
python3 src/expC_slope_check.py --gap-perbackbone results/expC_gap_perbackbone.csv \
  --label expC --out results/expC_slope_check.csv
echo "=== expC2_prep done $(date) ==="

#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J expD_triage
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_triage.out
# TASK B: validate the KL-triage METHOD on non-native backbones (issue #12).
set -e
source $SCRATCH/ftax/env.sh
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"

echo "=== POSITIVE CONTROL: crystal kl_triage (must reproduce committed capture@3 delta +0.098) ==="
python3 src/kl_triage.py --joined results/kl_detector_joined.csv --out results/kl_triage.csv

echo "=== Exp A (OpenFold3 predicted) — expA_kl_joined already has the schema ==="
python3 src/kl_triage.py --joined $SCRATCH/ftax/predicted/expA_kl_joined.csv --out results/kl_triage_expA.csv

echo "=== Exp D (AF2-multimer) — expD_kl_joined already has the schema (cleanest one-backbone arm) ==="
python3 src/kl_triage.py --joined $SCRATCH/ftax/expD/expD_kl_joined.csv --out results/kl_triage_expD.csv

echo "=== Exp C2 (generative) — build interface-formed per-position joined table, then triage ==="
python3 src/expD_build_triage_c2.py \
  --scored $SCRATCH/expC2/scored_positions.csv \
  --perbackbone results/expC2_gap_perbackbone.csv \
  --crystal results/kl_detector_joined.csv \
  --out $SCRATCH/ftax/expD/expC2_triage_joined.csv
python3 src/kl_triage.py --joined $SCRATCH/ftax/expD/expC2_triage_joined.csv --out results/kl_triage_expC2.csv
echo "expD_triage DONE $(date)"

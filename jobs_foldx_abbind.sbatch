#!/bin/bash
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH -J foldx_abbind
#SBATCH -o /scratch/users/cbertsch/ftax/_foldx_abbind_%j.log
# Task 2 — AB-Bind physics ΔΔG_bind (FoldX Lane A on antibody-antigen). Self-contained: prep(gate)->repair->ddg->analyse.
# Same FoldX binary as the SKEMPI Lane A campaign (v5_0/foldx_20261231) for comparability.
set -uo pipefail
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
source "$SCRATCH/ftax/env.sh"
export FOLDX_BIN="$SCRATCH/ftax/foldx/v5_0/foldx_20261231"
export FOLDX_WORKLIST="results/foldx_worklist_abbind.csv"
export FOLDX_PDBS="$HOME/ftax/data/ab-bind"                 # AB-Bind-specific structures (match the fixture numbering)
export FOLDX_REPAIR="$SCRATCH/ftax/foldx/repaired_abbind"   # separate cache: 1AK4 etc. collide with SKEMPI repairs
export FOLDX_TIMEOUT=1800
N=16

echo "=== PREP (idempotent; positive-control cross-check + worklist) ==="
python3 src/foldx_prep_abbind.py
if [ ! -s "$FOLDX_WORKLIST" ]; then echo "FATAL: worklist empty/missing — gate failed, aborting"; exit 1; fi
NROWS=$(tail -n +2 "$FOLDX_WORKLIST" | wc -l)
echo "worklist rows=$NROWS"; [ "$NROWS" -gt 0 ] || { echo "FATAL: 0 worklist rows"; exit 1; }

echo "=== PHASE 1: RepairPDB (sharded x$N) ==="
for i in $(seq 0 $((N-1))); do
  python3 src/foldx_ddg.py --mode repair --shard "$i" --nshards "$N" > "$SCRATCH/ftax/_abbind_repair_s${i}.log" 2>&1 &
done
wait
echo "AB-Bind repaired PDBs cached:"; ls "$FOLDX_REPAIR" 2>/dev/null | grep -cE "_Repair.pdb$" || true

echo "=== PHASE 2: BuildModel+AnalyseComplex ΔΔG_bind (sharded x$N, runs=5) ==="
for i in $(seq 0 $((N-1))); do
  python3 src/foldx_ddg.py --mode ddg --lane A --shard "$i" --nshards "$N" --runs 5 \
      --out "results/_foldx_abbind_A_s${i}.csv" > "$SCRATCH/ftax/_abbind_ddg_s${i}.log" 2>&1 &
done
wait
echo "shard outputs:"; ls -la results/_foldx_abbind_A_s*.csv 2>/dev/null | wc -l

echo "=== PHASE 3: ANALYSE (consolidate + detection metrics) ==="
python3 src/foldx_analyse_abbind.py

echo "=== DONE ==="

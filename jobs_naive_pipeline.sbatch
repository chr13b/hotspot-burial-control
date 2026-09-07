#!/bin/bash
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=naive_pipe
#SBATCH -o /scratch/users/cbertsch/ftax/jobs/naive_pipe_%j.log
# Phase 1+2 of the naive-guidance specificity arm (pre-reg: results/PREREG_naive.md), one detached CPU job:
#   generate the naive arm -> judge by non-self models -> paired specificity contrasts.
set -euo pipefail
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control

echo "=== [1/3] generate naive arm (frozen 60-complex ipTM subset, K=64, alpha in {0,2}) ==="
python3 src/cfg_steer_naive.py --subset results/iptm_subset.txt \
    --out results/cfg_steer_naive.csv --seqs-out results/cfg_steer_naive_seqs.csv

echo "=== [2/3] judge naive arm (ESM-IF1 + MIF non-self; ProteinMPNN self/ref) ==="
python3 src/judge_dumped.py --seqs results/cfg_steer_naive_seqs.csv --out results/cfg_naive_judge.csv

echo "=== [3/3] paired specificity contrasts (naive-random, L-naive, L-random) ==="
python3 src/analyse_naive.py --out results/cfg_naive_summary.csv

echo "=== DONE ==="

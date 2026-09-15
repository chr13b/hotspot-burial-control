#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 24:00:00
#SBATCH -J pred_fold_mgr
#SBATCH -o /scratch/users/cbertsch/ftax/logs/pred_fold_mgr_%j.out
# CONNECTION-INDEPENDENT manager for the predicted-backbone AF2 fold confirmation. Runs ENTIRELY on Slurm (compute
# node), so a broken laptop/SSH connection cannot stop it (unlike a login-node background loop, which dies on
# disconnect). Waits for the fold array, auto-retries any stuck/timed-out folds (idempotent — each resubmit skips
# folds whose rank_001 PDB already exists on disk), then runs parse->analyse inline and writes the full-n result.
set -uo pipefail
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
source "$SCRATCH/ftax/env.sh"
OUT=$SCRATCH/ftax/iptm/af2_out_pred
MAN=$SCRATCH/ftax/iptm/fastas_pred/manifest.txt
TOT=$(wc -l < "$MAN")
count() { local n=0 fid; while read -r fid; do
  ls "$OUT/$fid/${fid}_unrelaxed_rank_001_"*.pdb >/dev/null 2>&1 && n=$((n+1)); done < "$MAN"; echo "$n"; }
JID=$(cat "$SCRATCH/ftax/_pred_fold_jid.txt")
for round in 1 2 3 4; do
  echo "[mgr] round $round: watching fold array $JID ($(date))"
  while [ -n "$(squeue -h -j "$JID" 2>/dev/null)" ]; do sleep 300; done
  n=$(count); echo "[mgr] round $round: array $JID cleared -> $n/$TOT folds ($(date))"
  [ "$n" -ge "$TOT" ] && break
  JID=$(sbatch --parsable --array=0-81%6 --export=ALL,STRIDE=8 jobs_iptm_fold_pred.sbatch)
  echo "$JID" > "$SCRATCH/ftax/_pred_fold_jid.txt"
  echo "[mgr] round $round: $n/$TOT done -> resubmitted missing (idempotent) as $JID"
done
echo "[mgr] FINISHER on $(count)/$TOT folds ($(date))"
python3 src/parse_iptm.py --out-dir "$OUT" --subset results/iptm_predicted_subset_full.txt \
    --positions results/leverage_skempi_positions.csv --out results/iptm_predicted_steer.csv
python3 src/analyse_iptm.py --in results/iptm_predicted_steer.csv --out results/iptm_predicted.csv
echo "[mgr] DONE ($(date)) -> results/iptm_predicted{,_steer}.csv"
cat results/iptm_predicted.csv

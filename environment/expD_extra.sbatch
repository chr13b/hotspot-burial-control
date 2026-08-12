#!/bin/bash
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:20:00
#SBATCH -J expD_extra
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_extra.out
set -e
source $SCRATCH/ftax/env.sh
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"

echo "=== symmetric leverage incl AF2 (regenerate expD_leverage.csv with AF2 block) ==="
python3 src/expD_leverage.py \
  --expA-pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
  --of3-positions $SCRATCH/ftax/predicted/expA_p0_positions.csv \
  --af2-positions $SCRATCH/ftax/expD/expD_p0_positions.csv \
  --c2-perbackbone results/expC2_gap_perbackbone.csv \
  --out results/expD_leverage.csv

echo "=== AF2 prediction-confidence summary (memorization caveat) ==="
python3 - <<'PY'
import pandas as pd, numpy as np
c=pd.read_csv("results/expD_confidence.csv")
print("n complexes scored:", len(c))
for col in ["ptm","iptm","avg_plddt","interface_plddt","rmsd_ca_global","rmsd_ca_interface"]:
    if col in c.columns:
        v=pd.to_numeric(c[col],errors="coerce").dropna()
        print(f"  {col:20s} median={v.median():.3f}  IQR[{v.quantile(.25):.3f},{v.quantile(.75):.3f}]")
p=pd.to_numeric(c["ptm"],errors="coerce"); ri=pd.to_numeric(c["rmsd_ca_interface"],errors="coerce")
print(f"  frac ptm>=0.8: {(p>=0.8).mean():.2f} ; frac interface RMSD<2A: {(ri<2).mean():.2f}")
PY

echo "=== coverage: AF2 rank_001 predictions produced ==="
ls $SCRATCH/ftax/expD/af2_out/*/*_unrelaxed_rank_001_*.pdb 2>/dev/null | wc -l
echo "expD_extra DONE $(date)"

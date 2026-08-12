#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:30:00
#SBATCH -J expD_archiveA
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_archiveA.out
# TASK A (heavy lifting): tar backbones, copy scored tables + manifests, freeze methods-as-code,
# compute RFdiffusion provenance. Git/LFS operations are done separately on the login node.
set -uo pipefail
REPO=/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
cd "$REPO"; mkdir -p environment

echo "=== [1] tar backbones (PDBs; gzip compresses) ==="
tar -czf results/expC_backbones.tar.gz  -C $SCRATCH/expC/backbones .   && echo "expC ok"
tar -czf results/expC2_backbones.tar.gz -C $SCRATCH/expC2/backbones .  && echo "expC2 ok"
# Exp D: ONLY the rank_001 AF2 PDBs (not the 5GB af2_out tree of 5 models + MSAs + PAEs)
( cd $SCRATCH/ftax/expD/af2_out && find . -name '*_unrelaxed_rank_001_*.pdb' -print0 \
    | tar --null -czf "$REPO/results/expD_backbones.tar.gz" -T - ) && echo "expD rank_001 ok"
echo "  expD rank_001 count: $(ls $SCRATCH/ftax/expD/af2_out/*/*_unrelaxed_rank_001_*.pdb 2>/dev/null | wc -l)"

echo "=== [2] scored per-position tables (LFS via results/*_positions.csv) ==="
cp $SCRATCH/expC/scored_positions.csv       results/expC_scored_positions.csv
cp $SCRATCH/expC2/scored_positions.csv      results/expC2_scored_positions.csv
cp $SCRATCH/ftax/expD/expD_p0_positions.csv results/expD_scored_positions.csv

echo "=== [3] per-backbone manifests ==="
[ -f results/expC_gap_perbackbone.csv ]  && cp results/expC_gap_perbackbone.csv  results/expC_backbone_manifest.csv  || cp $SCRATCH/expC/scored_backbones.csv results/expC_backbone_manifest.csv
cp results/expC2_gap_perbackbone.csv results/expC2_backbone_manifest.csv
cp results/expD_confidence.csv       results/expD_backbone_manifest.csv
# nan-coordinate exclusions: Exp C had 18 at T40; C2/D none (documented in DATA.md, not in the manifest rows)

echo "=== [4] methods-as-code: SE3nv (RFdiffusion) env freeze ==="
set +u; source $SCRATCH/ftax/env_se3.sh >/dev/null 2>&1 || echo "(env_se3 source warned)"; set -u
conda list --explicit > environment/se3nv.explicit.txt 2>/dev/null && echo "explicit ok ($(wc -l < environment/se3nv.explicit.txt) lines)" || echo "conda list failed"
pip freeze > environment/se3nv.pip-freeze.txt 2>/dev/null && echo "pip freeze ok ($(wc -l < environment/se3nv.pip-freeze.txt) lines)" || echo "pip freeze failed"

echo "=== [5] copy sbatch job specs into environment/ ==="
cp $SCRATCH/ftax/jobs/expC_*.sbatch  environment/ 2>/dev/null || true
cp $SCRATCH/ftax/jobs/expC2_*.sbatch environment/ 2>/dev/null || true
cp $SCRATCH/ftax/jobs/of3_*.sbatch   environment/ 2>/dev/null || true
cp $SCRATCH/ftax/jobs/expD_*.sbatch  environment/ 2>/dev/null || true
echo "  environment/ now: $(ls environment/*.sbatch 2>/dev/null | wc -l) sbatch files"

echo "=== [6] RFdiffusion provenance (commit + checkpoint md5) ==="
echo "RFDIFF_COMMIT=$(git -C $SCRATCH/expC/RFdiffusion rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
echo "models dir:"; ls -la $SCRATCH/expC/RFdiffusion/models/ 2>/dev/null | head
for ck in Complex_base_ckpt.pt Base_ckpt.pt; do
  [ -f $SCRATCH/expC/RFdiffusion/models/$ck ] && echo "MD5 $ck = $(md5sum $SCRATCH/expC/RFdiffusion/models/$ck | awk '{print $1}')"
done

echo "=== [7] SIZES for the LFS budget report ==="
du -h results/expC_backbones.tar.gz results/expC2_backbones.tar.gz results/expD_backbones.tar.gz \
      results/expC_scored_positions.csv results/expC2_scored_positions.csv results/expD_scored_positions.csv 2>/dev/null
echo "  new-artifact total: $(du -ch results/expC_backbones.tar.gz results/expC2_backbones.tar.gz results/expD_backbones.tar.gz results/expC_scored_positions.csv results/expC2_scored_positions.csv results/expD_scored_positions.csv 2>/dev/null | tail -1)"
echo "expD_archiveA DONE $(date)"

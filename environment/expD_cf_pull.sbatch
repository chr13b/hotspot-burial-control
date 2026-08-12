#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -J expD_cf_pull
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_cf_pull.out
# Pull the official ColabFold (AF2-multimer) container. Apptainer uses its own TLS + the container
# brings modern glibc, sidestepping the el7 TLS1.3 and glibc-2.17 walls that broke the pip installs.
set -uo pipefail
mkdir -p $SCRATCH/ftax/.cache/apptainer $SCRATCH/ftax/tmp $SCRATCH/ftax/logs
export APPTAINER_CACHEDIR=$SCRATCH/ftax/.cache/apptainer
export APPTAINER_TMPDIR=$SCRATCH/ftax/tmp
export TMPDIR=$SCRATCH/ftax/tmp
SIF=$SCRATCH/ftax/colabfold.sif

echo "host $(hostname) $(date); $(apptainer --version)"
IMAGES=(
  "docker://ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2"
  "docker://ghcr.io/sokrypton/colabfold:1.5.3-cuda12.2.2"
  "docker://getwilds/colabfold:latest"
)
ok=0
for img in "${IMAGES[@]}"; do
  echo "=== pull $img $(date) ==="
  if apptainer pull --force "$SIF" "$img"; then
    echo "PULLED $img"; echo "$img" > $SCRATCH/ftax/colabfold_image.txt; ok=1; break
  fi
  echo "pull failed for $img, trying next"
done
[ $ok -eq 1 ] || { echo "ALL_PULLS_FAILED"; exit 1; }

echo "=== sif ==="; ls -la "$SIF"
echo "=== colabfold_batch --help inside container ==="
apptainer exec "$SIF" colabfold_batch --help 2>&1 | head -15 || echo "help nonzero (checked at runtime)"
echo "DONE_CF_PULL"

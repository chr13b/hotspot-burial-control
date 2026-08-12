#!/bin/bash
#SBATCH -p normal
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -t 6:00:00
#SBATCH -J expD_lcf_install
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_lcf_install.out
# Install localcolabfold (AF2-multimer) via the official install_colabbatch_linux.sh (Miniforge-based).
# github downloads are TLS1.2 which the el7 node curl/wget CAN do (the earlier failures were a 404 and
# pixi.sh's TLS1.3 requirement, not a network block). Produces colabfold_batch at
# $SCRATCH/localcolabfold/colabfold-conda/bin/ and pre-downloads AF2 weights to
# $SCRATCH/localcolabfold/colabfold (installer sed-redirects params off $HOME). Caches -> $SCRATCH.
set -uo pipefail
mkdir -p $SCRATCH/ftax/.cache/pip $SCRATCH/ftax/.cache/conda_pkgs $SCRATCH/ftax/.cache/xdg \
         $SCRATCH/ftax/tmp $SCRATCH/ftax/logs
export PIP_CACHE_DIR=$SCRATCH/ftax/.cache/pip
export CONDA_PKGS_DIRS=$SCRATCH/ftax/.cache/conda_pkgs
export XDG_CACHE_HOME=$SCRATCH/ftax/.cache/xdg
export TMPDIR=$SCRATCH/ftax/tmp

echo "host $(hostname) $(date)"; type wget >/dev/null && wget --version | head -1
echo "=== fetch installer (curl --tlsv1.2 from github) ==="
curl -fL --tlsv1.2 -sS -o $SCRATCH/ftax/tmp/install_colabbatch_linux.sh \
   https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh \
   || { echo "INSTALLER_DOWNLOAD_FAILED rc=$?"; exit 1; }

cd $SCRATCH
rm -rf $SCRATCH/localcolabfold
echo "=== run installer (Miniforge + conda env + pip colabfold + jax cuda12 + weight download) $(date) ==="
bash $SCRATCH/ftax/tmp/install_colabbatch_linux.sh
rc=$?
echo "installer exit rc=$rc $(date)"

CB=$SCRATCH/localcolabfold/colabfold-conda/bin/colabfold_batch
echo "=== verify ==="
if [ ! -x "$CB" ]; then
  echo "NO_COLABFOLD_BATCH at $CB"; find $SCRATCH/localcolabfold -maxdepth 4 -name colabfold_batch 2>/dev/null | head
  exit 1
fi
ls -la "$CB"
echo "--- params dir ($SCRATCH/localcolabfold/colabfold/params) ---"; ls -la $SCRATCH/localcolabfold/colabfold/params 2>/dev/null | head
"$CB" --help 2>&1 | head -8 || echo "help nonzero (checked at runtime)"
[ $rc -eq 0 ] && echo "DONE_LCF_INSTALL" || { echo "INSTALLER_NONZERO rc=$rc"; exit 1; }

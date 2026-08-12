#!/bin/bash
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -J expD_score
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expD_score_%j.out
# Score AF2-multimer backbones through the IDENTICAL Exp A pipeline -> Exp D readings.
set -e
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
PRED="$SCRATCH/ftax/expD"
OF3="$SCRATCH/ftax/predicted"
W="$PROTEINMPNN_DIR/vanilla_model_weights/v_48_020.pt"
mkdir -p "$PRED/PDBs"
ln -sf "$FTAX_DATA/skempi_v2.csv" "$PRED/skempi_v2.csv"   # p0 reads $data_dir/skempi_v2.csv for labels
export FTAX_MAX_BATCH=8

echo "=== [0/6] convert AF2 rank_001 PDBs -> crystal-keyed PDBs + confidence $(date) ==="
python -u src/expD_af2_to_pdb.py --pred-dir "$PRED/af2_out" --resmap results/expA_resmap.json \
    --crystal-dir "$FTAX_DATA/PDBs" --interface-csv results/p0_dssp_interface_resid.csv \
    --out-pdb-dir "$PRED/PDBs" --out-conf results/expD_confidence.csv \
    --complexes results/pair_complexes.txt

echo "=== [1/6] ProteinMPNN scoring on AF2 backbones $(date) ==="
python -u src/p0_burial_matched.py --data-dir "$PRED" --only-complexes results/pair_complexes.txt \
    --mpnn-weights "$W" --out "$PRED/expD_p0"

echo "=== [2/6] pydssp SS on AF2 backbones $(date) ==="
python -u src/patch_ss.py --positions "$PRED/expD_p0_positions.csv" --data-dir "$PRED"

echo "=== [3/6] KL detector on AF2 backbones $(date) ==="
python -u src/kl_detector.py --data-dir "$PRED" --out "$PRED/expD_kl" \
    --positions "$PRED/expD_p0_positions.csv" --complexes results/pair_complexes.txt \
    --mpnn-weights "$W" --threads 8

echo "=== [4/6] burial-matched gap reusing committed crystal pairs (AF2 backbone) $(date) ==="
python -u src/expA_gap_reuse_pairs.py --pred-positions "$PRED/expD_p0_positions.csv" \
    --crystal-positions results/p0_positions.csv --pairs-glob "results/p0_dssp_pairs_*.csv" \
    --confidence results/expD_confidence.csv --out results/expD_gap

echo "=== [5/6] KL delta-AUROC-over-burial, AF2 vs crystal $(date) ==="
python -u src/expA_kl_delta.py --pred-joined "$PRED/expD_kl_joined.csv" \
    --crystal-joined results/kl_detector_joined.csv --confidence results/expD_confidence.csv \
    --out results/expD_kl_delta

echo "=== [6/6] per-complex AF2-vs-OF3 deficit correlation (most-informative readout) $(date) ==="
python -u src/expD_af2_vs_of3_corr.py --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
    --af2-positions "$PRED/expD_p0_positions.csv" --of3-positions "$OF3/expA_p0_positions.csv" \
    --crystal-positions results/p0_positions.csv --out results/expD_af2_of3_corr.csv

cp "$PRED/expD_p0_summary.csv" results/expD_p0_summary.csv 2>/dev/null || true
cp "$PRED/expD_kl_summary.csv" results/expD_kl_summary.csv 2>/dev/null || true
echo "=== expD_score ALL DONE $(date) ==="

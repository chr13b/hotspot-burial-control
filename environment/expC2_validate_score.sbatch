#!/bin/bash
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 1:00:00
#SBATCH -J expC2_vscore
#SBATCH -o /scratch/users/cbertsch/ftax/logs/expC2_vscore.out
# Score the 3-complex pinning-validation backbones and print interface-formed + iRMSD by level.
set -e
source "$SCRATCH/ftax/env.sh"
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
python3 src/expC_score.py --backbones-dir "$SCRATCH/expC2/validate" --outmap results/expC_outmap.json \
  --crystal-dir "$FTAX_DATA/PDBs" --pairs results/p0_dssp_pairs_SECONDARY_B_any_interface.csv \
  --mpnn-weights "$SCRATCH/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt" \
  --out "$SCRATCH/expC2/validate_scored" --threads 4
echo "=== validate score done $(date) ==="
python3 - <<'PY'
import pandas as pd, os
S=os.path.expandvars("$SCRATCH/expC2")
bb=pd.read_csv(f"{S}/validate_scored_backbones.csv")
print("=== VALIDATION (10A hotspot pinning ON) interface-formed by level ===")
q=bb.groupby("partial_T").agg(n=("backbone_id","nunique"), iface_ok=("interface_ok","sum"),
    med_irmsd=("irmsd","median"), med_irmsd_binder=("irmsd_binder","median"),
    med_tgtrmsd=("tgt_rmsd","median")).reset_index()
q["iface_frac"]=(q.iface_ok/q.n).round(2)
print(q.to_string(index=False))
print("\n=== per-backbone ===")
cols=["backbone_id","partial_T","irmsd","irmsd_binder","tgt_rmsd","hot_contacts","interface_ok"]
print(bb[cols].to_string(index=False))
PY

# Sherlock task — predicted-backbone Cα-RMSD for the dose-curve placement (② quantitative)

**Paste to the Sherlock session.** Goal: measure, per complex, how far the OpenFold3 / AF2-multimer predicted
backbones used in the R2 result deviate from the crystal, so we can plot those points at their *true effective
σ* on the dose-law x-axis — turning "predicted backbones fall in the dose law's survival zone" from a
qualitative claim into a measured one. `SEED=20260803`. Repo:
`/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.

## 0. Sync (the repo is already unblocked from the CFG session)
```bash
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
git pull --no-edit origin main          # LFS filter is already set to passthrough from before
git rev-parse --short HEAD               # expect b2386c6 or newer
```
Get a compute node if the structure set is large (this is light — CPU is fine): `srun -c 4 --mem=8G -t 1:00:00 --pty bash`.

## 1. Locate the predicted backbones
They were produced for Exp A / Exp D (the R2 result). Their paths are in `results/expD_backbone_manifest.csv`
and/or on `$SCRATCH` under the Exp A/D output dirs (the OF3 positions came from
`/scratch/users/cbertsch/ftax/predicted/…` and AF2 from `/scratch/users/cbertsch/ftax/expD/…`, per the committed
`expD_leverage.py` command string). Crystal structures are at `~/ftax/data/PDBs/`. Confirm you can read one
predicted PDB and its crystal before batch-running.

## 2. Compute per-complex Cα-RMSD(predicted, crystal)
For each of the ~140 shared complexes in the R2 result:
- parse the crystal and the predicted structure (same chains/residue numbering; use the interface residue set
  from the committed `results/leverage_skempi_positions.csv` if numbering needs anchoring);
- superpose on **Cα atoms** (Kabsch/SVD) over the matched residues — do it two ways and report both:
  **(a) whole-complex Cα-RMSD** and **(b) interface-only Cα-RMSD** (the interface is what leverage reads);
- record per complex, for each predictor (OF3, AF2).

Write `results/predicted_backbone_rmsd.csv` with columns:
`complex_id, of3_ca_rmsd_all, of3_ca_rmsd_iface, af2_ca_rmsd_all, af2_ca_rmsd_iface, n_res, n_iface`.

**Positive control (rule 6):** superpose a crystal onto itself → RMSD must be ~0; superpose two chains that
should differ → nonzero. Report the median and IQR of the interface RMSDs; we expect most in the sub-Ångström
to ~1.5 Å range (that is the whole point — it should land on the surviving part of the dose curve).

## 3. Deliverables
Commit **only** `results/predicted_backbone_rmsd.csv` + a short `results/FINDINGS_rmsd_placement.md` (median/IQR
of interface RMSD per predictor, the positive-control result, `SEED`, exact command). `git add` those two files
by name (not `-A`), commit with the repo's two trailer lines, push. Then message me the median interface RMSD
so I can overlay the R2 points on the dose-law figure locally.

## Why this matters (for the write-up)
The dose law (Fig 3) shows CPI(L|geometry) survives ≤~0.5–0.75 Å of backbone jitter and collapses by ~1 Å
(MPNN). If the predicted backbones' *measured* interface RMSD sits in that surviving band, the dose law
**predicts** the R2 survival (+0.039/+0.032) rather than merely co-existing with it — a quantitative bridge
between §4 (dose law) and §6 (predicted-backbone leverage).

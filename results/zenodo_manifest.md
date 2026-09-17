# Zenodo bundle — hotspot-burial-control (large regenerable / license-gated artifacts)

**Purpose.** The anonymized GitHub repo ships only the *small, load-bearing* CSVs (paper-cited ∪ figure-read) so the
paper reproduces GPU-free / LFS-free / Sherlock-free. This bundle is the **durable home** for the big artifacts the
repo excludes or tracks via git-LFS — because the GitHub-LFS free tier (~1 GB/month bandwidth) will break public
clones. Nothing here is a paper number; every headline value is in a committed plain CSV in the repo.

**Deposit (operator step, their Zenodo account).** Upload as a **Restricted-access** record with a **reserved DOI**:
private now (no identity reveal for the double-blind submission), flipped to public + named at camera-ready. The
repo↔Zenodo split is specified in `notes/LOAD_BEARING.md`. Total bundle size: **779 MB, 12 files** (+ this MANIFEST
+ `SHA256SUMS.txt`). Verify integrity with `sha256sum -c SHA256SUMS.txt`.

## Contents

| file | size | sha256 | provenance | regenerable? |
|---|---|---|---|---|
| `expC_backbones.tar.gz` | 27 MB | `cde71c43a3798271…` | Exp C — RFdiffusion partial-diffusion backbones (SE3nv) | y — GPU (RFdiffusion; `environment/expC_*.sbatch`) |
| `expC2_backbones.tar.gz` | 62 MB | `4f7b4e7e53b27195…` | Exp C2 — interface-pinned partial-diffusion backbones | y — GPU (RFdiffusion; `environment/expC2_*.sbatch`) |
| `expD_backbones.tar.gz` | 9.6 MB | `3d8a0307c5477902…` | Exp D — AF2-multimer predicted backbones | y — GPU (ColabFold; `environment/expD_*.sbatch`) |
| `expC_scored_positions.csv` | 122 MB | `19d0d62ab4f5a5a8…` | Exp C per-position ProteinMPNN leverage on the C backbones | y — CPU from the C backbones (`leverage`/scoring) |
| `expC2_scored_positions.csv` | 300 MB | `5873106da196a1a0…` | Exp C2 per-position scored table | y — CPU from the C2 backbones |
| `expD_scored_positions.csv` | 58 MB | `cef61feee30b77a0…` | Exp D scored table — **also the AF2 geometry** consumed by `leverage_predicted.py --source af2` (byte-identical to `$SCRATCH/ftax/expD/expD_p0_positions.csv`) | y — CPU from the AF2 backbones |
| `expA_p0_positions_of3_geom.csv` | 58 MB | `ef97f8283e7621aa…` | OpenFold3 predicted-backbone geometry — input to `leverage_predicted.py --source of3` (`$SCRATCH/ftax/predicted/expA_p0_positions.csv`; not in repo) | y — CPU from the OF3 backbones |
| `p0_positions.csv` | 136 MB | `cbb61eadecee09b1…` | Crystal SKEMPI base geometry (rSASA/DSSP/neighbours over 345 crystals) | y — CPU (biopython + freesasa), but large |
| `predicted_backbones_of3_pdbs.tar.gz` | 11 MB | `73c0e9550bb98d9e…` | 140 OpenFold3 predicted PDBs (`$SCRATCH/ftax/predicted/PDBs`) | y — **GPU (OpenFold3)** |
| `predicted_backbones_af2_pdbs.tar.gz` | 9.5 MB | `53001d5f5988c428…` | 140 AF2-multimer predicted PDBs (`$SCRATCH/ftax/expD/PDBs`) | y — **GPU (AF2-multimer)** |
| `foldx_repaired_skempi_pdbs.tar.gz` | 21 MB | `cade9a0c1a83a673…` | FoldX-RepairPDB'd SKEMPI crystals (291) — `foldx_ddg.py --mode repair` cache | y — CPU but needs **license-gated FoldX 5.1** |
| `foldx_repaired_abbind_pdbs.tar.gz` | 2.1 MB | `8c9aa2fa8cdb40e4…` | FoldX-RepairPDB'd AB-Bind crystals | y — CPU but needs **license-gated FoldX 5.1** |

Full checksums in `SHA256SUMS.txt`. "Regenerable? y" everywhere — nothing here is irreplaceable; the bundle exists
so a public re-runner does not have to re-produce GPU backbones or hold a FoldX license just to re-run the pipeline.

## NOT included (redistribution-restricted; download from upstream — pinned versions/md5s in `environment/README.md`)
- **FoldX 5.1** — academic-license binary + `rotabase.txt`, non-redistributable (`foldxsuite.crg.eu`). Used for the
  `repaired/` caches above and the ΔΔG_bind numbers.
- **RFdiffusion weights** (`Complex_base`, md5 pinned in `environment/README.md`) — RosettaCommons/RFdiffusion.
- **ProteinMPNN** `vanilla_model_weights/v_48_020.pt` (6.7 MB) — dauparas/ProteinMPNN.
- **ESM-IF1** `esm_if1_gvp4_t16_142M_UR50`, **MIF**, **PiFold** — model weights via the upstream repos / torch-hub.
- **OpenFold3** (`of3-p2-155k`) and the **ColabFold** Apptainer image (`colabfold:1.5.5-cuda12.2.2`) — see
  `environment/README.md`.

## Backbone interface-QC labels
The committed `results/expC_interface_qc.csv` is a **per-`partial_T` QC summary** (columns
`partial_T,n,iface_ok,med_irmsd,med_tgtrmsd`), NOT a per-backbone formed/dissolved/nan manifest. The per-backbone
audit trail lives in the scored-positions tables above (join on backbone id); regenerate per-backbone labels from
`expC_scored_positions.csv` + the interface definition if a per-backbone table is needed.

# Data availability — hotspot-burial-control

Every number in `results/FINDINGS*.md` traces to a committed CSV in `results/`. Those CSVs are the
analysis-ready outputs and are sufficient to reproduce every figure and statistic **without a GPU**.
This file documents the larger raw artifacts that back them, and where they live.

## What is in git (this repo)

- All source (`src/`), pre-registrations (`results/PREREG*.md`), write-ups (`results/FINDINGS*.md`),
  and every summary/analysis CSV (`results/*.csv`) — including the per-backbone audit trails
  (`results/expC_gap_perbackbone.csv`, `results/expC_interface_qc.csv`).
- The label sets the experiments key against: `results/p0_dssp_pairs_*.csv`, `results/pair_complexes.txt`,
  `results/expC_complexes.csv`, and the residue maps (`results/expC_resmap.json`, `expC_outmap.json`).
- Methods-as-code for the GPU runs: see `environment/README.md` (the SE3nv build, the RFdiffusion
  commit + checkpoint identity, and the sbatch job specs). **TODO (Sherlock session):** drop the frozen
  env spec (`conda list --explicit` / `pip freeze` of SE3nv) and the 3 sbatch files in `environment/`.

## What is NOT in git (large raw artifacts — Zenodo deposit)

These are too large for git and are the wrong fit for GitHub LFS as an archival home (the free tier is
1 GB storage / 1 GB bandwidth per month, so a ~200 MB LFS repo breaks `git clone` for readers after a
handful of pulls). They are being deposited to **Zenodo** with a DOI, referenced here and in the paper's
Data Availability statement.

| Artifact | Size | Provenance | Regenerable? |
|---|---|---|---|
| Exp C RFdiffusion backbones (715 PDBs: 55 crystal controls + 660 ladder) | ~78 MB | `src/expC_prep_inputs.py` → RFdiffusion partial diffusion (SE3nv env) | Only on a GPU + the SE3nv env; **keep all** — the divergence/dissolution cases are themselves a finding |
| Exp C per-position scored table | ~122 MB | `src/expC_score.py` (ProteinMPNN v_48_020, 8 orders + KL pass) | CPU-regenerable from the backbones + `src/expC_score.py` |
| Exp A predicted-backbone tables | (on SCRATCH) | OpenFold3 predictions + `src/expA_*` | GPU (prediction) then CPU (scoring) |

**Manifest requirement:** the Zenodo archive must ship a manifest labelling each backbone
`interface-formed` / `dissolved` / `nan`, matching `results/expC_interface_qc.csv` and the exclusion list
in `results/FINDINGS_expC.md §7.4`. All 715 backbones are kept — the instability is a reported result,
not noise to be cleaned away.

> **⏳ TIME-SENSITIVE.** The raw artifacts currently live only on Sherlock `$SCRATCH/expC/`, which is
> purged **60–90 days** after last use. The run was **2026-08-10**; deposit to Zenodo **before
> ~2026-10-09** (conservative 60-day bound) or the raw output is lost. The derived CSVs in this repo
> survive regardless.

## Do not redistribute

- **RFdiffusion** (RosettaCommons) and its weights are third-party under their own license — not
  committed here. `environment/README.md` pins the exact commit and checkpoint (`Complex_base`) so the
  environment is reconstructable from upstream.

## Data Availability statement (paper draft)

> Analysis-ready data and all code are available at `github.com/chr13b/hotspot-burial-control`. Raw
> RFdiffusion backbones and per-position model scores are archived at Zenodo (DOI: _pending deposit_)
> with a manifest labelling interface-formed, dissolved, and divergent backbones. SKEMPI 2.0 is the
> upstream fixture (Jankauskaitė et al. 2019). RFdiffusion (Watson et al. 2023) and ProteinMPNN (Dauparas
> et al. 2022) are used under their respective licenses; the pinned commit and checkpoint are in
> `environment/README.md`.

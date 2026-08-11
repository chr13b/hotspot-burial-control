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

## What is NOT in git as plain files (large raw artifacts — two-phase plan)

**Phase 1 (now — purge-rescue): git-LFS in this working repo.** The artifacts must leave Sherlock before
the SCRATCH purge (below); pushing them to LFS now gets them into a durable store and keeps them one
`git pull` away during drafting. Fine at this stage — private iteration means few clones, so the LFS
free-tier bandwidth cap (1 GB/month) is not yet in play.

**Phase 2 (at submission — archival): Zenodo DOI + clean public repo.** GitHub LFS is the wrong *archival*
home — its 1 GB/month bandwidth cap breaks `git clone` for readers once a public repo is cloned a handful
of times. At submission the two big files move to a **Zenodo** record (DOI, 50 GB/record, what journals'
Data Availability statements expect), referenced from a clean public repo. Zenodo is **independent of
GitHub** (direct web/API upload — no chunking, no routing through git); the optional GitHub-release→Zenodo
integration is only for minting a DOI for the *code* snapshot.

| Artifact | Size | Provenance | Regenerable? |
|---|---|---|---|
| Exp C RFdiffusion backbones (715 PDBs: 55 crystal controls + 660 ladder) | ~78 MB | `src/expC_prep_inputs.py` → RFdiffusion partial diffusion (SE3nv env) | Only on a GPU + the SE3nv env; **keep all** — the divergence/dissolution cases are themselves a finding |
| Exp C per-position scored table | ~122 MB | `src/expC_score.py` (ProteinMPNN v_48_020, 8 orders + KL pass) | CPU-regenerable from the backbones + `src/expC_score.py` |
| Exp A predicted-backbone tables | (on SCRATCH) | OpenFold3 predictions + `src/expA_*` | GPU (prediction) then CPU (scoring) |

**Manifest requirement:** the Zenodo archive must ship a manifest labelling each backbone
`interface-formed` / `dissolved` / `nan`, matching `results/expC_interface_qc.csv` and the exclusion list
in `results/FINDINGS_expC.md §7.4`. All 715 backbones are kept — the instability is a reported result,
not noise to be cleaned away.

> **⏳ TIME-SENSITIVE.** The raw artifacts currently live only on Sherlock `$SCRATCH/expC/`, purged
> **60–90 days** after last use. The run was **2026-08-10** → get them off SCRATCH **before ~2026-10-09**
> (conservative 60-day bound). The Phase-1 LFS push satisfies this deadline; Zenodo (Phase 2) can follow
> at submission. Guardrail: watch the **1 GB free LFS storage** cap as C2 adds backbones — prune to the
> scored tables (regenerable from backbones) or add a data pack if needed. The derived CSVs survive regardless.

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

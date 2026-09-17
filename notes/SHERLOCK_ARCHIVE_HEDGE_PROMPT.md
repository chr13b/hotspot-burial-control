# Sherlock — archival + reproducibility hedge (run BEFORE losing the allocation) [paste-to-run]

Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`. `git pull --no-edit origin main`
FIRST. CLAUDE.md rules apply (never fabricate; `git add` by name; two trailer lines; push). Goal: make the project
reproducible **without Sherlock** and durable **without relying on the GitHub-LFS free tier**. Nothing here is a new
result — it is insurance. Report a short manifest at the end.

## Context (already true — verify, don't redo)
- Every paper number is a committed plain CSV → the paper reproduces GPU-free/LFS-free/Sherlock-free. Preserve that.
- ~856 MB is in git-LFS (backbones + scored tables) and fully pushed to origin. That is off-SCRATCH already, but
  the LFS free-tier bandwidth cap (~1 GB/month) will break public clones — Zenodo is the durable home.

## Step 1 — confirm nothing load-bearing is SCRATCH-only-and-unsaved
- `git status` clean; `git lfs push --all origin` (confirm "0 objects to push"). Push anything pending.
- List the gitignored, `$SCRATCH`-only derivable caches actually consumed by a committed script:
  `results/leverage_pq_predicted_*.csv`, `results/leverage_pq_*.csv`, `results/atlas_pq_*.csv`, the RepairPDB'd
  crystals under `$SCRATCH/ftax/foldx/repaired/`, and any predicted-backbone leverage dumps. For each, record the
  **exact regeneration command** (the `--stage score` invocation + which weights + which input PDBs). If any is
  **not** cheaply regenerable (e.g. needs a GPU rescore that would be lost with the allocation), stage it for the
  Zenodo bundle in Step 2 rather than trusting re-derivation.

## Step 2 — build the Zenodo bundle (durable archive)
Assemble one directory `$SCRATCH/ftax/zenodo_bundle/` containing:
- the LFS big artifacts (`git lfs pull` then copy): `exp{C,C2,D}_backbones.tar.gz`, `exp{C,C2,D}_scored_positions.csv`;
- the predicted-backbone + steering raw the paper's re-run needs that is not already a small committed CSV;
- the non-regenerable-or-expensive `$SCRATCH` caches from Step 1;
- `MANIFEST.md`: one line per file (path, size, sha256, provenance script, regenerable? y/n), and the backbone
  labels (interface-formed / dissolved / nan) from `results/expC_interface_qc.csv`.
Do **not** include: the FoldX binary or rotabase (license), RFdiffusion weights (third-party), model weights
(public downloads). Note their download URLs + pinned versions in `MANIFEST.md` instead.
**Upload to Zenodo is the operator's step** (their account) — produce the bundle + MANIFEST + a printed `du -sh`
and the file list so the operator can drag-drop or `zenodo` API upload. Report the total size. (Operator note:
deposit as a **Restricted-access** record with a **reserved DOI** — private now, no identity reveal for the
double-blind submission; flip to public + named at camera-ready. The repo/Zenodo split is in
`notes/LOAD_BEARING.md`: the anon repo ships load-bearing CSVs only, this bundle holds the big regenerable
artifacts the repo excludes.)

## Step 3 — prove reproducibility-without-Sherlock
- In a scratch dir, `git clone` origin fresh (no `git lfs pull`) and confirm `python3 src/fig_foldx.py`,
  `src/fig_ladder.py`, `src/foldx_analyse.py`, `src/foldx_laneA_deepen.py`, `src/leverage_ladder.py`,
  `src/analyse_predicted_steer.py` all run **from the committed CSVs alone** and reproduce the headline numbers
  (SEED=20260803) — i.e. the paper reproduces with no LFS/GPU. Report any script that secretly needs an LFS or
  `$SCRATCH` file (that is a reproducibility bug to fix before submission).
- Confirm `environment/README.md` + a frozen env spec (`pip freeze` / `conda list --explicit`) are committed for
  the GPU legs (SE3nv / ColabFold / ESM-IF1 / MIF / PiFold), with pinned versions + weight-download URLs.

## Deliverables (commit by name, two trailer lines, push)
`MANIFEST.md` (in the bundle dir path, and a copy at repo `results/zenodo_manifest.md`), any missing frozen env
spec under `environment/`, and a one-paragraph reproducibility note appended to `DATA.md` recording the fresh-clone
result. Message me: the bundle size, the list of any script that needed a non-committed file, and confirmation that
`git lfs push --all` shows nothing pending.

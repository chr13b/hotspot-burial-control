# GPU environments — methods as code

The CPU study runs under the project's `torch-2.0.1` env (see repo root). The GPU experiments each need
their own env: **A** (OpenFold3 prediction), **C/C2** (RFdiffusion partial diffusion), **D** (AF2-multimer).
This file records what the runs used so they are reconstructable from upstream. **The non-obvious build
fixes below cost the most time to discover and are the most valuable thing to preserve.**

> **Frozen from the Sherlock session (2026-08-11):** `environment/se3nv.explicit.txt` (`conda list
> --explicit`, 93 pkgs) and `environment/se3nv.pip-freeze.txt` (`pip freeze`, 63 pkgs) capture the SE3nv
> (RFdiffusion) env. The actual job specs are the committed `environment/*.sbatch` files (Exp A `of3_*`,
> Exp C `expC_*`, Exp C2 `expC2_*`, Exp D `expD_*`). RFdiffusion checkpoint md5s are pinned below. Raw
> artifacts (backbones, scored tables) are archived via git-LFS — see `DATA.md`.

## RFdiffusion (Experiments C and C2) — SE3nv env

- **Upstream:** RosettaCommons/RFdiffusion. **Commit:** installed from a release download, **not** a git
  checkout (`$SCRATCH/expC/RFdiffusion` has no `.git`), so the exact commit is not recoverable; the version
  is pinned instead by the checkpoint md5s below plus the upstream repo.
- **Checkpoints** (`$SCRATCH/expC/RFdiffusion/models/`, dated 2023-03-29):
  - `Complex_base_ckpt.pt` (used; trained on complexes + hotspot residues) — **md5 `7a5d99f3c8bede52d9240f79a99bc30b`**.
  - `Base_ckpt.pt` (Option-B fallback) — **md5 `4aa4a27ba280d23541e01860c106c7cc`**.
- **Note, corrected by Exp C2:** the runs used the checkpoint *without* `ppi.hotspot_res`. Exp C originally
  blamed the binder divergence on that absence, but **Exp C2 showed the opposite** — passing `hotspot_res`
  under partial diffusion *causes* catastrophic divergence (`results/FINDINGS_expC2.md §2`), and 21/55
  complexes dock with no pinning at all. Exp C2 reuses the same SE3nv env; its specs are `environment/expC2_*.sbatch`.
- **Partial-diffusion command shape** (contig holds the target, diffuses the binder):
  `diffuser.partial_T=<T>`, `contigmap.contigs="[<Lb>/0 B1-<Lt>]"`, `inference.num_designs=3/6`,
  binder-first input PDB. Exact per-complex commands are in the `command` column of `results/expC*_*.csv`.

### Build fixes that were required (reported by the run — verify against the frozen spec above)

These are the traps that made the difference between the run working and silently failing:

1. **CPU-only-torch trap (the big one).** The SE3nv env shipped a CPU-only torch, so RFdiffusion ran on
   CPU at ~115 s/step — which looked like a "RFdiffusion needs ~400 GPU-h" wall. Installing a CUDA
   torch (**`torch==1.9.1+cu111`**) dropped it to ~3 s/step and made the whole ladder ~12 GPU-h. Always
   assert `torch.cuda.is_available()` before a production run.
2. **libstdc++ CXXABI preload.** A `CXXABI_*` symbol mismatch required preloading the conda libstdc++
   (`LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`) before importing dgl/SE3Transformer.
3. **GPU compute-capability / V100 pin.** The build was pinned to `GPU_CC=7.0` (V100); the cu111 wheels
   match that arch. Record which partition/GPU the sbatch requested.
4. **Binder-first inputs.** Input PDBs are written binder-chain-first so the contig `"<Lb>/0 B1-<Lt>"`
   diffuses the binder and holds the target as a fixed motif.

### PDB-parsing hardening (scoring side, in `src/expC_score.py`)

RFdiffusion writes coordinates that overflow PDB's 8-column x/y/z fields at |coord| ≥ 1000 Å; a naive
biopython read silently dropped ~290 high-drift backbones — a **noise-correlated** bias. A manual
fixed-width-aware parser recovers them (scored coverage 352 → 642 of 660). This matters because dropping
exactly the diverged backbones would have biased the dose-response. Keep the manual parser.

## OpenFold3 (Experiment A)

- Predicted backbones for the 141 pair complexes; templates **OFF** (no leakage; the crystal control
  reproduces to 4e-16). See `results/PREREG_expA.md` and `results/FINDINGS_expA.md`. Job specs:
  `environment/of3_*.sbatch`. of3c conda env (openfold3 0.4.3, rdkit 2025.9.6); weights `of3-p2-155k`.
- **TODO (low priority):** pin the exact OpenFold3 commit + weights hash if a leakage-free re-run is needed.

## AF2-multimer (Experiment D) — Apptainer container

Second, architecturally-independent predictor (Evoformer + regression IPA) for the deficit-generality test
(`results/FINDINGS_expD.md`). **Sherlock is CentOS 7 (glibc 2.17; system OpenSSL ≤ TLS 1.2), which broke
every pip/installer route** — the renamed installer 404s, `pixi.sh` needs TLS 1.3, and Miniforge+pip hits
`manylinux_2_28` wheels that need glibc 2.28. **Fix: the official ColabFold Apptainer container** (`apptainer`
1.5.2 at `/usr/bin`, no module) — its bundled TLS + the container's modern glibc sidestep both walls:

```bash
apptainer pull docker://ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2   # -> $SCRATCH/ftax/colabfold.sif
apptainer exec --nv -B $SCRATCH:$SCRATCH $SIF colabfold_batch <fa> <out> --data <params> \
    --model-type alphafold2_multimer_v3 --msa-mode mmseqs2_uniref_env --num-models 5 --num-recycle 3
```

Templates OFF (no `--templates` flag; verified `"use_templates": false` in `config.json`). Job specs:
`environment/expD_*.sbatch`. This container pattern is the reusable fix for any modern-wheel GPU tool on this
cluster's el7 nodes.

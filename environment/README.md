# GPU environments — methods as code

The CPU study runs under the project's `torch-2.0.1` env (see repo root). The GPU experiments (A: OpenFold3
prediction; C: RFdiffusion partial diffusion) each need their own env. This file records what the runs
used so they are reconstructable from upstream. **The non-obvious build fixes below cost the most time to
discover and are the most valuable thing to preserve.**

> **TODO (fill from the Sherlock session — these files live on `$SCRATCH`, not yet in git):**
> - `environment/se3nv.explicit.txt` — `conda list --explicit` of the SE3nv env used for RFdiffusion.
> - `environment/se3nv.pip-freeze.txt` — `pip freeze` of the same env.
> - `environment/expC_run.sbatch`, `expC_score_array.sbatch` (+ Exp A jobs) — the actual job specs.
> - Confirm the RFdiffusion commit hash and checkpoint md5 below against the install on `$SCRATCH`.

## RFdiffusion (Experiment C) — SE3nv env

- **Upstream:** RosettaCommons/RFdiffusion. **Commit:** `TODO: <pin the exact commit used>`.
- **Checkpoint:** `Complex_base_ckpt.pt` (trained on complexes + hotspot residues). **md5:** `TODO`.
  Note the run used it *without* specifying hotspot residues, which is the cause of the binder
  divergence documented in `results/FINDINGS_expC.md §2` — see `notes/SHERLOCK_HANDOFF_C2.md` for the fix.
- **Partial-diffusion command shape** (contig holds the target, diffuses the binder):
  `diffuser.partial_T=<T>`, `contigmap.contigs="[<Lb>/0 B1-<Lt>]"`, `inference.num_designs=3`,
  binder-first input PDB. Exact per-complex commands are in the `command` column of `results/expC_*.csv`.

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
  reproduces to 4e-16). See `results/PREREG_expA.md` and `results/FINDINGS_expA.md`.
- **TODO (Sherlock session):** pin the OpenFold3 commit + weights identity and the prediction command here.

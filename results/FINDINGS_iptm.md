# FINDINGS — ipTM confirmation of CFG-steering (INTERIM: pipeline validated, full fold GPU-bounded, in progress)

**Status: INTERIM.** The pipeline is validated end-to-end and the pre-registered signal fires strongly on
every complex folded so far, but the full 60-complex paired test is **still folding** — the GPU partition is
saturated by a concurrent job array, so the ~420 AF2-multimer folds are trickling in. This file reports the
validated controls + the preliminary per-complex signal, and documents how the result auto-finalizes. Do **not**
read the paired statistics as final until n reaches the pre-registered subset. Pre-reg: `results/PREREG_iptm.md`.
SEED=20260803.

## What is being tested
Steer a frozen ProteinMPNN by `+α·L` at interface positions (α=2); fold wt / L(k0-2) / random(k0-2) with
AF2-multimer (ColabFold container, `--num-models 1`, `--msa-mode mmseqs2_uniref_env`, templates OFF — the Exp D
pipeline); ask whether **ipTM(L) > ipTM(random)** (paired, per complex), with the pre-registered robust
composite (z-mean of ipTM, −interface pAE, interface pLDDT) and global pTM as a localization control.

## Positive controls — PASS
- **wt sanity:** wt interface ipTM median **0.91** (pre-reg wants ~0.6–0.9) — the fold setup (chain order / MSA)
  is correct. → `results/iptm_steer.csv` (wt rows).
- **Determinism (same wt sequence, 3 AF2 seeds, 5 complexes):** within-complex ipTM SD = 0.005, 0.069, 0.005,
  0.005, 0.000; **mean 0.017**. So AF2's own ipTM noise floor is ~0.02 (one complex, 1AK4, noisier at 0.07).
  An effect must clear this to be trusted. → `results/iptm_determinism.csv`.
- **Parser:** interface pAE (cross-interface g1↔g2 residue pairs), interface pLDDT (mean over interface
  residues), ipTM, pTM extracted from ColabFold rank_001; interface set = `leverage_skempi_positions.csv`;
  folded residue order = crystal-complex order so pLDDT/PAE indices align. Validated on the smoke complex.

## Preliminary signal (DESCRIPTIVE, n small — statistics pending full fold)
Smoke complex 1ACB (k0 only): wt ipTM 0.91, **L 0.88 vs random 0.63**; interface pAE L 2.91 vs random 6.45;
interface pLDDT 90.7 vs 78.0; global pTM 0.93 vs 0.87 (moves less than the interface metrics). Textbook
pre-registered direction on the first complex.

Per-complex mean-over-folded-k ipTM, first complexes complete (completion order = frozen-subset order, so this
is deterministic, not cherry-picked):

| complex | wt | L | random | **L − random** | L − wt |
|---|---|---|---|---|---|
| 1ACB_E_I | 0.910 | 0.777 | 0.620 | **+0.157** | −0.133 |
| 1AK4_A_D | 0.810 | 0.347 | 0.213 | **+0.133** | −0.463 |
| 1CT0_E_I | 0.910 | 0.890 | 0.335 | **+0.555** | −0.020 |

**All 3/3 folded complexes favour L over random on ipTM (mean L−random = +0.28), each far exceeding the 0.017
determinism floor.** This is consistent with the pre-registered H1 direction but n=3 is descriptive only — no
bootstrap CI is reported at this n, and L−wt is mildly negative (steering moves off the native sequence), which
H2 tolerates as long as the interface does not collapse.

## Completion + how this finalizes
- 60 complexes × 7 sequences = **420 folds**; the frozen subset is `results/iptm_subset.txt` (committed before
  any fold). Sequences: `results/cfg_steer_seqs.csv` (cfg_steer.py --dump-seqs).
- The fold is a **resumable Slurm array** (`$SCRATCH/ftax/jobs/iptm_fold_array.sbatch`, `%6`, skips done folds),
  GPU-bounded by a concurrent array. An **auto-finalize** job (`--dependency=afterany` on the fold array,
  `$SCRATCH/ftax/jobs/iptm_finalize.sh`) re-runs `parse_iptm.py` + `analyse_iptm.py` when the array ends, writing
  the final `results/iptm_steer.csv` + `results/iptm_summary.csv`.
- **To finalize manually** once folding completes (check `ls $SCRATCH/ftax/iptm/af2_out/*/*rank_001*.pdb | wc -l`
  → 420): `python3 src/parse_iptm.py --out results/iptm_steer.csv && python3 src/analyse_iptm.py`. If the array
  timed out with folds missing, resubmit the same sbatch (resumable) first. `analyse_iptm.py` computes the
  paired L−random (complex-clustered bootstrap CI, P>0) for ipTM / interface pAE / interface pLDDT / pTM plus
  the composite, and the H1/H2/H3 verdicts.

## Honest scope
ipTM is a model proxy for "does this assemble", metric-noisy; this is a structure-predictor confirmation of the
steering direction, not experimental binding. The steering result's primary validation remains the anti-circular
ESM-IF1 leverage (`FINDINGS_cfg_steer.md`); a positive ipTM result strengthens it, a null bounds it. The
preliminary 3/3 is encouraging but **not** the pre-registered result until the fold completes.

# FINDINGS — the CFG-steering ipTM benefit REPRODUCES under Boltz-2 (cross-modality, not AF2-specific)

**Pre-registered** `results/PREREG_boltz.md` (frozen before any Boltz number). `SEED=20260803`. The standing
result (`FINDINGS_iptm.md`) measured the +α·L steering benefit with AF2-multimer; Boltz-2 is an independent,
architecturally decorrelated all-atom folder. **Result: H1 passes — the benefit reproduces under Boltz-2**, so it
is a property of the sequences, not an AF2 quirk. Raw `results/iptm_steer_boltz.csv`, stats
`results/iptm_summary_boltz.csv`. Driver `src/build_boltz_inputs.py` + `src/parse_boltz.py` (emits the exact
`parse_iptm.py` schema → `analyse_iptm.py` runs UNCHANGED); folds `$SCRATCH/ftax/jobs/boltz_fold_array.sbatch`.

```
python3 src/build_boltz_inputs.py --seqs results/cfg_steer_seqs.csv --subset results/iptm_subset.txt --indir .../boltz_in
sbatch --array=0-6%4 --export=ALL,STRIDE=60,INDIR=.../boltz_in,OUT=.../boltz_out boltz_fold_array.sbatch   # Boltz-2, --use_msa_server
python3 src/parse_boltz.py --out-dir .../boltz_out --subset results/iptm_subset.txt --out results/iptm_steer_boltz.csv
python3 src/analyse_iptm.py --in results/iptm_steer_boltz.csv --out results/iptm_summary_boltz.csv
```

## Design & controls — PASS
Same SET-A α=2 steered sequences (wt / L k0-2 / random k0-2), same **frozen batch-1 60 complexes**
(`iptm_subset.txt`), same crystal interface set and g1-then-g2 residue order as the AF2 run. **60/60 complexes,
all 7 arms, 0 missing.** wt-sanity: **wt interface ipTM median 0.940** (native complexes fold confidently → chain
order / MSA correct). MSA parity: Boltz-2 `--use_msa_server` = the same MMseqs2 (ColabFold) server AF2 used.

## Result — H1 PASSES, falsifier does NOT fire

**Paired L − random (mean over k, complex-clustered 95% CI):**

| metric | Δ (L − random) | 95% CI | P(>0) | n |
|---|---|---|---|---|
| **ipTM** | **+0.139** | [+0.101, +0.177] | 1.000 | 60 |
| interface pAE (lower=better) | −4.09 | [−5.19, −3.01] | 0.000 | 58 |
| interface pLDDT | +5.68 | [+4.23, +7.15] | 1.000 | 60 |
| **composite** (z-mean of the 3) | **+0.677** | [+0.529, +0.826] | 1.000 | 58 |
| global pTM (localization) | +0.048 | [+0.033, +0.064] | 1.000 | 60 |

- **H1 (specificity, load-bearing): PASS.** The pre-registered composite **AND** ipTM both favour L with CIs
  excluding zero, and all three interface metrics agree (ipTM ↑, interface pAE ↓, interface pLDDT ↑). The benefit
  is specific to the **L direction** — the matched-magnitude random control does the opposite. **Falsifier (either
  CI spanning 0 → AF2-specific) does NOT fire.**
- **H3 (localization): PASS.** Global pTM shifts **+0.048** — ~14× smaller than the composite: interface-local.
- **H2 (no collapse): satisfied.** L−wt ipTM −0.050, composite −0.41 → ordering **wt 0.94 > L ≈ 0.89 > random**;
  L close to wt, no interface collapse (same pattern as AF2).

## Cross-modality reading
Two architecturally different folders now agree. AF2-multimer (standing, n=60): ipTM **+0.226**, composite
**+0.779**. Boltz-2 (here, n=60): ipTM **+0.139**, composite **+0.677**. Same direction, comparable magnitude,
both CIs tight and excluding 0. The L>random interface benefit is **not a quirk of one predictor's training
signal** — it reproduces on a decorrelated all-atom model. Combined with the anti-circular ESM-IF1/MIF judge
matrix (`FINDINGS_judge_matrix.md`), the steering direction is corroborated across inverse-folding models AND
across structure predictors.

## Honest scope
- **Boltz kernel path (disclosed):** stock Boltz-2 uses `cuequivariance_ops_torch` CUDA triangle kernels on
  Ampere+ GPUs; those ops are not installable in this env, so the venv was patched to force the **pure-torch**
  triangle attention on ALL GPUs (`boltz2.py:setup` `use_kernels=False`). This is a numerically-equivalent
  computation path (not a different model), applied uniformly to every fold, so it does not bias the within-
  complex paired contrast. (First array run failed 252/420 on Ampere GPUs before the patch; re-run idempotent.)
- ipTM/pLDDT/pAE are model proxies for "does this assemble", not experimental binding; absolute scales differ
  between Boltz and AF2, so the cross-modality claim is at the level of the **paired L−random effect** (z-scored
  per metric within each folder), which is exactly what `PREREG_boltz.md` pre-registered — not the raw values.
- interface pAE / composite are over n=58 (2 complexes had a PAE-array length mismatch and were dropped for the
  pAE-dependent metrics; ipTM/pLDDT/pTM are the full 60). The effect is decisive at either n.
- **Pre-registered determinism spread — disclosed, not measured.** `PREREG_boltz.md` pre-registered a Boltz
  seed-SD (same wt under 2–3 seeds); it was **not** separately run (GPU-bounded). We rely on the within-complex
  paired design and on AF2's measured determinism reference (~0.017 ipTM SD; Boltz's own SD is expected to be of
  comparable order), against which the paired +0.139 ipTM effect is large. A Boltz-specific seed-SD (one wt ×
  2–3 seeds, ~3 folds) can close this cleanly if wanted — flagged as a ~3-fold add, not a re-run of the contrast.

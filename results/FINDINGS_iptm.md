# FINDINGS — the CFG-steering benefit TRANSFERS to AF2-multimer: L-steered interfaces score higher ipTM

**Pre-registered** `results/PREREG_iptm.md` (subset frozen before folding). **Result:** the pre-registered H1
passes decisively — steering a frozen ProteinMPNN by `+α·L` at interface positions produces sequences an
**independent structure predictor (AF2-multimer)** is significantly more confident assemble than a
matched-magnitude **random**-direction control. Confirmation of the CFG-steering result (`FINDINGS_cfg_steer.md`),
not load-bearing. SEED=20260803. n = **60 complexes** (the full frozen subset; all 420 folds complete).
Raw `results/iptm_steer.csv`, stats `results/iptm_summary.csv`.

```
python3 src/build_iptm_fastas.py --n 60 --max-res 600 ... # frozen subset -> iptm_subset.txt
# AF2-multimer fold (ColabFold container, --num-models 1, --msa-mode mmseqs2_uniref_env, templates OFF):
sbatch --array=0-52%6 --export=ALL,STRIDE=8 $SCRATCH/ftax/jobs/iptm_fold_array.sbatch
python3 src/parse_iptm.py --out results/iptm_steer.csv
python3 src/analyse_iptm.py --in results/iptm_steer.csv --out results/iptm_summary.csv
```

## Design
α=2 steered sequences (wt background, interface positions replaced by the `+α·L` ProteinMPNN samples,
k=0..2) vs the matched-magnitude random control (k=0..2) vs wt, from `cfg_steer.py --dump-seqs`. Each folded
with AF2-multimer (the Exp D pipeline). Aggregate k per (complex, direction) by mean (best-of-k in the CSV).
Paired **L − random** per complex (complex-clustered bootstrap 95% CI, 5000 resamples). Interface set =
`leverage_skempi_positions.csv`; folded residue order = crystal-complex order so pLDDT/PAE indices align.

## Positive controls — PASS
- **wt sanity:** wt interface ipTM median **0.88** (pre-reg ~0.6–0.9) → chain order / MSA correct.
- **Determinism** (same wt sequence, 3 AF2 seeds, 5 complexes): within-complex ipTM SD mean **0.017**
  (`results/iptm_determinism.csv`) — the model noise floor; the effect below clears it by ~13×.

## Result — H1 PASSES (all three interface metrics agree), falsifier does NOT fire

**Paired L − random (mean over k, n=60, complex-clustered 95% CI):**

| metric | better | Δ (L − random) | 95% CI | P(>0) |
|---|---|---|---|---|
| **ipTM** | higher | **+0.226** | [+0.172, +0.283] | 1.000 |
| interface pAE | lower | **−5.27** | [−6.51, −3.98] | 0.000 (i.e. L lower = better) |
| interface pLDDT | higher | **+9.47** | [+7.13, +11.91] | 1.000 |
| **composite** (z-mean of the three) | higher | **+0.779** | [+0.599, +0.955] | 1.000 |
| global pTM (localization control) | higher | +0.080 | [+0.058, +0.103] | 1.000 |

- **H1 (specificity, load-bearing): PASS.** The pre-registered composite **AND** ipTM both favour L with CIs
  excluding zero, and **all three interface metrics agree** (ipTM ↑, interface pAE ↓, interface pLDDT ↑),
  P(>0)=1.000 throughout. The benefit is specific to the **L direction** — a matched-magnitude random
  perturbation does the opposite. The +0.226 ipTM gain is ~13× the 0.017 determinism floor.
- **H3 (localization): PASS.** Global pTM shifts **+0.080** — ~3× smaller than the ipTM shift and ~10× smaller
  than the composite: the improvement is interface-specific, not a global fold change. (pTM is not perfectly
  flat — a better interface lifts global pTM a little — but it moves far less than the interface metrics.)
- **H2 (no collapse): satisfied, with honest disclosure.** L−wt ipTM = −0.097 [−0.140, −0.056], composite
  −0.39. So the ordering is **wt 0.88 > L ≈ 0.78 > random ≈ 0.56**: L sits between wt and random, **much closer
  to wt** (Δ0.095) than to random (Δ0.223). Steering to a non-native (binding-favorable) interface costs a
  little foldability relative to wt but does **not** collapse the interface (L ipTM ~0.79 is a confident
  assembly), whereas the random control tanks it. Consistent with the CFG result where native recovery was
  preserved under L but degraded under random.

## Reading
An independent structure predictor confirms the CFG-steering direction: sequences steered along ProteinMPNN's
own leverage `L` fold to interfaces AF2-multimer is markedly more confident about than random-of-matched-
magnitude, across three complementary interface metrics, localized to the interface, on 60 complexes with
tight CIs. This corroborates the anti-circular ESM-IF1 leverage result with a physics-adjacent readout.

## Honest scope
ipTM/pAE/pLDDT are model proxies for "does this assemble", not experimental binding, and are metric-noisy —
hence the multi-metric composite, the determinism floor, and the localization control, all pre-registered. This
strengthens the steering result; it does not replace the experimental question. `--num-models 1` per the
pre-reg (one model); the effect is far larger than the single-model seed noise. → `iptm_steer.csv`,
`iptm_summary.csv`, `iptm_determinism.csv`.

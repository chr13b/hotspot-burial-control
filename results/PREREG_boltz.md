# PRE-REGISTRATION — does the CFG-steering ipTM benefit survive a DECORRELATED structure predictor (Boltz-2)?

**Committed BEFORE any Boltz-2 number (CLAUDE.md rule 1). Frozen — do not edit after the first fold.**
`SEED=20260803`. This is the cross-modality confirmation of the CFG-steering result (`FINDINGS_cfg_steer.md`,
`FINDINGS_iptm.md`): the standing ipTM benefit was measured with AF2-multimer, itself an evoformer/MSA model in
the same family as much of the training signal. Boltz-2 is an independent, architecturally decorrelated
all-atom folder. If the L-direction benefit is a genuine property of the sequences (not an AF2 quirk), it should
reproduce under Boltz-2.

## Design (identical contrast to the AF2 ipTM run)
SET-A = ProteinMPNN `+α·L` steered sequences at α=2 (wt background, interface positions replaced by the steered
samples, k=0..2) vs the matched-magnitude **random**-direction control (k=0..2) vs **wt**, from
`cfg_steer.py --dump-seqs` (`results/cfg_steer_seqs.csv`). Batch-1 subset = the frozen 60 complexes in
`results/iptm_subset.txt`. Fold all 7 arms per complex (60 × 7 = 420 folds) with **Boltz-2**. Interface set =
the SAME crystal `leverage_skempi_positions.csv` used by `parse_iptm.py`; folded residue order = crystal-complex
order so pLDDT/PAE indices align. Aggregate k per (complex, direction) by mean (best-of-k secondary). Paired
**L − random** per complex, complex-clustered bootstrap 95% CI (5000 resamples).

## Metrics (identical to `analyse_iptm.py`)
Per fold: Boltz-2 native **ipTM**, global **pTM** (localization control), **interface pLDDT** (mean pLDDT over
interface residues), **interface pAE** (mean predicted-aligned-error over cross-interface g1↔g2 residue pairs).
Pre-registered robust **composite** = z-mean of (ipTM, −interface pAE, interface pLDDT) across all folds.

## Hypotheses
- **H1 (specificity, load-bearing).** Paired **composite(L) > composite(random)** AND **ipTM(L) > ipTM(random)**,
  both complex-clustered CI excluding 0. The benefit is specific to the **L direction** — a matched-magnitude
  random perturbation does not produce it.
- **H3 (localization).** Global **pTM** shifts far less than the interface composite / ipTM — the improvement is
  interface-localized, not a global fold change.
- **H2 (no collapse).** L−wt composite is not strongly negative — steering to a binding-favorable interface does
  not collapse foldability (reported honestly with the ordering wt ≥ L > random, whatever it is).

## Falsifier (report verbatim if it fires)
If the paired **composite** L−random **or** the paired **ipTM** L−random has a 95% CI that includes 0 (or is
negative), the ipTM gain **does not reproduce under Boltz-2** → it is (at least partly) **AF2-specific**, reported
verbatim as a bounded cross-modality result. A null here bounds the claim; it does not erase the standing AF2 +
ESM-IF1-judge result.

## Positive controls (rule 6 — BEFORE trusting any contrast)
1. **wt-sanity gate (STOP-gate).** Median wt interface ipTM on ≥5 wt complexes must be ~0.6–0.9. If systematically
   low, the chain-order / MSA setup for Boltz-2 is wrong — STOP and fix before folding the arms.
2. **Determinism spread.** Same wt sequence folded under 2–3 seeds → within-complex ipTM SD, kept visible next to
   the effect (an effect inside the seed-noise floor is not an effect).
3. **MSA parity.** Reuse the ColabFold-computed MSAs (a3m) where possible so Boltz-2 vs AF2 differ in the FOLDER,
   not the evolutionary input; disclose if single-sequence is used instead. (If Boltz-2 setup fights us, fall back
   to **Boltz-1** and note it in FINDINGS.)

Outputs: `results/iptm_steer_boltz.csv` (per fold: complex_id, direction, k, iptm, ptm, interface_pae,
interface_plddt), `results/iptm_summary_boltz.csv` (paired contrasts + CIs, the composite, the localization
control), `results/FINDINGS_boltz.md`. Script/driver committed under `src/`.

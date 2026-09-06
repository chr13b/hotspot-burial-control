# PRE-REGISTRATION — AF2 ipTM batch-2 (60 → 120): fresh-complex replication + tighter CIs

**Committed BEFORE folding batch-2 (CLAUDE.md rule 1). Frozen — do not edit after the first batch-2 fold.**
`SEED=20260803`. Extends the primary AF2-multimer ipTM headline (`PREREG_iptm.md`, `FINDINGS_iptm.md`; standing
n=60: paired L−random ipTM +0.226 [+0.172,+0.283], composite +0.779 [+0.599,+0.955], H1 PASS) to a second,
DISJOINT batch of 60 complexes — testing that the effect is not batch-1-specific and tightening the CIs on 120.

## Frozen subset (committed alongside this file, BEFORE any batch-2 fold)
`results/iptm_subset_batch2.txt` = 60 complexes, selected by the identical committed rule (`build_iptm_fastas.py`,
`SEED=20260803`): eligible = all 7 arms present in `cfg_steer_seqs.csv` AND wt residue count ≤ 600; then
**excluding** the 60 batch-1 complexes (`--exclude results/iptm_subset.txt`); SEED-shuffled first-60 of the
remaining 187 eligible. **Positive control (recorded):** re-running the builder with no exclusion reproduces the
committed batch-1 `iptm_subset.txt` EXACTLY (shuffle is deterministic); batch-1 ∩ batch-2 = 0. Same
ProteinMPNN-steered SET-A sequences (α=2) already in `cfg_steer_seqs.csv`; same AF2 pipeline, interface set,
parse, and `analyse_iptm.py` as batch-1.

## Hypotheses
- **H1 (fresh-complex replication, load-bearing).** On the batch-2-ONLY 60 complexes, paired **L − random**
  **ipTM > 0** AND **composite > 0**, complex-clustered 95% CI excluding 0. The steering benefit reproduces on
  complexes never used in the batch-1 headline.
- **H2 (tighter CIs).** On the combined 120, the paired L−random ipTM/composite CIs are tighter than at n=60,
  with the point estimates consistent (no drift outside the batch-1 CI).
- **H3 (localization).** Global pTM shift remains far smaller than the interface composite (as at n=60).

## Falsifier (report verbatim if it fires)
If the **batch-2-only** paired L−random **ipTM** or **composite** 95% CI includes 0 (or is negative), the effect
is (partly) **batch-1-specific** → reported verbatim; the n=60 headline stands but its generality is bounded.

## Positive controls (rule 6)
wt-sanity (median wt interface ipTM ~0.6–0.9 on batch-2) and the determinism SD (0.017, from batch-1) kept
visible next to the effect; idempotent skip means committed batch-1 folds are reused, never re-folded.

Outputs: `results/iptm_subset_batch2.txt` (frozen list), `results/iptm_steer_120.csv` (batch-1 committed rows +
batch-2, no re-fold of batch-1), `results/iptm_summary_120.csv` (`analyse_iptm.py` on 120 + a batch-2-only slice),
`results/FINDINGS_iptm_batch2.md`.

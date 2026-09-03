# PRE-REGISTRATION — does the confidence–leverage decomposition replicate on a SECOND natural fixture (ATLAS, TCR–pMHC)?

**Committed BEFORE any ATLAS leverage/CPI number (CLAUDE.md rule 1).** `SEED=20260803`. This is a *bonus*
replication attempt: SKEMPI 2.0 is the field's converged sink and no clean large non-overlapping natural
ΔΔG_bind fixture exists (see §9); ATLAS (TCR–peptide–MHC affinities + structures) is the cleanest genuinely
non-overlapping option, at the cost of a narrow, atypical biology. **A null here bounds generalization; it does
not refute the SKEMPI result.** Either outcome is reportable.

## Fixture
ATLAS (Borrman et al. 2017; atlas.ibbr.umd.edu / weng-lab/ATLAS): TCR–pMHC complexes with mutations and measured
binding affinities (mostly SPR), each with a solved structure. Expected scale ~700–1000 records over ~110–123
structures (to be confirmed on download — treat the exact count as unverified until fetched).

## The partner-ablation operator on TCR–pMHC
The two biological binding partners are the **TCR** (α+β chains) and the **peptide–MHC** (MHC α[+β2m] + peptide).
`X_complex` = the full TCR–pMHC; `X_monomer` = the TCR with the pMHC deleted (partner removed), on the *bound*
backbone — exactly the SKEMPI operator. Interface positions and leverage `L_i(a) = [logP(a|complex) −
logP(wt|complex)] − [logP(a|monomer) − logP(wt|monomer)]` are defined identically. Mutations in ATLAS are on the
TCR side, so the TCR is the scored chain group; if a subset mutates the peptide, score that side symmetrically.

## Hypotheses (pre-registered, mirroring the SKEMPI feature-class law)
- **H1 (leverage replicates).** Mutation-level **Spearman(L, ΔΔG_bind) < 0** and **CPI(L | geometry) > the
  placebo floor**, as on SKEMPI (there −0.30 / +0.059).
- **H2 (confidence blind).** Position-level **CPI(confidence | geometry) ≈ 0** (at or below the floor).
- **Falsifier.** If Spearman(L, ΔΔG) ≥ 0, **or** CPI(L | geometry) does not clear the placebo floor, the
  decomposition does **not** replicate on TCR–pMHC — reported verbatim as a bounded-generalization result.

## Positive controls (rule 6 — run BEFORE trusting any number)
1. **Scorer sanity:** ProteinMPNN/ESM-IF1 native recovery on a held ATLAS structure must be in the normal range
   (~0.3–0.5 interface), and a bit-identical re-score of one complex must reproduce.
2. **SKEMPI-overlap check (the critical one).** Cross-map ATLAS complexes/mutations against SKEMPI 2.0. Report
   the overlap; run H1/H2 on the **non-overlapping subset** (and on the full set, disclosing both). If ATLAS is
   substantially a SKEMPI subset, say so and treat it as non-independent.
3. **Interface definition control:** the TCR–pMHC interface residue set must be non-empty and biophysically
   sensible (CDR loops contacting peptide/MHC); spot-check one complex.

## Metrics (identical to SKEMPI, complex-clustered bootstrap, seed fixed)
Mutation-level Spearman(L, ΔΔG) and CPI(L | burial+nbr+ΔSASA); position-level CPI(confidence | geometry) and the
placebo floor on this fixture; report `n` mutations / `n` complexes and the overlap-with-SKEMPI fraction.

## Honest scope stated up front
TCR–pMHC is germline-biased, μM-affinity, peptide-in-groove recognition — genuinely different physics from
general PPI. This tests whether the decomposition is a property of inverse-folding on protein interfaces broadly.
A clean replication strengthens generality; a null is a real boundary and is reported as one, not buried.

Output: `results/atlas_leverage.csv` (+ `FINDINGS_atlas.md`). Script: `src/leverage_atlas.py` (to be written,
reusing the committed leverage machinery).

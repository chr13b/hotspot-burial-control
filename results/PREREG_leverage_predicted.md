# PRE-REGISTRATION — does leverage's binding signal survive on PREDICTED backbones? (§6)

**Frozen before any CPI number was computed** (this file + `src/leverage_predicted.py` committed prior to
running `--stage analyse`). BRIEF.md ground rule 1: no falsifier is moved after seeing a number.
SEED = 20260803. Script: `src/leverage_predicted.py`.

## Question

The crystal leverage result (CPI(L | geometry) > 0) and the crystal backbone-noise ladder (survives
≤0.5 Å, collapses ~1 Å for ProteinMPNN / ~1.5 Å for ESM-IF1) establish that the mixed derivative carries
binding information on native/accurate backbones. **Designers do not have crystals — they condition on a
folding model's backbone.** This tests whether CPI(L | geometry) and the geometry+|L| hotspot ranker
survive on OpenFold3 and AF2-multimer predicted backbones for the complexes shared with the SKEMPI ΔΔG
fixture.

## Design — one-variable manipulation

The SKEMPI interface-mutation fixture is held FIXED from the committed crystal analysis (same complexes,
same interface positions, same alanine-scan hotspot labels, same ΔΔG). Only two things are swapped, and
they are exactly what a designer working off a predicted backbone gets:

1. **Leverage L** is recomputed by the IDENTICAL committed scorer (`leverage_decomposition._score_one` →
   ProteinMPNN sequence-free unconditional marginal, complex vs partner-deleted monomer) reading the
   **predicted** complex PDB. The monomer is constructed the identical way (delete the partner chain from
   the same predicted PDB).
2. **Geometry** (burial = −rSASA_complex, neighbour count, ΔSASA) is taken **from the predicted structure**
   — the `expA_p0_positions.csv` (OF3) and `expD_p0_positions.csv` (AF2) files, built by
   `p0_burial_matched.py` on the same predicted PDBs. This is the honest baseline a designer has; the
   crystal geometry is NOT used for the predicted runs.

Interface membership and the hotspot label are properties of the ΔΔG data, not the backbone, so they are
inherited from the committed crystal fixture — keeping the sample identical across crystal / OF3 / AF2 so
the ONLY thing changing is the backbone the derivative is read from. Shared set: complexes present in
OF3 ∩ AF2 predicted geometry ∩ SKEMPI interface-mutation fixture (**140** at complex granularity; effective
mutation/position counts reported after position mapping + WT-match).

## Prediction

OpenFold3/AF2-multimer interface RMSD is typically ~0.5–1.5 Å — the regime where the crystal ladder shows
partial-to-full survival. **Predict roughly the σ ≈ 0.5–1.0 Å rung: CPI(L | geometry) in ~[+0.01, +0.04]**,
positive with the complex-bootstrap CI clearing the placebo floor (+0.0007), on both predictors, attenuated
relative to the crystal value.

## Positive control (gate — trust nothing until it passes)

`--source crystal` runs the identical pipeline on the crystal PDBs restricted to the shared 140. It must:
- reproduce the **committed** crystal leverage L per mutation (`leverage_skempi_mutations.csv`) and per
  position (`leverage_skempi_positions.csv`) to float32 MPNN precision (~1e-5); and
- land near the committed crystal CPI(L | geometry) — **≈ +0.059 mutation-level, ≈ +0.0048 position-level**
  (the shared-140 subset value may differ from the full-285 committed number; reported alongside).

If the crystal control fails, the predicted numbers are uninterpretable → stop and debug.

## Falsifier — either direction strengthens the paper, and neither will be softened

- **If CPI(L | geometry) COLLAPSES to the placebo floor on predicted backbones** (CI includes 0 on a
  powered sample): leverage does NOT generalize to the design regime. Report it as a **measured
  factorization tax on the real pipeline** — a *stronger* version of the thesis (the predicted backbone's
  interface error destroys the very signal that carries binding, which is precisely why staged
  backbone→sequence design misses hotspots). Do not soften.
- **If it SURVIVES** (CI > 0 on ≥1 predictor): the actionable claim generalizes past crystals, and the
  crystal dose law is conservative. Do not overstate — report the attenuation honestly and per predictor.

## Readouts (all with complex-clustered bootstrap, seed 20260803)

- CPI(L | burial+nbr+ΔSASA), **mutation-level** and **position-level**, per source (crystal, OF3, AF2) and
  pooled; plus the +confidence control and the drop-3-influential-complex robustness.
- Spearman(L, ΔΔG_bind) per source.
- The geometry+|L|_rms hotspot ranker: OOF cross-fit logistic AUROC (identical metric to
  `w4_combined_ranker.py`), ΔAUROC of geometry+|L| over geometry alone, paired complex-bootstrap.

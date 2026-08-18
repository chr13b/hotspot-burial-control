# Handoff — Project #3: "Is confidence the wrong derivative across structural generative models?"

**Status:** NOT STARTED. This is the explicit *next paper* after the current ICLR submission (do not fold it
into the current paper — different object, needs a GPU). Start on Sherlock the day after we submit.

**One-line thesis.** The current paper shows that for an *inverse-folding* model, per-residue **confidence is
blind to interface hotspots**, while the **partner-ablated mixed derivative** (leverage `L`) carries the
binding signal. Question: does the same dissociation hold for a *structure predictor* — **AlphaFold-Multimer**?
If pLDDT/PAE is blind but the partner-ablation derivative is not, the thesis generalizes from "a fact about
inverse folding" to "a fact about structural generative models." If AF-Multimer is partner-sensitive in its
confidence too (it explicitly models complexes), that *bounds* the thesis to IF — also a real, publishable
result. **Either outcome is a paper.**

Why it is a *separate* paper and not a column in the current one: AF-Multimer's "leverage" is a
**partner-ablation of a structure predictor** (predict bound vs unbound, read the change), not a **backbone
perturbation of a frozen IF model**. The W2 dose-law mechanism does not transfer; the object needs its own
methods, controls, and confounds. Folding it in would dilute a tight thesis into "two papers in a trench coat."

---

## The measurement (two tiers; start with Tier 1)

Let `i` be an interface residue; hotspot label from SKEMPI alanine ΔΔG_bind ≥ 1 kcal/mol (as in the main paper).

**Diagonal / "confidence" (one-pass, predict-blind analogue of IF confidence):**
- `pLDDT_i(complex)` and interface `PAE` to the partner chain, from the BOUND AF-Multimer prediction.
- Hypothesis (H0-confidence): blind to hotspots within a burial-matched control — mirrors the IF result.

**Mixed derivative / "leverage" (partner ablation — the load-bearing object):**
- **Tier 1 (cheap, START HERE): confidence-change on ablation.**
  `ΔC_i = f(i | bound complex) − f(i | monomer)` where `f` is pLDDT (and, separately, a local-structure change:
  per-residue Cα displacement between the bound-context and monomer-context predictions after alignment on `i`'s
  neighborhood). Tests whether the partner's *presence* changes the model's local certainty/geometry more at
  hotspots than at burial-matched controls. NO per-mutation predictions needed → ~1000 AF runs total.
- **Tier 2 (expensive, only if Tier 1 is promising): prediction-based ΔΔG proxy.**
  For each SKEMPI mutation, predict the mutant complex and read an interface-confidence / interface-contact
  change as a ΔΔG_bind proxy, partner-ablated (mutant-monomer as the reference). Correlate with SKEMPI ΔΔG.
  This is the *faithful* second-difference but costs thousands of predictions.

**The pre-registered discriminator (write this BEFORE running, mirror the main paper's ground rules):**
1. Burial-matched pairs within complex (rSASA ±0.05, same SS class, nbr ±1) — burial is the confound that
   *hides* the effect (buried = high pLDDT), exactly as in the IF paper.
2. Report the same readouts the main paper survived: **CPI** (conditional predictive impact, cross-fit,
   conditional-permutation within geometry strata) and within-geometry-stratum AUROC. NOT ΔAUROC-over-0.
3. Complex-clustered bootstrap; fix + record every seed. Every number → a committed CSV.
4. Positive control before trusting any zero (a known hotspot must score as a hotspot in the pipeline).

---

## Reusable dependencies (from the main repo `../`)

- **`../src/ftax_common.py`** — SKEMPI parsing (`parse_skempi`), complex/monomer loading with the partner
  chains deleted (`load_complex(..., group1, "")` gives the monomer — the SAME partner-ablation used here),
  burial/rSASA + neighbor-count features, hotspot labeling. The partner-ablation bookkeeping is already solved.
- **`../src/leverage_decomposition.py`** — the `cpi(y, g, Z, X, rng)` estimator and the within-stratum-AUROC
  and drop-3-influential robustness code. Reuse verbatim so the AF result is method-identical to the IF result.
- **`../src/p0_burial_matched.py`** — the burial-matched pairing (the whole controlled comparison).
- **`skempi_complexes.txt`** (this folder) — the 344 SKEMPI complex ids (`PDB_group1_group2`) already used,
  so the two papers are on the *same* complexes → directly comparable.
- Hotspot labels + geometry per position: `../results/leverage_skempi_positions.csv` (chain, resnum, is_hot,
  rsasa, nbr, is_interface) — reuse as the label/feature table; you only add AF-derived columns.

## What you must ADD (the GPU part, on Sherlock)

- AF-Multimer / ColabFold predictions: 344 bound complexes + ~700 monomers (each chain group unbound).
  MSA-dominated; precompute MSAs once and reuse (bound and unbound share single-chain MSAs).
- A parser for pLDDT (b-factor column of the AF PDB) and PAE (the `*_pae.json` / pickle) → per-residue table
  keyed by (chain, resnum) matching `leverage_skempi_positions.csv`.
- `starter_afm_mixed_derivative.py` (this folder) is a runnable SKELETON of the Tier-1 pipeline with the
  analysis wired to the reused CPI — fill in the two TODOs (AF runner + pLDDT/PAE parse).

## Compute estimate (Sherlock)

- Tier 1: ~1000 AF-Multimer/AF2 runs, MSA-dominated. With precomputed MSAs ≈ 5–15 GPU-min each →
  **~100–250 GPU-hours**, multi-day. Fits a Sherlock GPU allocation.
- Tier 2 (only if Tier 1 fires): thousands of mutant predictions → **500+ GPU-hours**. Decide after Tier 1.

## Decision points (resolve at kickoff)

1. AF stack: **ColabFold** (fast, MSA caching, easiest on Sherlock) vs full AlphaFold-Multimer (heavier). Default
   ColabFold unless a reviewer would object to MSA subsampling.
2. Which "confidence": pLDDT vs interface-PAE vs both. Default: report both; PAE is the more binding-specific.
3. Ablation reference for the monomer prediction: predict the isolated chain group with its own MSA
   (recommended) vs mask the partner in the complex MSA (leakier). Default: isolated-chain prediction.
4. Memorization confound (AF has seen many of these PDBs): include a held-out / recent-PDB subset, or a
   template-off ablation, mirroring the main paper's Exp A/D leakage controls.

## Prior art to position against (verify all DOIs with the reference checker before citing)

- AF-based ΔΔG_bind methods exist (e.g. AF-Multimer confidence as an affinity proxy) — cite and distinguish:
  we test the *decomposition* (is confidence blind, does the partner-ablation derivative carry it?), not a new
  ΔΔG predictor. See `../results/REFERENCES_verified.bib` and `../memory`-noted credits (AlphaFold-Multimer =
  bioRxiv 10.1101/2021.10.04.463034, verified).
- The current paper's decomposition/leverage credits (BA-Cycle 2410.09543; categorical Jacobian Zhang et al.
  10.1101/2024.01.30.577970; RedNet 10.64898/2026.05.09.722041) carry over as the conceptual lineage.

## First session checklist

1. Copy the pre-registration ground rules from `../CLAUDE.md` and `../BRIEF.md §4` into `PREREG.md` here; freeze
   the falsifiers BEFORE any AF run.
2. Set up ColabFold + MSA cache on Sherlock; smoke-test on one complex (e.g. `1BRS_A_D`, barnase–barstar) both
   bound and unbound; confirm the pLDDT/PAE parse aligns to `leverage_skempi_positions.csv`.
3. Run the Tier-1 confidence-blindness + partner-ablation-derivative CPI on ~30 complexes as a powered pilot;
   go/no-go on the full 344 and on Tier 2.

# DMS as a powered independent fixture — follow-up plan (PARKED)

*Status: parked as a later bonus / follow-up-paper centerpiece (decision 2026-09-05). Prioritised now:
second steered model (ESM-IF1), second/third judge (PiFold), second folder (Boltz-2), 60→120. This file
preserves the plan so it is not lost.*

## The question DMS could answer
"**Does leverage `L` add binding information beyond geometry on a powered, independent fixture?**" — the one
thing the 3-complex ATLAS fixture cannot power (its geometry-controlled CPI placebo floor is ~60× SKEMPI's).
DMS supplies thousands of independent variants, so the conditional predictive impact (CPI) test *can* be
powered there.

**Honest nuance on "…on TCR–pMHC specifically":** the large, clean DMS resources (AbAgym / BindingGYM) are
mostly **antibody–antigen**, not TCR–pMHC. So:
- DMS cleanly answers *"does L add beyond geometry on a powered, independent fixture."*
- Keeping it **TCR–pMHC** depends on a TCR–pMHC DMS existing at sufficient scale — must be **verified**, not
  assumed (rule 5: no citing a dataset from memory). If we relax to antibody–antigen DMS, it is powered but
  changes the biological system from ATLAS's TCR–pMHC.

## Two lanes (both runnable through the same CPI machine; independent datasets)
- **Lane 1 — TCR–pMHC DMS:** the direct ATLAS follow-up ("does L add beyond geometry *on TCR–pMHC*"),
  *if* a scaled TCR–pMHC DMS is verified to exist.
- **Lane 2 — antibody–antigen DMS (AbAgym / BindingGYM):** a powered, genuinely non-overlapping fixture;
  answers the general question, in a different biological system than ATLAS.

## Steps to run it
1. **Dataset selection + provenance** — a DMS with (i) a *binding-specific* readout, (ii) a solved or
   reliably-predictable WT complex structure, (iii) enough independent positions to power the CPI, and
   (iv) **verified non-overlap with SKEMPI**.
2. **WT-identity gate** — same hard gate as ATLAS; map every variant onto the complex structure.
3. **Geometry features** on the WT complex (rSASA, neighbour count, ΔSASA, secondary structure).
4. **Leverage `L` per variant** under ProteinMPNN + ESM-IF1 (partner-ablation cycle, decoding-order
   averaged) — the compute bulk (thousands of variants × 2 models).
5. **Label QC + calibration** (below).
6. **CPI pipeline** — the existing machine (cross-fit, GroupKFold, conditional permutation within geometry
   strata, complex-clustered bootstrap, placebo floor).
7. **Sensitivity + write-up.**

## Why it is so much effort
Not the CPI (that is our existing machine). It is (a) dataset wrangling + the non-overlap and
binding-specificity verification, (b) per-variant leverage across two models **at DMS scale**, and (c) the
calibration argument, which is a mini-study reviewers will scrutinise because the *measurement modality*
differs from ΔΔG.

## The calibration argument you must make
A DMS reports an **enrichment / fitness score**, which is: (i) in arbitrary units; (ii) often
**nonlinear / saturating** in ΔΔG (ceiling & floor censoring); and (iii) frequently **confounded by
expression / stability, not pure binding**. CPI does not need kcal/mol — it asks whether `L` adds signal
beyond geometry *for the label* — but the label must be a valid binding readout. So show:
- **Monotonicity** — the DMS score is monotone in binding affinity for the in-scope mutations (anchor
  against any overlapping K_D / ΔΔG).
- **Binding-specificity** — the readout reflects *binding*, not expression (ideally a **Tite-Seq K_D**
  dataset, which is thermodynamic and makes the calibration burden minimal).
- **Censoring-robustness** — use a **rank-based** CPI so ceiling/floor censoring does not drive the result.

## Pro / con + recommendation
**Pros:** genuinely non-overlapping with SKEMPI; **powers** the geometry-controlled CPI (what ATLAS cannot);
design-relevant modality; would let us claim *"L adds beyond geometry on a second, independent, powered
fixture"* — a real ceiling-raise.
**Cons:** modality mismatch → the calibration argument is under scrutiny; large effort; "TCR–pMHC
specifically" may lack a scaled dataset; expression/stability confound risk; next-paper-scale.

**Recommendation: not for this paper.** Make it the centerpiece of a follow-up (or the rebuttal response if a
reviewer demands a powered second fixture). ATLAS already delivers honest **bounded generalization** (the
leverage direction and the confidence-blindness carry to TCR–pMHC; the geometry-controlled add-on is
indeterminate there at 3 non-overlapping complexes, not absent).

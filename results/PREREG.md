# Pre-registration — analysis choices fixed BEFORE any number was computed

Written 2026-08-03, after STEP 0 (preprint fetch) and after downloading SKEMPI 2.0 + ProteinMPNN,
but **before running any scoring or statistics**. BRIEF.md §4 falsifiers F0/F1/F2 are unchanged and
are not restated here except where an operational detail was underspecified.

Purpose: BRIEF.md §4 fixes the falsifiers but leaves several operational choices open. Fixing them
here removes the degree of freedom to choose, after seeing a number, the variant that gives the
answer I want.

## 1. ΔΔG convention

`ΔΔG_bind = RT·ln(Kd_mut / Kd_wt)`, R = 1.9872036e-3 kcal/mol/K, T from the SKEMPI `Temperature`
column (numeric prefix; rows without a parseable T use 298 K and are flagged). **Positive ΔΔG =
mutation weakens binding.** Only rows with both `Affinity_mut_parsed` and `Affinity_wt_parsed`
present and > 0.

## 2. Labels

- **Hotspot (loose, primary):** single mutation to alanine, wild-type ≠ Ala, ΔΔG > 1.0 kcal/mol.
- **Hotspot (strict, ProBID-Net comparability):** same but ΔΔG > 2.0 kcal/mol.
- **Null:** single mutation to alanine, wild-type ≠ Ala, |ΔΔG| < 0.25 kcal/mol.
- Where a position has multiple alanine measurements, the **median** ΔΔG over rows is used, and the
  position is dropped if the measurements straddle the hotspot and null bands.

## 3. Interface definition

A residue is interface if `ΔrSASA = rSASA_free − rSASA_complex > 0.05`, where `rSASA_free` is
computed with the residue's own SKEMPI chain-group in isolation and `rSASA_complex` with both
groups present. Absolute SASA from freesasa (Shrake–Rupley), normalised by Tien et al. 2013
*theoretical* per-residue maxima. SKEMPI's own `iMutation_Location(s)` (COR/SUP/RIM) is recorded
alongside but is **not** the primary definition.

## 4. Matching (the experiment)

Burial variable for matching is **`rSASA_complex`** — the burial the model actually experiences,
since it conditions on the bound complex backbone.

Constraints, all within the same complex, exactly as BRIEF.md §4:
- |Δ rSASA_complex| ≤ 0.05
- identical secondary-structure class (H / E / L)
- |Δ neighbour count| ≤ 1

Neighbour count = number of other residues in the complex whose Cβ (Cα for Gly) lies within 10.0 Å
of this residue's Cβ.

Pairing is **optimal 1:1 assignment** (`scipy.optimize.linear_sum_assignment`) minimising
|Δ rSASA_complex|; each control is used at most once. This is fixed now so that a greedy variant
cannot be substituted later.

**Control-pool hierarchy, declared now, in order.** The verdict on F0 is taken from the PRIMARY.
The others are reported for power and robustness and cannot replace the primary.

| Tier | Control pool | Rationale |
|---|---|---|
| **PRIMARY** | Positions labelled **null** (measured Ala ΔΔG, \|ΔΔG\| < 0.25) | Both members of the pair carry experimental evidence; the control is *known* not to be a hotspot |
| SECONDARY-A | Any interface position with a measured Ala mutation that is not a hotspot | Larger pool, weaker control label |
| SECONDARY-B | Any interface position at all (no measurement required) | Largest pool, control label is "unmeasured", so a hotspot may be hiding in it — biases toward the null |

## 5. Model and scoring

- Checkpoint: `vanilla_model_weights/v_48_020.pt` (primary; the ProteinMPNN default),
  `v_48_002.pt` reported as a robustness check because RedNet's SKEMPI table used σ = 0.02.
- `augment_eps = 0.0` at inference (no backbone noise added on top of the checkpoint).
- Score = per-position teacher-forced autoregressive conditional log-probability of the **native**
  residue, `log p(s_i | backbone_bound_complex, s_{<i in decoding order})`, from `model.forward()`.
- **8 decoding orders**, seeds 0–7, supplied explicitly via `use_input_decoding_order=True`.
- Primary per-position score = mean over the 8 orders.
- **Decoding-order spread is assessed on the ESTIMATE, not the position:** the entire paired-gap
  analysis is re-run independently within each of the 8 orders, and the SD of those 8 gap estimates
  is reported. Per BRIEF.md §5.6, a gap smaller than that SD is not a result.
- Order-independent secondary: unconditional (backbone-only, no sequence context) log-probability,
  single pass, no decoding-order variance at all.

## 6. Statistics

- Paired difference `d = logp(hotspot) − logp(control)`. **A negative mean d is the hypothesised
  direction** (hotspots are harder / lower native log-prob).
- **Complex-level bootstrap:** resample complexes with replacement (complex = independent unit),
  recompute the mean of all pair differences in the resampled set. **10,000 replicates, seed 20260803.**
  95% CI = percentile interval. F0 fires if this CI contains zero.
- Headline restricted per BRIEF.md §5.4 to 293–303 K. pH is not a parseable SKEMPI column; a pH
  restriction is therefore **not applied**, and this deviation is reported.

## 7. F1 — partial Spearman

- Quantity: inverse-folding **log-odds** of the mutation, `ℓ_i(mut) − ℓ_i(wt)`, versus SKEMPI
  ΔΔG_bind, over single mutations at interface positions.
- Burial-controlled = **partial Spearman controlling for `rSASA_complex`** (rank-transform all three,
  regress out the control linearly, correlate residuals).
- The hypothesis predicts a **negative** correlation (model dislikes the mutation ⇒ ΔΔG large).
  **F1 fires on |ρ_partial| ≥ 0.35.** Absolute value is used so that a strong correlation of either
  sign refutes blindness.

## 8. F2 — N_hot

- T = 0.1. Hotspot constellation per complex = its loose-threshold hotspot positions.
- `δ_i = log p_mode − log p_native` at hotspot i (nats), from the 8-order mean conditionals.
- **Analytic (BRIEF formula):** `N_hot = exp(Σ_i δ_i / T)`.
- **Analytic (exactly normalised):** `N_hot = 1 / Π_i p_T,i(native)` with
  `p_T,i(a) = p_i(a)^{1/T} / Σ_b p_i(b)^{1/T}`. Reported alongside because the BRIEF formula is the
  mode-dominance approximation to it.
- **Direct:** sample K sequences from `model.sample()` at T = 0.1 and count draws recovering the
  **full** constellation. Seeds fixed. Valid only where at least one recovery is observed; where
  zero are observed only a lower bound is reported, and it is reported as a bound, not a value.
- Discrepancy = `log10(N_direct) − log10(N_analytic)` on complexes where both are observable.
- F2 fires only on the conjunction: median log10 N_hot < 2 **AND** the F0 CI contains zero.

## 9. Deviations from BRIEF.md, declared up front

- **No DSSP binary is installable** (no sudo on this machine). Secondary structure is computed with
  a self-implemented Kabsch–Sander hydrogen-bond assignment reduced to 3 classes (H/E/L). It is
  validated against a known all-α and a known all-β structure before use; that validation is
  reported.
- **pH filtering is not applied** — SKEMPI 2.0 has no pH column.
- Phase 2 is not started (no GPU), per CLAUDE.md.

# PRE-REGISTRATION — "the model knows WHERE the interface is, not WHAT binding requires"

**Registered 2026-08-13, BEFORE running the analysis** (Big Idea 1 from the 2026-08-13 novelty brief).
Seed 20260803. Data: Bennett et al. 2023 de-novo binder SSM (already parsed; `src/bennett_kl_detector.py`,
`results/bennett_kl_positions.csv`). Analysis script: `src/bennett_knows_where.py` (written after this file).

## Hypothesis
An inverse-folding model's partner-conditioning is **geometric (occlusion), not energetic**. It predicts
which substitutions are tolerated at positions governed by its training objective (fold **stability**), but
is at/near chance at positions governed by **binding** — and conditioning on the target shifts its
distribution (KL 0.363 interface vs 0.016 elsewhere) **without** improving agreement with measured binding
preference. This explains why our learned KL ≈ ΔSASA: the model responds to the partner's *presence*, not
its energetics.

## Data & scores
- For each SSM position: the **19 non-native substitutions** with Kd bounds. A substitution **retains
  binding** if `kd_lb < highest_conc` (per-library cap; the assay's measurable-binding threshold), else
  **abolishes**. Native = the SSM-excluded aa (= PDB residue, verified 1.000).
- Model score at a position = `p(aa | complex backbone)[sub]` [**P**] and `p(aa | binder-alone backbone)[sub]`
  [**Q**], ProteinMPNN **unconditional** (sequence-free, backbone-only). Higher = model favours the substitution.

## Pre-specified strata (positional layer)
- **interface**: ΔSASA (surface buried by partner) > 5 Å².
- **core (non-interface)**: ΔSASA ≤ 1 Å² AND rSASA(complex) < 0.15 (buried in the binder fold).
- **surface (non-interface)**: ΔSASA ≤ 1 Å² AND rSASA(complex) > 0.40 (exposed).
- Positions between strata are excluded (clean layers).

## Baselines the model must beat to be "knowing" (sequence-only, no structure)
BLOSUM62(native, sub); volume similarity −|V(sub)−V(native)|; hydropathy match −|H(sub)−H(native)|.

## Primary estimator
Pooled AUROC( score predicts *retains-binding* ) within each layer, over all (position, substitution)
pairs; **design-clustered bootstrap** (resample the 73 parents), 2000 reps, seed 20260803. Layers require
≥50 pooled pairs with both classes present.

## Pre-registered predictions (falsifiers fixed NOW)
- **P1**: interface AUROC(P) > 0.5 (CI excludes 0.5) — the model encodes *some* interface tolerance signal.
- **P2 (the dissociation)**: interface AUROC(P) < core AUROC(P) — the model predicts **stability better than
  binding**. Bootstrap difference CI excludes 0.
- **P3 (the energetics test)**: interface AUROC(P) vs interface AUROC(Q) — does conditioning on the partner
  improve *binding* prediction over the binder-alone distribution? Bootstrap difference CI.

## Decision rule (declared in advance — cannot lose)
- **P2 fires + P3 null** → HEADLINE: "the model knows *where* (occlusion) not *what* (energetics);
  partner-conditioning is geometric." Explains KL ≈ ΔSASA.
- **P3 fires** → DIFFERENT, equally reportable headline: "partner-conditioning *does* add binding
  information beyond geometry" — which would also partially rescue KL-as-signal.
- Either way we report P1/P2/P3 and the model-vs-baseline comparison. No falsifier moves after seeing a number.

## Caveats stated in advance
SSM labels convolve display/fold-stability with binding — the **three-way stratification is the confound
control** (core = stability is the built-in positive control). 4 targets / one epitope each →
design-clustered bootstrap; report per-target signs. Parent sequences are ProteinMPNN outputs (native-biased)
— but we rank the 19 *alternatives*, and the bias is identical across strata.

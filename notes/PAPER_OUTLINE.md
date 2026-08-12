# Paper outline — the conditioning-set spine

Working draft, 2026-08-10. Iterate as Experiment C lands. Every claim maps to a committed CSV; the
`→` tags name the source. Target: **ICLR main track** (fallback TMLR). ~9 pages + appendix.

## Working title

**"Inverse folding is not blind to binding hotspots — until you take away the crystal.
The factorization tax lives in the conditioning set."**

Alt: *"The hotspot design gap is a conditioning-set artifact: hidden by native backbones, exposed by
the predicted backbones designers use."*

## One-paragraph abstract (sketch)

Staged binder design — generate a backbone, then inverse-fold a sequence — is widely believed to
struggle at interface *hotspots*, the few residues that carry most of the binding free energy, and a
published result (ProBID-Net) reports inverse-folding sequence recovery of 0.334 at hotspots vs 0.472
elsewhere. We show, pre-registered across five inverse-folding architectures on SKEMPI 2.0, that on
**native crystal backbones this gap is entirely a burial confound**: matched within complex for
solvent exposure, secondary structure and packing, hotspots are recovered no worse than non-hotspot
interface positions. But native crystal backbones are carved by the very side chains being predicted.
On **OpenFold3-predicted** backbones of the same complexes the burial-matched deficit **appears**
(−0.19 nats, paired Δ −0.15), as large where the prediction is accurate as where it is not. A
**sequence-free** signal — the partner-induced shift in the model's backbone-only distribution —
predicts where, strengthens on the predicted backbones, and **transfers to RFdiffusion generative
backbones** wherever a physical interface forms [Exp C]; on those same generative backbones the raw
log-prob deficit is present only at large backbone drift and the binding-energy ranking collapses off
the native manifold — a mixed result we report as such. The tax is
real but it is a property of the **conditioning set**, not of hotspot chemistry: the field benchmarks
on crystal backbones, which hide it, while designers work on non-native backbones, where it bites.
Two proposed alternative mechanisms — a low-temperature constellation cost and a commitment-ordering
schedule effect — are each measured and neither survives (the constellation cost is generic to any
multi-position set; commitment reordering is inert on both an autoregressive and a coupled
co-design model).

## Section map

**1. Introduction.**
- The staged pipeline; the frustration hypothesis (hotspots are buried polars / strained rotamers
  that buy affinity, not stability); the intuition that inverse folding, lacking a binding-energy
  term, misses them. → BRIEF §2.1.
- ProBID-Net's 0.334/0.472 gap and its dynamics attribution; the confound nobody controlled: hotspots
  are buried, and burial is where inverse folding is *most* confident.
- Contributions: (i) the gap is a burial artifact on crystal backbones (5 models, pre-registered) —
  and the burial-matched matched-pair design is offered as a reusable **protocol/benchmark** for
  evaluating inverse folding at interface hotspots without the confound; (ii) the tax is real and lives
  in the conditioning set — a burial-matched deficit appears on the backbones of TWO independent structure
  predictors (cross-predictor per-complex ρ=+0.57), is absent on noised-crystal generative backbones
  (so it is *independent-reconstruction*, not distance-from-native), and a sequence-free **partner-
  sensitivity** signal detects it across four backbone classes — while the model's own **confidence** does
  NOT (near-chance; naively hurts); (iii) that signal is actionable as a fixed-budget triage readout
  (KL+burial capture@3 +0.098 [+0.021,+0.175], n=106; crystal proof-of-concept, non-native validation
  pending #12); (iv) two competing mechanisms (constellation cost, commitment ordering) measured and refuted.

**2. Setup and pre-registration.**
- SKEMPI 2.0; hotspot = Ala-scan ΔΔG>1 (and strict >2, ProBID-Net's threshold); null |ΔΔG|<0.25.
- The matched-pair design (rSASA ±0.05 / pydssp SS / neighbour ±1, optimal 1:1) — the whole
  experiment. Complex-level bootstrap. → PREREG.md.
- Five architectures: ProteinMPNN (vanilla+soluble), ESM-IF1 142M, PiFold (one-shot, deterministic),
  MIF. Validation gate (positive control on every path). → §1 of FINDINGS, validate.py.

**3. On crystal backbones, the hotspot gap is a burial artifact.** *(the correction)*
- The confound, drawn: recovery rises monotonically with hotspot strength AND burial rises in
  lockstep (0.347→0.529 recovery, 0.218→0.080 rSASA). → §2.1.
- Matched: SECONDARY-B −0.042 [−0.222, +0.129]; recovery +0.005; AA-adjusted +0.11; Holm-null across
  8 variants; regression estimator +0.059 [−0.051, +0.167], MDE 0.156. → §2.3–2.6, regression.
- **Five architectures agree** (every PRIMARY CI contains zero); junction control on the single-chain
  models. → §2.5b. ProBID-Net's gap reproduced uncontrolled (+0.08 here) then dissolved under
  matching; four of five candidate causes for their sign eliminated. → §2.2, §4.2e.
- **Figure 1:** the confound (recovery vs burial by hotspot class) + the matched-pair forest plot
  across 5 models.

**4. The mechanism is real and localized to the conditioning set.** *(the pivot)*
- The interaction: hotspots gain +0.27 nats [+0.15, +0.41] (ΔΔrSASA-adjusted) more from the partner's
  presence than matched controls — the frustration signature, measured. Externally validated vs
  experimental ΔΔG_bind (ρ +0.28, adds beyond burial + log-odds). → §3.2, hardening.
- A sequence-free detector: KL(p(·|complex backbone) ‖ p(·|monomer backbone)), no residue identity,
  = the sequence-aware statistic (Δ +0.001), beats a contact-count baseline, adds to burial
  (ΔAUROC +0.048). **The nugget:** the model's own **confidence** is near-chance for hotspots (AUROC 0.527)
  and naively adding it to burial even HURTS (ΔAUROC −0.048 [−0.083,−0.014], P=0.002 — it is near-noise);
  only **partner-sensitivity** adds (burial+KL +0.064 [+0.037,+0.091]). *The quantity a designer would
  naively trust is useless; the free, sequence-free partner-sensitivity signal is what carries it.*
  → confidence_antipredicts.csv, kl_analysis.
- **The detector is actionable (candidate method contribution).** Framed as design-time triage — a
  fixed budget of k interface positions per complex to receive expensive binding-aware optimization —
  ranking by KL+burial captures significantly more experimental hotspots than the burial heuristic:
  capture@3 0.237 vs 0.139 (Δ +0.098 [+0.021,+0.175], P=0.99, n=106); capture@25% Δ +0.089
  [+0.009,+0.169]. The gain is a general additive property of KL, **not** specific to positions the
  model is uncertain about (niche AUROC Δ −0.002, null) — reported as such. Crystal proof-of-concept;
  the design-time claim is validated on the predicted/generative KL tables (where KL strengthens)
  before it enters as a method. → src/kl_triage.py, results/kl_triage.csv. [crystal done; non-native
  validation PENDING]
- Why crystal backbones hide it: they are carved by the side chains being predicted, so the model
  already has the partner information that makes the frustrated residue favourable and never pays for
  it. → §3.2.
- **Figure 2:** the bound-vs-unbound 2×2 interaction + the KL detector AUROC panel.

**5. The tax appears on *independently-reconstructed* backbones — and it is predictor-general.** *(the central positive)*
- **The headline is cross-predictor reproducibility, not any single deficit.** On backbones from TWO
  architecturally-independent folders — OpenFold3 (Exp A) and AF2-multimer (Exp D) — the burial-matched
  hotspot deficit appears (SECONDARY-B −0.191 [−0.37,−0.004] and −0.233 [−0.44,−0.035]; crystal ≈ 0,
  reproduces to 4e-16), and — the decisive readout — **the two predictors' per-complex deficits correlate
  ρ = +0.565 [+0.40,+0.71] (Pearson +0.62, n=127): the SAME complexes are hard under both.** A per-predictor
  memorization/architecture artifact would give *disjoint* deficits; instead two independent reconstructions
  agree, in magnitude and per complex — a **quasi-ensemble** result that is the robust anchor of the claim.
  → FINDINGS_expA.md, FINDINGS_expD.md §4.
- **Honest framing, up front (answers the adversarial review).** Each single predictor's deficit is
  MARGINAL — neither survives dropping its top-3 supporting complexes (the SAME 3 under both) — so the claim
  rests on cross-predictor agreement, not a lone −0.19/−0.23. The symmetric leverage jackknife was applied
  identically to OF3, AF2 AND C2: no pre-registration asymmetry. → FINDINGS_expD.md §5.
- **It is independent-reconstruction, not distance-from-native.** On partial-diffusion GENERATIVE backbones
  (Exp C2 — *noised crystals* at the same iRMSD) the deficit is ABSENT (binned gap flat/positive); the
  pre-registered slope "fired" but was a near-crystal leverage artifact (all three views — slope, leverage
  sensitivity, flat-positive bins — reported). So the deficit tracks the *type* of non-nativeness (independent
  reconstruction), not how far the backbone drifted. → FINDINGS_expC2.md.
- **The sequence-free KL detector generalises across FOUR backbone classes** — crystal +0.048, OpenFold3
  +0.062, generative/C2 +0.06–0.07, AF2-multimer +0.054 (every CI excludes zero) — and is STRONGER where the
  backbone is well-predicted (+0.092 at high pTM): it works best exactly where designers have good backbones.
  → FINDINGS_expD.md §3, kl_analysis.
- Binding-relevant readout: ProteinMPNN's rank-correlation with experimental ΔΔG_bind collapses −0.236
  (crystal) → ≈−0.05 off the native manifold — the binding-relevant face. → FINDINGS_expC2.md §6.
- **Figure 3 (the money figure): "what survives as the backbone leaves the native manifold."** LEFT: the
  AF2-vs-OF3 per-complex deficit scatter (ρ=+0.57) — two independent predictors agree on which complexes are
  hard. RIGHT: KL ΔAUROC holds across the four backbone classes while the sequence-COUPLED readouts (ΔΔG
  rank-corr; the log-prob deficit off the prediction manifold) decay. The robust, transferable signal is the
  sequence-free detector; the sequence-coupled quantities degrade with the backbone.

**6. It is the conditioning geometry, not the schedule or the sample budget.** *(ruling out competitors)*
- N_hot: the T=0.1 constellation cost is ~10^10 but statistically identical at burial-matched
  *control* constellations (median Δ 0.000, p 0.90) — generic to low-temperature sampling, not a
  hotspot tax. The "no oversampling recovers it" punch survives as a general statement; the barrier
  is the temperature exponent (a 3× ranking error → 10^5 at T=0.1). → §4.2b–4.3.
- Commitment ordering: on ProteinMPNN (fixed backbone) the oracle order is inert (DiD −0.002, K1
  null); on MultiFlow (coupled co-design, first such measurement) structure commits before sequence
  but only marginally (F3) and the unmasking-order knob is inert (F4). The schedule mechanism is ruled
  out on both an autoregressive and a coupled model. → §4.4, FINDINGS_expB.md.
- **Note:** this is where we do NOT overturn MultiFlow's purity unmasking — we report the ordering
  knob as inert. Honest, and it sharpens the conditioning-set claim by elimination.

**7. Related work and positioning.**
- ProBID-Net (the phenomenon; we correct the attribution and the benchmark). RedNet (frames
  ProteinMPNN's interface blindness as a decoding problem; we show the blindness is conditioning, not
  decoding, and our sequence-free KL reaches their zero-shot ΔΔG range with no new model). MultiFlow
  (purity unmasking; we measure its commitment ordering and find the knob inert). StaB-ddG (occupies
  the Tsuboyama+SKEMPI fixture with a folding-energy-difference parameterization; distinct question).
  Refolding-limitations (self-consistency oracles are biased; why we report recovery + a binding
  readout, not a fold-and-score number as primary).

**8. Limitations (stated up front, not buried).**
- Training leakage: every SKEMPI complex predates every checkpoint — but this makes the predicted-
  backbone result *conservative* (the predictor near-reconstructs them and the deficit appears anyway).
- Exp A/C are mechanistic brackets: OpenFold3/partial-diffusion of *known* complexes, ProteinMPNN
  scoring — a strictly better proxy for design-time conditioning than the crystal, but not a full
  RFdiffusion→design→wet-lab loop.
- Recovery/log-prob is the primary readout; the ΔΔG correlation is the binding-relevant anchor, not a
  wet-lab measurement.
- The pre-registered strict-control PRIMARY tier is underpowered; the verdict rests on the
  higher-powered tiers (declared in advance).

## Figure inventory (4 main + appendix)
1. The burial confound + 5-model matched forest plot. *(have)*
2. The bound-vs-unbound interaction 2×2 + KL detector AUROC. *(have)*
3. **Deficit vs backbone-distance-from-native (crystal→OpenFold3→partial-diffusion).** *(Exp C pending — the money figure)*
4. The two competing-mechanism nulls (N_hot control; commitment ordering on 2 model families). *(have)*
Appendix: junction sensitivity, TOST/Holm, external ΔΔG validation, per-model panel, decoding-order spread.

## What must land before submission
- ~~Exp C dose-response~~ — DONE, landed **mixed**: KL transfers; the log-prob gap is suggestive-not-
  decisive and the clean dose-response did NOT materialize. See Exp C2 below.
- ~~Binding readout~~ — DONE: ΔΔG rank-corr collapses −0.24 → −0.05 off-manifold. The binding gap is closed.
- **Exp C2 (hotspot-conditioned re-run)** — the remaining ICLR lever: resolve the log-prob gap in the
  physical-drift regime the unstable generator left unsampled. Pre-registers BOTH a clean dose-response
  and a TOST null, so either outcome is publishable (notes/SHERLOCK_HANDOFF_C2.md). ~15–20 GPU-h.
- **Validate the KL-triage method on predicted/generative backbones** (reuse the Exp A/C per-position
  KL tables; CPU, cheap). Crystal proof-of-concept is significant (capture@3 Δ +0.098); the design-time
  method claim needs the non-native backbones where KL strengthens. Turns the detector into a method.
- **Archive raw artifacts** before ~2026-10-09 (SCRATCH purge): git-LFS in the working repo now
  (purge-rescue), Zenodo DOI + clean repo at submission (archival + avoids the LFS bandwidth cap). See DATA.md.
- Optional/lower-priority: replicate the correction on ProBID-Net's own fixture (CPU-mostly, strongest
  rebuttal to "single fixture"); AF2-multimer ipTM design-loop readout (~15–40 GPU-h, C2-gated).

## Honest self-assessment of the ICLR case
FOR: a methodological correction with teeth (the benchmark hides the effect); a positive mechanistic
result located in the conditioning set; an actionable sequence-free signal (KL) now validated across
crystal, predicted AND generative backbones; a binding-relevant readout that degrades off-manifold as
predicted; three competing mechanisms measured and adjudicated; unusually disciplined pre-registration
and self-correction (incl. reporting Exp C's log-prob gap as suggestive-not-decisive rather than
upgrading it). AGAINST: single fixture (SKEMPI); recovery/log-prob primary readout; the clean
design-regime *dose-response* did not land — the log-prob gap is confounded with interface dissolution
and the generator was unstable. NET after Exp C: TMLR strong (~0.85, complete and honest); ICLR ~0.33,
gated on the C2 hotspot-conditioned re-run landing a clean physical-regime result (or on the
KL-transfer + benchmark-correction story carrying it without the dose-response).

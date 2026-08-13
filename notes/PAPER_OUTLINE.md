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
Even on that crystal backbone, asking *what does* flag a hotspot exposes the core asymmetry: **the one
quantity a designer would naively trust — the model's own confidence — is near-chance at hotspots (AUROC
0.53), and it ranks hotspots BELOW random for top-3 triage (0.064 vs 0.084) — carrying no information
beyond geometry; the signal that works needs no network at all: a trivial partner-contact-area feature
(ΔSASA) predicts hotspots as well as any learned signal.** [combiner-free; the z-sum "hurts" number retired
per the baseline audit — restate rigorously via conditional predictive impact (CPI).]
On **OpenFold3-predicted** backbones of the same complexes the burial-matched deficit **appears**
(−0.19 nats, paired Δ −0.15), as large where the prediction is accurate as where it is not. A
**sequence-free** signal — the partner-induced shift in the model's backbone-only distribution —
predicts where, strengthens on the predicted backbones, and **transfers to RFdiffusion generative
backbones** wherever a physical interface forms [Exp C]; on those same generative backbones the raw
log-prob deficit is present only at large backbone drift and the binding-energy ranking collapses off
the native manifold — a mixed result we report as such.
Crucially the deficit is **predictor-general, not a per-model artifact**: on backbones from a second,
architecturally-independent folder (AF2-multimer) the same burial-matched deficit emerges (−0.23 nats),
and the two predictors' per-complex deficits correlate (ρ = +0.57, n=127) — the *same complexes* are hard
under both. It is absent on noised-crystal generative backbones at equal distance, so it tracks
*independent reconstruction*, not distance-from-native. And the sequence-free KL signal is not merely
diagnostic: as a fixed-budget design-time triage ranker it captures significantly more experimental
hotspots than the burial heuristic on the predicted backbones designers actually use (capture@3
+0.08–0.09) — **but this KL-triage method claim is WITHDRAWN (weak-baseline artifact; see reframe note below).** The tax is
real but it is a property of the **conditioning set**, not of hotspot chemistry: the field benchmarks
on crystal backbones, which hide it, while designers work on non-native backbones, where it bites.
Two proposed alternative mechanisms — a low-temperature constellation cost and a commitment-ordering
schedule effect — are each measured and neither survives (the constellation cost is generic to any
multi-position set; commitment reordering is inert on both an autoregressive and a coupled
co-design model).

> **⚠️ REFRAME PENDING (2026-08-13).** The sequence-free KL detector was found to largely recapitulate ΔSASA
> (partner-contact area, trivial geometry): on crystal KL adds only +0.007 (ns) over a full cheap-geometry
> baseline (burial+nbr+ΔSASA), and the KL-triage method claims (capture@k *and* Lever-2 kcal/mol) are
> baseline artifacts — computed vs integer `nbr`; null vs rSASA/full-geometry (free geometry even beats the
> `nbr` baseline by more than KL does). **KL-as-method is WITHDRAWN.** The NUGGET survives and *sharpens*:
> adding the model's own confidence significantly HURTS even a free-geometry ranker (−0.020, P=0.009;
> results/nugget_partner_sensitivity.csv). This abstract / §4 / contributions (iii) will be rewritten
> **nugget-forward, KL demoted to a learned partner-sensitivity probe, matched-pair design framed as a
> diagnostic protocol** (not a benchmark; ProtDBench exists) — AFTER the decisive predicted-backbone ΔSASA
> control (R1, notes/SHERLOCK_HANDOFF_dsasa.md; CPU-only), which decides ICLR (KL earns its keep off the
> native manifold, unifying the story on one axis) vs TMLR (honest correction). idea-critic verdict: REFINE.
> **R1 RESOLVED (2026-08-13, c59688e).** KL adds ≈0 over full geometry on OF3 (+0.008) and AF2 (+0.005) too
> — scalar-KL-as-method is demoted on ALL 4 backbone classes (crystal/OF3/AF2/Bennett). **But Big Idea 1
> (P3-fire) supplies a POSITIVE** — partner-conditioning adds interface-binding information (the *full*
> distribution, not the scalar KL). So the venue no longer hinges on this; the reframe can now proceed
> **nugget-forward + Big-Idea-1-forward**, with the scalar KL demoted to "a learned frustratometer that
> equals the classical geometry."

## Section map

**1. Introduction.**
- The staged pipeline; the frustration hypothesis (hotspots are buried polars / strained rotamers
  that buy affinity, not stability); the intuition that inverse folding, lacking a binding-energy
  term, misses them. → BRIEF §2.1.
- ProBID-Net's 0.334/0.472 gap and its dynamics attribution; the confound nobody controlled: hotspots
  are buried, and burial is where inverse folding is *most* confident.
- Contributions: (i) the gap is a burial artifact on crystal backbones (five architectures, pre-registered) —
  and the burial-matched matched-pair design is offered as a reusable **protocol/benchmark** for
  evaluating inverse folding at interface hotspots without the confound; (ii) the tax is real and lives
  in the conditioning set — a burial-matched deficit appears on the backbones of TWO independent structure
  predictors (cross-predictor per-complex ρ=+0.57), is absent on noised-crystal generative backbones
  (so it is *independent-reconstruction*, not distance-from-native), and a sequence-free **partner-
  sensitivity** signal detects it across four backbone classes — while the model's own **confidence** does
  NOT (near-chance; naively hurts); (iii) that signal is a **validated design-time method** — as a
  fixed-budget triage ranker KL+burial captures more experimental hotspots than the burial heuristic,
  holding on crystal AND on the OpenFold3 + AF2-multimer predicted backbones designers use (capture@3
  +0.08–0.10, CIs exclude zero); (iv) two competing mechanisms (constellation cost, commitment ordering)
  measured and refuted.

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
- **ProBID-Net's own released voxel-CNN, run on our fixture (#4-full) — a reproduce-and-dissolve correction to
  ProBID-Net on its own model (audited + independently re-verified 2026-08-13).** The port is faithful (overall interface
  recovery 0.472 = their reported non-hotspot number). Their hotspot deficit **does reproduce** on our
  fixture under the like-for-like per-residue estimator, concentrated in comprehensively-Ala-scanned
  complexes (≥5 measured hotspots: −0.113 [−0.208,−0.022], p=0.007); the reproduction is robust to leave-one-out (LOO means stay in [-0.13,-0.08]) and deepens with scan
  depth (Spearman -0.20, p=0.06). The whole-fixture average is diluted toward zero by sparsely-scanned
  complexes where a "hotspot" is 1-2 noisy residues (hotspot-weighted +0.014 [−0.052,+0.087]). It is largely a **residue-composition** effect — ProBID recall
  spans 0.17 (R) to 0.98 (P) and hotspots are enriched in its worst types (WYFRMH 47% vs 22%, GP 3% vs
  12%) — plus burial. It **dissolves under confound-matching**: matching residue type flips it positive (AA-matched +0.120
  [−0.060,+0.300], n=25; burial-matched −0.038 [−0.139,+0.071]; hydrophobicity-matched −0.051), every CI
  spanning zero. **Correction:** an earlier draft called this an "opposite-sign, fixture-specific" sixth
  architecture null (+0.098); that was a complex-averaging + AA-composition artifact and is **withdrawn**.
  The honest reading: their deficit reproduces and is a composition/burial confound — consistent with the
  thesis via a *different* confound than burial alone, not a clean sixth null (the comprehensively-scanned stratum is post-hoc but robust and principled). → probid_gap_estimators.csv, §4.2e.
- **Residue-type composition is stereotyped but NOT the deficit driver (composition_confound.csv).** Per-
  residue-type recall is remarkably consistent across ESM-IF1 / MIF / ProteinMPNN-soluble / PiFold (Spearman
  0.87–0.90) — all inverse-folding models recover the same types (M/Q/K/H/R/W) worst. But on SKEMPI the
  uncontrolled hotspot deficit is POSITIVE (burial-driven, +0.02…+0.08) and composition-predicted deficits
  are small/negative — so **BURIAL, not composition, is the dominant confound here**; ProBID-Net's
  composition-driven deficit reflects its voxel-CNN's unusually extreme type-dependence (0.17–0.98), not a
  general law. Reinforces burial as *the* confound. → §2.2, composition_confound.csv.
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
- **The detector is actionable — a validated design-time method (#12).** Framed as design-time triage — a
  fixed budget of k interface positions per complex to receive expensive binding-aware optimization —
  ranking by KL+burial captures significantly more experimental hotspots than the burial heuristic:
  capture@3 0.237 vs 0.139 (Δ +0.098 [+0.021,+0.175], P=0.99, n=106); capture@25% Δ +0.089
  [+0.009,+0.169]. The gain is a general additive property of KL, **not** specific to positions the
  model is uncertain about (niche AUROC Δ −0.002, null) — reported as such. **VALIDATED as a design-time
  method (#12 discharged):** the capture@k advantage holds on the non-native backbones designers use —
  OpenFold3 +0.083 [+0.013,+0.155] and the independent AF2-multimer +0.087 [+0.018,+0.158] (both budgets,
  CIs exclude zero, at crystal magnitude; generative arm positive but underpowered, n=9). It enters the
  paper as a validated design-time method, not crystal-only. **In binding units (Lever 2):**
  among interface residues with Ala-scan data, the top-3 KL+burial positions capture more of the complex's
  total *experimental* binding free energy than burial alone — 51.3% vs 49.5%, +0.32 kcal/mol [+0.01,+0.65]
  (fractional +1.8pp, P=0.96; modest, small-budget-only, null at 25%) — **[WITHDRAWN 2026-08-13 — baseline artifact: the +0.32 kcal is KL+burial vs the *nbr* baseline; against
  rSASA / full-geometry KL adds nothing (−0.007 / −0.003, ns). KL-as-method demoted.]** → src/kl_triage_energy.py, results/kl_triage_energy.csv. → src/kl_triage.py, results/kl_triage_exp{A,D}.csv,
  FINDINGS_kl_triage.md §4.
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
  agree, in magnitude and per complex. We call this **cross-predictor reproducibility** — independent
  replication across two architectures, the robust anchor of the claim. (Precise term deliberately: it is
  *not* an ensemble in the modelling sense — we do not combine predictors into one, we show two
  independently-trained ones agree; "ensemble" would misdescribe it.) → FINDINGS_expA.md, FINDINGS_expD.md §4.
- **Honest framing, up front (answers the adversarial review).** Each single predictor's deficit is
  MARGINAL — neither survives dropping its top-3 supporting complexes (the SAME 3 under both) — so the claim
  rests on cross-predictor agreement, not a lone −0.19/−0.23. The symmetric leverage jackknife was applied
  identically to OF3, AF2 AND C2: no pre-registration asymmetry. → FINDINGS_expD.md §5.
- **Exploratory mechanistic hint — the deficit is a burial phenomenon (post-hoc, labelled).** The
  cross-predictor per-complex deficit is itself predictable from structure: more-buried interfaces carry
  LARGER deficits (mean neighbour count ρ = −0.21 [−0.38,−0.04]; mean rSASA ρ = +0.34 [+0.19,+0.49];
  n=127), even though the deficit is already burial-*matched within* each complex. The predicted-backbone
  deficit thus concentrates in deeply-buried interfaces — exactly where inverse folding is most confident
  and a predicted backbone's small errors bite hardest. KL and hotspot-count do NOT predict it; flagged
  exploratory (post-hoc, not pre-registered). → deficit_predictors.csv.
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
- **Prior art we MUST cite and differentiate (added 2026-08-13; ✎ verify each URL before submission —
  BAIF/DeSAE/CPI abstracts fetched, the rest search-only/unverified):** *Frustratometer* (Ferreiro/Parra,
  NAR 2012 & 2016) — partner-induced local-frustration change at interfaces is a 2012 statistical-mechanics
  result; our KL is essentially a **learned frustratometer**, and KL≈ΔSASA means the neural version does not
  beat the classical geometry. Cite as the physics ancestor, not an ignored competitor. *BAIF* (arXiv
  2410.09543, 2024) — inverse-folding log-likelihood over bound-vs-unbound states (the same two conditioning
  sets) for ΔΔG; **closest prior art.** We differ: a per-*substitution* experimental-binding test of the
  full conditional distribution, stratified stability-vs-binding (Big Idea 1), not a mutation-level ΔΔG
  cycle. *HotPoint/DBAC* (2010–11) — burial-based hotspot prediction beat ML a decade ago, so geometry is a
  **cautionary contrast** against the current IF-for-hotspots wave, NEVER "geometry predicts hotspots."
- **The BindCraft hook (field relevance):** the leading one-shot binder pipeline hard-codes a **4 Å
  interface freeze that forbids inverse folding at the interface** — the field's *implicit admission* of our
  thesis. We give that hack its measurement (confidence ranks interface hotspots below random; ΔSASA beats
  the 4 Å contact set at matched budget) and a principled replacement.
- **Conditioning-aware IF (concurrent, must position against):** AlphaFold-DB debiasing (DeSAE, arXiv
  2506.08365), target-conditioned inverse folding, UMA-Inverse — methods that *presuppose* the conditioning-
  set problem; we *measure* it and show the benchmark hides it. Surf2Spot (supervised hotspot predictor)
  owns the method lane → we stay a measurement/correction. ProtDBench occupies the benchmark slot → we offer
  the matched-pair design as a drop-in **protocol**.

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
3. **Deficit vs backbone-distance-from-native (crystal→OpenFold3→partial-diffusion).** *(landed: Exp C/C2/D — the money figure)*
4. The two competing-mechanism nulls (N_hot control; commitment ordering on 2 model families). *(have)*
Appendix: junction sensitivity, TOST/Holm, external ΔΔG validation, per-model panel, decoding-order spread.

## What must land before submission

**Status update (2026-08-12).** Several "pending" items below have since landed — the list beneath is the
original plan; current reality: Exp C2 ✅ (KL generalises; log-prob deficit is a prediction-specific null;
the "pinning fix" was refuted — all reported honestly). KL-triage validation ✅ (#12: capture@k holds on
OpenFold3 + AF2-multimer). Exp D / AF2-multimer 2nd predictor ✅ (D-PERSIST; cross-predictor ρ=+0.57).
#4-full ✅ but DOWNGRADED after 2026-08-12 audit (ProBID-Net's deficit reproduces on our fixture and is a
residue-composition + burial confound; the earlier "opposite-sign / fixture-specific" claim was withdrawn —
see probid_gap_estimators.csv).
Archive ✅ LFS purge-rescue done. **Still open:** Zenodo DOI at submission; optional AF2 ipTM design-loop
readout; optional 2nd hotspot-label source (ASEdb/BID); and the lightweight **Bennett de-novo KL-detector
check** (design-regime detector validation, in progress this session).
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
and the generator was unstable. NET (post-C2 / post-Exp D / post-#12): TMLR strong (~0.85 — complete,
honest, self-correcting). ICLR ~0.45. The C2 lever resolved as a *split*: the clean log-prob dose-response
is a prediction-specific null (reported as such, not upgraded), but the load-bearing spine strengthened —
(a) the burial correction spans **five architectures** (ProBID-Net's own model, run separately, shows its
deficit is reproduces its published deficit and shows it is a residue-composition + burial confound — see §3, corrected after audit);
(b) the sequence-free **KL detector generalises across all four backbone classes**; (c) KL-triage is
**validated as a design-time method** on the non-native backbones designers use (#12, capture@3 +0.08–0.09
on OpenFold3 and AF2-multimer); (d) **cross-predictor reproducibility** (Exp D: OpenFold3 and AF2-multimer
agree per complex, ρ=+0.57) defuses the memorization/architecture confound that previously gated Result 3.
The case is no longer hostage to a single unlanded experiment. Residual ceiling: single fixture (SKEMPI),
recovery/log-prob primary readout, and no full generate→design→wet-lab loop.

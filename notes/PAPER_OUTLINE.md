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
(−0.19 nats, paired Δ −0.15), as large where the prediction is accurate as where it is not; on
partial-diffusion **generative** backbones it grows monotonically with distance from native
[Exp C]; and a **sequence-free** signal — the partner-induced shift in the model's backbone-only
distribution — predicts where, and strengthens on exactly these non-native backbones. The tax is
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
- Contributions: (i) the gap is a burial artifact on crystal backbones (5 models, pre-registered);
  (ii) the tax is nonetheless real and lives in the conditioning set — it appears on predicted and
  generative backbones and a sequence-free signal predicts it; (iii) two competing mechanisms
  (constellation cost, commitment ordering) measured and refuted.

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
  (ΔAUROC +0.048). The model's own *confidence* is useless (AUROC 0.54); its *partner-sensitivity*
  carries the signal. → §4.2d-bis, kl_analysis.
- Why crystal backbones hide it: they are carved by the side chains being predicted, so the model
  already has the partner information that makes the frustrated residue favourable and never pays for
  it. → §3.2.
- **Figure 2:** the bound-vs-unbound 2×2 interaction + the KL detector AUROC panel.

**5. The tax appears on non-native backbones.** *(the central positive result)*
- Exp A: on OpenFold3-predicted backbones (templates off; crystal control reproduces to 4e-16) the
  burial-matched deficit appears, SECONDARY-B −0.19 [−0.37, −0.004], paired Δ −0.15 [−0.28, −0.03],
  as large at high pTM as low (backbone-error artifact ruled out); KL strengthens (+0.062 vs +0.048)
  and is more robust to backbone noise than burial. → FINDINGS_expA.md.
- Exp C: on partial-diffusion generative backbones, the deficit grows monotonically with interface
  RMSD-from-native, starting at ~0 on the crystal — the conditioning-set mechanism drawn as a
  dose-response. → FINDINGS_expC.md [PENDING].
- Binding-relevant readout: the model's ability to rank experimental ΔΔG_bind degrades as the
  backbone becomes non-native. → Exp C secondary [PENDING].
- **Figure 3 (the money figure):** deficit vs backbone-distance-from-native — crystal (~0) →
  OpenFold3 (−0.15) → partial-diffusion ladder (dose-response) — with the KL ΔAUROC overlaid.

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
- **Exp C** (the dose-response) — turns the central result from one point into a curve. Highest value.
- **Binding readout** (ΔΔG-vs-backbone correlation) — the binding-relevant anchor. Cheap.
- Optional: AF2-multimer readout; a second predictor for Exp A. Hardening, not novelty.

## Honest self-assessment of the ICLR case
FOR: a methodological correction with teeth (the benchmark hides the effect); a positive mechanistic
result located in the conditioning set; an actionable sequence-free signal that works on the noisy
backbones designers use; three competing mechanisms measured and adjudicated; unusually disciplined
pre-registration and self-correction. AGAINST: single fixture (SKEMPI), recovery-based primary readout,
mechanistic-bracket backbones rather than a full design loop. Exp C + the binding readout move this
from "strong TMLR" to "credible ICLR."

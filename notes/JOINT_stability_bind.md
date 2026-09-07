# Two legs of one cycle — feedback for `idea1_stability_manifold`, a joint-paper vision, and a combined package

*Written 2026-09-07 from the factorization-tax (ΔΔG_bind) side, after a read-only look at the stability project.
Point the stability session at this file if useful. Nothing in the stability repo was modified.*

## The relationship in one line
`ΔΔG_total(mutation) = ΔΔG_fold + ΔΔG_bind`. The **stability project isolates ΔΔG_fold** (a frozen-ESMFold
internal stability axis: first-order probe on activation deltas, geometry partialled out, causal *held-geometry*
patch). **This project isolates ΔΔG_bind** (the partner-ablation **mixed second derivative** of an
inverse-folding likelihood = the classifier-free-guidance direction). Both use the **same ESM-IF1 checkpoint**,
and both are instances of one operator: **freeze the confound, differentiate the readout along the target
direction.**

## Feedback / recommended directions for the stability paper
1. **Make the steering actually generative.** Your activation-patch is (honestly) a *readout handle* — linear
   regime, zero clean fold-flips. Ours steers a *generative* model's **logits** (`+α·L`) and is actionable
   (recovery preserved, ipTM ↑). If you want generative control, the move is to **steer an inverse-folding
   model's logits toward your stability axis** (the `+α·û` → `+α·L` analogue on a *decoder*, not a patch on
   ESMFold internals). That reframes "we can read ΔΔG_fold" into "we can *design toward* ΔΔG_fold."
2. **Take the second derivative.** Your probe is first-order (readout ↔ one mutation). Our binding result
   *reaches the second mixed derivative* (partner-ablated pairwise couplings ↔ experimental epistasis). Ask:
   does the stability axis capture **fold epistasis** in double mutants (MEGAscale has them)? Our couplings
   pipeline is a drop-in template.
3. **Keep the honest linear-regime caveat** — it's a strength; the field over-claims steering.
4. **Lean into the calibration.** Your α→experimental-ΔΔG (Pearson 0.73) is a genuine "the internal axis *is*
   ΔΔG_fold in physical units" result — foreground it.
5. **Cross-model localization ladder** (OF3 < OmegaFold < ESM-2 < ESMFold) is a nice template we may borrow for
   *where the binding cross-term lives* — reciprocal citation-worthy.

## What the stability project gives THIS (binding) project
- **Independent, fold-side support** for "the raw IF likelihood is a lossy scalar": your S669 result (raw
  ESM-IF LL-ΔΔG |ρ|≈0.42, beaten by a geometry-partialled internal direction ≈0.60) is the fold analogue of our
  "confidence is blind to binding; take the derivative." Different confound (geometry vs partner), same moral.
- **A calibration recipe** to turn our "estimates −ΔΔG_bind *up to temperature*" into a **measured slope**
  (regress `+α·L` magnitude on SKEMPI ΔΔG, as you did for fold). [next-paper]
- **An anti-symmetry control:** your linear axis satisfies ΔΔG(X→Y)=−ΔΔG(Y→X) exactly; our `L` should too —
  a cheap drop-in check.

## A joint paper — "Frozen networks carry the thermodynamic cycle"
**Thesis:** a frozen structure/inverse-folding network internally represents *both* legs of the mutational
thermodynamic cycle — **ΔΔG_fold** (stability, readable via a held-geometry probe) and **ΔΔG_bind** (binding,
readable via a partner-ablation mixed derivative / CFG direction) — and *neither* is visible to the scalar
readouts the field uses; both require the right *derivative/partial* of the likelihood. One operator (freeze the
confound, differentiate along the direction), two quantities, one cycle: `ΔΔG_total = ΔΔG_fold + ΔΔG_bind`,
testable directly where both are measured (e.g. SKEMPI mutations whose fold effect is also known).
- **Killer experiment:** predict `ΔΔG_total` by *summing the two independently-read legs* and show it beats
  either leg alone and beats the raw scalar — the cycle closes numerically inside a frozen model.
- **Why it's more than the sum:** it's a general **representation-learning** claim (what un-trained
  thermodynamics a conditional generative/predictor model encodes, and how to *read and steer* it), not two
  protein results stapled together.
- **Novelty guard:** grep-confirmed the CFG/mixed-second-derivative/SKEMPI framing is *absent* from the
  stability repo; the fold-probe/held-geometry-patch is absent here. Same author ⇒ cross-citation is
  self-citation — mind double-blind for the ICLR submission (cite only if public), keep fold/bind crisp.

## A combined package — `thermocycle` (or `frozen-thermo`), umbrella over two modules
Two shippable packages already planned (this repo's `leverage`; the stability repo's `frozen-ddg`). A thin
umbrella could unify them under the shared operator:
```
thermocycle/
  fold/   ->  frozen-ddg      # ΔΔG_fold: stability axis probe + held-geometry patch
  bind/   ->  leverage        # ΔΔG_bind: partner-ablation mixed derivative + +α·L steering
  cycle.py                    # ΔΔG_total = fold + bind; the shared Model protocol; the freeze-confound-
                              # differentiate operator; joint calibration to physical units
```
- **Shared `Model` protocol** (score `p(seq|structure)` / read an internal axis) so the same adapters
  (ProteinMPNN, ESM-IF1, ESMFold) serve both modules — they already share the ESM-IF1 checkpoint.
- Ship the two as independent installs *and* the umbrella as an optional meta-package, so each paper's package
  stands alone while `thermocycle` tells the unified story. Post-acceptance work; both stand on their own for
  their respective deadlines.
- Related: [[sibling-stability-manifold]], [[PACKAGE_PLAN]].

# FINDINGS — the fail-fast Bennett-hardening battery (venue = ICLR / Spine B)

**Pre-registered:** `results/PREREG_bennett_hardening.md` (committed 8c0306b, BEFORE any number).
**Trigger:** two blind adversarial reviews (research-strategist + idea-critic) converged that as-framed the
paper was a strong TMLR but a likely-reject ICLR, and that the decision hinged on whether the Bennett
positive survives an all-atom occlusion baseline and a non-parent scorer. We pre-registered a 5-test battery
with frozen decision rules and ran it. **5/5 landed on-thesis; every major kill-shot is defused.**

## Scorecard

| test | kill-shot | result | verdict |
|---|---|---|---|
| **T1** all-atom occlusion (decider) | #2 "weak baseline; repack and +0.025 evaporates" | 95.1% of subs have ZERO post-repack clash; on a STRONGER all-atom baseline (0.619>0.587) **P adds ΔAUROC +0.0182 [+0.0145,+0.0220]** P=1.000 | **SURVIVES → ICLR** |
| **T2** non-parent ESM-IF1 | #3 circularity (parents are ProteinMPNN outputs) | ESM-IF1 reproduces all three: AUROC(P) 0.625 [0.608,0.641]; P−Q +0.079 [+0.069,+0.090]; ΔAUROC over occlusion +0.0160 [+0.0129,+0.0194] | **DEFUSED** |
| **T4/R2** cross-model confidence | #4 "confidence≠competence is the field's premise" | interface-hotspot confidence-AUROC 0.50–0.54 across ALL 5 IF models (MPNN 0.538, ESM-IF1 0.517, PiFold 0.499, MIF 0.509, ProBID 0.536) | **property of inverse folding** |
| **T5** burial-residualize ρ | #5 "ρ=0.57 is recursive burial" | partial ρ(d_of3,d_af2 \| burial) **+0.529 [0.354,0.678]**; +0.533 after top-3 drop | **DEFUSED** |
| **T3** confidence fork (theory) | — (secondary; constraint-vs-leverage) | de-novo confidence-AUROC 0.596 (logp) / 0.627 (negentropy) > SKEMPI 0.538 | **theory FIRES** |

## What each buys

- **T1 (`bennett_occlusion_allatom.csv`, FINDINGS_occlusion_allatom.md).** The most lethal objection
  backfires: doing the critic's own operation (repack the rotamers, min over 40 conformers) makes steric
  occlusion nearly *vanish* (95.1% zero clash; all-atom clash predicts binding at 0.519 = chance). On a
  geometry baseline that is now genuinely all-atom and stronger than the original, P still adds. The model
  encodes per-substitution binding **energetics beyond all-atom steric occlusion**. This +0.018-over-a-strong
  -baseline **supersedes** the old +0.025-over-a-weak-baseline as the load-bearing number. Pre-registered
  RMSD validity gate passed (native reconstruction median 0.278 Å). *Process honesty:* the first run's gate
  was mis-coded as a clash-Spearman (failed for lack of dynamic range — 95% of natives don't clash);
  corrected to the pre-registered RMSD gate; stayed blind to ΔAUROC until it passed.
- **T2 (`bennett_nonparent.csv`).** A model that did NOT generate the parents (ESM-IF1) reproduces the whole
  Big-Idea-1 signal — so it is not ProteinMPNN scoring around its own mode. Cross-model + non-circular.
- **T4/R2 (`xmodel_confidence.csv`).** "Confidence is not competence" holds across five architectures →
  a claim about inverse folding, not one network. (A broken ProBID join was caught by a positive control and
  fixed before the zero was trusted.)
- **T5 (`deficit_burial_residualize.csv`).** Two predictors agree on which complexes are hard *beyond*
  burial — the conditioning-set signal is not a recursive burial confound.
- **T3 (`bennett_conf_fork.csv`).** Confidence predicts binding-hotspots above chance in the binding-dominated
  de-novo regime (0.60–0.63) but is at chance on natural SKEMPI interfaces (0.538) — the constraint-vs-leverage
  theory's key prediction. Converts the paper from measurements to measurements + a confirmed model.
  *Caveats:* 2-point comparison; hotspot-label constructions differ across fixtures; a full
  obligate→transient→de-novo gradient within SKEMPI would make it a clean 3-point law.

## Consequence
Venue fork resolves to **ICLR / Spine B**: the positive — the model encodes per-substitution binding
energetics beyond all-atom occlusion, cross-model and non-circular, plus a confirmed constraint-vs-leverage
gradient — is the ICLR-native contribution. Remaining ceiling: modest effect sizes (~0.016–0.018, tight CIs),
one main fixture (SKEMPI) + Bennett, and the §-body rewrite. Commits: 8c0306b (PREREG), f751784 (T4/R2),
0b06582 (T5), e42950e (T1), b2cfca6 (T2), + T3.

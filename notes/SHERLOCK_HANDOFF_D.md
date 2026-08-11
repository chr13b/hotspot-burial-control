# Sherlock handoff — Experiment D: the 2nd-predictor discriminator (does the deficit survive a different architecture?)

Paste the block below into a fresh Claude Code session on Sherlock, in a clone of
`github.com/chr13b/hotspot-burial-control`. This is the experiment that **decides the venue**.

**Why.** The project's central positive (Exp A: a burial-matched hotspot log-prob deficit −0.19 appears on
**OpenFold3-predicted** backbones of the 141 pair complexes) has one unresolved objection an adversarial
reviewer will raise, and a 2026 preprint ("Generalise or Memorise?") arms it: the deficit might track
**OpenFold3's specific error/memorization structure**, not non-nativeness — the complexes are pre-cutoff
(memorized), and "deficit as large at high pTM" rules out backbone *error* but not memorization (high pTM =
better memorized). It also **vanishes** on partial-diffusion backbones at the same iRMSD (Exp C2). The clean
discriminator: recompute the deficit on a **second, independent, different-architecture predictor
(AF2-multimer)**. Fits the ICLR 2027 window (paper deadline **Sept 25 2026**, verified iclr.cc/Conferences/2027/Dates);
~1 GPU-day, a config re-run of the Exp A pipeline. The de-novo/stable-generator arm (FIX 1) is the schedule
risk and is NOT this experiment.

**Honest scope, stated up front:** all SKEMPI complexes predate BOTH predictors' training, so this does
**not** fully separate memorization from non-nativeness (only post-cutoff or genuinely de-novo backbones
would). What it *does* decide: whether the deficit is **OpenFold3-architecture-specific** or a **general
property of independently-reconstructed backbones**. Replication on a second independent architecture
substantially defuses the "OpenFold3 artifact" objection even though it is not a complete memorization
exclusion — say exactly this in the write-up.

---

```
Read BRIEF.md, CLAUDE.md, results/FINDINGS.md, results/FINDINGS_expA.md, results/FINDINGS_expC2.md in full
before writing code. Context: on native crystal backbones there is NO burial-matched hotspot recovery
deficit (5-6 models; ProBID-Net's gap is a burial artifact). On OpenFold3-PREDICTED backbones a
burial-matched deficit APPEARS (Exp A, SECONDARY_B -0.191 [-0.373,-0.004]; paired vs crystal -0.154). It
does NOT appear on RFdiffusion partial-diffusion (noised-crystal) backbones at comparable iRMSD (Exp C2).
The open objection: the Exp A deficit may be OpenFold3-architecture/memorization-specific, not
non-nativeness. Experiment D discriminates with a SECOND predictor. Pre-register everything below in
PREREG_expD.md and commit it BEFORE predicting any structure.

=== TASK 1 (GPU, ~1 day) — AF2-multimer 2nd-predictor deficit + KL ===
1. Predict each of the 141 pair complexes (results/pair_complexes.txt) with AF2-multimer (ColabFold),
   TEMPLATES OFF (no leakage, as Exp A). Keep predicted N/CA/C/O; align to crystal by held target for
   bookkeeping only. Record pLDDT / pTM / iptm / interface Ca-RMSD per complex.
2. Recompute, on the AF2-multimer backbones, the IDENTICAL Exp A pipeline: (a) the burial-matched deficit
   reusing the SAME committed pydssp matched pairs (src/expA_gap_reuse_pairs.py, keyed by chain,resnum);
   (b) the KL detector (src/kl_detector.py) and its dAUROC-over-burial (src/expA_kl_delta.py), canonical
   label==hot_strict set. Complex-level bootstrap, seed 20260803. Same decoding: v_48_020, 8 orders,
   teacher-forced native log-prob.

PRE-REGISTERED READINGS (fix before running):
  - D-PERSIST: the burial-matched deficit CI EXCLUDES zero on AF2-multimer backbones (point estimate
    comparable to OpenFold3's -0.19) -> the deficit is a GENERAL property of independently-reconstructed
    backbones, NOT OpenFold3-specific -> the conditioning-set headline holds; Result 3 is ICLR-ready.
  - D-VANISH: the deficit CI CONTAINS zero on AF2-multimer while present on OpenFold3 -> the deficit is
    OpenFold3-architecture-specific -> report as such; BOUND/withdraw the log-prob-deficit generality claim.
    The paper's ICLR core becomes Results 1+2 (burial correction + KL detector), which are unaffected.
  - D-KL: KL dAUROC-over-burial CI excludes zero on AF2-multimer (expected if KL is general, as on crystal
    +0.079 / OpenFold3 +0.062 / generative +0.06-0.07). Report.
  - POSITIVE CONTROL: crystal reproduces the committed within-complex null; templates-off confirmed; report
    the deficit stratified by pTM/pLDDT AND the PER-COMPLEX correlation of the AF2 deficit with the
    OpenFold3 deficit (same complexes carry it under both predictors -> strong general signal; disjoint ->
    predictor-specific noise). This correlation is the most informative single readout.

=== TASK 2 (CPU, quick) — the SYMMETRIC leverage check (defuse the pre-reg-asymmetry charge) ===
An adversarial review (idea-critic) noted we accept Exp A's fire but call Exp C2's pre-registered slope-fire
a leverage artifact, and applied the leverage diagnostic only to C2. Apply the SAME robustness scrutiny to
Exp A's -0.19:
  - Using the Exp A per-pair gaps on $SCRATCH (d = logp(hot)-logp(ctl) per matched pair, predicted backbone),
    jackknife over complexes: recompute the paired mean deficit dropping each complex; report the influence
    distribution and whether dropping the top-3/top-5 most-influential complexes keeps the CI excluding zero.
  - Do the identical jackknife for Exp C2's within-binder gap for visible symmetry. Write results/expD_leverage.csv.
  - READING: if Exp A's deficit survives leave-k-complexes-out (CI still excludes zero) it is NOT a leverage
    artifact -> the asymmetry charge is answered; if it does NOT survive, say so and demote it.

Standing rules (CLAUDE.md): pre-register readings before any number; positive controls through every path;
complex-level bootstrap; >=8 decoding orders; write raw outputs to results/ as CSV with exact commands;
write results/FINDINGS_expD.md with a one-line verdict from the pre-registered readings; do not move a
reading after seeing a number.
```

---

## What each outcome means for the paper (decided in advance)
- **D-PERSIST + D-KL fire** → the memorization objection is substantially defused; the conditioning-set
  headline (deficit on independently-reconstructed backbones + KL detector) is ICLR-ready. Push Results
  1+2+3 to ICLR (deadline Sept 25); go/no-go on the optional de-novo arm at ~3 weeks.
- **D-VANISH** → the log-prob deficit is OpenFold3-specific; withdraw its generality claim honestly. The
  ICLR core is Results 1+2 (burial correction + cross-backbone KL detector), which the strategist judges
  unscooped and ICLR-viable on their own; Result 3 becomes a bounded observation or a TMLR-fallback detail.
- Either way the paper is honest and shippable; D removes the single biggest reviewer crack before submission.

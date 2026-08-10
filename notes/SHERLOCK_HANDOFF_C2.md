# Sherlock handoff — Experiment C2: the physical-drift dose-response (the ICLR lever)

Paste the block below into a fresh Claude Code session on Sherlock, in a clone of
`github.com/chr13b/hotspot-burial-control`. This is the follow-up to Experiment C
(`results/FINDINGS_expC.md`), declared there as the next step.

**Why C2.** Experiment C could not power the burial-matched log-prob gap in the regime that decides the
paper — backbones that are genuinely non-native **but still form a physical interface** (interface
Cα-RMSD ≈ 2–8 Å). Hotspot-*free* partial diffusion of a binder against a held target **diverges on
62–75 % of designs** (binder coordinates blow up to 10³–10⁷ Å), so C only ever sampled *near-native-and-
physical* (iRMSD 1–3 Å, gap ≈ 0) or *far-and-marginal* (iRMSD ≈ 18 Å, gap −0.88 but interface half-gone,
fails the pre-registered iRMSD<3 Å confound). The deciding middle band was never populated. **C2 fixes
the instability by pinning the binder to the interface**, so the backbone can drift while the interface
stays real — turning the underpowered, confounded C-PRIMARY into a clean test.

**GPU estimate: ~15–20 GPU-h lean** (C's ladder was ~1 min/backbone on V100, faster than first estimated).
One `--gres=gpu:1`. AF2 readout optional and separate (~15–25 GPU-h).

---

```
Read BRIEF.md, CLAUDE.md, results/FINDINGS.md, results/FINDINGS_expA.md, results/FINDINGS_expC.md, and
results/PREREG_expC.md in full before writing any code. Context: on native crystal backbones there is NO
burial-matched hotspot recovery penalty (5 architectures; ProBID-Net's gap is a burial artifact). On
OpenFold3-PREDICTED backbones a burial-matched deficit APPEARS (Exp A). On RFdiffusion generative
backbones (Exp C) the sequence-free KL detector TRANSFERS (C-KL fired) but the burial-matched log-prob
gap was only SUGGESTIVE — significant just at non-physical drift (iRMSD~18A), non-monotone, not
surviving the iRMSD<3A confound — because hotspot-free partial diffusion diverges on 62-75% of designs
and never populated the physical-drift regime.

Experiment C2 re-runs the partial-diffusion ladder with the binder PINNED to the interface so it stays
physical while the backbone drifts, sampling the deciding regime (iRMSD 2-8A, interface FORMED).
Pre-register everything below BEFORE generating any backbone; write PREREG_expC2.md and commit it first.
Inherit PREREG_expC.md wholesale except the two changes marked [C2].

=== [C2] THE FIX — keep the interface formed ===
Two independent ways to stop the binder floating off; pre-register which is PRIMARY:
  (A) PRIMARY: specify RFdiffusion `ppi.hotspot_res` = the TARGET-chain residues that the labelled
      binder hotspots contact in the crystal (compute from the crystal: target residues within 5A Cbeta
      of any labelled binder hotspot). This tells the Complex_base checkpoint to keep the binder docked
      to that patch. NOTE: hotspot_res are TARGET residues (a GEOMETRY/targeting constraint), NOT the
      binder's own residue identities — so it constrains where the interface sits, not what the binder
      sequence is. That distinction is what makes the recovery measurement still valid; the leakage
      control (KILL C2b) checks it empirically.
  (B) FALLBACK (if A still diverges >40%): the Base checkpoint instead of Complex_base, same contig.
  Report the interface-formed fraction per noise level; if the fix works it should be >>C's 25-38%.

=== [C2] THE LADDER — sample the physical regime densely ===
Noise ladder partial_T in {0, 5, 10, 15, 20, 30} of 50, N>=6 backbones per (complex, level) over the
same 55 L<=400 complexes (results/expC_complexes.csv). More samples + the pinning together aim for
>=30 interface-formed complexes per physical level (C had only 12-13). partial_T=0 = crystal control.
Dose variable = interface Ca-RMSD to crystal, same as C.

=== SCORING — identical to Exp C (reuse src/expC_score.py, src/expC_analyze.py verbatim) ===
Re-key each backbone to crystal via the committed residue maps; ProteinMPNN v_48_020, 8 decoding orders
teacher-forced native log-prob + unconditional KL pass; burial = Cbeta neighbour count; SECONDARY_B
within-binder pairs + the EXPC_within_binder re-match robustness variant. Complex-level bootstrap, 10,000
gap / 2,000 KL reps, seed 20260803. [C2 fix on KL bootstrap: compute p_gt0 nan-aware (np.mean over
non-nan reps) and REPORT the fraction of degenerate resamples dropped, per level.]

PRE-REGISTERED READINGS (fix before running; both primaries are publishable):
  - C2-PRIMARY (positive): on interface-FORMED, iRMSD<8A backbones the burial-matched SECONDARY_B gap
    becomes more negative with iRMSD AND its 95% CI excludes zero in the physical band (iRMSD 3-8A,
    interface formed) -> the tax reaches the design regime on physical generative backbones; the
    conditioning-set claim is complete. This is the clean result C could not power.
  - C2-NULL (equally reportable): if the gap in the physical band (iRMSD<8A, formed) is TOST-equivalent
    to zero (CI inside +-0.15, the Exp A predicted-backbone effect size), the log-prob tax is
    PREDICTION-scale only and does not appear in the physical generative-drift regime -> honest bound;
    KL remains the transferable signal. NOT a failure -- a pre-registered negative that sharpens scope.
  - C2-KL: KL dAUROC-over-burial CI excludes zero at each physical level (replicating C).
  - KILL C2a (mandatory control): partial_T=0 reproduces the ~zero crystal within-binder deficit
    (Exp C T0 was +0.303 [-0.19,+0.80], CI contains zero). If not, pipeline broken -- stop.
  - KILL C2b (leakage control, NEW): the burial-matched gap at hotspot positions whose target contacts
    WERE passed to ppi.hotspot_res must EQUAL the gap at a held-out set of interface hotspots NOT passed
    (pre-split 50/50 per complex, seed recorded). If conditioning leaked interface information into
    recovery, conditioned positions would show a smaller deficit than held-out -- pre-register that they
    are statistically indistinguishable (paired CI contains zero). If they differ, C2-PRIMARY is
    confounded by the conditioning and must be reported as such.
  - CONFOUND (as Exp C): report the gap restricted to well-formed-interface backbones; the signal must
    NOT live only in the marginal-interface tail.

=== SECONDARY (CPU, free) === recompute partial-rho(experimental SKEMPI ddG_bind, ProteinMPNN log-odds |
burial) per level on the C2 backbones (src/expC_secondary.py). Prediction unchanged: collapses off the
native manifold. Now check whether it collapses GRADUALLY across the denser physical ladder or (as in C)
is already gone by partial_T=5.

=== OPTIONAL (only if C2-PRIMARY lands clean, ~15-25 GPU-h) === AF2-multimer ipTM on a ~20-complex
subset: ProteinMPNN-design K=8 on crystal vs highest physical-noise backbone, fold, compare interface
ipTM/pAE at hotspots.

Standing rules (CLAUDE.md): pre-register both readings + the two kill criteria before any number; run
partial_T=0 through the identical path; complex-level bootstrap; >=6 samples/condition, report spread +
the interface-formed fraction and every exclusion; write raw outputs to results/ as CSV with exact
commands; write results/FINDINGS_expC2.md with a one-line verdict strictly from the pre-registered
readings; do NOT move a reading after seeing a number.
```

---

## Notes

- **Archival (do this in the same session — see `DATA.md`, time-sensitive).** SCRATCH purges 60–90 d
  after the 2026-08-10 run; before **~2026-10-09** deposit the Exp C (and C2) backbones + scored tables
  to Zenodo with a formed/dissolved/nan manifest, and commit the frozen SE3nv spec + sbatch files as
  plain text to `environment/` (the `environment/README.md` scaffold lists exactly what to drop in).
- **Why both readings are pre-registered as publishable.** After Exp C, either outcome advances the
  paper: a clean physical-regime dose-response upgrades Fig 3 to the monotone-deficit curve and is the
  ICLR clincher; a precise physical null bounds the claim to prediction-scale and leaves the KL detector
  (which transferred across all three backbone classes) as the transferable positive. Under
  pre-registration this is win–win — which is exactly why it is worth the ~15–20 GPU-h.
- **If GPU-constrained:** the leanest informative version is fix (A), ladder {0,10,20} × N=6, 55
  complexes (~9 GPU-h) — enough to test whether pinning lifts the interface-formed fraction and whether
  the physical-band gap moves. Extend the ladder only if the interface-formed fraction confirms the fix.

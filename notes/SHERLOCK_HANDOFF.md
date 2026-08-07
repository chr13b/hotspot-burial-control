# Sherlock handoff — the two GPU experiments that decide TMLR vs ICLR

Paste the block below into a fresh Claude Code session on Sherlock, in a clone of
`github.com/chr13b/hotspot-burial-control`. Everything the CPU could do is done and pushed;
these two need a GPU. One `--gres=gpu:1` (V100/A100) suffices for both — this is inference and
sampling, not training.

Priority order is deliberate: **Experiment A** is the cheaper validity check and it de-risks the
whole C5 story; **Experiment B** is the one that can lift the paper from TMLR to ICLR. Do A first;
if A collapses the C5 claim, B's framing changes, so don't run them in parallel blind.

---

```
Read BRIEF.md, CLAUDE.md, and results/FINDINGS.md in full before writing any code. This repo is a
completed CPU study of whether inverse-folding models are "blind" at protein-protein interface
hotspots. Headline results, all pre-registered and pushed:
  - No burial-matched hotspot recovery penalty on the BOUND complex backbone, across 4
    architectures (ProteinMPNN vanilla/soluble, PiFold, MIF). F0 fired; the mechanism as stated is
    refuted on this fixture.
  - The signal that DOES survive: a residue-agnostic, sequence-free quantity
      KL_i = KL( p(.|bound complex backbone) || p(.|own-chain backbone) )
    computed from two ProteinMPNN UNCONDITIONAL forward passes, identifies experimental hotspots
    with burial-orthogonal information, is not a contact-count proxy, and matches a sequence-AWARE
    statistic (removing the sequence costs nothing: ΔAUROC +0.001 [-0.020,+0.023]). But its
    per-complex top-k precision gain is NOT significant, so it is a diagnostic, not yet a detector.
  - At T=0.1 the constellation barrier is manufactured by the temperature exponent, not model
    uncertainty (median deficit at a missed hotspot is only 1.1 nats = 3x in probability; at T=1
    the same conditionals give ~8 draws instead of 10^10).

Two open questions need a GPU. Do them in this order.

=== EXPERIMENT A — predicted-backbone transfer (the validity condition for C5, ~2 GPU-hr) ===
Every backbone scored so far is a NATIVE co-crystal backbone, carved by the very side chains being
predicted. The reviewer-critical question is whether the KL signal and the burial-matched null
survive on backbones the model did NOT get from the native side chains.

Do:
1. For the 141 pair complexes (results/pair_complexes.txt), predict each complex backbone with
   AF2-multimer (ColabFold) or ESMFold or Boltz — whichever is already on the cluster. Keep the
   predicted N/CA/C/O; align to the crystal (Kabsch) only for bookkeeping, do not use crystal
   coords in scoring.
2. Recompute, on the PREDICTED backbones, exactly the src/ pipeline: (a) the burial-matched
   hotspot gap (src/p0_burial_matched.py machinery — reuse the SAME pydssp matched pairs from
   results/p0_dssp_pairs_*.csv, keyed by residue id, so the comparison is matched), and (b) the KL
   detector (src/kl_detector.py) and its AUROC vs burial baseline (src/kl_analysis.py).
3. Report, with complex-level bootstrap: the AUROC drop for KL, and whether a burial-matched
   hotspot deficit APPEARS on predicted backbones (predicted from d_bind_local ~0.3-0.5 nats).

PRE-REGISTERED READINGS (fix before running):
  - If KL's ΔAUROC-over-burial survives on predicted backbones (CI still excludes 0): C5 is a real
    design-time signal, not a native-crystal artifact. This is the strong outcome.
  - If it collapses to zero: C5 is a property of crystal backbones only; report it as such — still a
    finding about what native backbones encode, but not a design-time tool.
  - If a burial-matched hotspot deficit APPEARS on predicted backbones where it was absent on
    crystals: that is the project's central positive result — the tax is real precisely when the
    backbone is not native — and it reframes the whole paper around the conditioning set.
Kill/caveat: AF2 backbones near-memorise many of these complexes; report pLDDT/pTM and stratify by
prediction confidence so "predicted" does not secretly mean "reconstructed crystal".

=== EXPERIMENT B — commitment ordering on a COUPLED co-design model (the ICLR move, ~4-8 GPU-hr) ===
BRIEF §2.3's mechanism is "decide the discrete variable while the continuous channel is still hot."
ProteinMPNN has NO continuous channel (fixed backbone), so the CPU ordering experiment (src/decoding/,
tested and ready) can only test a weaker claim and CANNOT overturn MultiFlow's purity-unmasking
choice. Running the same intervention on a model with a real SE(3) continuous channel makes §2.3
testable for the first time.

Do:
1. Get a public MultiFlow-family co-design checkpoint running (MultiFlow, or Chroma, or
   ProteinGenerator). FIRST verify the BRIEF §4 implementation assumption: the discrete (CTMC) and
   continuous (SE(3) flow) corruption processes must be coupled ONLY through a shared time index,
   with independently specifiable rate functions. If they are entangled, report and stop — the
   experiment is not well-defined.
2. Measure commitment times t*_seq and t*_str (0.5-crossing of normalised agreement between the
   endpoint prediction and the realised value: token-argmax agreement for sequence, TM-score or
   contact-map overlap for structure), ≥3 seeds, ≥2 length bins. [KILL F3: t*_seq ≤ t*_str + 0.05
   under the default schedule means joint models already decide sequence first — diagnosis wrong.]
3. Sweep the discrete unmasking-rate exponent AT INFERENCE with the continuous schedule fixed, no
   retraining. Metric: hotspot-restricted recovery = recovery at SKEMPI hotspots minus recovery at
   the burial-matched controls (reuse results/p0_dssp_pairs_*). Compare KL-first / purity-first
   (MultiFlow's default) / burial-first / random / oracle orders. [KILL F4: a full-range sweep
   moves hotspot-restricted recovery by less than seed-to-seed SD → the knob is inert.]
4. ADD ONE BINDING-RELEVANT READOUT (the second ICLR requirement): AF2-multimer ipTM/interface-pAE,
   or FoldX/Rosetta ΔΔG_bind, on the designs from each ordering arm. Sequence recovery alone will
   not satisfy an ICLR binder-design reviewer.

PRE-REGISTERED FRAMING: this contradicts MultiFlow's PUBLISHED finding that purity (confidence-
ordered) unmasking is beneficial. Burden of proof is on us; the comparison must be scrupulously
matched. Carry over the CPU pilot's free by-product — teacher-forced evaluation ranks the ordering
choice OPPOSITE to how it performs under real sampling (§4.2c exposure bias) — and record
teacher-forced log p under every order in the same pass; it needs no extra sampling.

Standing rules (CLAUDE.md): pre-register kill criteria before seeing numbers; run a positive control
through every path; average over ≥8 sampling orders / ≥3 seeds and report the spread — a result
inside seed-to-seed SD is not a result; write raw outputs to results/ as CSV with the exact command.
```

---

## Cluster notes

- One GPU, `--gres=gpu:1`. Both experiments are inference/sampling.
- Phases 0/1 need **no** data movement — the CPU results are already in the repo. Only the two
  experiments above need the GPU.
- The tested CPU ordering harness is `src/decoding/mpnn_steer.py` (+ `test_steer.py`, all pass). If a
  Sherlock GPU is scarce, running the ProteinMPNN ordering experiment there is still worthwhile as
  the fixed-backbone control for Experiment B (PiFold, being one-shot deterministic, is the other
  control — it has no decoding order at all).

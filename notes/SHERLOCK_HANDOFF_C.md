# Sherlock handoff — Experiment C: the de-novo backbone test (the decisive ICLR experiment)

Paste the block below into a fresh Claude Code session on Sherlock, in a clone of
`github.com/chr13b/hotspot-burial-control`. This is the follow-up to Experiment A
(`results/FINDINGS_expA.md`), which showed the burial-matched hotspot deficit appears on
**OpenFold3-predicted** backbones of *known* complexes. The reviewer objection A cannot answer:
OpenFold3 near-reconstructs these pre-2021 complexes (median interface RMSD 1.3 Å), so the backbone
is barely "non-native," and it is a prediction of a *known* complex, not a backbone from a generative
design model. **Experiment C closes that gap with backbones a generative model produced — and does it
as a dose–response, which is much harder to dismiss than a single point.**

**GPU estimate: ~20 GPU-hours for the core (one A100-day); ~40 with the optional AF2 readout.** One
`--gres=gpu:1` (A100 preferred; V100 fine, ~1.5× slower). Breakdown in the cluster notes at the end.

---

```
Read BRIEF.md, CLAUDE.md, results/FINDINGS.md, and results/FINDINGS_expA.md in full before writing
any code. Context: this project pre-registered whether inverse-folding models are "blind" at
protein-protein interface hotspots. On NATIVE crystal backbones there is NO burial-matched hotspot
recovery penalty across 5 architectures (the published ProBID-Net gap is a burial artifact). But the
native backbone secretly encodes the side chains being predicted. Experiment A showed that on
OpenFold3-PREDICTED backbones of the same complexes, a burial-matched hotspot deficit APPEARS
(SECONDARY_B -0.191 [-0.373,-0.004]; paired delta vs crystal -0.154 [-0.279,-0.028]), as large at
high prediction confidence as low, and a sequence-free KL signal predicts it. The tax is a property
of the CONDITIONING SET, not hotspot chemistry.

Experiment C tests the claim in the setting designers actually work in: backbones a GENERATIVE model
produced, which the scorer has never seen. Pre-register everything below BEFORE generating any
backbone; write PREREG_expC.md and commit it first.

=== PRIMARY — partial-diffusion dose-response (label-preserving, the powered result) ===
The problem with "de novo" is that a from-scratch backbone has no SKEMPI hotspot labels. Partial
diffusion solves it: RFdiffusion partial diffusion perturbs an input backbone by a controllable
number of noising steps while PRESERVING residue registration (same chain lengths, same residue
array), so the committed hotspot/control labels (results/p0_dssp_pairs_*.csv, keyed by chain,resnum)
transfer EXACTLY. Sweeping the noise level gives backbones progressively further from native — a
dose-response of the deficit vs backbone-distance-from-native.

Do:
1. For each of the ~60 SKEMPI pair complexes with L<=400 (results/pair_complexes.txt, filtered by
   length; use the same set Experiment A scored so the crystal and OpenFold3 points are comparable),
   run RFdiffusion PARTIAL DIFFUSION at a ladder of noise steps -- e.g. partial_T in
   {0, 5, 10, 20, 40} out of 50 -- generating N=3 backbones per (complex, noise level). Diffuse the
   BINDER chain(s) while holding the target chain(s) fixed (motif/partner held), so the interface
   context is preserved and only the binder backbone drifts. partial_T=0 is the crystal (positive
   control: must reproduce ~no deficit).
2. For each generated backbone, compute interface Ca-RMSD to the crystal (the dose variable) and
   confirm the interface still forms (>=5 inter-chain contacts at the labelled hotspot positions;
   drop backbones where the interface has dissolved and REPORT how many).
3. Score each backbone with the committed pipeline: src/p0_burial_matched.py machinery reusing the
   SAME committed pydssp pairs (src/expA_gap_reuse_pairs.py is exactly this -- reuse it), plus the KL
   detector (src/kl_detector.py) and its ΔAUROC-over-burial (src/expA_kl_delta.py). Score at
   T-teacher-forced native log-prob, same as everywhere.
4. Report, complex-level bootstrap:
   - the burial-matched SECONDARY_B gap AS A FUNCTION OF interface-RMSD bin (the dose-response);
   - the KL ΔAUROC-over-burial at each noise level;
   - the crystal (partial_T=0) point, which must reproduce the committed crystal ~0 deficit.

PRE-REGISTERED READINGS (fix before running):
  - C-PRIMARY: the burial-matched gap becomes MORE negative monotonically with interface-RMSD, and at
    the highest realistic-interface noise level its CI excludes zero. -> the tax scales with how
    non-native the backbone is; the conditioning-set claim is complete in the design regime.
  - C-KL: KL ΔAUROC-over-burial CI excludes zero on the generated backbones (as it did on OpenFold3).
    -> the sequence-free signal is usable on generative-model backbones.
  - KILL C1: if partial_T=0 does NOT reproduce ~zero crystal deficit, the pipeline is broken -- stop
    and debug (this is the mandatory positive control).
  - KILL C2: if the gap is flat across the whole RMSD ladder (no dose-response) AND the highest level
    contains zero, the deficit is specific to OpenFold3-style predictions and does not generalise to
    generative backbones -- an honest negative that bounds the claim to prediction, not design.
  - CONFOUND to rule out (as in Exp A): stratify by backbone quality proxies; the deficit must NOT be
    carried only by backbones where the interface half-dissolved. Report the gap restricted to
    backbones with a well-formed interface (contacts preserved, iRMSD < 3 A).

=== SECONDARY (binding-relevant readout, cheap, CPU -- the other ICLR gap) ===
Everything so far is recovery/log-prob, not a binding number. On the crystal, ProteinMPNN's
per-mutation log-odds correlate with experimental SKEMPI ΔΔG_bind at partial rho +0.18-0.28
(results/hardening_external.csv, src/hardening.py). Recompute that SAME correlation using log-odds
computed on the PREDICTED (Exp A) and PARTIAL-DIFFUSION backbones. Prediction: the model's ability to
rank experimental binding energy DEGRADES as the backbone becomes non-native -- the binding-relevant
face of the deficit. This needs no extra GPU (reuses SKEMPI ΔΔG + the scored positions). Report
partial-rho(ΔΔG, log-odds | burial) as a function of interface-RMSD.

=== OPTIONAL (heavier, only if GPU budget allows) -- AF2-multimer readout ===
For a ~20-complex subset, ProteinMPNN-design K=8 sequences on the crystal vs the highest-noise
partial-diffusion backbone, fold each design with AF2-multimer (or OpenFold3), and compare interface
ipTM / interface-pAE at the hotspot positions. Prediction: designs from non-native backbones fold to
worse interfaces at the hotspots. ~15-25 GPU-hours; do only after the primary lands.

Standing rules (CLAUDE.md): pre-register kill criteria before seeing numbers; run the partial_T=0
positive control through the identical path and require it to reproduce the crystal; complex-level
bootstrap (complexes are the unit); >=3 samples per condition, report the spread; write raw outputs
to results/ as CSV with the exact command. Write results/FINDINGS_expC.md with a one-line verdict
derived strictly from the pre-registered readings. Do NOT move a reading after seeing a number.
```

---

## Why partial diffusion, not full de-novo binder design

Full RFdiffusion binder design against the target is the *literal* workflow, but a from-scratch
backbone has **no hotspot labels** — you cannot ask "is the burial-matched deficit there?" because
there is no burial-matched pair set. Partial diffusion keeps the residue registration, so the
committed SKEMPI hotspot/control labels transfer exactly, and you get a **dose–response** (deficit vs
backbone drift) instead of a single unlabelled point. A monotone dose–response that starts at ~0 on
the crystal and grows with RMSD is the single most reviewer-proof figure this project can produce:
it *is* the conditioning-set mechanism, drawn as a curve. Keep full de-novo binder design as the
optional qualitative coda (does KL still concentrate at the generated interface?), not the primary.

## Cluster notes / GPU estimate

- **RFdiffusion partial diffusion:** ~2–3 min per backbone (A100) for a binder chain of ~100–150 res
  with the target held; partial (≤40 of 50 steps) is faster than full generation. 60 complexes × 5
  noise levels × 3 samples ≈ 900 backbones × ~2.5 min ≈ **~37 GPU-h at full scale**; a lean
  40-complex × 4-level × 2-sample design ≈ 320 backbones ≈ **~13 GPU-h**. Start lean, extend if the
  dose-response is clean.
- **ProteinMPNN scoring + KL:** seconds per backbone, negligible; runs on the same GPU or CPU.
- **Binding-readout (ΔΔG correlation):** CPU, free — reuses SKEMPI and the scored positions.
- **Optional AF2-multimer readout:** ~15–25 GPU-h for a 20-complex subset (design × fold × score).
- **Core total: ~15–20 GPU-h (one A100-day).** With the AF2 readout: ~35–45 GPU-h (two A100-days).

Environment: RFdiffusion needs its own conda env (SE3Transformer + dgl); it does not share the
MultiFlow/OpenFold envs. The scoring scripts (`src/expA_gap_reuse_pairs.py`, `src/kl_detector.py`,
`src/expA_kl_delta.py`) already exist and run under the project's torch-2.0.1 env. Reuse
`results/p0_dssp_pairs_*.csv` and `results/pair_complexes.txt` verbatim — they are the label set.

# Pre-registration — Experiment C: partial-diffusion dose-response (generative backbones)

Written 2026-08-10 on Sherlock, **after** reading BRIEF.md / CLAUDE.md / results/FINDINGS.md /
results/FINDINGS_expA.md and fixing the L≤400 complex set + binder assignment (`results/expC_complexes.csv`,
`src/expC_setup.py`), but **before generating a single RFdiffusion backbone**. Companion to
`PREREG_expA.md`; §6 scoring choices are inherited. No reading is moved after seeing a number.

## 0. The question

Experiment A showed a burial-matched hotspot log-prob deficit **appears** on OpenFold3-*predicted*
backbones of known complexes (SECONDARY_B −0.191 [−0.373,−0.004]; paired vs crystal −0.154
[−0.279,−0.028]) where it is **absent** on native crystals (−0.042 [−0.222,+0.129]). Objection A cannot
answer: OpenFold3 near-reconstructs these pre-2021 complexes (median iRMSD 1.3 Å), so the backbone is
barely non-native and is a prediction of a *known* complex. **Experiment C tests the claim on backbones
a generative model produced, as a dose–response of the deficit vs backbone-distance-from-native.**

## 1. Generator and the label-preserving trick

- **RFdiffusion partial diffusion** (RosettaCommons/RFdiffusion). Partial diffusion perturbs an input
  backbone by `partial_T` of 50 noising steps while **preserving residue registration** (same chains,
  same residue array), so the committed pydssp hotspot/control pairs (`results/p0_dssp_pairs_*.csv`,
  keyed by `chain,resnum`) transfer **exactly**.
- **Diffuse the BINDER, hold the TARGET.** Per complex the BINDER = the chain group carrying the
  labelled positions (majority; `results/expC_complexes.csv`), TARGET = the other group, held fixed as
  a motif so the interface context is preserved and only the binder backbone drifts.
- **Complex set (FIXED):** the **55** pair complexes with **L≤400** that carry labelled pairs
  (`results/expC_complexes.csv`; median L 262, median binder 141 res) — the same set Experiment A
  scored, so crystal and OpenFold3 points are comparable.
- **Noise ladder (FIXED):** `partial_T ∈ {0, 5, 10, 20, 40}` of 50, **N=3** backbones per
  (complex, level), seeds fixed and recorded. If GPU-constrained, a pre-declared lean interim runs
  `{0,10,20,40} × N=2` over all 55 complexes and is extended toward the full design; the *design* is
  the full ladder and no level is added/removed after seeing numbers.
- **`partial_T=0` = the crystal backbone**, taken directly from the crystal PDB (0 noising = identity)
  and pushed through the **identical** convert+score path — the mandatory positive control (KILL C1).

## 2. Per-backbone geometry (the dose variable + interface QC)

- **Interface Cα-RMSD to crystal** (the dose variable): superpose the generated complex onto the
  crystal by the **held target** Cα (near-identity, since the target is fixed), then RMSD over the
  labelled binder interface Cα. Recorded per backbone.
- **Interface QC:** at the labelled hotspot positions, count inter-chain (binder–target) Cβ contacts
  within 10 Å; a backbone passes if ≥ 5 such contacts remain (interface still formed). Backbones that
  fail (interface dissolved) are **dropped and their count REPORTED** per noise level.

## 3. Scoring (inherited from PREREG_expA / FINDINGS §6, applied to each generated backbone)

- RFdiffusion output is **backbone-only** (N/CA/C/O) — sufficient for ProteinMPNN teacher-forced
  log-prob and the unconditional KL passes (both use backbone only). Re-key each backbone to crystal
  `(chain,resnum,icode)` via `results/expC_resmap.json` (registration is preserved, so positional).
- ProteinMPNN `v_48_020.pt`, `augment_eps=0`, per-position teacher-forced native log-prob, mean over 8
  decoding orders; unconditional pass for KL. Same as Experiments A/B.
- **Gap (C-PRIMARY):** reuse the committed pydssp matched pairs via `src/expA_gap_reuse_pairs.py`,
  restricted to pairs with **both** hot and control positions on the **diffused binder** (report the
  retained pair count and how many are excluded for spanning the held target). `d = logp(hot) −
  logp(ctl)` on the generated backbone; complex-level bootstrap (10,000 reps, seed 20260803).
  Verdict tier = **SECONDARY_B** (the power tier, as in Exp A); all committed tiers reported.
- **KL (C-KL):** `KL_i = KL(p(·|generated complex backbone) ‖ p(·|chain-deleted generated backbone))`
  (`src/kl_detector.py`), and its `ΔAUROC = AUROC(burial+KL) − AUROC(burial)` (`src/expA_kl_delta.py`),
  paired complex bootstrap. **Burial baseline = Cβ-neighbour count** computed on the generated backbone
  (design-time, backbone-only; the pipeline's `neighbour_counts`), consistent across all arms including
  `partial_T=0`; the rSASA-based Exp-A baseline is also reported for cross-reference (absolute AUROCs
  are not directly comparable across the two burial definitions — the reading is the *trend across
  noise levels*).

## 4. Metrics reported (complex-level bootstrap; ≥3 samples/condition, spread reported)

1. burial-matched **SECONDARY_B gap as a function of interface-RMSD bin** (the dose–response);
2. **KL ΔAUROC-over-burial at each `partial_T`** level;
3. the **`partial_T=0` (crystal) point** (must reproduce the committed crystal ~0 deficit);
4. the gap **restricted to well-formed-interface backbones** (contacts preserved AND iRMSD < 3 Å) —
   the confound control.

## 5. Pre-registered readings (fixed before running)

- **C-PRIMARY:** the burial-matched SECONDARY_B gap becomes **more negative monotonically** with
  interface-RMSD, and at the highest realistic-interface noise level its 95% CI **excludes zero** →
  the tax scales with how non-native the backbone is; the conditioning-set claim is complete in the
  design regime.
- **C-KL:** KL ΔAUROC-over-burial CI **excludes zero** on the generated backbones (as on OpenFold3) →
  the sequence-free signal is usable on generative-model backbones.
- **KILL C1 (mandatory positive control):** if `partial_T=0` does **not** reproduce the ~zero crystal
  SECONDARY_B deficit (committed −0.042 [−0.222,+0.129]), the pipeline is broken — **stop and debug**.
- **KILL C2:** if the gap is **flat** across the whole RMSD ladder (no dose–response) **and** the
  highest level's CI **contains zero**, the deficit is specific to OpenFold3-style predictions and does
  not generalise to generative backbones — an honest negative that bounds the claim to prediction, not
  design.
- **CONFOUND (as in Exp A):** the deficit must **not** be carried only by backbones where the interface
  half-dissolved. Report the gap restricted to well-formed-interface backbones (contacts preserved,
  iRMSD < 3 Å); it must survive there.

## 6. Secondary — binding-relevant readout (CPU, no extra GPU)

On the crystal, ProteinMPNN per-mutation log-odds correlate with experimental SKEMPI ΔΔG_bind at
burial-partial Spearman ≈ −0.247 (F1; `results/hardening_external.csv`, `src/hardening.py`).
**Recompute the same `partial-ρ(ΔΔG_bind, log-odds | burial)`** using log-odds
`ℓ(mut) − ℓ(wt)` computed on the **predicted (Exp A)** and **partial-diffusion** backbones (both store
per-position `lp_<AA>`), as a function of interface-RMSD. **Prediction:** the model's ability to rank
experimental binding energy **degrades** (|ρ| toward 0) as the backbone becomes non-native — the
binding-relevant face of the deficit. Exploratory beyond the sign prediction; no falsifier attaches.

## 7. Optional (only after the primary lands; ~15–25 GPU-h)

AF2-multimer / OpenFold3 ipTM on a ~20-complex subset: ProteinMPNN-design K=8 sequences on the crystal
vs the highest-noise partial-diffusion backbone, fold, compare interface ipTM/pAE at hotspots. Not run
unless the primary is clean and GPU budget allows.

## 8. Deviations / operational items, declared up front

- RFdiffusion is not installed on this cluster; it is built from source in its own SE3 conda env and
  the exact partial-diffusion command (contig holding the target, diffusing the binder) is **recorded**
  in `results/expC_*.csv` `command` columns once fixed.
- Backbones failing the interface QC (§2) or that RFdiffusion fails to produce are **listed** with the
  reason and excluded; the excluded set is reported, never silently dropped (CLAUDE.md rule 6).
- Burial on generated backbones uses the Cβ-neighbour proxy (§3) because the output is backbone-only;
  this deviation from Exp A's all-atom rSASA is stated and both are reported.

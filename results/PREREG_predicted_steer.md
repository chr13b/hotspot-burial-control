# PRE-REGISTRATION — Predicted-backbone steering check (does +α·L steer on the backbones designers actually use?)

Registered **before any steering number**. `SEED=20260803`, `NBOOT=5000`, complex-clustered (unit = complex)
throughout. CLAUDE.md rules apply: never fabricate; every number → committed CSV; positive controls before trusting
a zero; falsifier below is fixed and will not be moved after seeing a value.

## Question
Our steering result (a `+α·L` tilt on a frozen ProteinMPNN raises an independent model's binding-leverage and ipTM
vs a matched random direction) is measured on **crystal** backbones. This run tests whether it survives when the
leverage is **computed and applied on the predicted backbone** — the actual staged-design regime. The **detection**
side already survives there (mixed-derivative CPI on OF3/AF2 backbones +0.039 [+0.026,+0.050], §6/§8; predicted
backbones fall on the surviving part of the dose law). This tests the **intervention** side.

## Inputs (cached; not regenerated)
- Predicted backbones: `results/expC2_backbones.tar.gz` + `results/expC2_backbone_manifest.csv` (OpenFold3 and
  AF2-multimer, 140 complexes shared with SKEMPI). Backbones are NOT re-folded.
- Predicted-backbone leverage: reuse `src/leverage_predicted.py` outputs (per-position L on the predicted
  structure). Interface positions defined on the **predicted** structure (the honest design-time set).
- Complex set: **intersection** of the 60-complex ipTM steering set (`results/iptm_subset.txt`) and the 140
  predicted backbones. `n` reported; if the 60 are not fully covered, the overlap is used and stated. Both OF3 and
  AF2 backbones used where available (a within-fold replication).

## Phase 1 — DECISIVE, CPU-only: steer on the predicted backbone, judge anti-circularly
Reuse `src/cfg_steer.py` / `src/cfg_steer_naive.py` with the backbone pointed at the **predicted** structure. Arms
**wt / L / random** (+ **naive** = the confidence tilt, matched per-position magnitude, if cheap). α as in the
crystal run (`--alphas 0,2`, `--K 64`).
- **Anti-circular judging (never steer-X / judge-X):** steered interface residues scored by an *independent* IF
  model's leverage — **ESM-IF1, MIF, and PiFold** (the three non-self judges from the crystal judge matrix; crystal
  steer-ProteinMPNN → ESM-IF1 **+0.77**, MIF **+0.71**, PiFold **+0.89**). Paired, complex-clustered bootstrap
  (`NBOOT`), 95% CI: **L − random** (primary), **L − naive** (secondary), per judge. → `results/cfg_steer_predicted.csv`.
- **Native-recovery-preserved control:** report interface recovery on the predicted backbone per arm (steering must
  not wreck recovery), as the crystal run does.
- **Positive control (before trusting the predicted contrast):** the *same pipeline on the crystal backbone* for
  this complex subset must reproduce the committed crystal judge numbers (L−random ESM-IF1 ≈ +0.77). Included as a
  `backbone=crystal` block in the same CSV — run and checked before the predicted-backbone contrast is trusted.

## Phase 2 — CONFIRMATION, GPU: structure-predictor metrics on the predicted-backbone steered sequences
Fold the Phase-1 steered sequences (wt / L / random, k as in the crystal ipTM run) with **AF2-multimer** (primary)
and **Boltz-2** (second, decorrelated). Reuse the crystal pipeline UNCHANGED: `src/build_iptm_fastas.py` → fold →
`src/parse_iptm.py` → `src/analyse_iptm.py`, so the metric set is identical.
- **Metrics (exact `parse_iptm.py` columns, no new metrics):** **ipTM**, **global pTM**, **interface pLDDT**,
  **interface pAE**.
- **Pre-registered robust COMPOSITE (unchanged):** z-mean of (ipTM, −interface pAE, interface pLDDT). **global pTM
  is the localization control**, not part of the composite.
- **Aggregation (unchanged):** mean-over-k **primary**, best-of-k **secondary**; then paired complex-clustered
  bootstrap **L − random** (and **L − naive**), 95% CI + P(>0), for **each** of ipTM, interface pAE, interface
  pLDDT, global pTM, and the composite. **L − wt** reported (the no-collapse check). → `results/iptm_predicted.csv`.
- **Coherence caveat:** AF2-multimer and Boltz-2 ipTM are differently calibrated — magnitudes are NOT cross-folder
  comparable; the claim is the **direction replicating within each folder**, read against its own zero. Each folder
  in its own block.
- **Anchor to crystal:** each predicted-backbone contrast tabulated next to its committed crystal value (ipTM
  L−random +0.235 AF2 / +0.139 Boltz-2; composite +0.82 / +0.68) so the attenuation is explicit.

## Pre-registered falsifier + confirmation criteria (reported verbatim if fired)
- **Primary (Phase 1, decisive):** if **L − random ≤ 0 at the judge level** (CI includes/below 0, across ESM-IF1 /
  MIF / PiFold), the steering benefit is a **crystal-backbone artifact** and does not transfer to the design regime
  — reported plainly, bounding the steering claim to crystals.
- **Confirmation (Phase 2):** the structure-level criterion is the crystal run's **H1** — composite(L) > random
  **and** ipTM(L) > random, both CI>0. If Phase 1 clears but Phase 2 H1 does not, report exactly as the naive/ipTM
  story already does (judge-level binding-specificity holds; the fold-level effect is bounded), **not** as a failure.
- **H2** L − wt not strongly negative (steering doesn't collapse the fold). **H3** |global-pTM shift| ≪ |composite
  shift| (the effect is localized to the interface, not a global foldability change).
- Neither outcome touches the crystal steering results or the detection lanes. **Expectation (given the surviving
  dose law): L − random > 0 but likely ATTENUATED vs crystal — reported honestly, not inflated.**

### Deliverables
`results/PREREG_predicted_steer.md` (this), `results/cfg_steer_predicted.csv` (judge-level ESM-IF1/MIF/PiFold
L−random & L−naive + the crystal positive-control block), `results/iptm_predicted.csv` (per-folder AF2 + Boltz-2:
ipTM, global pTM, interface pLDDT, interface pAE, composite; L−random, L−naive, L−wt; mean-over-k + best-of-k;
H1/H2/H3 verdicts), `results/FINDINGS_predicted_steer.md`. Paper left untouched.

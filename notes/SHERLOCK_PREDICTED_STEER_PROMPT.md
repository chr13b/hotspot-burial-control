# Sherlock — Predicted-backbone steering check: does +α·L steer on the backbones designers actually use? [paste-to-run]

`SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.
`git pull --no-edit origin main` FIRST. CLAUDE.md rules apply (never fabricate; every number → committed CSV;
positive controls before trusting a zero; `git add` by name; two trailer lines; push). **Pre-register
`results/PREREG_predicted_steer.md` before any steering number**, with the falsifier stated verbatim below.

## Why (the softest reviewer attack this closes)
Our steering result (`+α·L` tilt on a frozen ProteinMPNN raises independent-model leverage and ipTM vs a random
direction) is measured on **crystal** backbones. A reviewer will ask: *does it survive on **predicted** backbones —
the actual staged-design regime?* We already know the **detection** side survives there (the mixed derivative on
OpenFold3/AF2 backbones: CPI(L|geom) +0.039 [+0.026,+0.050], §6/§8; predicted backbones fall on the *surviving*
part of the dose law). This run tests the **intervention** side: steering computed and applied **on the predicted
backbone**.

## Phase 0 — inputs (mostly cached; confirm, don't regenerate)
- Predicted backbones already exist: `results/expC2_backbones.tar.gz` + `results/expC2_backbone_manifest.csv`
  (OpenFold3 and AF2-multimer, 140 complexes shared with SKEMPI). **Do not re-fold backbones.**
- Predicted-backbone leverage already exists via `src/leverage_predicted.py` (the §8 detection result). Reuse its
  per-position `L` on the predicted structure; if a needed dump is absent, regenerate with that script (records
  its own PREREG). Interface positions are defined on the **predicted** structure (the honest design-time set).
- Complex set: the **intersection** of the steering set and the 140 predicted backbones. Report `n`; if the 60
  ipTM steering complexes are not fully covered, use the overlap and say so. Use **both** OpenFold3 and AF2
  backbones where available (a within-folder replication, like the two structure predictors already in the paper).

## Phase 1 — DECISIVE, CPU-only: steer on the predicted backbone, judge anti-circularly
Reuse `src/cfg_steer.py` / `src/cfg_steer_naive.py` but point the backbone at the **predicted** structure (not the
crystal PDB). For each complex, arms **wt / L / random** (and **naive** if cheap — the confidence tilt, matched
per-position magnitude, `src/cfg_steer_naive.py`), α as in the crystal run (`--alphas 0,2 --K 64`).
- **Judge anti-circularly** (never steer-X/judge-X): steered residues scored by an *independent* IF model's
  leverage — **ESM-IF1, MIF, and PiFold** (all three non-self judges from the crystal judge matrix, for coherence:
  crystal steer-ProteinMPNN → ESM-IF1 **+0.77**, MIF **+0.71**, PiFold **+0.89**). Paired, complex-clustered
  bootstrap (NBOOT=5000), 95% CI: **L − random** (primary), **L − naive** (secondary), per judge.
  → `results/cfg_steer_predicted.csv`.
- **Native recovery preserved** control: report recovery on the predicted backbone for each arm (steering must not
  wreck recovery), exactly as the crystal run does.
- **Positive control:** the *same pipeline on the crystal backbone* for this complex subset must reproduce the
  committed crystal judge numbers (L−random ESM-IF1 ≈ +0.77) — run it on the overlap as a sanity anchor before
  trusting the predicted-backbone contrast. → include as a `backbone=crystal` block in the same CSV.

## Phase 2 — CONFIRMATION, GPU: structure-predictor metrics on the predicted-backbone steered sequences
Fold the Phase-1 steered sequences (wt / L / random, k as in the crystal ipTM run) with **AF2-multimer** (primary)
and **Boltz-2** (second, decorrelated folder), reusing the **exact crystal pipeline unchanged** so the metric set
stays identical: `src/build_iptm_fastas.py` → fold → `src/parse_iptm.py` → `src/analyse_iptm.py`.
- **Same metric set as the crystal run (do not invent new metrics):** per fold record **ipTM** (interface pTM),
  **global pTM**, **interface pLDDT** (mean pLDDT over interface residues), and **interface pAE** (mean PAE over
  cross-interface g1↔g2 residue pairs, both directions; lower = better) — the `parse_iptm.py` columns exactly.
- **Pre-registered robust COMPOSITE (unchanged):** z-mean of (ipTM, −interface pAE, interface pLDDT) across all
  folds — the same three interface metrics `analyse_iptm.py` uses. **global pTM is the localization control**, not
  part of the composite.
- **Aggregation (unchanged):** mean-over-k **primary**, best-of-k **secondary**; then the **paired, complex-clustered
  bootstrap L − random** (and **L − naive**), 95% CI + P(>0), for **each** of ipTM, interface pAE, interface pLDDT,
  global pTM, and the composite. Report **L − wt** too (the no-collapse check). → `results/iptm_predicted.csv`.
- **Same three pre-registered hypotheses as the crystal ipTM run** (state PASS/FAIL): **H1** composite(L)>random
  **and** ipTM(L)>random, both CI>0 (the confirmation criterion); **H2** L−wt not strongly negative (steering
  doesn't collapse the fold); **H3** |global-pTM shift| ≪ |composite shift| (the effect is *localized* to the
  interface, not a global foldability change).
- **Coherence caveat (as in the paper):** AF2-multimer and Boltz-2 ipTM are **differently calibrated** — magnitudes
  are **not** cross-folder comparable; the claim is the **direction replicating within each folder**, read against
  its own zero. Report each folder in its own block; do not compare the two magnitudes.
- **Anchor to crystal:** tabulate each predicted-backbone contrast next to its committed crystal value (ipTM
  L−random +0.235 AF2 / +0.139 Boltz-2, composite +0.82 / +0.68) so the attenuation is explicit.

## Pre-registered falsifier + confirmation criteria (report verbatim if fired)
- **Primary (Phase 1, decisive):** if **L − random ≤ 0 at the judge level** (CI includes/below 0, across ESM-IF1 /
  MIF / PiFold), the steering benefit is a **crystal-backbone artifact** and does not transfer to the design regime
  — reported plainly, bounding the steering claim to crystals.
- **Confirmation (Phase 2):** the structure-level confirmation criterion is the crystal run's **H1** — composite(L)
  > random **and** ipTM(L) > random, both CI>0. If Phase 1 clears but Phase 2 H1 does not, report it exactly as the
  naive/ipTM story already does (judge-level binding-specificity holds; the fold-level effect is bounded), not as a
  failure.
- Neither outcome touches the crystal steering results or the detection lanes. Expected, given the surviving dose
  law: L − random **> 0 but likely attenuated** vs crystal — report the attenuation honestly, don't inflate.

## Compute (estimate)
Phase 1 is **CPU-only** and cheap (ProteinMPNN + ESM-IF1/MIF/PiFold judges over the overlap × arms × K ≈ 1–3
CPU-hours) — it is the decisive lane. Phase 2 is a **moderate GPU job array** (AF2-multimer folds primary, Boltz-2
second — roughly the combined scale of the two original crystal ipTM runs, ~tens of GPU-hours; Boltz-2 optional if
GPU is tight, AF2 alone still confirms). Run Phase 1 first; only launch Phase 2 if Phase 1 clears zero.

## Deliverables (commit by name, two trailer lines, push)
`PREREG_predicted_steer.md`, `cfg_steer_predicted.csv` (judge-level: ESM-IF1/MIF/PiFold L−random & L−naive, incl.
the crystal positive-control block), `iptm_predicted.csv` (per-folder AF2-multimer + Boltz-2, carrying **ipTM,
global pTM, interface pLDDT, interface pAE, composite** — L−random, L−naive, L−wt, mean-over-k + best-of-k, with the
H1/H2/H3 verdicts), `FINDINGS_predicted_steer.md` (what replicated, attenuation vs crystal, falsifier/H1 status).
Message me the Phase-1 **L − random (judge, per model)** and, if run, the Phase-2 **ipTM and composite L − random**
per folder. Leave the paper (`notes/PAPER_DRAFT.md`) untouched — I fold in Methods/refs/§4 myself.

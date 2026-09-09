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
  leverage — ESM-IF1 and MIF (as in the crystal judge matrix). Paired, complex-clustered bootstrap (NBOOT=5000),
  95% CI: **L − random** (primary), **L − naive** (secondary). → `results/cfg_steer_predicted.csv`.
- **Native recovery preserved** control: report recovery on the predicted backbone for each arm (steering must not
  wreck recovery), exactly as the crystal run does.
- **Positive control:** the *same pipeline on the crystal backbone* for this complex subset must reproduce the
  committed crystal judge numbers (L−random ESM-IF1 ≈ +0.77) — run it on the overlap as a sanity anchor before
  trusting the predicted-backbone contrast. → include as a `backbone=crystal` block in the same CSV.

## Phase 2 — CONFIRMATION, GPU: ipTM on the predicted-backbone steered sequences
Fold the Phase-1 steered sequences (wt/L/random, best-of-k as in the crystal ipTM run) with **AF2-multimer**
(primary; Boltz-2 optional), reusing `src/build_iptm_fastas.py` → `src/analyse_iptm.py` → `src/parse_iptm.py`.
Report paired complex-clustered **interface ipTM L − random** on predicted-backbone designs. → `results/iptm_predicted.csv`.

## Pre-registered falsifier (report verbatim if it fires)
If **L − random ≤ 0 at the judge level (Phase 1, CI includes/below 0)**, the steering benefit is a **crystal-backbone
artifact** and does not transfer to the design regime — reported plainly, bounding the steering claim to crystals.
(This does not touch the crystal steering results or detection.) Expected, given the surviving dose law: L−random
**> 0 but likely attenuated** vs crystal — report the attenuation honestly, don't inflate.

## Compute (estimate)
Phase 1 is **CPU-only** and cheap (ProteinMPNN + leverage judge over the overlap × arms × K ≈ 1–3 CPU-hours) — it
is the decisive lane. Phase 2 is a **moderate GPU job array** (AF2-multimer folds of the steered sequences, same
scale as the original ipTM steering run, ~tens of GPU-hours). Run Phase 1 first; only launch Phase 2 if Phase 1
clears zero.

## Deliverables (commit by name, two trailer lines, push)
`PREREG_predicted_steer.md`, `cfg_steer_predicted.csv` (judge-level, incl. the crystal positive-control block),
`iptm_predicted.csv`, `FINDINGS_predicted_steer.md` (what replicated, the attenuation vs crystal, falsifier
status). Message me the Phase-1 **L − random (judge)** on predicted backbones and, if run, the Phase-2 ipTM L−random.
Leave the paper (`notes/PAPER_DRAFT.md`) untouched — I fold in Methods/refs/§4 myself.

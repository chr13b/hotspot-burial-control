# Pre-registration — Experiment A: predicted-backbone transfer

Written 2026-08-09 on Sherlock, **after** repairing the checkout, staging SKEMPI 2.0 + ProteinMPNN,
and confirming `src/validate.py` prints ALL PASS on this environment, but **before any
predicted-backbone number was computed**. Fixes every operational choice so the variant that gives a
desired answer cannot be chosen after the fact. Companion to `PREREG.md` (Phase 0/1), whose §5–§6
scoring and statistics choices are inherited verbatim unless overridden here.

## 0. The question

Every backbone scored in the committed study is a **native co-crystal backbone**, carved by the very
side chains ProteinMPNN is asked to predict. Experiment A asks whether the two surviving signals —
(i) the **absence** of a burial-matched hotspot log-prob deficit, and (ii) the **KL detector's**
AUROC-over-burial (finding C5) — persist on backbones the model did **not** obtain from the native
side chains. This is the reviewer-critical validity condition for C5.

## 1. Structure predictor

- **OpenFold3** (`openfold3` 0.4.3, `run_openfold`; AlphaFold3-class, all-atom, multi-chain), the
  complex predictor already installed on this cluster. GPU inference via Slurm `-p gpu`.
- **Targets:** the 141 complexes in `results/pair_complexes.txt` (the complexes that contribute
  matched pairs).
- **Input sequences:** for each complex, the per-chain sequences **exactly as extracted by
  `ftax_common.load_complex` from the SKEMPI cleaned PDB** — i.e. the identical residue set, per
  chain, in the identical order, that the crystal run scored. This makes the predicted↔crystal
  residue correspondence 1:1 and positional (predicted residue *j* of chain *X* ↔ crystal
  `(chain X, resnum, icode)` at position *j*), so SKEMPI labels and the committed matched pairs
  transfer without alignment ambiguity.
- **No structural-template leakage.** OpenFold3 is run **template-free** (no deposited structure fed
  as a template); MSAs (sequence homologs, no coordinates) are permitted and the exact MSA/template
  setting is recorded from the emitted `experiment_config.json`. Rationale: a template would let the
  model copy the crystal coordinates, defeating the experiment. MSAs leak no coordinates.
- **Seeds:** fixed (recorded). Per-complex the top-ranked sample (by OpenFold3's own aggregated
  confidence) is the primary predicted backbone; the full sample set is retained.
- **Coordinates used in scoring are the predicted coordinates only.** Kabsch alignment of the
  predicted backbone to the crystal is computed **solely** to report RMSD-to-crystal (a leakage /
  confidence diagnostic) and is **never** fed to ProteinMPNN.

## 2. Scoring (inherited from PREREG.md §5, applied to the predicted backbone)

- ProteinMPNN `vanilla_model_weights/v_48_020.pt`, `augment_eps = 0.0`.
- Per-position teacher-forced conditional log-prob of the **native** residue, mean over **8 decoding
  orders** (seeds 0–7); order spread assessed on the estimate.
- Order-free unconditional (backbone-only) log-prob as the decoding-order-independent secondary and
  as the KL detector's distribution.
- `rSASA_complex`, `rSASA_free`, ΔrSASA, secondary structure (pydssp), and neighbour count are
  recomputed on the **predicted all-atom structure** (OpenFold3 emits side chains). The native
  residue identity is unchanged (it is the SKEMPI/crystal wild type); only geometry changes.

## 3. Analysis 1 — burial-matched hotspot gap on predicted backbones

- **Reuse the committed pydssp matched pairs** `results/p0_dssp_pairs_{VARIANT}.csv`, keyed by
  `(complex_id, chain, resnum)`. These pairs were matched on **crystal** burial / SS / neighbour
  count; reusing them keyed by residue id isolates the effect of swapping the backbone on the
  identical hot/control residue set.
- For each pair, `d_logp = logp_native(hot | predicted) − logp_native(ctl | predicted)`. Negative
  mean `d` is the hypothesised direction (hotspots harder).
- **Complex-level bootstrap**, 10,000 replicates, seed 20260803 (PREREG.md §6), percentile 95% CI.
- Reported for every committed variant; **PRIMARY = `PRIMARY_loose_null`** (verdict tier per
  PREREG §4), with **SECONDARY_B_any_interface** carried for power exactly as in the crystal study.
- Headline is the **paired predicted−crystal difference** per variant, where the crystal arm is the
  committed pairs' own `d_logp` **and** the crystal arm recomputed on this environment (they must
  agree — positive control §6).
- Secondary (reported, not verdict-bearing): pairs **re-matched on predicted burial/SS/nbr** with the
  same constraints, answering "if a designer matched on the predicted backbone, is there a gap?"

## 4. Analysis 2 — KL detector AUROC on predicted backbones

- `KL_i = KL( p(· | predicted complex backbone) ‖ p(· | chain-deleted predicted backbone) )`, both
  from ProteinMPNN unconditional passes — the identical construction as `src/kl_detector.py`, with
  the monomer `Q` obtained by **chain-deletion of the predicted complex** (not a re-predicted apo
  monomer), matching the crystal pipeline exactly.
- Label = strict hotspot (Ala ΔΔG > 2). Interface set and burial baseline (`−rSASA_complex`) are
  computed on the **predicted** backbone (the design-time realistic choice).
- AUROC with **complex-level bootstrap**, 2,000 replicates, seed 20260803: report burial baseline,
  KL alone, and burial+KL (rank-average), plus the **paired** `ΔAUROC = AUROC(burial+KL) −
  AUROC(burial)` with its bootstrap CI and `P(Δ>0)` (`src/kl_analysis.py` Q1 machinery).
- Headline is `ΔAUROC-over-burial` on **predicted** vs the same quantity on **crystal**, both
  recomputed here for an apples-to-apples drop.

## 5. Pre-registered readings (fixed before running)

1. **KL's ΔAUROC-over-burial survives on predicted backbones (CI still excludes 0):** C5 is a real
   design-time signal, not a native-crystal artifact. **Strong outcome.**
2. **It collapses to zero:** C5 is a property of crystal backbones only — still a finding about what
   native backbones encode, but not a design-time tool.
3. **A burial-matched hotspot deficit APPEARS on predicted backbones where it was absent on
   crystals** (magnitude anticipated ~0.3–0.5 nats from the crystal `d_bind_local`): the project's
   **central positive result** — the factorization tax is real precisely when the backbone is not
   native — reframing the paper around the conditioning set.

## 6. Positive control (env + code fidelity) — gates everything downstream

- `src/validate.py` prints ALL PASS on this environment. **[done: ALL PASS, recovery 0.650, SASA
  0.000%, ΔΔG D39A 6.79 — matches committed]**
- The pipeline re-run on **crystal** backbones reproduces the committed KL AUROC and burial-matched
  gaps on this environment (max |ΔKL| vs committed within numerical tolerance). The crystal re-run is
  the baseline arm of every predicted−crystal delta; no predicted-backbone delta is reported until
  the crystal arm reproduces.

## 7. Kill / caveat (pre-registered) — memorization

AlphaFold-class predictors near-memorize these pre-2021 complexes. Therefore:

- **Report pLDDT, pTM, ipTM, and backbone RMSD-to-crystal** (global and interface-restricted) per
  complex, from OpenFold3's own confidence outputs and a Kabsch fit.
- **Stratify every headline** (Analysis 1 gap; Analysis 2 ΔAUROC) by prediction confidence — at
  minimum a high-vs-low split on interface pLDDT / pTM and on interface RMSD-to-crystal.
- A signal that appears **only** at low confidence, or that **vanishes as RMSD→0**, is reported with
  that dependence stated. "Predicted" must not silently mean "reconstructed crystal"; equally, near-
  perfect reconstruction is the **conservative** regime for reading 3 (a deficit that appears there
  is the least confoundable).

## 8. Deviations / open operational items, declared up front

- The exact OpenFold3 MSA source (server vs local mmseqs) and template flag are set at runtime and
  **recorded** in `results/expA_predict_config.json`; the template-free requirement (§1) is
  non-negotiable and is verified against the emitted config.
- Complexes OpenFold3 cannot predict (length caps, failures) are **listed** with the reason and
  excluded; the excluded set is reported, never silently dropped (CLAUDE.md ground rule 6).

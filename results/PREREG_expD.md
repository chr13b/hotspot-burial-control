# Pre-registration — Experiment D: second-predictor discrimination + symmetric leverage check

Written 2026-08-11 on Sherlock, **before any AF2-multimer structure was predicted** and **before any
Task-2 jackknife number was computed**. Companion to `PREREG_expA.md` (predicted-backbone transfer)
and `PREREG.md` (Phase 0/1), whose scoring and statistics choices are inherited verbatim unless
overridden here. Fixes every operational choice so no variant can be chosen after seeing a number.

## 0. The question

Experiment A found that a **burial-matched hotspot log-prob deficit appears on OpenFold3-predicted
backbones** (SECONDARY_B `d_pred = −0.191 [−0.373, −0.004]`; paired vs crystal `−0.154 [−0.279,
−0.028]`) where none exists on native crystal backbones, and is **as large at high prediction
confidence as at low** (interface RMSD ~1 Å). Experiment C2 found it does **not** appear on
RFdiffusion partial-diffusion (noised-crystal) backbones at comparable iRMSD.

**The open objection (adversarial review):** the Exp A deficit may be **OpenFold3-specific** —
an artifact of *that* architecture (AF3-class: pairformer + diffusion decoder) or its particular
memorization of these pre-2021 complexes — rather than a general property of a backbone that was
**independently reconstructed** instead of carved by the native side chains.

**Experiment D discriminates with a second, architecturally independent predictor.** A second task
answers a distinct, self-inflicted charge of **pre-registration asymmetry** (below).

## 1. Structure predictor (Task 1)

- **AF2-multimer** via **localcolabfold** (`colabfold_batch`, AlphaFold-Multimer `_multimer_v3`
  weights; Evoformer + **regression** IPA structure module — **not** a diffusion decoder). This is
  the maximally architecture-independent second predictor available: OpenFold3 is AF3-class
  (pairformer + diffusion), so a deficit shared between OF3 and AF2-multimer cannot be an
  AF3-diffusion-family artifact.
  - **Not pre-installed on this cluster** (verified: `ml spider colabfold`/`alphafold` → "Unable to
    find"; no group/scratch install). localcolabfold is installed from scratch onto `$SCRATCH`
    (conda env + jax/CUDA + AF params), analogous to how OpenFold3 was stood up for Exp A. The exact
    install (installer version, jax/jaxlib/CUDA versions, params release) is recorded in
    `results/expD_predict_config.json`. **This install is a pre-data operational step, declared here
    before any structure is predicted (§9).**
- **Targets:** the same **141** complexes in `results/pair_complexes.txt` as Exp A.
- **Input sequences:** the **identical per-chain sequences already used for Exp A**
  (`results/expA_queries.json`, extracted by `ftax_common.load_complex` from the SKEMPI cleaned PDB).
  Reusing the exact same inputs makes the predicted↔crystal residue correspondence 1:1 and positional
  (predicted residue *j* of chain *X* ↔ crystal `(chain X, resnum, icode)` at position *j*), so the
  committed matched pairs and SKEMPI labels transfer without alignment ambiguity, and OF3 vs AF2 is a
  same-input comparison.
- **No structural-template leakage.** AF2-multimer is run **template-free** (`--templates` OFF; no
  deposited coordinates fed as a template). MSAs (ColabFold `mmseqs2_uniref_env` server — sequence
  homologs, **no coordinates**) are permitted, exactly as Exp A's OF3 run. The template flag and MSA
  mode are recorded from the emitted config and **verified OFF** before any number is reported.
- **Seeds:** fixed (recorded). Per complex the **top-ranked model** (by AF2-multimer's own ranking =
  0.8·ipTM + 0.2·pTM) is the primary predicted backbone; all models retained.
- **Coordinates used in scoring are the AF2 predicted coordinates only.** A Kabsch fit to the crystal
  is computed **solely** to report interface/global Cα-RMSD-to-crystal (a leakage/confidence
  diagnostic) and is **never** fed to ProteinMPNN. No Amber relaxation (ProteinMPNN scores backbone
  N/CA/C/O only; relax is off to avoid a confound and an OpenMM dependency).

## 2. Scoring (inherited from PREREG_expA.md §2, applied to the AF2 backbone)

- ProteinMPNN `vanilla_model_weights/v_48_020.pt`, `augment_eps = 0.0`.
- Per-position teacher-forced conditional log-prob of the **native** residue, mean over **8 decoding
  orders** (seeds 0–7); order spread reported.
- Order-free unconditional (backbone-only) log-prob as the KL detector's distribution.
- `rSASA_complex`, `rSASA_free`, ΔrSASA, secondary structure (**pydssp**), neighbour count recomputed
  on the **AF2 predicted structure**. Native residue identity is unchanged (SKEMPI/crystal wild type);
  only geometry changes.

## 3. Analysis 1 — burial-matched hotspot gap on AF2 backbones

- **Reuse the identical committed pydssp matched pairs** `results/p0_dssp_pairs_{VARIANT}.csv`, keyed
  by `(complex_id, chain, resnum)` — the SAME pairs Exp A reused (matched on **crystal** burial / SS /
  neighbour count). Reusing them isolates the effect of swapping the backbone on the identical
  hot/control residue set, and makes AF2 vs OF3 an apples-to-apples comparison.
- For each pair, `d = logp_native(hot | AF2) − logp_native(ctl | AF2)`. Negative mean `d` is the
  hypothesised direction (hotspots harder).
- **Complex-level bootstrap**, 10,000 replicates, seed 20260803, percentile 95% CI (`src/expA_gap_reuse_pairs.py`).
- **Verdict tier = `SECONDARY_B_any_interface`** — the highest-powered tier, on which Exp A's central
  −0.19 deficit was measured; the D-PERSIST / D-VANISH readings are read from it, for a like-for-like
  comparison to the very number the objection concerns. **All committed tiers are reported**
  (PRIMARY_loose_null, strict, SECONDARY_A, HYDROMATCHED, SENS_nbr_tol2, AAMATCHED); the pre-registered
  PRIMARY_loose_null tier is underpowered and carried a positive-side quirk in **both** the crystal and
  Exp A runs — it is reported, not verdict-bearing, exactly as documented in FINDINGS_expA §3.
- Headlines: (i) `d_af2` per tier; (ii) the paired **AF2−crystal** delta (crystal arm = committed
  pairs' own `d_logp`, and the crystal arm recomputed on this env — they must agree, §6); (iii) the
  **AF2 vs OF3** comparison per tier (same pairs, both predicted backbones).

## 4. Analysis 2 — KL detector AUROC on AF2 backbones

- `KL_i = KL( p(· | AF2 complex backbone) ‖ p(· | chain-deleted AF2 backbone) )`, both from
  ProteinMPNN unconditional passes — identical construction to `src/kl_detector.py`, monomer `Q` by
  **chain-deletion of the AF2 complex** (not a re-predicted apo monomer), matching crystal/Exp A exactly.
- Label = **canonical `label=="hot_strict"`** (Ala ΔΔG > 2), the same set `kl_detector.py` uses and the
  set the corrected C2-KL used (NOT the matched-pairs subset). Interface set and burial baseline
  (`−rSASA_complex` / Cβ neighbour count) computed on the **AF2** backbone.
- AUROC with **complex-level bootstrap**, 2,000 replicates, seed 20260803: burial baseline, KL alone,
  burial+KL (rank-average), and the paired `ΔAUROC = AUROC(burial+KL) − AUROC(burial)` with CI and
  `P(Δ>0)` (`src/expA_kl_delta.py`).
- Headline: `ΔAUROC-over-burial` on **AF2** vs the same quantity on **crystal** and **OF3** (Exp A),
  all recomputed/committed here.

## 5. Pre-registered readings (fixed before running)

- **D-PERSIST.** The burial-matched deficit's 95% complex-bootstrap CI **excludes zero** on AF2
  backbones (SECONDARY_B, point estimate comparable to OF3's −0.19) → the deficit is a **general**
  property of independently-reconstructed backbones, **not** OpenFold3-specific → the conditioning-set
  headline (FINDINGS_expA Result 3) holds and is ICLR-ready.
- **D-VANISH.** The deficit CI **contains zero** on AF2 while present on OF3 → the deficit is
  **OpenFold3-architecture-specific** → report as such and **BOUND / withdraw the log-prob-deficit
  generality claim**. The paper's ICLR core then rests on Results 1+2 (the burial correction and the
  KL detector), which are unaffected. **D-VANISH is a-priori as publishable as D-PERSIST** — an honest
  null that bounds a claim, in the spirit of this project's F0.
- **D-KL.** The KL `ΔAUROC-over-burial` CI **excludes zero** on AF2 backbones (expected if KL is
  general: crystal +0.048, OF3 +0.062, generative +0.06–0.07). Report.
- **POSITIVE CONTROL / most-informative readout.** (a) The crystal arm reproduces the committed
  within-complex null and KL AUROC on this env (gate; already established in Exp A). (b) Templates-off
  is verified from the AF2 config. (c) The deficit is stratified by AF2 pTM / interface-pLDDT /
  interface-RMSD. (d) **The single most informative readout:** the **per-complex correlation of the AF2
  deficit with the OpenFold3 deficit** across the shared complexes — if the **same complexes** carry
  the deficit under both predictors, that is strong evidence of a general signal; if the two are
  **disjoint**, the per-predictor deficits are predictor-specific noise. Report Spearman + Pearson with
  complex-bootstrap CI.

## 6. Positive control (env + code fidelity) — gates everything downstream

- `src/validate.py` prints ALL PASS on this environment (established: recovery 0.650, SASA 0.000%,
  ΔΔG D39A 6.79).
- The pipeline re-run on **crystal** backbones reproduces the committed KL AUROC and burial-matched
  gaps (Exp A established: KL ΔAUROC +0.0484 vs committed +0.048; every gap tier's crystal `d_cry`
  reproduces committed `d_logp` to max|Δ| = 4.4e-16). This crystal arm is the baseline of every
  AF2−crystal delta; no AF2 delta is reported until the crystal arm reproduces.
- **Same-input positive control:** AF2 and OF3 receive the byte-identical per-chain sequences
  (`results/expA_queries.json`); the AF2 residue re-keying must match SKEMPI wild type at 100% of
  mapped positions (as OF3 did: 2167/2167), or the complex is excluded and reported.

## 7. Task 2 — the SYMMETRIC leverage check (CPU, no prediction)

Defuses a self-inflicted asymmetry charge: Exp A's `−0.19` slope-of-nothing was *accepted* while Exp
C2's pre-registered slope-fire was demoted as a near-crystal **leverage artifact** — and the leverage
diagnostic (`src/expC2_slope_diag.py`) was applied only to C2. Apply the **same** robustness scrutiny
to Exp A's deficit, and (for visible symmetry) to C2's gap.

- **Exp A.** Per-pair predicted gaps `d = logp(hot|OF3) − logp(ctl|OF3)` on the committed SECONDARY_B
  pairs (`$SCRATCH/ftax/predicted/expA_p0_positions.csv`). **Jackknife over complexes:** recompute the
  pooled paired mean dropping each complex; report the **signed influence distribution** (full mean −
  leave-one-complex-out mean). Then **drop the top-3 and top-5 most-influential complexes** (those that
  most *support* the deficit, i.e. most negative influence) and recompute the complex-bootstrap 95% CI
  on the reduced set.
- **Exp C2.** The identical jackknife on C2's within-binder interface-formed generative gap
  (`results/expC2_gap_perbackbone.csv`, `interface_ok==1 & partial_T>0`, per-complex aggregated), for
  visible method symmetry. (C2's point estimate is positive; drop the top-k that most support *its*
  sign.)
- Output: `results/expD_leverage.csv` (full estimate + CI, per-complex influence, drop-top-k CIs).
- **READING (D-LEVERAGE).** If Exp A's SECONDARY_B deficit **survives leave-3/5-complexes-out** (CI
  still excludes zero, or the point estimate holds its sign and magnitude with the CI only marginally
  crossing) it is **not** a leverage artifact → the asymmetry charge is answered and Exp A's −0.19 is
  reported as robust. If it does **not** survive → say so plainly and **demote** the −0.19 to the same
  "carried by a few complexes" status as C2's slope. Fixed before the number is seen.

## 8. Kill / caveat (pre-registered) — memorization

AF2-multimer, like OF3, near-memorizes these pre-2021 complexes (AF-Multimer training set predates
2021). Therefore, exactly as PREREG_expA §7: report pTM / ipTM / interface-pLDDT / Cα-RMSD-to-crystal
(global + interface) per complex; **stratify every headline** (Analysis 1 gap; Analysis 2 ΔAUROC) by
prediction confidence (high-vs-low split on interface pLDDT / pTM / interface RMSD). A signal appearing
**only** at low confidence, or vanishing as RMSD→0, is reported with that dependence. Near-perfect
reconstruction is the **conservative** regime for D-PERSIST (a deficit there is least confoundable).
Leakage is symmetric with Exp A and does not distinguish D-PERSIST from D-VANISH — both predictors have
seen these complexes — so the discrimination is **architecture**, not leakage.

## 9. Deviations / open operational items, declared up front

- **Predictor availability (pre-data).** The task named "AF2-multimer (ColabFold)"; ColabFold/AF2 are
  **not installed** on this cluster (verified via `ml spider`). Resolution, chosen before any structure
  was predicted: **install localcolabfold from scratch** (the user's explicit decision, matching the
  spec and maximizing architectural independence vs the already-installed AF3-class Chai-1). Recorded
  here as a pre-data deviation; the install provenance goes to `results/expD_predict_config.json`.
- The exact AF2 MSA source (ColabFold `mmseqs2_uniref_env` server) and the template flag (OFF) are set
  at runtime and **recorded**; template-free (§1) is non-negotiable and verified against the config.
- Complexes AF2-multimer cannot predict (length/memory caps, failures) are **listed** with the reason
  and excluded; the excluded set is reported, never silently dropped (CLAUDE.md ground rule 6). If the
  AF2-scored complex set differs from OF3's, the AF2−OF3 comparison and the per-complex correlation are
  computed on the **intersection**, with the intersection size stated.
- Bootstrap seed 20260803 throughout (10,000 gap replicates, 2,000 KL replicates), matching Exp A.

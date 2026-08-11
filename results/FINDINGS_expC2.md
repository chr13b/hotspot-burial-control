# FINDINGS — Experiment C2: the tax on properly-sampled generative backbones, and a pinning correction

Run 2026-08-10/11 on Sherlock. Analysis choices were fixed in [`PREREG_expC2.md`](PREREG_expC2.md)
**before any C2 backbone**; the two mid-course changes (the 10 Å contact cutoff and the drop of the
`hotspot_res` fix) are pre-data / validation-stage deviations, each recorded in PREREG_expC2 §8–§9 with its
evidence, and **no verdict statistic was observed before it was fixed**. Every number below traces to a
committed CSV.

> **VERDICT — a clean split, plus a correction to Exp C.**
>
> **(A) Methodological, strong — the pre-registered "fix" is REFUTED.** Pinning the binder with
> `ppi.hotspot_res` does not prevent divergence, it **causes** it: the 3 complexes validated (1A22, 1H9D,
> 5M2O), which Exp C docked cleanly at interface Cα-RMSD **1.0–6.2 Å**, all blew up to **10³–10⁶ Å** once
> hotspots were specified (interface_ok **0/12**; the only difference vs Exp C is `ppi.hotspot_res`). This
> **corrects Exp C's FINDINGS**, which blamed its 62–75 % divergence on the *absence* of hotspot residues —
> in fact **21/55 complexes dock with no pinning at all**, and specifying hotspots is what destroys docking.
>
> **(B) C2-KL FIRES — the sequence-free detector survives on generative backbones (the positive).** Using
> the strict (ΔΔG>2) interface-hotspot set that defines the project's canonical KL detector
> (`kl_detector.py`, `label=="hot_strict"`), the burial-orthogonal ΔAUROC-over-burial excludes zero at
> **every** physical level: **+0.072 / +0.073 / +0.071 / +0.070 / +0.059** at `partial_T` 5/10/15/20/30
> (n_cx 20–21). The residue-agnostic hotspot signal now holds across **crystal → predicted (Exp A) →
> generative** backbones.
>
> **(C) C2-PRIMARY — no burial-matched log-prob deficit in the physical generative regime (the null).** The
> pre-registered slope statistic technically fired (physical slope **−1.25 [−2.14, −0.18]**) **but is a
> near-crystal leverage artifact**: it collapses to **−0.26 (p=0.61)** once iRMSD ≥ 1.5 Å and Theil-Sen
> flips to **+0.14** by 2 Å; the robust, binned gap is **flat and positive** across iRMSD 1–8 Å
> (**+0.17 … +0.55** — hotspots slightly *easier*, like the crystal). So the Exp A predicted-backbone deficit
> (−0.19) does **not** extend to partial-diffusion backbones. **KILL C2a passed** (T0 within-binder
> **+0.303 [−0.19, +0.80]**, identical to the committed crystal). **KILL C2b moot** (no conditioning).
> **SECONDARY**: ProteinMPNN's ΔΔG_bind rank-correlation collapses **−0.236 → ≈ −0.08** off-manifold.
>
> **Net.** The **burial-orthogonal KL signal generalises to generative design**, but the **log-prob tax is
> prediction-specific**: a burial-matched deficit appears on OpenFold3-*predicted* backbones (Exp A) and not
> on RFdiffusion partial-diffusion backbones at comparable iRMSD, because the latter are *noised crystals*
> that keep native character. Partial diffusion cannot reach a physical *de-novo design* regime — it yields
> near-crystal (docked, behaves like the crystal) or divergent (10³–10⁷ Å, unscoreable) backbones — and the
> pinning route that might have reached one is the one that breaks docking.

## 1. What was run

```bash
# Pre-registration (before any backbone) + setup; 10A pre-data cutoff correction; validation; pivot.
#   PREREG_expC2.md (da00c16) ; setup+10A (2e4125e) ; validation pivot (ea4ac7a)
python3 src/expC2_hotspot_res.py --cutoff 10.0        # 50/50 split + ppi.hotspot_res (later dropped)
#   (C2-PRIMARY slope computed in finalize by src/expC2_slope_check.py -> results/expC2_slope_check.csv;
#    the committed Exp C continuous-slope decomposition src/expC_slope_check.py -> results/expC_slope_check.csv
#    — merged from the Exp C write-up — already gives the flat generated-physical slope -0.009 [P=0.52],
#    which this run's leverage analysis (§4) independently confirms.)

# VALIDATION (Option A) -> REFUTED: hotspot_res breaks docking (results/expC2_pinning_validation.csv)
OUTDIR=$SCRATCH/expC2/validate ONLY_CIDS="1A22_A_B 1H9D_A_B 5M2O_A_B" TLEVELS="5 20" NDES=2 \
    sbatch --array=0-0 expC2_run.sbatch     # all 12 diverged (iRMSD 1e3-1e6 A)

# SALVAGE ladder (PI decision): denser, NO hotspot_res, Complex_base (the config that docks 21/55).
USE_HOTSPOTS=0 TLEVELS="5 10 15 20 30" NDES=6 sbatch --array=0-9%10 expC2_run.sbatch   # job 38553431
sbatch --dependency=afterany:38553431 --array=0-7%8 expC2_score_array.sbatch           # 38553432
sbatch --dependency=afterany:38553432 expC2_finalize.sbatch                            # 38553433
python3 src/expC2_kl_loose.py       # canonical (label==hot_strict) + loose interface-formed KL -> C2-KL
python3 src/expC2_slope_diag.py     # slope leverage-artifact diagnostic
```

1705 backbones scored (55 T0 crystal controls + 5 levels × 330 noised). Bootstrap seed 20260803
(10,000 gap / 2,000 KL reps). Numbers trace to `results/expC2_{dose,rematch_dose,slope_check,leakage,
secondary,interface_qc,gap_perbackbone,kl_hotpoor,pinning_validation}.csv`.

## 2. The pinning "fix" breaks docking — a correction to Exp C (`results/expC2_pinning_validation.csv`)

`ppi.hotspot_res` was passed correctly (11–26 target tokens each). Result vs the identical complexes in
Exp C (same inputs, contig, checkpoint; no `hotspot_res`):

| complex | Exp C, no pinning — iRMSD / interface-formed | C2, hotspot_res ON (10 Å) — iRMSD / interface-formed |
|---|---|---|
| 1A22_A_B  T5 / T20 | 1.0–1.3 / 2.4–3.6 Å · **12/12** | **1.3×10³ / 1.6×10⁶ Å · 0/12** |
| 1H9D_A_B  T5 / T20 | 1.3–1.7 / 3.2–6.2 Å · **10/12** | 1.2×10³ / 1.4×10⁶ Å · 0/12 |
| 5M2O_A_B  T5 / T20 | 0.9–1.1 / 2.5–4.5 Å · **10/12** | 1.4×10³ / 1.5×10⁶ Å · 0/12 |

The only functional difference is `ppi.hotspot_res`, so passing it to `Complex_base` under partial diffusion
**causes** catastrophic divergence. Consequences: (i) Exp C's stated cause of divergence was backwards —
21/55 complexes dock with *no* hotspot residues; (ii) Exp C's real deficiency was **undersampling** the
docked iRMSD 1–8 Å regime (N=3, 13 complexes), which C2 fixes with N=6. Per the pre-registered >40 %-
divergence rule, Option A was dropped for the PI-approved no-pinning ladder (PREREG_expC2 §9).

## 3. Generation reality — small, near-crystal, matched-pair-poor docked subset (`results/expC2_interface_qc.csv`)

| partial_T | produced | interface-FORMED | median tgt-RMSD | median iRMSD (interface-formed) |
|---|---|---|---|---|
| 0 (crystal) | 55 | 50 / 55 (0.91) | 0.0 Å | 0.0 Å |
| 5  | 330 | **119 / 330 (0.36)** | 0.13 Å | ~1.1 Å |
| 10 | 330 | **120 / 330 (0.36)** | 0.13 Å | ~1.5 Å |
| 15 | 330 | **121 / 330 (0.37)** | 0.13 Å | ~2.1 Å |
| 20 | 330 | **118 / 330 (0.36)** | 0.13 Å | ~3.6 Å |
| 30 | 330 | **98 / 330 (0.30)** | 0.13 Å | ~9.2 Å |

No pinning → ~36 % interface-formed (as in Exp C), but N=6 gives ~120 docked backbones/level. The target is
held perfectly (tgt-RMSD 0.13 Å). Interface-formed backbones are **near-crystal** (median iRMSD 1–9 Å;
noised crystals, not de-novo designs) over ~20 complexes carrying strict interface hotspots (§5).

## 4. Reading C2-PRIMARY — the pre-registered slope FIRED but is a near-crystal leverage artifact

`d = logp(hotspot) − logp(burial-matched control)`, within-binder SECONDARY_B pairs, interface-formed
backbones. Pre-registered statistic = slope of `d` vs `log10(interface-RMSD)`, interface-formed, iRMSD ≤ 8 Å,
`partial_T ≠ 0` (`results/expC2_slope_check.csv`): **−1.248 [−2.143, −0.184]**, p_boot 0.020 (n_bb 268,
n_cx 11). Naive all-backbone slope −0.026 (not evidence, per pre-reg). Taken literally C2-PRIMARY fires, but
it does not survive scrutiny (`src/expC2_slope_diag.py`):

| iRMSD floor | OLS slope | p | Theil-Sen |  | iRMSD bin | interface-formed gap d [95 % CI] |
|---|---|---|---|---|---|---|
| ≥ 0.5 Å | −1.248 | 0.0001 | −1.382 |  | <1 Å | +0.168 [−0.269, +0.588] |
| ≥ 1.0 Å | −0.840 | 0.024 | −0.946 |  | 1–2 Å | +0.374 [−0.002, +0.760] |
| **≥ 1.5 Å** | **−0.262** | **0.61** | −0.106 |  | 2–3 Å | +0.372 [−0.152, +0.877] |
| ≥ 2.0 Å | −0.186 | 0.79 | **+0.136** |  | 3–5 Å | +0.022 [−0.449, +0.522] |

The whole slope is carried by near-crystal T5 designs at iRMSD 0.6–1.5 Å with extreme `d` (e.g. 1LFD_A_B T5
at iRMSD ~0.7 Å, d = +2.7…+3.4). Remove iRMSD < 1.5 Å → non-significant; per-complex own-slopes are
inconsistent (median −0.27, range −2.68…+1.61). The binned gap is **flat and positive** — no deficit, on the
*opposite* side from one. Per-level interface-formed gaps are all positive too: T5 **+0.549 [+0.055,+1.079]**,
T10 +0.386, T15 +0.251, T20 +0.399, T30 +0.420 (`results/expC2_dose.csv`; re-match agrees). **Neither slope
reading fires cleanly** (C2-PRIMARY's positive is an artifact; the robust slope is too imprecise for a formal
±0.10 TOST), but the **substantive C2-NULL outcome holds: no burial-matched deficit in the physical
generative regime.**

## 5. Reading C2-KL — the sequence-free detector SURVIVES on generative backbones (`results/expC2_kl_hotpoor.csv`)

KL detector = `KL(p(·|generated complex) ‖ p(·|chain-deleted))`, burial = Cβ neighbour count; metric =
paired ΔAUROC = AUROC(burial+KL) − AUROC(burial) for strict hotspots over interface positions,
complex-bootstrapped, on interface-formed backbones:

| partial_T | 0 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| **strict ΔAUROC [95 % CI]** | +0.079 [+0.046,+0.114] | **+0.072 [+0.035,+0.110]** | **+0.073 [+0.038,+0.110]** | **+0.071 [+0.040,+0.111]** | **+0.070 [+0.037,+0.106]** | **+0.059 [+0.014,+0.117]** |
| n_cx / n_hot | 50 / 122 | 20 / 56 | 21 / 57 | 21 / 57 | 21 / 57 | 21 / 57 |
| loose ΔΔG>1 (exploratory) | +0.058 | +0.060 | +0.057 | +0.045 | +0.054 | +0.053 |

**Every physical-level CI excludes zero → C2-KL fires**, and the loose (ΔΔG>1) set agrees. The KL detector
adds ~+0.07 AUROC over burial on docked generative backbones — the reviewer-critical validity condition for
the main study's finding C5, now met on crystal, predicted (Exp A +0.062), *and* generative backbones.

> **Implementation note (disclosed in full).** `src/expC_analyze.py`'s KL step keyed `is_hot` off the
> **matched-pairs** strict set (`p0_dssp_pairs_strict_hot2_null`, which requires a burial-matched null
> control per complex) rather than the canonical `label=="hot_strict"` used by `src/kl_detector.py`. On C2's
> draw only 2 such matched-pairs hotspots land on interface-formed noised backbones, so `expC_analyze` alone
> returned no value at T5–30 (`src/expC2_kl_debug.py`). The numbers above use the **canonical** label-based
> set (`src/expC2_kl_loose.py`) — the same definition as `kl_detector.py` — which is the pre-registered KL
> detector. This also means Exp C's reported C-KL (matched-pairs subset, +0.083–0.094) fired on a fortunate
> draw; the canonical computation is the robust one, and C2 confirms it fires.

**KILL C2a — PASSED.** `partial_T=0` reproduces the committed within-binder deficit exactly:
**+0.303 [−0.193, +0.800]**, CI contains zero. **KILL C2b — MOOT** (no conditioning applied; run for
completeness the arms are indistinguishable — paired diff −0.73 [−1.63, +0.16], n_cx 2).

## 6. Secondary — binding-energy ranking collapses off the native manifold (`results/expC2_secondary.csv`)

Partial-ρ(experimental SKEMPI ΔΔG_bind, ProteinMPNN log-odds `ℓ(mut)−ℓ(wt)` | burial), 1088 single
mutations / 55 complexes:

| partial_T | 0 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| partial-ρ | **−0.236** [−0.33,−0.16] | −0.089 [−0.16,−0.00] | −0.085 [−0.16,+0.01] | −0.079 [−0.15,+0.04] | −0.071 [−0.14,+0.04] | −0.089 [−0.15,+0.02] |

The model's ability to rank experimental binding energy collapses toward zero the moment the backbone leaves
the native manifold (−0.236 → ≈ −0.08), as in Exp C and as pre-registered (exploratory, no falsifier).

## 7. What this means, and the honest caveats

1. **The KL detector generalises; the log-prob tax does not.** The residue-agnostic, burial-orthogonal KL
   signal survives on crystal, predicted, and generative backbones (+0.05…+0.10 throughout) — it is the
   robust, transferable design-time signal. The burial-matched *log-prob* deficit, by contrast, appears only
   on OpenFold3-*predicted* backbones (Exp A, −0.19 at iRMSD ~1.3 Å) and **not** on partial-diffusion
   backbones at comparable iRMSD (C2, +0.3). The distinguishing variable is the **type** of non-nativeness:
   predictions are independent reconstructions; partial-diffusion backbones are noised crystals that retain
   native character. The tax tracks prediction-type deviation, not distance-from-native per se.
2. **Partial diffusion cannot reach a physical de-novo design regime.** It produces near-crystal (docked,
   behaves like the crystal) or divergent (10³–10⁷ Å, unscoreable) backbones — nothing far-from-native and
   physical. C2 bounds the log-prob tax to prediction; it cannot, with this generator, test a true design
   regime.
3. **The pinning route that might have reached a design regime is the one that breaks docking** (§2). A
   different generator/protocol would be needed to make binders drift far while staying physical.
4. **Power / scope, stated plainly.** The physical-regime gap rests on **11** within-binder complexes; the
   pre-registered slope statistic is artifactual; C2-KL rests on 20–21 complexes and is robust. The
   log-prob null is clear in sign but modest in n. Reported as bounded, not upgraded.
5. **Coverage.** 1705 backbones produced and scored; no `nan`-coordinate exclusions this run (unlike Exp C's
   T40 tail). Interface-dissolved backbones dropped per pre-reg with counts in §3.

### Files
- `results/expC2_kl_hotpoor.csv` — C2-KL: canonical-strict + loose interface-formed ΔAUROC by level (the reading).
- `results/expC2_dose.csv` / `results/expC2_rematch_dose.csv` — burial-matched gap (all / interface-formed / by-iRMSD).
- `results/expC2_slope_check.csv` (`src/expC2_slope_check.py`) — C2-PRIMARY physical + naive slope; `results/expC_slope_check.csv` (`src/expC_slope_check.py`, merged Exp C decomposition) — the flat generated-physical slope (−0.009) this run independently confirms.
- `results/expC2_secondary.csv` — ΔΔG_bind partial-ρ by level; `results/expC2_interface_qc.csv` — interface-formed per level.
- `results/expC2_leakage.csv` — KILL C2b (moot); `results/expC2_pinning_validation.csv` — hotspot_res-breaks-docking evidence.
- `results/expC2_gap_perbackbone.csv` — per-backbone audit trail (the slope input). Backbones + scored tables on `$SCRATCH/expC2/` (gitignored).

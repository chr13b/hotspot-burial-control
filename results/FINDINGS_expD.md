# FINDINGS — Experiment D: second-predictor discrimination, and a symmetric leverage check

Run 2026-08-11 on Sherlock. Analysis choices were fixed in [`PREREG_expD.md`](PREREG_expD.md)
**before any AF2-multimer structure was predicted** and **before any Task-2 jackknife number was
computed**; no reading was moved. Every number below traces to a committed CSV or one on `$SCRATCH`.

> **VERDICT — D-PERSIST fires: the burial-matched deficit is GENERAL, not OpenFold3-specific.**
> A second, architecturally-independent predictor (**AF2-multimer**: Evoformer + regression IPA, not
> AF3-class diffusion) **reproduces** the Exp A deficit, and — the decisive readout — the two
> predictors' **per-complex deficits correlate strongly**. The paper's Result 3 (the conditioning-set /
> predicted-backbone log-prob deficit) survives the architecture-specificity objection.
>
> **D-PERSIST (fires).** On the identical committed pydssp pairs, AF2-multimer's highest-powered tier
> (SECONDARY_B, 382 pairs / 127 cx) gives **d_af2 = −0.233 [−0.440, −0.035]** (CI excludes zero),
> essentially reproducing OpenFold3's **−0.191 [−0.373, −0.004]**; paired AF2−crystal delta
> **−0.196 [−0.358, −0.039]**. SENS_nbr_tol2 agrees (**−0.196 [−0.378, −0.018]**). Crystal control
> reproduces committed to **4.4e-16**.
>
> **Most-informative readout (decisive).** The **per-complex AF2-vs-OF3 deficit correlation** is
> **Spearman +0.565 [+0.395, +0.707]**, Pearson **+0.620 [+0.422, +0.766]** over 127 shared complexes:
> the **same complexes** carry the deficit under both independent architectures. This — not any single
> marginal point estimate — is the robust evidence that the deficit is a real, general property of
> independently-reconstructed backbones, not per-predictor noise.
>
> **D-KL (fires).** KL ΔAUROC-over-burial on AF2 backbones = **+0.0536 [+0.029, +0.079]**,
> P(>0)=1.000 (larger than the crystal +0.048). The sequence-free detector now holds on
> **crystal → OpenFold3 → generative (C2) → AF2-multimer** backbones.
>
> **D-VANISH — refuted.**
>
> **Task-2 leverage (D-LEVERAGE), applied symmetrically to BOTH predictors.** Each predictor's
> *single-point* deficit is marginal: neither survives dropping the 3 most-supporting complexes
> (OF3 → −0.115 [−0.29, +0.06]; AF2 → −0.135 [−0.31, +0.04], both CIs include zero). **But the
> top-supporting complexes are the SAME set under both predictors** (1JRH_LH_I, 1JTD_A_B, 1Z7X_W_X),
> both survive the |influence|-drop, and the per-complex deficits correlate at +0.57. So the "fragility"
> is not randomness — two independent reconstructions agree on *which* complexes are hard. **The honest
> claim rests on cross-predictor reproducibility, not on either −0.19/−0.23 in isolation.**

---

## 1. What was run

```bash
# AF2-multimer via the official ColabFold Apptainer container (sokrypton/colabfold:1.5.5-cuda12.2.2),
# TEMPLATES OFF, ColabFold mmseqs2 MSA server, 5 models, rank_001 kept. Same 141 complexes and the
# BYTE-IDENTICAL per-chain input sequences OpenFold3 received in Exp A (results/expA_queries.json).
apptainer exec --nv -B $SCRATCH:$SCRATCH colabfold.sif colabfold_batch <fa> <out> \
    --model-type alphafold2_multimer_v3 --msa-mode mmseqs2_uniref_env --num-models 5 --num-recycle 3
# Convert rank_001 PDB -> crystal-keyed PDB; score through the IDENTICAL Exp A pipeline; analyse.
python3 src/expD_af2_to_pdb.py --pred-dir $SCRATCH/ftax/expD/af2_out --resmap results/expA_resmap.json ...
python3 src/p0_burial_matched.py --data-dir $SCRATCH/ftax/expD ...      # ProteinMPNN v_48_020, 8 orders
python3 src/patch_ss.py ... ; python3 src/kl_detector.py ...
python3 src/expA_gap_reuse_pairs.py --pred-positions .../expD_p0_positions.csv --pairs-glob 'results/p0_dssp_pairs_*.csv' ...
python3 src/expA_kl_delta.py ... ; python3 src/expD_af2_vs_of3_corr.py ...   # gap, KL ΔAUROC, AF2-vs-OF3 corr
```

**Predictor.** AF2-multimer (`alphafold2_multimer_v3`) in the ColabFold container — chosen (over the
already-installed AF3-class Chai-1) precisely because it is **architecturally independent** of OpenFold3.
Standing it up required an Apptainer container: Sherlock is CentOS 7 (glibc 2.17; system TLS ≤ 1.2), which
broke every pip/installer route (installer renamed → 404; `pixi.sh` needs TLS 1.3; the Miniforge route hit
glibc-2.17-vs-`manylinux_2_28` wheel builds). The container brings its own glibc + TLS. Templates verified
OFF (`"use_templates": false`).

**Scale / coverage.** **141/141** complexes predicted (0 failures, no MSA-server rate-limiting);
**140** scored (1 excluded: `3SE4_B_A`, chain group empty after conversion — the same exclusion as Exp A).
SKEMPI wt matched the AF2 structure at 100% of mapped positions.

**Positive control (gates everything).** Every gap tier's crystal `d_cry` reproduces the committed pair
`d_logp` to **max|Δ| = 4.4e-16**; crystal KL ΔAUROC reproduces to +0.0484 (committed +0.048). AF2 deltas are
trustworthy.

**Prediction quality — nearly identical to OpenFold3, a clean matched comparison**
(`results/expD_confidence.csv`, n=141): median **pTM 0.850** (IQR 0.76–0.91), ipTM 0.840, avg pLDDT 91.8,
interface Cα-RMSD-to-crystal **1.33 Å** (IQR 0.72–6.23), global 2.49 Å; 67% pTM ≥ 0.8, 61% interface RMSD < 2 Å.
Compare OpenFold3 (Exp A): median pTM 0.857, interface RMSD 1.31 Å, 62% / 60%. The two predictors reconstruct
these (memorised, pre-2021) complexes to the **same quality** — leakage is symmetric and does not distinguish
D-PERSIST from D-VANISH.

## 2. D-PERSIST — the burial-matched deficit reproduces on AF2-multimer (`results/expD_gap_summary.csv`)

Reusing the **identical committed pydssp matched pairs**, `d = logp(hot) − logp(ctl)` on each backbone
(negative = hotspots harder), complex bootstrap 10,000 reps, seed 20260803:

| tier | pairs / cx | **d_af2 (AF2)** | d_of3 (Exp A) | d_cry (crystal) | delta = AF2 − crystal |
|---|---:|---|---:|---:|---|
| **SECONDARY_B_any_interface** (verdict) | 382 / 127 | **−0.233 [−0.440, −0.035]** | −0.191 | −0.042 | **−0.196 [−0.358, −0.039]** |
| SENS_nbr_tol2 | 466 / 133 | **−0.196 [−0.378, −0.018]** | −0.150 | −0.021 | −0.176 [−0.333, −0.027] |
| HYDROMATCHED | 188 / 84 | −0.191 [−0.454, +0.063] | −0.272 | −0.003 | −0.194 [−0.425, +0.029] |
| SECONDARY_A_measured_nonhot | 128 / 57 | +0.088 [−0.326, +0.462] | +0.224 | +0.337 | −0.256 [−0.518, −0.017] |
| PRIMARY_loose_null | 46 / 30 | +0.179 [−0.394, +0.638] | +0.468 | +0.420 | −0.262 [−0.731, +0.153] |
| strict_hot2_null | 21 / 16 | −0.377 [−1.17, +0.30] | +0.180 | +0.145 | −0.522 [−1.37, +0.083] |

**On the verdict tier and SENS the AF2 deficit CI excludes zero**, at a magnitude matching OpenFold3's,
where the crystal is at zero. Every tier's AF2−crystal delta is negative (four exclude zero). The
underpowered PRIMARY tier carries the same positive-side quirk documented for crystal and OF3 — reported,
not verdict-bearing, exactly as pre-registered.

**Confidence stratification — one honest difference from OpenFold3.** For AF2 the SECONDARY_B deficit is
**larger at low pTM** (ptm<0.84: −0.370 [−0.687, −0.077]) than at high pTM (ptm≥0.84: −0.116 [−0.380, +0.140],
CI includes zero). OpenFold3's deficit was **confidence-flat** (Exp A: high −0.235, low −0.145). So AF2's
deficit carries a **modest backbone-quality component** that OF3's did not — the high-pTM point estimate is
still negative but underpowered at that reduced n. This tempers, but does not overturn, the "not a pure
backbone-error artifact" reading: the overall CI excludes zero and the cross-predictor correlation (§4) holds.

## 3. D-KL — the sequence-free detector survives on AF2 backbones (`results/expD_kl_delta_summary.csv`)

`KL_i = KL(p(·|AF2 complex) ‖ p(·|chain-deleted AF2))`; canonical `label=="hot_strict"`; paired
ΔAUROC = AUROC(burial+KL) − AUROC(burial), complex bootstrap:

| arm | burial | KL | burial+KL | **ΔAUROC-over-burial** | n_cx |
|---|---:|---:|---:|---|---:|
| **crystal** (control) | 0.689 | 0.694 | 0.737 | **+0.0484 [+0.022, +0.075]** | 141 |
| **AF2 (all)** | 0.647 | 0.675 | 0.701 | **+0.0536 [+0.029, +0.079]** | 140 |
| AF2, pTM ≥ 0.85 | 0.658 | 0.743 | 0.749 | +0.0918 [+0.064, +0.117] | 73 |
| AF2, pTM < 0.85 | 0.630 | 0.590 | 0.637 | +0.0070 [−0.023, +0.047] | 67 |
| AF2, interface RMSD < 1.52 Å | 0.656 | 0.747 | 0.751 | +0.0943 [+0.068, +0.120] | 73 |
| AF2, interface RMSD ≥ 1.52 Å | 0.639 | 0.601 | 0.648 | +0.0090 [−0.022, +0.045] | 67 |

**The AF2 ΔAUROC CI excludes zero (+0.054), larger than crystal.** As on OpenFold3, KL is *stronger* on
well-predicted backbones and degrades on the poor tail — a geometry-carried signal, not a distance-from-native
one. **D-KL fires.**

## 4. The decisive readout — per-complex AF2-vs-OF3 agreement (`results/expD_af2_of3_corr.csv`)

Per complex, the burial-matched deficit is the mean over its SECONDARY_B pairs of `d`, on each predictor.
Across the 127 shared complexes:

| statistic | value |
|---|---|
| mean deficit: AF2 / OF3 / crystal | **−0.221 / −0.187 / −0.081** |
| **Spearman(d_af2, d_of3)** | **+0.565 [+0.395, +0.707]**, P(>0)=1.000 |
| **Pearson(d_af2, d_of3)** | **+0.620 [+0.422, +0.766]**, P(>0)=1.000 |

**The same complexes carry the deficit under two architecturally-independent predictors.** A per-predictor
artifact (memorization quirk, architecture bias) would give disjoint per-complex deficits; instead they
correlate at ρ = +0.57 with a CI comfortably excluding zero over 127 complexes. This is the single most
informative result of Experiment D, and it is not a marginal statistic.

## 5. Task 2 — the symmetric leverage check, now covering both predictors (`results/expD_leverage.csv`)

The pre-registered jackknife (drop the top-k complexes most *supporting* the deficit) applied identically to
OF3 and AF2:

| block | full | drop top-3 supporters | drop top-3 by \|influence\| |
|---|---|---|---|
| **OF3** SECONDARY_B (Exp A −0.19) | −0.191 [−0.373, **−0.004**] | −0.115 [−0.29, +0.06] · **not survive** | −0.173 [−0.337, −0.009] · survives |
| **AF2** SECONDARY_B (Exp D) | −0.233 [−0.440, −0.035] | −0.135 [−0.31, +0.04] · **not survive** | −0.197 [−0.375, −0.019] · survives |

**Both predictors behave identically**, which is itself the point: (i) each single-point deficit is marginal
and does not survive dropping its 3 most-supporting complexes → *neither −0.19 nor −0.23 is robust in
isolation*; (ii) neither is a single-outlier artifact (max leave-one-out influence ≈ 0.035 ≪ the estimate;
both survive the |influence|-drop); (iii) **the top-3 supporters are the SAME complexes for both predictors**
(1JRH_LH_I, 1JTD_A_B, 1Z7X_W_X). The leverage points coincide because the underlying signal — which complexes
are hard — is shared, exactly as the ρ=+0.57 correlation says. This **answers the pre-registration-asymmetry
charge** (we no longer treat Exp A's fire and C2's slope differently — both, and now AF2, get the same
scrutiny) and **reframes the claim correctly**: the deficit is established by cross-predictor reproducibility,
not by any one marginal number. (C2's within-binder gap +0.344 [incl 0] likewise fails supporters-drop —
symmetric.)

## 6. What this means, and the honest caveats

1. **The deficit is a property of INDEPENDENT RECONSTRUCTION, and it is general.** It appears on OpenFold3-
   *and* AF2-multimer-predicted backbones — two independent architectures that fold from sequence+MSA — at a
   consistent −0.19/−0.23, correlated per-complex at +0.57; it is absent on the native crystal and on
   RFdiffusion partial-diffusion (noised-crystal) backbones (Exp C2). The distinguishing variable is
   *independent reconstruction*, not distance-from-native or any single architecture. **Result 3 survives the
   architecture-specificity objection.**
2. **The robust evidence is cross-predictor, and we say so.** Any single predictor's burial-matched deficit is
   modest and marginal (§5). What is robust is that two independent predictors agree, in magnitude and per
   complex. The paper should lead with the correlation (§4) and the two-predictor replication, not with a lone
   −0.19.
3. **One asymmetry with OpenFold3, disclosed:** the AF2 deficit is somewhat backbone-quality-dependent (larger
   at low pTM; §2), whereas OF3's was confidence-flat. AF2's high-confidence deficit is negative but
   underpowered. So the "appears even on near-perfect reconstructions" claim is strong for OF3 and weaker for
   AF2 — reported, not hidden.
4. **Memorization is symmetric** (both predictors near-reconstruct these pre-2021 complexes to the same
   quality), so leakage cannot manufacture the AF2-vs-OF3 agreement; it is not what separates D-PERSIST from
   D-VANISH. The decisive leakage-free test named in the main study (post-cutoff complexes) still applies to
   the magnitude, not to the generality established here.
5. **Scope.** ProteinMPNN scores; the backbones are predictions of *known* complexes, not de-novo designs. The
   deficit is a mechanistic bracket for design-time conditioning (a non-native, independently-reconstructed
   backbone), now shown predictor-general.

### Falsifier-style readings, as pre-registered (PREREG_expD §5, §7)
| Reading | Pre-registered condition | Measured | Fires? |
|---|---|---|:--:|
| **D-PERSIST** | AF2 burial-matched deficit CI excludes 0 (≈ OF3 −0.19) | SECONDARY_B −0.233 [−0.44, −0.035]; SENS −0.196 [−0.38, −0.02] | ✅ **YES** |
| **D-VANISH** | AF2 deficit CI contains 0 while OF3's excludes | did not occur | — |
| **D-KL** | AF2 KL ΔAUROC CI excludes 0 | +0.0536 [+0.029, +0.079] | ✅ **YES** |
| **most-informative** | AF2 & OF3 per-complex deficits correlate (same complexes) | Spearman +0.565 [+0.40, +0.71] | ✅ general signal |
| **D-LEVERAGE** (Task 2) | OF3 −0.19 survives leave-3/5-out? | no (−0.115 [incl 0]); AF2 same; but shared leverage set + ρ=+0.57 | demote single estimate; **cross-predictor robust** |

### Files
`results/expD_gap_summary.csv` (Analysis 1, all tiers + pTM strata), `results/expD_kl_delta_summary.csv`
(Analysis 2 + strata), `results/expD_af2_of3_corr.csv` (+`_percomplex.csv`, the correlation),
`results/expD_confidence.csv` (pTM/ipTM/pLDDT/RMSD per complex), `results/expD_leverage.csv`
(+`_influence.csv`, symmetric OF3/AF2/C2 leverage). Large tables (`expD_p0_positions.csv`,
`expD_kl_positions.csv`) and the AF2 outputs on `$SCRATCH/ftax/expD/`, regenerable from the committed scripts
+ `results/expA_queries.json`. Seeds: bootstrap 20260803. Container: `ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2`.

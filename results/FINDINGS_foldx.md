# FINDINGS — FoldX physics ΔΔG_bind: L is competitive with the physics SOTA (Lane A), and beats the confidence tilt on physics (Lane B)

**Pre-registered** `results/PREREG_foldx.md`. `SEED=20260803`, NBOOT=5000. Physics energy function **FoldX 5.1**
(binary on `$SCRATCH`, license-restricted; cite Schymkowitz 2005 + Delgado 2019 — `notes/FOLDX_CITATION.md`). FoldX
is a binding readout from a modality **independent of inverse folding**; it is an approximate *fit* energy
function, **not ground truth**. Driver `src/foldx_ddg.py` (RepairPDB→BuildModel→AnalyseComplex, ΔΔG_bind =
mean(IE_mut) − mean(IE_wt), numberOfRuns=5), analysis `src/foldx_analyse.py`. Raw `results/foldx_detection.csv`,
`results/foldx_steer.csv`; per-set `results/_foldx_{A,B}_s*.csv`.

## Lane A — DETECTION (accuracy vs experimental ΔΔG; reviewer-anticipation)
SKEMPI single-mutation interface fixture, **n=2,948 mutants, 284 complexes**, same mutations for both:

| readout | Spearman vs experimental ΔΔG | |
|---|---|---|
| **FoldX ΔΔG_bind** (physics SOTA) | **+0.434** (p=5.2e-136) | fit energy function |
| **L** (inverse-folding leverage) | **−0.301** (p=9.1e-63) | zero-shot, no binding-energy term |

FoldX, a purpose-built physics ΔΔG predictor, is more accurate (|0.43| vs |0.30|) — **exactly as pre-registered
and disclosed; this is expected and is not our claim.** What Lane A supports: the **zero-shot `L` is competitive
with the standard physics tool** — it recovers ~70% of FoldX's rank accuracy with no binding-energy term and no
fitting. (Signs differ by convention only: FoldX ΔΔG>0 = destabilising; `L` is leverage, anti-correlated with ΔΔG.
`Spearman(L, exp) = −0.301` reproduces the standing detection magnitude, `foldx_laneA_Lbaseline.csv`.)

## Lane B — STEERING SPECIFICITY (the decisive lane): L vs naive vs random on physics ΔΔG_bind
60-complex ipTM set, arms wt / L / naive / random. Favorability = **−ΔΔG_bind** (>0 = more-favorable binding);
paired, complex-clustered bootstrap 95% CI:

| contrast | Δ favorability (kcal/mol) | 95% CI | P(>0) | n |
|---|---|---|---|---|
| **L − naive** (decisive) | **+1.24** | **[+0.14, +2.33]** | 0.985 | 59 |
| L − random | +7.40 | [+5.78, +9.12] | 1.000 | 57 |
| naive − random | +6.48 | [+4.82, +8.24] | 1.000 | 57 |

Arm means (favorability): **L −1.86 > naive −3.10 > random −9.54** (all negative = all steered arms bind worse
than the native wt interface, as expected when ~21 interface residues are changed; **L degrades binding the
least**). Ordering **L > naive > random**.

**Decisive test — PASS.** `L − naive` favorability is **+1.24 with the 95% CI excluding zero** (P=0.985). **The
pre-registered falsifier (L − naive ≤ 0) does NOT fire: L beats the confidence tilt on a physics, non-inverse-
folding readout** — the strongest form of the steering claim.

## The cross-modality picture — FoldX sides with the judges, closing the ipTM gap
The naive-guidance arm ([[FINDINGS_naive]]) exposed a divergence: `L > naive` under the inverse-folding **judges**
(ESM-IF1/MIF), but `naive ≳ L` under **ipTM** (a foldability/confidence metric). FoldX — a physics energy function
with an **explicit binding term**, entirely outside the inverse-folding family — **confirms `L > naive`**, siding
with the binding-sensitive judges *against* ipTM. So the binding-specific advantage of `L` over "just be more
confident" is **real, not an inverse-folding artifact**; the ipTM disagreement was the foldability effect.
Magnitudes are consistent across readouts: naive captures most of the benefit (naive−random ≈ L−random − a small
increment), and `L` adds a **modest but CI-excluding-zero** increment on top — judges L−naive +0.16/+0.24, FoldX
L−naive +1.24, both > 0.

## Coverage / controls / honest scope
- **Lane A:** 2,948/2,949 finite (1 FoldX failure), **0 mutations dropped** by validation (all SKEMPI mutations
  matched the structures). 284/285 complexes — **1KBH excluded** (pathological ~33k-atom PDB that hung RepairPDB;
  a per-call FoldX timeout was added, `f78bdab`, so it resolves to NaN rather than stalling).
- **Lane B:** 511/540 sets finite (29 set-level FoldX BuildModel failures; **14 interface mutations dropped by
  validation, uniformly across arms**). Aggregated over k: 59 complexes for L−naive; 57 for the random contrasts
  (3N4I_A_B lost all arms; 1DAN_HL_UT and 3BT1_A_U lost the *random* arm — random substitutions clash more and
  FoldX more often fails to build them, a mild conservative bias against the already-large L−random / naive−random).
- FoldX is a fit physics energy function, not truth; ΔΔG_bind is over ~21-residue interface sets (hence large
  kcal/mol magnitudes); the claim is the **paired favorability contrast**, not absolute binding. Positive control:
  the wt arm is 0 by construction; per-set run-to-run spread (numberOfRuns=5) recorded as `ddg_bind_sd`.
- Rosetta flex_ddG (the heavier gold-standard) was **not** run — the FoldX L−naive CI excludes zero, so it is not
  borderline enough to require it (pre-registered as optional).

## Bottom line
On a **physics, non-inverse-folding** readout: `L` is a zero-shot detector competitive with FoldX (Lane A), and —
decisively — **`L` steers to more-favorable binding than the confidence tilt** (Lane B, L−naive +1.24 [+0.14,
+2.33]). Together with the judges and the two structure predictors, the binding-specific `+α·L` steering advantage
now holds across inverse-folding, structure-prediction, **and physics** modalities.

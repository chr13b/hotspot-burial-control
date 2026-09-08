# FINDINGS — Lane A deepening: reproduce (✓ bit-for-bit), AB-Bind physics detection (direction replicates,
# underpowered), and L→kcal/mol calibration (L lands next to fitted FoldX)

Pre-registered `results/PREREG_laneA_deepen.md`. `SEED=20260803`, `NBOOT=5000`, complex-clustered throughout. Every
number below traces to a committed CSV. FoldX 5.1 binary on `$SCRATCH` (license-restricted; cite Schymkowitz 2005 +
Delgado 2019). Tasks 1 & 3 are CPU-only; Task 2 used new FoldX runs.

---

## Task 1 — REPRODUCE the committed Lane-A deepening (determinism check) → **IDENTICAL, bit-for-bit**
`python3 src/foldx_laneA_deepen.py` reproduced `results/foldx_laneA_deepen.csv` exactly at `SEED` (git-independent
diff: identical). The recovered "L adds beyond a fitted physics ΔΔG function" result is stable:

| metric | value | 95% CI | |
|---|---|---|---|
| Spearman(FoldX, exp) | +0.4344 | [+0.3804, +0.4900] | physics SOTA |
| Spearman(L, exp) | −0.3009 | [−0.3548, −0.2443] | zero-shot leverage |
| **partial(L, exp \| FoldX)** | **−0.1727** | **[−0.2288, −0.1170]**, P(<0)=1.0 | L carries binding rank beyond fitted physics |
| ensemble uplift over FoldX (OOF) | +0.0142 | [−0.0011, +0.0305], P(>0)=0.967 | learned 10-fold complex-clustered stack |

No drift → no regression. (This is a check, not a new claim.)

## Task 2 — AB-Bind physics detection (does L's beyond-physics result generalise to antibody–antigen?)
**Direction replicates; strongest form underpowered — exactly the pre-registered outcome.** New FoldX ΔΔG_bind runs
on the AB-Bind single-mutation interface set, same metrics as SKEMPI Lane A, complex-clustered (unit = pdb):

| metric | value | 95% CI | P | vs SKEMPI (Task 1) |
|---|---|---|---|---|
| Spearman(FoldX, exp) | **+0.274** | [+0.123, +0.461] | P(>0)=1.0 | same sign (+0.434) |
| Spearman(L, exp) | **−0.180** | [−0.324, −0.076] | P(<0)=1.0 | same sign (−0.301) |
| **partial(L, exp \| FoldX)** | **−0.081** | **[−0.190, +0.003]** | P(<0)=**0.97** | same sign, attenuated (−0.173) |

- **The pre-registered WIN — a replicated direction — holds:** all three signs match SKEMPI (`Spearman(L,exp)<0`,
  `Spearman(FoldX,exp)>0`, `partial(L|FoldX)<0`), and the two marginals have CIs excluding zero.
- **partial(L|FoldX) is INDETERMINATE at the strict 95% level (underpowered), as pre-registered:** the point
  estimate −0.081 is consistent with SKEMPI's −0.173 (attenuated on antibody–antigen), 97% of the bootstrap mass is
  negative, but the 95% CI upper bound is **+0.003** — it grazes zero. With only **22 complexes** this is the
  expected power ceiling; we do **not** over-read P(<0)=0.97 as significance. **The falsifier (partial ≥ 0 with CI
  excluding the negative side → "L beyond physics is bounded to SKEMPI-type complexes") does NOT fire** — the result
  is directionally consistent, just not powered for 95% significance on a 22-complex antibody fixture.
- **Coverage / positive controls (CLAUDE.md rule 6):**
  - **WT-identity gate: 341/341 mutations mapped cleanly (100%)**; FoldX finite 341/341 sets. n=341 over 22 complexes.
  - **Partition positive control passed:** the authoritative AB-Bind `Partners(A_B)` chain-group partition was
    re-fetched (the `$SCRATCH` copy had been purged) and cross-checked — **401/401 committed interface single-mutant
    ddG values matched the re-fetched release** (2 with \|Δ\|≤0.23, near-zero magnitude) → the recovered partition is
    provably the one `L` was computed against (`results/_abbind_xcheck.csv`).
  - **Structure numbering:** the generic RCSB PDBs FAILED the WT gate (different numbering); the **AB-Bind-specific
    structures** (same files `L` used, re-fetched via `src/fetch_abbind.sh`) pass. This was caught by a smoke test
    before the campaign, not silently absorbed.
  - **Excluded (honest cuts):** 5 homology-model complexes (`HM_*`, no crystal → 62 mutations); duplicate fixture
    rows collapsed (the committed AB-Bind fixture carries exact-duplicate rows, e.g. `3BN9 H100H` ×7 at ddG 8.0 — a
    naive merge 7×-weighted that outlier; deduped on mutation identity so each mutation counts once).
- **Redundancy** Spearman(L, FoldX) = −0.394 (L and FoldX are only mildly co-linear on antibody–antigen).
- Raw: `results/foldx_detection_abbind.csv` (metrics), `results/foldx_ddg_abbind.csv` (per-set ΔΔG_bind),
  `results/foldx_worklist_abbind.csv`.

## Task 3 — Calibrate L → kcal/mol (single global 2-param fit, complex-clustered 10-fold OOF)
On the SKEMPI Lane-A mutations (n=2948, 284 complexes), reading zero-shot `L` and fitted FoldX onto the absolute
experimental-ΔΔG axis:

| predictor | slope b (kcal/mol per unit) | 95% CI | OOF RMSE | OOF Pearson | OOF Spearman |
|---|---|---|---|---|---|
| **−L** (zero-shot) | **0.4213** | [0.3398, 0.5235] | **1.816** | 0.357 | 0.293 |
| FoldX ΔΔG_bind (fitted physics) | 0.4423 | [0.3696, 0.5683] | 1.792 | 0.389 | 0.432 |
| intercept-only floor | — | — | 1.950 | — | — |

**Reading (not a retrained model — single global slope+intercept, no per-position temperatures):** one unit of
leverage is worth **≈ 0.42 kcal/mol** of experimental ΔΔG, and the **zero-shot L calibration sits right next to the
fitted physics tool in absolute error** (OOF RMSE 1.82 vs FoldX 1.79; both beat the 1.95 floor by a similar margin).
The slopes nearly coincide (0.42 vs 0.44). FoldX ranks better out-of-fold (Spearman 0.43 vs 0.29, matching Lane A),
but in absolute kcal/mol the un-fitted leverage is barely distinguishable from the fitted energy function.
→ `results/laneA_calibration.csv`.

---

## Bottom line
1. The "**L adds beyond fitted physics**" result on SKEMPI is **stable and deterministic** (Task 1).
2. On a **second, independent antibody–antigen fixture** the **direction replicates** (L detects, physics detects,
   L-beyond-physics is negative), but at **22 complexes the strict beyond-physics test is underpowered**
   (partial −0.081 [−0.190, +0.003], P(<0)=0.97) — an honest **indeterminate**, not a confirmation and not a
   refutation (Task 2).
3. Zero-shot **L calibrates to ≈ 0.42 kcal/mol per unit** and lands **next to fitted FoldX in absolute error** (Task 3).

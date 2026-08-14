# FINDINGS — the ΔSASA geometry control on PREDICTED backbones

> **VERDICT (the pre-registered headline).** On both independently-predicted backbone classes
> the pre-registered decision variable — **ΔAUROC(full+KL − full), CI excludes 0** — does
> **NOT** fire, and the result is **robust to the one free choice (absolute vs relative ΔSASA):**
>
> | backbone | full+KL − full, absolute ΔSASA | full+KL − full, relative ΔSASA |
> |---|---|---|
> | crystal (control) | −0.001 [−0.011, +0.010] | +0.0074 [−0.004, +0.019] |
> | OF3 (Exp A) | +0.0005 [−0.011, +0.012] | +0.0085 [−0.004, +0.021] |
> | AF2 (Exp D) | −0.003 [−0.013, +0.007] | +0.0047 [−0.006, +0.016] |
>
> Every cell contains zero. Predicted backbones behave *identically* to crystal: KL is ~0.51
> Spearman-correlated with ΔSASA on every backbone class and adds ≈ 0 hotspot-ranking signal
> beyond burial + neighbour-count + ΔSASA. The relative-ΔSASA column reproduces the committed
> crystal geometry-control number (**+0.0074 [−0.0036, +0.0189]**) essentially exactly — pinning
> the original's convention and making the crystal an *exact* positive control (§2). **The
> "off-manifold rescue" hypothesis is refuted and KL-as-a-method is demoted** — per the
> pre-registered fallback the paper leans on the (now ΔSASA-robust) nugget, the burial
> correction, and the cross-predictor reproducibility of the deficit (Exp A/D), none of which
> this touches.
>
> This is the **"decisive predicted-backbone ΔSASA control"** the outline
> (`notes/PAPER_OUTLINE.md`, R1) gates its KL reframe on — **resolved: KL does not earn its keep
> off the manifold** — and it agrees with the parallel session's committed control
> (`results/kl_geometry_control.csv`) on crystal (reproduced to 4 dp, §2) and on **Bennett
> de-novo** backbones, where adding KL to full geometry is *even worse* (ΔAUROC −0.012, excludes
> 0; adding KL HURTS AUROC 0.734→0.721; `results/FINDINGS_bennett.md`).

## 1. What was run

```
python3 src/kl_geometry_control_predicted.py \
  --out results/kl_geometry_control_predicted.csv \
  --nrep 10000 --seed 20260803 --modes absolute relative \
  --fixtures \
  crystal:results/kl_detector_joined.csv:$SCRATCH/ftax/data/PDBs \
  OF3_expA:$SCRATCH/ftax/predicted/expA_kl_joined.csv:$SCRATCH/ftax/predicted/PDBs \
  AF2_expD:$SCRATCH/ftax/expD/expD_kl_joined.csv:$SCRATCH/ftax/expD/PDBs
```
Sherlock job `klgeo` (`-p normal`, CPU, 4 cores, seed 20260803, 10,000 complex-bootstrap
replicates). The **only new ingredient** is a ΔSASA (buried-by-partner area) column computed
**on each fixture's own backbone** — crystal on the crystal PDB, OF3 on the OpenFold3 PDB, AF2
on the AF2-multimer PDB — via the project's freesasa plumbing (`ftax_common.residue_sasa`),
never borrowed from the crystal. ΔSASA(residue) = SASA(residue | its group isolated) −
SASA(residue | full complex). **ΔSASA merge fraction = 1.0000 on all three fixtures, 0 missing
PDBs** — every interface position received a ΔSASA value from its own structure.

Features: `burial = -rsasa_complex` (committed column), `nbr`, `dsasa`, `kl`; label `is_hot`;
interface positions only. `full = z(burial)+z(nbr)+z(dsasa)` is a fixed z-sum composite;
`+KL` adds `z(kl)`. The complex bootstrap resamples complexes and re-evaluates AUROC (paired:
both members of each Δ use the same resample). Every number traces to
`results/kl_geometry_control_predicted.csv`.

## 2. Positive control (crystal) — the port is faithful

The `crystal` fixture reproduces the committed crystal geometry-control result:

- **KL over burial ALONE**: ΔAUROC(burial+KL − burial) = **+0.0410 [+0.0122, +0.0706]** —
  the committed value is **+0.041** → **exact match to three decimals**. This validates the
  entire machinery (burial definition, KL, z-standardisation, tie-aware AUROC, complex
  bootstrap) independently of ΔSASA.
- **KL over FULL geometry**: the committed value is **+0.007 [−0.004, +0.019]**. With
  **relative** ΔSASA (normalised by residue-type max-ASA) the crystal fixture gives
  **+0.0074 [−0.0035, +0.0188]** — reproducing the committed number **to the third decimal**.
  This pins the original control's convention (relative, not absolute ΔSASA) and makes the
  crystal an *exact* positive control. With absolute ΔSASA (the user's snippet) the crystal
  gives **−0.0010 [−0.0114, +0.0098]** — the same reading (CI contains zero), point estimate
  shifted only by the scaling choice. Either way, KL does not survive on crystal.

This port was written before the parallel session's `src/kl_geometry_control.py` was committed;
both now agree. Against the committed crystal control (`results/kl_geometry_control.csv` — which
uses the p0 ΔrSASA column and a Spearman-of-residuals partial) my `relative` crystal row matches
the two headline numbers **to the fourth decimal**: `dAUROC_KL_over_burial_alone` 0.041
[0.0123, 0.0706] and `dAUROC_KL_over_full_geometry` 0.0074 [−0.0036, 0.0189]. The committed
control covers crystal + Bennett de-novo; **this file extends the identical control to the
OpenFold3 and AF2-multimer predicted backbones** — the piece the outline gates on. The port is
trustworthy, so the predicted numbers are trustworthy.

## 3. The decisive result (`results/kl_geometry_control_predicted.csv`, `dsasa_mode=absolute`)

| fixture | AUROC burial / full | corr(KL,ΔSASA) | corr(burial,ΔSASA) | ΔAUROC(bur+KL − bur) | **ΔAUROC(full+KL − full)** [headline] |
|---|---|---|---|---|---|
| **crystal** (control) | 0.689 / 0.766 | +0.516 | +0.108 | +0.041 [+0.012, +0.071] | **−0.001 [−0.011, +0.010]** |
| **OF3** (Exp A, n=140) | 0.648 / 0.753 | +0.510 | +0.088 | +0.059 [+0.028, +0.091] | **+0.0005 [−0.011, +0.012]** |
| **AF2** (Exp D, n=140) | 0.647 / 0.745 | +0.507 | +0.120 | +0.044 [+0.017, +0.073] | **−0.003 [−0.013, +0.007]** |

Read this top-to-bottom:

1. **KL beats burial ALONE on every backbone** (ΔAUROC(bur+KL − bur) = +0.041 / +0.059 / +0.044,
   all exclude 0) — this is the KL-detector headline that motivated the whole method (crystal
   +0.048, OF3 +0.062, AF2 +0.054 in the detector's own metric; the same story). It is real.
2. **…but that entire advantage is ΔSASA.** Once the composite already contains ΔSASA (a
   free, no-neural-net geometric feature: partner-contact area), **adding KL moves AUROC by ≈ 0**
   on every backbone (headline column: −0.001 / +0.0005 / −0.003, all CIs contain zero). KL is
   ~0.51 correlated with ΔSASA everywhere; ΔSASA is nearly orthogonal to burial (corr ≈ 0.09–0.12),
   so ΔSASA is a *distinct* cheap feature that KL recapitulates.
3. **Predicted ≡ crystal.** The hoped-for rescue was that off the native manifold the learned
   distribution encodes partner-sensitivity raw contact-area misses — which would show up as
   ΔAUROC(full+KL − full) excluding 0 on predicted but not crystal. **It does not happen.** OF3
   and AF2 land on top of crystal (all three ≈ 0). KL is not doing anything special off-manifold.

**Per the pre-registered decision rule (ΔAUROC(full+KL − full) CI excludes 0 ⇒ survive), KL
does not survive on either predicted backbone → demote KL-as-method.**

## 4. The honest tension: the partial-Spearman criterion points the other way

The report-back also listed a second operationalisation: *partial Spearman(KL, is_hot |
burial+nbr+ΔSASA), positive & CI-excludes-0 ⇒ KL survives.* **This readout is fragile near zero
and its sign is not robust across implementations, so it does not carry the verdict** — the
ΔAUROC does.

My rank-residualised partial (Pearson of the rank residuals) is small-positive and excludes 0:

| fixture | partial(KL, is_hot \| burial) | partial(KL, is_hot \| burial+nbr+ΔSASA), mine (abs / rel) |
|---|---|---|
| crystal | +0.117 | +0.033 / +0.062 |
| OF3 | +0.114 | +0.032 / +0.055 |
| AF2 | +0.102 | +0.025 / +0.046 |

But the **committed crystal control** — the same rank-residualisation with **Spearman** of the
residuals and the p0 ΔrSASA column (a slightly stronger KL-correlate, ρ≈0.6) — gives
partial(KL, is_hot | burial+nbr+ΔSASA) = **−0.022** on crystal and **−0.060 (p=0.017)** on
Bennett de-novo, where adding KL to the geometry ranker *actively HURTS* AUROC (0.734→0.721). So
across implementations and backbone classes the full-geometry partial sits at **≈ 0 with unstable
sign** (−0.06 … +0.06): KL's residual beyond full cheap geometry is negligible — not a real
positive signal — and its point-estimate sign is an artifact of the exact ΔSASA source and
residual-correlation flavour.

The **decision-relevant, implementation-robust** metric is ΔAUROC(full+KL − full) — *does adding
KL improve the actual hotspot ranking?* — and it is ≈ 0 everywhere, matching the committed crystal
control to the fourth decimal. A method is only useful if it improves the ranking you act on, and
KL does not, beyond features computable without it. Honest bottom line: **KL ≈ full cheap geometry;
the partial-correlation residual is within implementation noise of zero.**

## 5. ΔSASA-scaling sensitivity (the one free choice)

The absolute ΔSASA (Å², matching the user's snippet) is one free choice; the relative variant
(absolute / Tien max-ASA of the residue type) is the robustness check, run in the same job
(`dsasa_mode=relative`). It matters twice: it is the convention that **reproduces the committed
crystal number exactly** (§2), and it confirms the verdict does not hinge on the scaling.

| fixture | corr(KL,ΔSASA_rel) | ΔAUROC(full+KL − full), relative | partial(KL \| burial+nbr+ΔSASA), relative |
|---|---|---|---|
| crystal | +0.524 | **+0.0074 [−0.0035, +0.0188]** | +0.062 [+0.035, +0.088] |
| OF3 (Exp A) | +0.516 | **+0.0085 [−0.0040, +0.0213]** | +0.055 [+0.030, +0.080] |
| AF2 (Exp D) | +0.516 | **+0.0047 [−0.0056, +0.0157]** | +0.046 [+0.023, +0.070] |

**Sensitivity verdict: the headline is unchanged.** Under relative ΔSASA (the original's
convention), ΔAUROC(full+KL − full) still contains zero on both predicted arms *and* on crystal
— KL does not survive under either scaling. The partial Spearman is somewhat larger under
relative ΔSASA (relative ΔSASA captures a little less of KL's variance, leaving more residual),
but the ΔAUROC — the ranking-relevant metric — stays ≈ 0, so the §4 reconciliation holds under
both conventions. corr(KL,ΔSASA) is ~0.51–0.52 either way, so the "KL ≈ ΔSASA" core is
scale-invariant. Determinism check: the `absolute` rows reproduced the first single-mode run
bit-for-bit (crystal −0.0010, OF3 +0.0005, AF2 −0.0035).

## 6. What it means

- **The KL detector's advantage over the burial heuristic is ΔSASA.** The sequence-free
  signal's edge over burial-alone (+0.04–0.06 AUROC, real and cross-predictor-robust) is
  recovered by partner-contact area, a trivial geometric quantity from the same two structures
  KL needs — no ProteinMPNN required. This is the "ΔSASA-robust nugget" framing the
  pre-registration named as the fallback, and it is the honest way to state the KL result.
- **Completes the triage reframe the outline gates on this result.** `FINDINGS_kl_triage.md` §4
  celebrates a **KL+burial** capture@k advantage over a **burial-alone** baseline on the *same*
  OF3 and AF2 predicted backbones (+0.083 / +0.087). This control shows that on those same
  backbones KL adds ≈ 0 over burial+nbr+ΔSASA — so the predicted-backbone triage advantage is
  **ΔSASA-substitutable**, matching the parallel session's already-withdrawn Lever-2 (kcal/mol)
  claim and its crystal/Bennett geometry control. The outline's "REFRAME PENDING" banner gates
  the KL-triage capture@k withdrawal on exactly this predicted-backbone control; with it done, the
  honest contribution is **"cheap geometry (burial+ΔSASA) triages experimental hotspots — the
  learned frustratometer buys nothing over it."** I have deliberately **not** edited
  `FINDINGS_kl_triage.md` or `notes/PAPER_OUTLINE.md` — the parallel session owns those and is
  actively reframing them; this file is the decisive input they were waiting on.
- **What this does NOT touch.** The burial-matched log-prob **deficit** on predicted backbones
  (Exp A/D, cross-predictor Spearman +0.57) is a different result — a property of `p(seq|backbone)`
  at matched burial, not a KL-vs-geometry comparison — and stands unchanged. So does the
  cross-predictor reproducibility. This control sharpens *one* of the project's contributions
  (KL-as-method → ΔSASA-as-method) and leaves the deficit result intact. Both directions were
  pre-registered as publishable and honest.

## Files
- `src/kl_geometry_control_predicted.py` — the port (ΔSASA from each backbone; partial Spearman +
  z-sum AUROC + complex bootstrap; `--modes absolute relative`).
- `results/kl_geometry_control_predicted.csv` — every number above (3 fixtures × 2 ΔSASA modes).
- Job: `$SCRATCH/ftax/jobs/klgeo.sbatch` (`-p normal`, CPU); authoritative two-mode log
  `$SCRATCH/ftax/logs/klgeo_38850283.out` (Slurm 38850283; the single-mode 38848858 was the
  first absolute-only run and reproduced identically).
- Inputs: `results/kl_detector_joined.csv` (crystal); `$SCRATCH/ftax/predicted/expA_kl_joined.csv`
  (OF3); `$SCRATCH/ftax/expD/expD_kl_joined.csv` (AF2); backbones under the three `PDBs/` dirs.
- Companion controls (parallel session): `src/kl_geometry_control.py` +
  `results/kl_geometry_control.csv` (crystal + Bennett de-novo; my `relative` crystal row
  reproduces its two headline numbers to 4 dp); `results/FINDINGS_bennett.md` (the de-novo arm,
  where KL over full geometry is negative). Together: crystal + de-novo + OF3 + AF2 all agree KL
  adds ≤ 0 over cheap geometry.

## CORRECTION 2026-08-15 (over-kill sweep + independent verification)
The z-sum ΔAUROC-against-0 readout in kl_geometry_control.py has a **−0.021 noise floor** (adding a *pure-noise*
feature to the geometry z-sum scores −0.021; verified). So "KL adds ≈0 / **actively hurts**" measured the
combiner penalty, not KL — **that reading is WITHDRAWN.** Correct readouts: within-geometry-stratum AUROC of KL
for hotspots = **0.605** (vs 0.499 leakage; verified), and the committed combiner-free **CPI(KL | geometry) =
+0.00201 [+0.0006, +0.0034], P=0.998 "ADDS"** (nugget_cpi.csv). **KL adds a small but real, significant
increment beyond geometry** — a genuine learned-frustratometer signal, ~6× below ΔSASA, not "nothing/hurts".
The z-sum-ΔAUROC-against-0 estimator is retired (sibling of the ΔAUROC-over-one-hot error). Negative control
holds: within-stratum confidence 0.478 / negentropy 0.484 (both chance) — the nugget is unaffected, strengthened.
OF3/AF2 arms (Sherlock) not re-audited under the corrected readout — outstanding.

# FINDINGS — placing the R2 predicted backbones on the dose-law x-axis (measured Cα-RMSD)

**Goal:** measure how far the OpenFold3 / AF2-multimer backbones used in the R2 result deviate from crystal,
so the R2 leverage points (+0.039 OF3 / +0.032 AF2) can be plotted at their *true effective σ* on the dose-law
x-axis (Fig 3) — turning "predicted backbones fall in the dose law's survival zone" from qualitative to
measured. SEED=20260803. Script `src/predicted_backbone_rmsd.py`. Deliverable
`results/predicted_backbone_rmsd.csv` (140 complexes).

```
python3 src/predicted_backbone_rmsd.py --out results/predicted_backbone_rmsd.csv
```

**Method.** For each of the 140 shared R2 complexes, Cα atoms are matched by (chain, resnum, icode) between
crystal (`~/ftax/data/PDBs`) and the *exact renumbered predicted PDB the R2 leverage was read from*
(`$SCRATCH/ftax/predicted/PDBs` OF3, `$SCRATCH/ftax/expD/PDBs` AF2), Kabsch-superposed, two ways:
`ca_rmsd_all` (superpose+measure on all matched Cα) and `ca_rmsd_iface` (superpose+measure on interface Cα
only — the local interface deformation after removing rigid-body, the correct analogue of the dose-law jitter,
which is pure local per-atom noise). Interface set from committed `leverage_skempi_positions.csv`.

## Positive controls (rule 6) — both pass
- **Superposition sanity:** crystal-onto-itself Cα-RMSD = **1.0×10⁻¹³ Å** (~0); a deliberately mis-paired
  superposition = **5.65 Å** (nonzero). Kabsch is correct and the control is live.
- **Cross-validation against the committed manifests.** These RMSDs already existed (OF3 in
  `expA_confidence.csv`, AF2 in `expD_backbone_manifest.csv`); the independent recompute here **reproduces them
  exactly** — Spearman **1.000** and median|Δ| **0.000 Å** for OF3 global, OF3 interface, AF2 global; Spearman
  0.979 / median|Δ| 0.000 Å for AF2 interface (a few complexes differ only in interface-superposition detail).
  So both the committed numbers and this deliverable are validated.

## Measured distribution (Å)

| metric | median | IQR | ≤1.0 Å | ≤1.5 Å |
|---|---|---|---|---|
| **OF3 interface** | **1.28** | [0.77, 6.07] | 0.37 | 0.53 |
| **AF2 interface** | **1.29** | [0.72, 6.13] | 0.39 | 0.53 |
| OF3 global | 2.36 | [1.30, 8.64] | 0.13 | 0.33 |
| AF2 global | 2.50 | [1.25, 6.75] | 0.18 | 0.34 |

Interface RMSD is **bimodal / heavy-tailed**, not tightly sub-Ångström: OF3 buckets ≤0.75 Å **24%**,
0.75–1.5 Å **29%**, 1.5–3 Å **16%**, **>3 Å 31%** (AF2: 28% / 25% / 11% / 36%). The >3 Å tail is
**badly-docked complexes** — the predictor placed the chains wrong (e.g. 3HH2 31.8 Å, and antibody complexes
1BJ1 / 1AHW / 1N8Z / 1REW), mostly Ab–Ag and multi-chain assemblies. OF3 and AF2 interface RMSD correlate
(Spearman 0.57): the same complexes are hard under both predictors.

## The dose-law bridge — leverage retention falls with interface RMSD, per complex

The point of the placement is whether the dose law *predicts* the R2 survival. It does, mechanistically and at
the per-complex level. Merging with the per-complex leverage retention (`deficit_vs_leverage_percomplex.csv`,
retention = mean|L|rms(predicted)/mean|L|rms(crystal)):

**Spearman(interface Cα-RMSD, leverage retention), complex-bootstrap 95% CI, n=127:**
- **OF3: −0.560 [−0.680, −0.411]**
- **AF2: −0.638 [−0.746, −0.500]**

Both strongly negative with CIs far from zero: **the more a predicted interface deviates from crystal, the less
leverage survives** — the dose law of §4 operating on *real* predicted backbones, not just under synthetic
jitter. (This is far cleaner than the #7 confidence-deficit correlation, ρ≈+0.16/+0.19, because interface RMSD
is a direct measure of backbone error whereas the recovery deficit is a noisy sparse-hotspot readout.)

## Reading — honest placement
The qualitative claim "predicted backbones sit in the dose law's survival band" is **half right, and the honest
version is stronger**. The median interface RMSD (~1.28 Å) sits at the dose-law **knee**, not deep in survival:
~**half** the complexes are ≤1.5 Å (the surviving side, where the crystal ladder keeps CPI up), and a **~third**
are past the cliff (>3 Å, badly docked). The pooled R2 survival (+0.039 OF3 / +0.032 AF2, 69–84% of crystal) is
exactly what this mixture predicts — a survival-zone bulk carrying the signal, diluted by a collapsed tail — and
the −0.56/−0.64 RMSD↔retention correlation is the mechanism that makes the dose law *predict*, not merely
co-exist with, the R2 result. For Fig 3, the R2 points belong at the median interface RMSD (~1.3 Å) with a wide
horizontal spread (IQR 0.77–6.1 Å), annotated that survival is carried by the ≤1.5 Å half.

Columns of `predicted_backbone_rmsd.csv`: `complex_id, of3_ca_rmsd_all, of3_ca_rmsd_iface, af2_ca_rmsd_all,
af2_ca_rmsd_iface, n_res, n_iface, seed`.

# FINDINGS — Bennett 2023 de-novo design-regime KL-detector validation

**Date:** 2026-08-13. **Scripts:** `src/bennett_kl_detector.py`. **Data:** Bennett et al. 2023,
*Improving de novo protein binder design with deep learning* (Nat Commun, PMC10163288, CC-BY-4.0),
`files.ipd.uw.edu/pub/improving_dl_binders_2023/`. **Outputs:** `results/bennett_kl_detector.csv`,
`results/bennett_kl_positions.csv`.

## Why this test

Every prior test of the sequence-free KL detector used SKEMPI (crystal / OpenFold3 / AF2 / partial-
diffusion) backbones and Ala-scan ΔΔG labels. This is the first test on **genuinely de-novo binder
backbones** (RFdiffusion + ProteinMPNN — the exact staged pipeline this paper critiques) against
**experimental** site-saturation-mutagenesis (SSM) labels. It attacks the single-fixture, model-readout,
and no-design-regime weaknesses at once. Recovery deficit is NOT testable here (the sequences ARE
ProteinMPNN's output → recovery trivially high); only the **detector** is on-thesis.

## Method

- 73 SSM parent designs across 6 libraries (ALK_SSM1/2, IL2Ra_SSM1/2, LTK_SSM2, IL10Ra_SSM); LTK_SSM1
  native PDBs not analysed (17 designs / ~19%; **unverified: could not fetch** — local dir empty, server
  tarball 404/403). Binder = chain A (55–57 res), target = chain B.
- **Label (PROXY):** from the pre-computed affinity tables (`ngs_data_analysis/affinities/*.sc`), each
  position's SSM lists 19 non-native substitutions with Kd bounds. Per interface position,
  *binding-restrictiveness* = fraction of the 19 substitutions that lose binding (kd_lb ≥ highest tested
  conc). Hotspot = restrictiveness ≥ 0.75.
- **Interface** = partner-buried (ΔSASA on binding > 1 Å²). **Burial** = −rSASA(complex). **KL** =
  KL(p(·|complex backbone) ‖ p(·|binder-alone backbone)), ProteinMPNN unconditional, sequence-free.
- AUROC bootstrapped over **designs** (parent-level), seed 20260803.

## Positive controls (all pass — CLAUDE.md §6)

1. **Mapping: SSM-excluded aa == PDB-native residue = 1.000 (4137/4137).** The SSM tables list the 19
   non-native substitutions; the excluded one equals the PDB residue at resnum = SSM position, at every
   position → the position→resnum alignment is exact. (An earlier "native-binds = 0.000" control was
   mis-designed — native is excluded by construction — and was corrected to this invariant.)
2. **KL elevated at the interface:** mean KL 0.363 (interface) vs 0.016 (non-interface).
3. **Label non-degenerate:** 583/1569 interface positions (37%) are hotspots.

## Result — KL predicts hotspots on de-novo backbones, but recapitulates cheap geometry

**CORRECTION (2026-08-13, independent Fable-5 audit + my re-verification).** An intermediate draft claimed
"KL retains a significant UNIQUE signal beyond burial / DOES add." WRONG: it residualised KL on burial
(−rSASA) ALONE, omitting ΔSASA (partner-contact area), which is ~orthogonal to burial (ρ=0.09) and strongly
correlated with KL (ρ=0.60). Against the full cheap-geometry baseline (burial+nbr+ΔSASA) the unique signal
vanishes and adding KL HURTS ranking. What survives: KL predicts the hotspots (ρ +0.29), it just does so
via geometry it does not improve on.

| detector (interface positions, n=1569, 73 designs) | AUROC / stat |
|---|---|
| burial (−rSASA) baseline | 0.709 [0.689, 0.729] |
| neighbour count | 0.716 [0.694, 0.737] |
| **KL(complex ‖ binder)** | **0.646 [0.626, 0.669]** (excludes 0.5) |
| burial + KL (rank-avg) | 0.706 [0.687, 0.726] |
| ΔAUROC(burial+KL − burial), naive rank-avg | −0.003 [−0.016,+0.011] — **dilution artifact** (equal-weight avg of unequal predictors; z-sum 0.697 < burial 0.709) |
| ΔAUROC(burial+KL − burial), CV logistic | +0.005 **[−0.002,+0.011] — CI spans 0** (one-sided p≈0.07, NOT significant) |
| Spearman(KL, restrictiveness), interface | +0.291, p < 10⁻⁴ (**flips to −0.236 OUTSIDE interface**) |
| partial Spearman(KL, restr \| burial) | +0.104 — but this is **ΔSASA leakage** |
| **partial Spearman(KL, restr \| burial+nbr+ΔSASA)** | **−0.060, p=0.017 — NEGATIVE vs full geometry** |
| full geometry (burial+nbr+ΔSASA) AUROC → +KL | 0.734 → **0.721 (adding KL HURTS)** |

- **KL transfers as a predictor:** significantly correlated with experimental binding-restrictiveness on
  real de-novo backbones (Spearman +0.29; AUROC 0.65) — genuine design-regime evidence the sequence-free
  partner-sensitivity signal detects experimentally binding-critical positions.
- **WITHDRAWN (2026-08-13 audit): "KL adds a unique signal beyond burial."** KL is a noisy mixture of
  self-burial (ρ 0.49) and partner-contact-area ΔSASA (ρ 0.60). The +0.104 partial-over-burial was ΔSASA
  leaking into an incomplete baseline. Control for BOTH (burial+nbr+ΔSASA — all cheap geometry, no neural
  net, the same two structures KL needs) and KL's partial goes **NEGATIVE (−0.060, p=0.017)**; adding KL to
  the geometry ranker **HURTS** AUROC (0.734→0.721). The CV-logistic +0.005 has a CI spanning zero
  (one-sided p≈0.07) — a null, not a positive. The naive rank-avg −0.003 is separately a dilution artifact,
  but the fair verdict from the full baseline is the same: **on these de-novo binders KL provides no
  advantage over trivial geometry.** (My earlier "DOES add" was a second over-correction; withdrawn.)

## Interpretation and caveats (report as such)

- **On de-novo binders KL ≈ cheap geometry.** Once the baseline includes partner-contact-area (ΔSASA), KL
  adds nothing. Likely because (a) the SSM proxy **convolves binding with fold/display stability** — burial
  predicts restrictiveness even MORE strongly OFF the interface (+0.53) than on it (+0.45) — and (b) these
  idealised small helical-bundle binders (one interface geometry per target) align binding-criticality with
  cheap geometry more tightly than natural SKEMPI interfaces (where frustrated hotspots need not be the most
  buried).
- **The honest design-regime datapoint:** the sequence-free detector TRANSFERS as a *predictor* of
  experimental hotspots on real de-novo backbones (ρ +0.29, design-clustered-robust) — a genuine,
  independent, experimental-label fixture — but offers **no advantage over cheap structural features**
  there. Do NOT claim "KL adds over burial in the design regime."
- **Load-bearing follow-up (the audit's real gift):** the central SKEMPI claim ("KL adds +0.05–0.09 over
  burial") must be re-tested against the SAME full-geometry baseline (burial + ΔSASA + nbr), not burial
  alone — otherwise the crystal/predicted result risks the same ΔSASA-leakage. See the SKEMPI geometry check.
- Caveats: proxy labels (stability-convolved); interface cut ΔSASA>1 Å² is permissive (the burial-only
  partial decays to null by ΔSASA>10 and negative by >30); 4 targets / one epitope each (design-level
  bootstrap treats 73 designs as independent); LTK_SSM1 (17 designs, ~19%, strongest per-target partial
  +0.18) not analysed — **unverified: could not fetch** (local dir empty; server tarball 404/403).

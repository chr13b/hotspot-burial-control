# FINDINGS — the mixed derivative's binding signal SURVIVES on predicted backbones (R2, design regime)

**Pre-registered** in `results/PREREG_leverage_predicted.md`, frozen (commit `58f2cde`) before any CPI
number. **Script** `src/leverage_predicted.py`. **Outputs** `results/leverage_predicted.csv` (CPI + controls)
and `results/leverage_predicted_ranker.csv` (AUROC lift). SEED 20260803. ProteinMPNN sequence-free marginal,
identical scorer to the crystal result; complex-clustered bootstrap throughout.

## Question and design

The crystal result and the backbone-noise ladder show CPI(L | geometry) > 0 on native/accurate backbones.
Designers condition on a **folding model's** backbone. This asks whether the signal survives there.
**One-variable manipulation:** the SKEMPI interface-mutation fixture is held fixed (same complexes, positions,
alanine-scan hotspot labels, ΔΔG); only (1) leverage L is recomputed by the identical scorer reading the
**predicted** complex PDB (monomer = delete partner chain from the same predicted PDB), and (2) geometry
burial/nbr/ΔSASA is taken **from the predicted structure** (`expA_p0_positions.csv` for OF3,
`expD_p0_positions.csv` for AF2 — the honest baseline a designer has). Shared set = OF3 ∩ AF2 predicted
geometry ∩ SKEMPI interface fixture = **140 complexes** (2,520 mutations after WT-match; 5,690 interface
positions, 325 hotspots).

## Positive control — PASSES (gate)

- **Crystal re-score reproduces the committed leverage L exactly**: mutation-level max|ΔL| = **1.13e-05**,
  position-level max|Δ(L_ala, L_rms)| = **1.12e-05** — float32 MPNN precision. WT-match and position map-rate
  = 1.000 on all three sources (the predicted structures carry the native sequence, so residue identity is
  intact).
- **Crystal-on-140 lands solidly positive**: CPI(L | geom) mutation-level **+0.0462** [+0.0332, +0.0600],
  position-level L(→Ala) **+0.0094** [+0.0062, +0.0128], |L|_rms **+0.0056** [+0.0036, +0.0079]; ranker
  ΔAUROC **+0.0142** [+0.0043, +0.0246]. (The committed full-285 values are +0.059 mut / +0.0048 pos /
  ~+0.013–0.016 ranker; the shared-140 subset differs in point estimate but is the correct matched baseline
  for the predicted comparison — same 140 complexes.) Gate passes → predicted numbers are interpretable.

## Result — SURVIVES

**Mutation-level CPI(L | burial+nbr+ΔSASA)** — the headline. All positive, P(>0)=1.000, all survive dropping
the 3 most-influential complexes:

| source | CPI(L \| geom) | 95% CI | drop-3 | Spearman(L, ΔΔG) | % of crystal-140 |
|---|---|---|---|---|---|
| **crystal (140)** | **+0.0462** | [+0.0332, +0.0600] | +0.0264 SURVIVES | −0.279 | 100% |
| **OF3** | **+0.0389** | [+0.0265, +0.0504] | +0.0243 SURVIVES | −0.272 | 84% |
| **AF2** | **+0.0320** | [+0.0226, +0.0409] | +0.0185 SURVIVES | −0.243 | 69% |
| **pooled OF3+AF2** | **+0.0362** | [+0.0258, +0.0476] | +0.0213 SURVIVES | −0.257 | 78% |

The predicted CPIs land squarely in the pre-registered **[+0.01, +0.04]** band, positive with the CI clearing
the +0.0007 placebo floor by 30–70×. **L still adds beyond confidence** on predicted backbones
(CPI(L | geom+confidence): OF3 +0.0461, AF2 +0.0358, pooled +0.0427, all CI>0) — so this is not confidence
in disguise. Spearman(L, ΔΔG) is barely touched (−0.279 → −0.24…−0.27), consistent with the ESM-IF1 dose-law
finding that the rank correlation is more backbone-robust than the CPI.

**Position-level CPI (| burial+nbr+ΔSASA)** — attenuates more but stays CI>0:

| source | L(→Ala) | |L|_rms |
|---|---|---|
| crystal (140) | +0.0094 [+0.0062, +0.0128] | +0.0056 [+0.0036, +0.0079] |
| OF3 | +0.0071 [+0.0045, +0.0099] | +0.0022 [+0.0006, +0.0038] |
| AF2 | +0.0024 [+0.0002, +0.0044] | +0.0021 [+0.0008, +0.0035] |
| pooled | +0.0044 [+0.0022, +0.0067] | +0.0022 [+0.0008, +0.0037] |

**The geometry+|L| ranker lift attenuates to marginal — reported, not softened.** OOF cross-fit logistic
AUROC of is_hot (identical metric to `w4_combined_ranker.py`):

| source | AUROC geom | geom+\|L\|_rms | ΔAUROC | 95% CI | P(>0) |
|---|---|---|---|---|---|
| crystal (140) | 0.726 | 0.741 | **+0.0142** | [+0.0043, +0.0246] | 0.997 |
| OF3 | 0.705 | 0.711 | +0.0067 | [−0.0011, +0.0149] | 0.952 |
| AF2 | 0.672 | 0.677 | +0.0047 | [−0.0019, +0.0121] | 0.912 |
| pooled | 0.690 | 0.696 | +0.0057 | [−0.0009, +0.0129] | 0.952 |

So the **information content survives** (CPI CI>0 on both predictors, mutation and position level, drop-3
robust), while the **single-number ranking gain over geometry becomes marginal** on predicted backbones
(CI includes 0, P(>0) ≈ 0.91–0.95). Both facts stand.

## Reading

1. **The actionable claim generalizes to the design regime, and the crystal dose law is conservative.** The
   binding signal in the mixed derivative is not a crystallography artifact: it is present, positive, and
   beyond-geometry on the OpenFold3 and AF2-multimer backbones a designer actually gets. The attenuation is
   exactly the pre-registered ~0.5–1.0 Å rung of the crystal ladder.
2. **OF3 retains more signal than AF2** (84% vs 69% of the crystal-140 CPI), matching Exp A/D independently:
   OF3 backbones are slightly more interface-native than AF2 (Exp A burial-matched deficit −0.19 vs Exp D
   AF2 −0.23; per-complex OF3–AF2 corr +0.57). The two predictors agree on the *ranking* of backbone fidelity.
3. **This is the counterpoint to the burial-matched hotspot deficit, and the two together sharpen the thesis.**
   On the same predicted backbones, the *confidence-type* readout degrades (Exp A/D: hotspots recovered worse,
   −0.19/−0.23) while the *mixed-derivative* readout survives (this file). Confidence is fragile to backbone
   error where leverage is robust — precisely the decomposition's claim that they are different derivatives.
4. **The ranker caveat is the honest limit.** Using |L| as an extra *ranking feature* on top of geometry buys
   a significant AUROC gain on crystals but only a marginal one on predicted backbones. The mixed derivative
   remains the right object to *read binding from*; as a plug-in ranking feature its crystal-grade lift does
   not fully transfer.

## Reproduce

```
python3 src/leverage_predicted.py --stage score --source crystal --threads 8   # + of3, af2
python3 src/leverage_predicted.py --stage analyse --out results/leverage_predicted.csv
```
Score writes the gitignored PQ caches (`results/leverage_pq_predicted_{source}.csv`); analyse writes the two
committed summary CSVs. Predicted PDBs: `$SCRATCH/ftax/predicted/PDBs` (OF3), `$SCRATCH/ftax/expD/PDBs` (AF2);
predicted geometry from the committed expA/expD p0 positions files. ~2 min/source on 8 CPU threads (no GPU).

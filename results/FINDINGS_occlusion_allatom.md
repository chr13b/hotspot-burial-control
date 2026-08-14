# FINDINGS — T1: does P add beyond an ALL-ATOM occlusion baseline? (venue decider)

**Pre-registered:** `results/PREREG_bennett_hardening.md` (committed 8c0306b, before any number).
**Script:** `src/p_bennett_occlusion_allatom.py`. **Outputs:** `results/bennett_occlusion_allatom.csv`
(+ `_pairs.csv`). Bennett de-novo SSM interface, 27,170 (position,substitution) pairs, 73 designs;
ProteinMPNN complex-conditioned P; design-clustered bootstrap (3000), seed 20260803.

## Why
Idea-critic kill-shot #2 (the most lethal specific objection): the published +0.025 "energetics beyond
geometry" (`bennett_occlusion_energetics.csv`) rests on a *volume x contact* occlusion baseline the paper
itself called weak — "repack the rotamers and it evaporates." This rebuilds occlusion as a real ALL-ATOM,
min-over-rotamer van-der-Waals clash (rdkit ETKDG rotamers, each Kabsch-superposed on the N,CA,C backbone
of the SSM position on the de-novo binder, clashed against every heavy atom of the partner), and re-runs the
identical CV-logistic ΔAUROC(P over geometry) test on a STRONGER baseline.

## Process honesty (recorded deliberately)
- The **first run's gate deviated from the pre-registration**: it was coded as a clash-correlation Spearman
  instead of the pre-registered *native side-chain reconstruction RMSD*. It FAILED (ρ=0.258) — but on
  diagnosis this was because **95% of native side chains do not clash** (relaxed interfaces), so the metric
  had no dynamic range. That gate could not certify anything here.
- Corrected to the **pre-registered RMSD gate**: the builder reconstructs native side chains to **median
  0.278 Å** (p90 0.78), Gly clash 0 → **GATE PASS**. The builder is valid.
- **I was blind to the ΔAUROC throughout**: the first run never computed it (gate blocked it); the second
  computed it only after the corrected, pre-registered gate passed. The result decision rule (lo>0 AND
  point ≥ +0.010) was frozen in the pre-registration and untouched.

## Result
| quantity | value |
|---|---|
| native reconstruction RMSD (gate) | **0.278 Å** median (< 1.0 ✓) |
| **post-repack occlusion prevalence** | **95.1% of substitutions have ZERO all-atom clash** |
| standalone AUROC — P | 0.615 |
| standalone AUROC — all-atom clash | 0.519 (near chance) |
| standalone AUROC — ΔSASA | 0.585 |
| all-atom geometry baseline (clash + clash×contact + ΔSASA + vol + contact) | 0.619 |
| geometry + P | 0.637 |
| **ΔAUROC(P over ALL-ATOM geometry)** | **+0.0182 [+0.0145, +0.0220], P(>0)=1.000** |
| pre-registered decision | **SURVIVES → Spine B / ICLR** (lo>0 ✓ and ≥ +0.010 ✓) |

## Reading
Kill-shot #2 is **defused, and it backfires on the critic's own proposed operation.** When you actually
repack the rotamers — min over 40 conformers, the best steric fit — **steric occlusion nearly vanishes**
(95.1% zero clash; all-atom clash predicts binding at 0.519, essentially chance). Occlusion therefore
*cannot* be the source of P's interface-binding signal. And on a geometry baseline that is now genuinely
all-atom and **stronger** than the original (0.619 vs 0.587), P still adds **+0.0182 [+0.0145, +0.0220]**.
The model encodes per-substitution binding **energetics beyond all-atom steric occlusion** — the effect is
smaller than the original +0.025 (weaker baseline) but far more defensible, pre-registered, and gate-validated.

## Caveats
- The effect is modest (+0.018). It is tight (CI excludes 0 at P=1.000) and on a strong baseline, but modest.
- The rotamer model is min-over-40 ETKDG conformers with a soft vdW-overlap penalty — a repacking proxy, not
  a full molecular-mechanics force field. A physics FF (Rosetta/FoldX) could shift the baseline; but the
  95%-zero prevalence means there is little occlusion signal for any clash model to capture.
- **Single scorer = ProteinMPNN, whose outputs are the SSM parents.** This is the circularity kill-shot (#3);
  T2 (`src/p_bennett_nonparent.py`, ESM-IF1) re-scores with a non-parent model to confirm the signal is not
  the model ranking substitutions around its own mode.

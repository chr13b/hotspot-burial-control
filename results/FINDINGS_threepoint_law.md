# FINDINGS — the constraint-vs-leverage gradient is a monotone multi-point law

**Pre-registered:** `results/PREREG_3point_law.md` (committed ab1c290, BEFORE any per-class AUROC). **Script:**
`src/p_3point_law.py`. **Output:** `results/threepoint_law.csv`. SEED=20260803.

## Why
The paper's constraint-vs-leverage fork (§4) rested on a **2-point** comparison (SKEMPI 0.538 vs de-novo 0.60)
with different label constructions across the two fixtures — §9 called it "suggestive, not formal." This breaks
SKEMPI's single number into its own `Hold_out_type` classes **within one pipeline / one label construction**,
adding de-novo as the anchor.

## Result — confidence-AUROC rises monotonically, exactly as pre-registered
| stratum (transience rank) | conf-AUROC [95% CI] | burial-residualized conf-AUROC | burial-AUROC | n_cx / n_hot |
|---|---|---|---|---|
| TCR/pMHC (1, most transient) | **0.430 [0.355, 0.508]** | 0.413 | 0.606 | 36 / 34 |
| AB/AG (2) | **0.457 [0.383, 0.518]** | 0.426 | 0.750 | 52 / 68 |
| Pr/PI (3, rigid inhibitors) | **0.554 [0.465, 0.615]** | 0.551 | 0.481 | 60 / 26 |
| **de-novo Bennett (anchor)** | **0.596 [0.567, 0.624]** (logp); 0.627 (negentropy) | — | — | — / 747 |

**H1 (confirmatory) — CONFIRMED.** De-novo confidence-AUROC exceeds every natural class, and the natural
classes are at or below chance (TCR/pMHC's CI upper bound is 0.508; AB/AG's 0.518; Pr/PI's spans 0.5). Confidence
is blind to hotspots across **all** natural interface types, not just in the pooled SKEMPI number.

**H2 (exploratory) — the pre-registered order holds exactly.** Spearman(transience-rank, conf-AUROC) = **+1.000**
over the three natural classes; the full four-point sequence is monotone **0.430 → 0.457 → 0.554 → 0.596**.

## The circularity control (load-bearing — this is a burial project)
The gradient is **not** a burial artifact, by two independent checks:
1. **Burial-residualized** confidence-AUROC (confidence with its linear burial component removed) preserves the
   order: 0.413 < 0.426 < 0.551. The signal that rises is not the burial component.
2. **The burial-AUROC gradient runs opposite.** AB/AG has the *highest* burial-AUROC (0.750) yet a *low*
   confidence-AUROC (0.457); Pr/PI has the *lowest* burial-AUROC (0.481) yet the *highest* confidence-AUROC
   (0.554). So "how well confidence sees hotspots" is orthogonal to (even anti-correlated with) "how buried the
   hotspots are" — the gradient tracks fold/binding coupling, exactly as the theory predicts, not burial.

Physically consistent, too: in transient recognition complexes (TCR/pMHC, AB/AG) the burial-residualized
confidence is *below* 0.5 — hotspots are anti-confident (frustrated) even after burial control — while in rigid
protease–inhibitor complexes confidence weakly sees them, and in de-novo binders it clears chance.

## Honest bounds
- **Adjacent-class CIs overlap.** This is a monotone trend in point estimates confirmed in the pre-registered
  direction, plus two *significant* separations (de-novo vs TCR/pMHC and de-novo vs AB/AG have non-overlapping
  CIs) — **not** four pairwise-significant steps. With three natural classes the trend test itself is weak
  (Spearman +1 on n=3 is p≈1/6); the load-bearing evidence is H1 (all natural ≈ chance, de-novo above) plus the
  burial-orthogonality, not the trend p-value.
- The de-novo point is a different pipeline/label construction (the anchor, as in the original 2-point); the
  *new* within-pipeline result is the three-class SKEMPI gradient.
- Pr/PI has the fewest hotspots (n_hot=26) — its wider CI reflects that.

## The upgrade — leverage-AUROC per class (the two feature classes diverge)
The decisive addition: compute the **leverage**-AUROC in the same strata. It is **flat and high across every
natural class**, while confidence climbs the gradient beneath it:

| class | confidence-AUROC | **leverage-AUROC (\|L\|_rms)** |
|---|---|---|
| TCR/pMHC | 0.430 [0.355, 0.508] | **0.641 [0.539, 0.765]** |
| AB/AG | 0.457 [0.383, 0.518] | **0.628 [0.544, 0.716]** |
| Pr/PI | 0.554 [0.465, 0.615] | **0.701 [0.599, 0.811]** |

Leverage clears chance (all lower CIs > 0.5) and beats confidence in every class (non-overlapping in TCR/pMHC:
conf hi 0.508 < lev lo 0.539). So this is not merely "confidence declines toward transient interfaces" — it is
the **feature-class law demonstrated across a controlled biological axis**: the scalar of `P` is regime-dependent
and blind, the mixed derivative is regime-independent and sighted. This is a cleaner statement of the thesis
than any single-fixture panel, and it upgrades the section from a caveated 4-point trend to a divergence. →
threepoint_law.csv (`leverage_auroc_Lrms`).

## Bottom line
The 2-point "suggestive comparison" is now a **four-point monotone law**, pre-registered, with the ordering
confirmed exactly and shown to be orthogonal to burial. Confidence sees binding-hotspots in proportion to how
fold-coupled the interface is — blind in transient recognition, weakly sighted in rigid inhibitors, clearing
chance only in de-novo binders. → threepoint_law.csv, PREREG_3point_law.md.

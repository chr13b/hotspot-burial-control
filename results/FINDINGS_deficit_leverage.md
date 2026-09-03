# FINDINGS — #7 (exploratory): do the confidence deficit and leverage stress move together per complex?

**Exploratory, pre-declared honest-null.** SEED=20260803. Script `src/deficit_vs_leverage.py`.
Deliverable `results/deficit_vs_leverage_percomplex.csv` (per complex); stats
`results/deficit_vs_leverage_percomplex_stats.csv`.

```
python3 src/deficit_vs_leverage.py --out results/deficit_vs_leverage_percomplex.csv
```

## Question
The decomposition has two readouts. On predicted backbones, the **confidence** readout degrades into a
burial-matched hotspot-recovery deficit (§6: pooled −0.19 OF3 / −0.22 AF2), while the **mixed-derivative
(leverage)** readout largely survives (R2: CPI retains 84% OF3 / 69% AF2). Do the two degrade *together*
across complexes — a within-complex confirmation that they are the same phenomenon — or are they *separable*?

## Two per-complex quantities (n = 127 shared complexes)
- **Confidence deficit:** `d_of3`, `d_af2` from `results/expD_af2_of3_corr_percomplex.csv` (signed;
  more-negative = larger burial-matched hotspot-recovery deficit).
- **Leverage retention:** per complex, position-matched `mean|L|_rms(predicted) / mean|L|_rms(crystal)` over
  interface positions, from the R2 frames `leverage_predicted_{crystal,of3,af2}_positions.csv`. (A per-complex
  *CPI* is not well-defined — CPI needs a cross-complex bootstrap — so the magnitude-retention proxy the
  handoff sanctioned is used; stated, not hidden.) A raw `mean|L|_rms(predicted)` proxy is reported too.

## Positive controls (rule 6) — both pass
- **Aggregate reproduces R2 exactly.** Re-running `leverage_predicted.py --stage analyse` on the on-disk pq
  caches reproduces the committed pooled CPI(L|geom): crystal **+0.0462**, **OF3 +0.0389**, AF2 +0.0320
  (crystal re-score matches committed L to 1.1×10⁻⁵). So the leverage frames are the R2 data.
- **Crystal control ~0.** `d_crystal` median **+0.013** (mean −0.08), far smaller than the predicted
  deficits (−0.19/−0.22) — the crystal backbone carries essentially no deficit, as §6 requires.

## Result — a WEAK "degrade-together" tendency, significant for AF2 only

Spearman(deficit, leverage-retention), matched predictor, complex-bootstrap 95% CI (5,000 resamples).
Because `d` is signed (negative = deficit) and retention is high = leverage-survived, a **positive** ρ means
big-deficit complexes also have low retention — the two **degrade together**.

| test | Spearman ρ | 95% CI | P(>0) | reading |
|---|---|---|---|---|
| OF3 deficit vs OF3 retention | **+0.162** | [−0.005, +0.322] | 0.972 | degrade-together direction, **CI spans 0** |
| AF2 deficit vs AF2 retention | **+0.192** | [+0.021, +0.356] | 0.986 | degrade-together, **CI excludes 0** |
| OF3 deficit vs raw mean\|L\|rms (proxy) | +0.152 | [−0.021, +0.320] | 0.959 | same direction, CI spans 0 |
| AF2 deficit vs raw mean\|L\|rms (proxy) | +0.179 | [+0.008, +0.346] | 0.980 | same direction, CI excludes 0 |

By deficit quintile the direction is visible in the tails (OF3): the largest-deficit fifth of complexes has
mean retention **0.851**; the smallest-deficit fifth **0.988** — but the middle quintiles are non-monotone
(0.961, 0.914, 0.888), consistent with a weak effect.

## Reading — honest, not spun
Both predictors and both leverage metrics give a **positive correlation in the degrade-together direction**
(bigger confidence deficit ↔ lower leverage retention), but it is **weak**: significant for AF2 (CI excludes
0), not significant for OF3 (CI spans 0, P(>0)=0.97). This is **suggestive-but-underpowered** evidence that
the confidence deficit and leverage stress partly co-occur within complexes — leaning toward "same
phenomenon" rather than "cleanly separable", but far from decisive. Three reasons to keep it exploratory:
1. **n = 127 with sparse hotspots** — the pre-registered underpowered regime the handoff anticipated.
2. **Per-complex leverage magnitude is largely preserved** (retention median 0.98 OF3 / 0.95 AF2), so there is
   little variance in the x-axis to correlate against — the effect lives in a minority of hard complexes.
3. **The retention proxy is magnitude, not discrimination.** The pooled CPI (binding-*discrimination*) drops
   to 84%/69% while per-complex magnitude barely moves — so this proxy understates leverage stress, and a
   discrimination-based per-complex metric (unavailable at n≈few mutations/complex) might correlate more
   strongly. The direction, not the magnitude, is the reportable content here.

**Bottom line:** a weak, direction-consistent hint that the two readouts of the decomposition degrade together
across complexes (AF2 CI>0, OF3 CI spans 0). Reported as exploratory; neither a clean confirmation nor a clean
dissociation.

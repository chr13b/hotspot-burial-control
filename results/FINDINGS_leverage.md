# FINDINGS — the Confidence–Leverage Decomposition (the paper's new spine)

**Script:** `src/leverage_decomposition.py` (built by a Fable-5 max-effort agent). **Summary:**
`results/leverage_decomposition.csv` (88 rows). Caches: `leverage_{pq,skempi_mutations,skempi_positions,
abbind_mutations,bennett_pairs}.csv`. seed 20260803. **Independently re-verified 2026-08-15** (numbers below
reproduce to ≤1e-3).

## The theorem
Per position i, two orthogonal terms of the IF log-likelihood's interaction expansion (partner indicator λ):
- **Confidence (the diagonal)** — a scalar summary of the single distribution p(·|X_bound) at fixed λ=1
  (log p(native), negentropy). Estimates positional constraint.
- **Leverage (the mixed second difference)** `L_i(a) = [log p(a|X_cplx) − log p(wt|X_cplx)] − [log p(a|X_mono)
  − log p(wt|X_mono)]` = the response to ablating the partner. ∝ −ΔΔG_bind (up to an unknown temperature β).

**Blindness proof (existence claim, verified in data):** confidence is a functional of p(·|bound); L is not,
provided p(·|bound) does not determine p(·|mono). Empirical existence: matching interface positions on the FULL
bound distribution (TV<0.02) gives 904 pairs with median |Δconfidence| = 0.004 nats yet median |ΔL_rms| = 0.282
(≈30% of the total leverage spread survives inside a confidence-matched pair). So confidence **cannot express**
leverage — blind by construction, not by failure.

## The result — a FEATURE-CLASS law, not a regime law (the prediction inverted, for the better)
CPI(feature | burial+nbr+ΔSASA). **Position level, SKEMPI natural, 13,401 interface positions, 327 hotspots**
(directly comparable to committed nugget_cpi.csv):

| feature | CPI \| geometry (verified) | within-stratum AUROC |
|---|---|---|
| **confidence** (diagonal) | **+0.00023 [−0.00021, +0.00068]** — BLIND | 0.456 (chance) |
| scalar KL (contraction of L) | +0.00099 [+0.00036, +0.00170] | 0.603 |
| **leverage L(→Ala)** (mixed deriv.) | **+0.00485 [+0.00326, +0.00649]** — ADDS (~21× conf, ~5× KL) | 0.623 |

**Mutation level, SKEMPI:** Spearman(L, ΔΔG_bind) = **−0.295** (verified; Fable −0.301). CPI(L | geometry) =
**+0.0588 [+0.0457, +0.0727]**; survives geometry+BLOSUM+Δvol+Δhydro (+0.0468), its own components
(all-controls +0.0504), and drop-3 influential complexes (+0.0518).

**The regime comparison** (CPI(L|geom)): SKEMPI natural +0.0588 ≫ Bennett de-novo +0.0108 (both survive drop-3);
AB-Bind +0.0155 does NOT survive drop-3 (**indeterminate — 27 complexes, use neither way**). Predicted
natural<de-novo; measured **natural ≫ de-novo**. The "de-novo-specific positive" was a property of the *probe*
(scalar), not the *regime*. **Corrected law:** on natural complexes, *scalar* summaries of the IF distribution
reduce to geometry (confidence exactly, KL nearly); the *mixed derivative* does not.

## Positive controls (all pass)
Re-scoring reproduces committed kl_detector KL (5,742 positions, agreement 1.000, max|ΔKL| 5.4e-7). Algebraic
identity `KL(P‖Q) = E_{a∼P}[L(a)] + [log P(wt)−log Q(wt)]` holds to 8.9e-7 — the formal statement that scalar KL
is ONE functional of the leverage vector. SKEMPI: 4,193 single mutations, 100% mapped, WT-match 100%
(shuffled-position control 18.8%). **Rule-6 catch:** first AB-Bind run returned zero rows (regex omitted the
colon in `D:A488G`) — a false zero avoided.

## Honesty caveats (must survive review)
1. **Orthogonal ≠ independent.** Spearman(confidence, |L|) = +0.075. Say confidence "cannot express" leverage,
   NOT "is uncorrelated with" it.
2. **The log-Z point is OURS, not the free-energy paper's.** arXiv:2506.05596's normaliser is *global
   per-sequence over conformation space* and yields an amino-acid *marginal prior* for **ΔΔG_fold, not binding**
   — do NOT cite it for a per-position pseudo-free-energy. Our genuine point: in L both per-position log-Z terms
   cancel (each bracket is within-conditioning), so **L is better-posed than confidence**, which mixes native
   free energy with a position-dependent normaliser.
3. **Temperature β unknown** → L ∝ −ΔΔG only up to scale; no kcal/mol readout. All our readouts are scale-invariant.
4. **Rigid backbone:** X_mono = complex backbone minus partner; mutant assumed WT backbone.
5. **CPI is not formally commensurable across fixtures** (per-obs log-loss); base rates (48–54%) and geometry
   baselines (0.67–0.70) are close, so the 5× gap isn't a scale artifact, but the natural≫de-novo gap is
   suggestive, not a formal difference test. Label quality differs (SKEMPI quantitative, Bennett binary).
6. One model (ProteinMPNN v_48_020), backbone-only marginals.

## Provenance — credit the score, claim the decomposition
The operator L is **BA-Cycle** (Jiao, Mao, Jin et al. 2024, arXiv:2410.09543, *"Boltzmann-Aligned Inverse
Folding…"* — the paper's own name is BA-Cycle, not "BAIF"; their Eq. 10 rearranges to our double difference).
Credit it outright. Verified by exhaustive enumeration of their tables/figures/appendix: **they run no
beyond-geometry control (no burial/rSASA/ΔSASA/contact) and no natural-vs-de-novo / feature-class split.** So
the decomposition, the blindness theorem, the beyond-geometry control, and the feature-class law are **not
scooped**; the score is theirs. They use whole-sequence autoregressive likelihoods; we use per-position
sequence-free marginals (design-time usable, order-free).

## Draft corrections this forces
- **§4 "Why de-novo, and only de-novo" is FALSIFIED by our own primary fixture** (L on SKEMPI +0.0588). Rewrite
  as the feature-class law; add the §4a decomposition (theorem + this table).
- **AB-Bind logP null is readout-dependent:** abbind_bigidea1.csv ΔAUROC +0.008 ("adds nothing") becomes CPI
  +0.0417 [+0.022, +0.061] on the same fixture. The ΔAUROC readout is the one with the −0.021 noise floor.
- Abstract + §1: promote the decomposition; the model DOES know binding on natural complexes, in the mixed
  derivative, invisible to confidence and scalar summaries.

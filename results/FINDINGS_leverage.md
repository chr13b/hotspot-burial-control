# FINDINGS — the Confidence–Leverage Decomposition (the paper's new spine)

**Script:** `src/leverage_decomposition.py` (built by a Fable-5 max-effort agent). **Summary:**
`results/leverage_decomposition.csv` (88 rows). Caches: `leverage_{pq,skempi_mutations,skempi_positions,
abbind_mutations,bennett_pairs}.csv`. seed 20260803. **Independently re-verified 2026-08-15** (numbers below
reproduce to ≤1e-3). **Adversarially audited 2026-08-15 (Fable-5): VERDICT SOUND** — code, measurements,
sign-chain (mean L −1.21 destabilising / +0.49 stabilising), monomer ablation and no-circularity all correct;
the CPI conditional-independence test does NOT leak (pure-noise features → ~0 spanning 0; y itself → +0.388;
null floor ≈1e-4, 16× below the position headline / 60× below the mutation headline). Every defect the audit
found was in the write-up and is corrected above/below.

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
(committed values from leverage_decomposition.csv):

| feature | CPI \| geometry (committed) | within-stratum AUROC (hotspot-predicting orientation) |
|---|---|---|
| **confidence** (diagonal) | **+0.00019 [−0.00025, +0.00061]** — BLIND | 0.471 [0.436, 0.508] — chance |
| scalar KL (contraction of L) | +0.00093 [+0.00028, +0.00163] | 0.598 |
| **leverage L(→Ala)** (mixed deriv.) | **+0.00484 [+0.00328, +0.00649]** — ADDS (~5× KL; confidence adds nothing) | 0.624 |

NB the 13,401-position sample is NOT on the same per-observation CPI scale as committed nugget_cpi.csv (5,742
positions / 141 complexes / higher base rate) — the two are not numerically comparable, only qualitatively.
On the *identical* 5,742-position nugget sample (results/leverage_nugget_match.csv, committed
2026-08-16): confidence +0.00043 [−0.00028, +0.00113] (CI spans 0 — conditionally independent), scalar KL
+0.00192 [+0.00054, +0.00332], leverage L(→Ala) **+0.00918 [+0.00621, +0.01237]** — L is **4.8× the scalar KL**
and confidence adds nothing beyond geometry. (Within-stratum AUROC column corrected 2026-08-15: earlier draft
mixed +feature/−feature orientations; each row now in its hotspot-predicting sign.)

**Mutation level, SKEMPI:** Spearman(L, ΔΔG_bind) = **−0.301** [−0.354, −0.243] (committed). CPI(L | geometry) =
**+0.0588 [+0.0457, +0.0727]**; survives geometry+BLOSUM+Δvol+Δhydro (+0.0468), its own components
(all-controls +0.0504), and drop-3 influential complexes (+0.0518).

**The regime comparison** (CPI(L|geom)): SKEMPI natural +0.0588 ≫ Bennett de-novo +0.0108 (both survive drop-3);
AB-Bind +0.0155 [−0.0002, +0.0321] is conditionally independent already at the primary bar — CI spans 0 (**indeterminate — 27 complexes, use neither way**). Predicted
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

**RedNet (contrastive decoding, 2026) is the design-time twin of L — credit it too, and DO NOT build a
"leverage-tilted decoder."** Verified from their released code (github.com/zw2x/rednet_public, fetched
2026-08-15): `sampling_utils.py` decodes with `logits = (1+α)·logit_bound − α·logit_apo` and `infer_pipeline.py`
builds the apo contrast by deleting the partner chain — i.e. p̃ ∝ p(·|complex)·exp(α·L), our mixed derivative
applied at sampling time, with an α-sweep in their configs. So a training-free leverage-tilted decoder is
**already published** (KILL, per idea-critic 2026-08-15), months before ICLR 2027. It does NOT scoop the
diagnostic paper: RedNet runs no beyond-geometry control, no decomposition, no scalar-vs-mixed / regime split.
Position (now in §8): credit RedNet as the actionable operationalization of the leverage; keep our theorem +
beyond-geometry control + feature-class law as the contribution. (They use their trained net for the contrast,
not a frozen off-the-shelf model — the only unrun cell, and a thin one, not a paper.)

## Draft corrections this forces
- **§4 "Why de-novo, and only de-novo" is FALSIFIED by our own primary fixture** (L on SKEMPI +0.0588). Rewrite
  as the feature-class law; add the §4a decomposition (theorem + this table).
- **AB-Bind logP null is readout-dependent:** abbind_bigidea1.csv ΔAUROC +0.008 ("adds nothing") becomes CPI
  +0.0417 [+0.022, +0.061] on the same fixture. The ΔAUROC readout is the one with the −0.021 noise floor.
- Abstract + §1: promote the decomposition; the model DOES know binding on natural complexes, in the mixed
  derivative, invisible to confidence and scalar summaries.

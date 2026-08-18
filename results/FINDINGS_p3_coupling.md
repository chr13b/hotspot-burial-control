# Phase 3 — Coupling extension: does the model know binding EPISTASIS?

**Claim tested.** StaB-ddG (arXiv:2507.05502) is an inverse-folding-as-folding-energy ΔΔG_bind predictor;
its pairwise/epistasis behaviour is untested. We measure directly whether the inverse-folding likelihood
carries pairwise *binding epistasis*: the model's SECOND mixed derivative (partner-ablated pairwise
coupling `C_ij`) vs the experimental epistasis energy `g_ij = ΔΔG_ab − ΔΔG_a − ΔΔG_b` from SKEMPI double
mutants whose two singles are also measured. The operator is the partner-ablated analogue (for inverse
folding and *binding*) of the **categorical Jacobian** — Zhang, Wayment-Steele, …, Ovchinnikov 2024,
bioRxiv 10.1101/2024.01.30.577970 (VERIFIED; used there to read coevolutionary couplings from a protein
LANGUAGE model). That a second sequence-difference measures epistasis is Nambiar 2025's framing
(bioRxiv 10.1101/2025.09.14.676130).

## Measurement
Conditional (autoregressive, teacher-forced) ProteinMPNN v_48_020. `C_ij(a,b)` = the shift in the
(a-vs-wt) conditional log-odds at position i when the input residue at j is set to its mutant b, for
decode orders where j precedes i, symmetrised over the two directions and averaged over 8 decode orders.
Because C is a difference of log-*odds* at a fixed position, the per-position normalisation cancels (the
log-softmax conditionals difference directly). By the thermodynamic cycle `L ~ -ΔΔG`, so `C_lev ~ -g` and
we EXPECT a **negative** Spearman(C, g).

Partner ablation:
 * **cross-interface** pair (i in group1, j in group2): no single monomer contains both, so `C_monomer = 0`
   and `C_lev = C_complex`. The clean set.
 * **same-side** pair (both in one group): `C_lev = C_complex − C_monomer` (removes intra-fold coupling).

Commands (SEED=20260803):
```
python3 src/p3_coupling.py --stage score  --seeds 8 --order-batch 2 --threads 6 --max-residues 800 \
        --out results/p3_coupling.csv
python3 src/p3_coupling.py --stage analyse --out results/p3_coupling.csv
python3 src/p3_sign_verify.py            # sign-channel + double-count audit responses
python3 src/p3_coupling_biascheck.py     # exclusion-bias + size-dependence
```
Data: 557 triangles over 61 complexes (383 cross-interface, 174 same-side) after canonicalising
swapped-order double mutants to one physical pair each (5 duplicate pairs merged; the model coupling is
bit-identical under the swap — an order-invariance check that passes exactly). OOM-guard dropped 14
complexes >800 residues (27 triangles — mostly TCR/pMHC + antibody Fabs; a lower bound on coverage).

## Result — a real, modest, distance-independent binding-epistasis signal

| set | n | complexes | Spearman(C_lev,g) | **partial \| distance** | P(<0) |
|---|---|---|---|---|---|
| all | 557 | 61 | −0.138 [−0.228,−0.065] | **−0.120 [−0.213,−0.052]** | 0.998 |
| **cross-interface** | 383 | 28 | −0.143 [−0.250,−0.049] | **−0.129 [−0.252,−0.039]** | 0.997 |
| same-side | 174 | 44 | −0.106 [−0.237,+0.012] | −0.118 [−0.253,+0.019] | 0.955 |

The distance control barely moves the estimate (−0.138→−0.120), so the signal is **not** merely "are they
in contact." CI excludes 0 for the full and cross-interface sets.

**Method-consistent CPI** (project estimator, binary outcome |g|>0.5, control = Cβ–Cβ distance):
 * all: CPI(|C_lev| | dist) = **+0.0218 [+0.0125,+0.0334]** P(>0)=1.000
 * cross: CPI = **+0.0168 [+0.0071,+0.0289]** P(>0)=1.000
   → drop 3 most-influential complexes (1BRS, 1KNE, 1LFD): **+0.0069 [+0.0003,+0.0161] — SURVIVES**

**Not a sign-skew artifact** (the obvious objection: C is 76% positive while g is 54% negative, so a pure
magnitude relation plus skew could manufacture a negative Spearman). It does not: the distance-controlled
partial ρ is negative in *both* sign strata — g<0: −0.119 (all) / −0.098 (cross); g>0: −0.088 / −0.093.
Under the artifact hypothesis the g>0 stratum would be *positive*.

## Partner ablation does the work (the mechanism, not an artifact)
Same-side pairs, identical 174 rows, partial | distance:
 * un-ablated `C_complex`: **+0.014 [−0.152,+0.147]** P(<0)=0.447 — **null**
 * ablated   `C_lev`:      **−0.118 [−0.253,+0.019]** P(<0)=0.955 — signal

Subtracting the monomer (intra-fold) coupling exposes the binding coupling — parallel to single mutations.

## Positive controls (rule 6) — all pass
1. **Additivity, distance-controlled.** Model coupling magnitude tracks epistasis magnitude *beyond*
   distance: partial Spearman(|C|,|g| | dist) = **+0.214** (all) / **+0.180** (cross). (The raw |g|-tertile
   means 0.10→0.18→0.29 are partly distance-driven — |g| tertiles differ in mean Cβ distance — so the
   distance-controlled partial is the honest number; it is monotone within every distance quartile.)
2. **Direction symmetry.** Spearman(C_{i→j}, C_{j→i}) = **+0.610** (n=553). The two directions are read
   from *disjoint* decode-order subsets, so this is simultaneously a symmetry check and a decoding-order
   stability check.
3. **Contact split** (cross-interface, partial | distance). Strongest *within* contacts
   (−0.156 [−0.332,−0.030], n=137) and weaker but same-sign in non-contacts (−0.079 [−0.264,+0.058]) —
   not an artifact of the contact boundary.

## Honest limitations
 * **Modest.** partial-Spearman ≈ −0.12 to −0.13 is about **half** the single-mutant leverage's −0.30.
   Pairwise epistasis is a smaller, subtler object than single-site ΔΔG. Independent floor: SKEMPI's own
   reproducibility on the 5 pairs measured twice is mean |Δg| = 0.23, **max 1.03 kcal/mol** for the *same
   physical pair* — so a modest correlation is close to what the data can support.
 * **Sign is near chance; only a small genuine sign channel survives.** Per-pair sign accuracy is 0.542
   (all pairs). It does NOT improve in a meaningful sense on large-|g| or large-|C| subsets: those raw
   accuracies (0.62–0.68) sit *at or below* the trivial majority-class baseline (|g|>1.0: model 0.625 vs
   majority 0.694; |C|>p90: 0.679 vs 0.696), because those subsets are class-imbalanced — a class-imbalance
   artifact, not recovery. The genuine, chance-corrected channel is small: partial rank-corr(C_lev, 1[g<0]
   controlling |g| AND distance) = **+0.079 [+0.011,+0.174]** (all; cross +0.088 [−0.003,+0.214], marginal),
   and model-side |C|>p75 gives balanced accuracy 0.604 / MCC +0.224. So the model carries ~2–3 points of
   real sign information, not 15; the correlation is carried by magnitude, not per-pair sign.
 * **Coverage limitation (low power), no detectable outcome-side difference.** The 14 complexes / 27
   triangles dropped by the memory guard (n>800 res) are a lower bound. On the available (weak) evidence
   they are not a biased slice — their experimental epistasis distribution is not distinguishable from the
   retained set (KS |g| p=0.63, Mann-Whitney p=0.44; n=27, low power), and within the retained set the
   effect is if anything *stronger* in larger complexes (partial ρ: small n≤372 −0.077, large 372<n≤800
   −0.186), so excluding the largest is more likely conservative than inflationary. But the dropped regime
   (mostly flat CDR-dominated TCR/pMHC + Fab interfaces, where ProteinMPNN is weakest) is not directly
   measurable, so this is a coverage caveat, not a proof of no bias.
 * **Effective sample is concentrated.** The cross set's 383 pairs are dominated by a few deeply-scanned
   complexes (1JTG, 3S9D, 1BRS ≈ 43%); the complex-clustered bootstrap (why the CI is wide) and the
   drop-3-influential test are the correct responses, and it survives both. 28 cross-interface complexes.

## Bottom line
First direct measurement of whether an inverse-folding likelihood carries pairwise *binding* epistasis:
on natural complexes it carries a **real but modest** binding-coupling signal that **is not reducible to
inter-residue distance**, **requires partner ablation to surface**, and **is not a sign-skew artifact** —
though it is a signal about coupling *magnitude*, with only a small genuine per-pair *sign* channel. It
extends the paper's thesis from the first mixed derivative (single-site leverage) to the second (epistasis):
what the model knows about binding lives in the derivative structure of its distribution, not its confidence.

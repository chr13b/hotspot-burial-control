# Phase 3 — Coupling extension: does the model know binding EPISTASIS?

**Claim tested.** StaB-ddG (arXiv:2507.05502, appendix B) claims — untested — that inverse-folding
likelihoods carry pairwise *binding epistasis*, not just single-mutant effects. We measure it directly:
the model's SECOND mixed derivative (partner-ablated pairwise coupling `C_ij`) vs the experimental
epistasis energy `g_ij = ΔΔG_ab − ΔΔG_a − ΔΔG_b` from SKEMPI double mutants whose two singles are also
measured. Second-difference = epistasis credited to Nambiar 2025 (bioRxiv 10.1101/2025.09.14.676130).

## Measurement
Conditional (autoregressive, teacher-forced) ProteinMPNN v_48_020. `C_ij(a,b)` = shift in the
(a-vs-wt) conditional log-odds at position i when the input residue at j is set to its mutant b,
symmetrised over the two directions and averaged over 8 decode orders (only orders with the conditioner
decoded before the conditionee contribute). Because `C` is a difference of log-*odds*, per-position
normalisation cancels — raw conditional logits suffice. By the thermodynamic cycle `L ~ -ΔΔG`, so
`C_lev ~ -g` and we EXPECT a **negative** Spearman(C, g).

Partner ablation:
 * **cross-interface** pair (i in group1, j in group2): no single monomer contains both, so
   `C_monomer = 0` and `C_lev = C_complex`. The clean binding-epistasis set.
 * **same-side** pair (both in one group): `C_lev = C_complex − C_monomer` (removes intra-fold coupling).

Commands (SEED=20260803):
```
python3 src/p3_coupling.py --stage score  --seeds 8 --order-batch 2 --threads 6 --max-residues 800 \
        --out results/p3_coupling.csv
python3 src/p3_coupling.py --stage analyse --out results/p3_coupling.csv
```
Data: 562 triangles over 61 complexes (388 cross-interface, 174 same-side). OOM-guard dropped 14
complexes >800 residues (28 triangles — mostly TCR/pMHC and antibody Fabs; a lower-bound on coverage,
logged): 1BD2 1YY9 2NYY 3D3V 3LZF 3QDG 3QDJ 3VR6 4CVW 4FTV 4GNK 4GXU 4K71 4L3E.

## Result — the model carries a real, modest, distance-independent binding-epistasis signal

| set | n | complexes | Spearman(C_lev,g) | **partial \| distance** | P(<0) |
|---|---|---|---|---|---|
| all | 562 | 61 | −0.151 [−0.239,−0.075] | **−0.132 [−0.223,−0.063]** | 0.999 |
| **cross-interface** | 388 | 28 | −0.160 [−0.265,−0.056] | **−0.144 [−0.263,−0.050]** | 0.998 |
| same-side | 174 | 44 | −0.106 [−0.237,+0.012] | −0.118 [−0.253,+0.019] | 0.955 |

The distance control barely moves the estimate (−0.151→−0.132), so the signal is **not** merely "are
they in contact." CI excludes 0 for the full and cross-interface sets.

**Method-consistent CPI** (project estimator, binary outcome |g|>0.5, control = Cβ–Cβ distance):
 * all: CPI(|C_lev| | dist) = **+0.02251 [+0.01398,+0.03187]** P(>0)=1.000
 * cross: CPI = **+0.01547 [+0.00746,+0.02394]** P(>0)=1.000
   → drop 3 most-influential complexes (1BRS, 1JTG, 4G0N): **+0.00725 [+0.00191,+0.01468] — SURVIVES**

## Partner ablation does the work (the mechanism, not an artifact)
Same-side pairs, partial | distance:
 * un-ablated `C_complex`: **+0.014 [−0.152,+0.147]** P(<0)=0.447 — **null**
 * ablated   `C_lev`:      **−0.118 [−0.253,+0.019]** P(<0)=0.955 — signal

Subtracting the monomer (intra-fold) coupling is what exposes the binding coupling — exactly parallel to
the single-mutant leverage story.

## Positive controls (rule 6) — all pass
1. **Additivity dose-response.** mean|C_lev| rises monotonically with |g| tertile:
   0.103 (low) → 0.175 (mid) → 0.289 (high). Model coupling magnitude tracks epistasis magnitude.
2. **Direction symmetry.** Spearman(C_{i→j}, C_{j→i}) = **+0.604** (n=558) — the operator is ~symmetric;
   no directional bug.
3. **Contact split** (cross-interface, partial | distance). Signal is *strongest within contacts*
   (−0.166 [−0.349,−0.042], n=139, 22 cplx) and weaker but same-sign in non-contacts
   (−0.095 [−0.282,+0.053], n=249) — not an artifact of the contact boundary.

## Honest limitations
 * **Modest.** partial-Spearman ≈ −0.14 is about **half** the single-mutant leverage's −0.30. Pairwise
   epistasis is a smaller, subtler object than single-site ΔΔG and is predicted less well.
 * **Sign of individual pairs is weak.** sign(−C_lev)=sign(g) only ~0.53 (barely above chance): the model
   ranks coupling *magnitude* (CPI, dose-response) far better than it calls the *sign* (cooperative vs
   buffering) of any single pair. The rank correlation is carried by magnitude, not per-pair sign.
 * **Coverage.** 14 large complexes (28 triangles) dropped by the memory guard — a lower bound.
 * **28 cross-interface complexes** for the clustered bootstrap (moderate, not large).

## Bottom line
First direct measurement of StaB-ddG's untested epistasis claim: on natural complexes the inverse-folding
distribution carries a **real but modest** binding-coupling signal that **is not reducible to inter-residue
distance** and **requires partner ablation to surface**. It extends the paper's thesis from the first mixed
derivative (single-site leverage) to the second (epistasis): what the model knows about binding lives in
the derivative structure of its distribution — at both orders — not in its confidence.

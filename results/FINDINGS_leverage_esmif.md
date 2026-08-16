# FINDINGS — the Confidence–Leverage Decomposition REPLICATES under ESM-IF1 (model-generality)

**Script:** `src/leverage_esmif.py` (pure scorer swap; reuses leverage_decomposition's verified CPI/leverage
code). **Outputs:** `results/leverage_esmif.csv` (39 rows), `leverage_esmif_{mutations,positions}.csv`. Cache
`leverage_pq_skempi_esmif.csv` (gitignored). Model: `esm_if1_gvp4_t16_142M_UR50` (GVP-transformer, 142M,
causal, native-teacher-forced conditional readout). **337/344 SKEMPI complexes** (7 giants >1200 residues
dropped for memory: 1KBH, 2NYY, 2NZ9, 3VR6, 4GXU, 4LRX, 4NM8), mean top-1 recovery 0.598. seed 20260803.

This answers the "one model (ProteinMPNN)" limitation (§9f). ESM-IF1 is doubly different: a different
architecture (GVP-transformer vs MPNN GNN) AND a different per-position readout (native-teacher-forced
*conditional* vs sequence-free *unconditional* marginal). The decomposition holds under both.

## Verdict: REPLICATES (direction + structure), with somewhat smaller magnitudes

### Position level (hotspot-ness | burial+nbr+ΔSASA), the nugget unit
| feature | ProteinMPNN (committed) | ESM-IF1 |
|---|---|---|
| **confidence** (diagonal) | +0.00019 [−0.00025, +0.00061] — BLIND | **−0.00004 [−0.00026, +0.00019] — BLIND** |
| scalar KL | +0.00093 | +0.00103 |
| **leverage L(→Ala)** | +0.00484 [+0.0033, +0.0065] | **+0.00424 [+0.0027, +0.0057]** (survives drop-3 → +0.0035) |

→ The feature-class law replicates cleanly: confidence blind, leverage adds ~4× the scalar KL.

### Mutation level (destabilising ΔΔG_bind ≥ 1 | geometry), 2,900 mutations / 280 complexes
| quantity | ProteinMPNN | ESM-IF1 |
|---|---|---|
| Spearman(L, ΔΔG) | −0.301 | **−0.255 [−0.316, −0.190]** (theory: <0) ✓ |
| CPI(L \| geometry) | +0.0588 | **+0.0350 [+0.026, +0.045]** ✓ |
| CPI(L \| geom+BLOSUM+ΔVol+ΔHydro) | +0.0468 | +0.0270 ✓ |
| CPI(L \| geom+**confidence**) | +0.0558 | **+0.0180 [+0.012, +0.024]** ✓ (L adds beyond confidence) |
| CPI(L \| ALL: geom+subst+conf+KL) | +0.0504 | **+0.0099 [+0.005, +0.015]** ✓ (still CI>0) |

→ Leverage adds binding information beyond geometry AND beyond every scalar (including confidence) under both
models. This is the decomposition's core empirical content and it replicates.

### Blindness-by-construction (theorem demo)
ESM-IF1: 2,520 distribution-matched interface-position pairs (TV<0.02), median |ΔL_rms| = 1.272 against overall
SD(L_rms) = 1.749 (**73% of the leverage spread survives confidence/distribution matching**) while median
|Δconfidence| within pairs = 0.0037. Confidence cannot express leverage — as for ProteinMPNN (~30–44%), only
more so. Identity KL = E_P[L] + [logP(wt)−logQ(wt)] holds to 3.6e-15.

## Honest differences to state (not problems, but report them)
1. **Magnitudes are smaller under ESM-IF1** (CPI(L|geom) +0.035 vs +0.059; Spearman −0.26 vs −0.30). Direction,
   structure, and all control-set survivals replicate; the point estimate is architecture-dependent.
   **The 7 dropped giants are NOT the cause** (drop-bias control, leverage_dropcheck.csv): re-running MPNN on
   the same 337-complex subset gives CPI(L|geom) +0.0601 and Spearman −0.303 — unchanged from full-344 (+0.0584,
   −0.301); the 7 are only 2.7% of positions / 1.7% of mutations. So the gap is a genuine model/readout effect.
   Remaining candidate causes (not a confound, just the mechanism): the native-teacher-forced *conditional*
   readout vs MPNN's sequence-free *unconditional* marginal, and ESM-IF1 using only N/CA/C atoms (no O, no CB).
2. **ESM-IF1 confidence is a STRONGER mutation-level constraint signal than MPNN's.** At the mutation level
   CPI(confidence|geom) = **+0.0226** for ESM-IF1 vs **+0.0100** for MPNN, and Spearman(confidence, ΔΔG) = +0.245
   vs +0.184. This is NOT a contradiction: confidence = fold-stability *constraint*, and removing a
   strongly-preferred native residue is destabilising, so confidence carries per-mutation ΔΔG signal in *both*
   models. ESM-IF1's readout is teacher-forced on native context, a richer constraint estimate, so it carries
   more. The decomposition's claim is intact because **leverage still adds beyond confidence** (+0.018, CI>0).
   The "confidence is blind" headline is a *position-level hotspot* claim (CPI −0.0000 here), which replicates.
   Scalar KL, by contrast, is blind at the ESM-IF1 mutation level (−0.0013, CI spans 0) — scalars stay weak.
3. **Confidence–leverage correlation is model-dependent:** Spearman(confidence, |L|) = +0.075 (MPNN) → +0.305
   (ESM-IF1). Orthogonal ≠ independent, more so for ESM-IF1 — hence the "cannot express" (not "uncorrelated")
   framing, now stated in §9(a).

## Draft updates made
§9(f) rewritten (model-generality replicated), §9(a) correlation range, §4 corroboration sentence
("not a ProteinMPNN artifact… a property of the inverse-folding class"). → leverage_esmif.csv.

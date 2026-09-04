# FINDINGS — ATLAS (TCR–pMHC): a BOUNDED-GENERALIZATION result, not an independent replication

**Pre-registration** `results/PREREG_atlas.md` (frozen; committed before any number). `SEED=20260803`.
This was a *bonus* second-fixture test (CLAUDE.md order-of-work): does the confidence–leverage decomposition
hold on a genuinely non-overlapping natural ΔΔG_bind fixture? **Verdict: the leverage RANK-DIRECTION replicates
in both models, but the pre-registered geometry-controlled CPI criterion is not met — and the fixture is too
small and too SKEMPI-overlapping to power an independent test. A null that bounds generality; it does NOT refute
the SKEMPI result (rule 2).** Raw: `results/atlas_fixture.csv`, `results/atlas_leverage.csv` (per-mutation, both
models), `results/atlas_summary.csv` (every stat below).

```
# fixture (direct author-number map + mandatory WT-identity gate); score both models; analyse
python3 src/leverage_atlas.py --stage build-fixture
srun -p dev  ... python3 src/leverage_atlas.py --stage score --model mpnn   --device cpu   # env.sh
srun -p normal ... source $SCRATCH/ftax/env_esmif.sh; python3 src/leverage_atlas.py --stage score --model esmif --device cpu
python3 src/leverage_atlas.py --stage analyse
```

## Fixture & the mapping (Phase 1–2)
ATLAS Mutants_052915.tsv = 418 rows → 207 with a real ΔΔG_bind and a TCR mutation → **171 single-point**
(36 multi-point/non-canonical skipped, first pass). Structure per mutation = the block's WT-row `true_PDB`
(mutant rows carry `true_PDB=N/A`); chain groups from `CDR_seqs.txt` (g1 = TCR, g2 = pMHC — deleted for the
monomer pass); scored chain = the row's `TCR_PDB_chain`.

**Mapping method: DIRECT** — ATLAS `num` is the WT-PDB **author residue number** (verified: 1MI5 chain D res 25
= SER = the wt of `S25A`). **No ANARCI needed.** The prompt's suspected "numbering ≠ raw-PDB" was a chain/icode
artifact, not a scheme mismatch.

**WT-identity gate (mandatory).** After mapping, the structure residue's identity MUST equal the mutation's wt
letter, else dropped+logged. **Pass rate = 91.5 % (97/106 unique substitutions; 157/171 = 91.8 % at row level).**
The 9 dropped substitutions are systematic within a few atypical structures (3PL6 = class-II, offset numbering;
3MV8/3MV9 = renumbered; a couple of positions already-Ala in the deposited crystal) — never force-mapped.

## Positive controls (rule 6) — ALL PASS
1. **Scorer sanity.** ProteinMPNN native (native-context, 8-order) recovery on TCR interface = **0.352**;
   ESM-IF1 = **0.338** (142 positions, 7 complexes; pre-reg band ~0.3–0.5 — TCR CDRs run at the low end). ✓
2. **Determinism.** Bit-identical re-score, both models: `max|Δlogp| = 0.00e+00`. ✓
3. **SKEMPI-overlap (the critical control) — FIRES, as feared.** **85 % of ATLAS interface mutations (81/95) sit
   on structures already in SKEMPI 2.0** (1AO7, 1MI5, 2AK4, 2VLR). 1AO7 alone is 46 of them. The genuinely
   **non-overlapping remainder is 16 substitutions over 3 structures, dominated by one complex (2VLJ = 11)** — so
   ATLAS **cannot** provide a powered *independent* replication. This was pre-registered as the make-or-break
   control and it is decisive.

## Result — H1 half-met, H2 confirmed (both models agree)

**H1a — Spearman(L, ΔΔG_bind) < 0: REPLICATES (both models).** Complex-clustered bootstrap 95 % CI.

| model | subset | n (cx) | Spearman(L, ΔΔG) | 95% CI |
|---|---|---|---|---|
| MPNN | full | 76 (7) | **−0.220** | [−0.621, −0.151] |
| ESM-IF1 | full | 76 (7) | **−0.324** | [−0.447, −0.061] |
| MPNN | non-overlap | 12 (3) | −0.585 | [−1.000, −0.427] |
| ESM-IF1 | non-overlap | 12 (3) | −0.371 | [−1.000, −0.151] |

Both full-set CIs exclude 0 and bracket the SKEMPI value (−0.30). The **confidence/leverage dissociation holds
directionally**: in both models `confidence` and `scalarKL` correlate **positively** with ΔΔG (+0.19/+0.17 MPNN,
+0.19/+0.15 ESM-IF1) while `L` is negative — confidence points the wrong way, leverage the right way, as on
SKEMPI. The non-overlap CIs hit −1.0 (3 complexes → uninformative width).

**H1b — CPI(L | burial+nbr+ΔSASA) > placebo floor: NOT MET (both models).**

| model | CPI(L\|geometry) | 95% CI | P(>0) | placebo floor | clears? |
|---|---|---|---|---|---|
| MPNN | −0.0017 | [−0.027, +0.014] | 0.43 | +0.0445 | **no** |
| ESM-IF1 | +0.0095 | [−0.015, +0.035] | 0.85 | +0.0495 | **no** |

Both point estimates ≈ 0 and below the floor. **But the floor is the story:** at +0.045–0.049 it is ~60× the
SKEMPI floor (+0.0007) — inflated by the complex-clustered bootstrap over **only 7 complexes**. It sits *on top of*
the SKEMPI-sized effect (CPI ≈ +0.036–0.059), so this test **cannot resolve an effect of the SKEMPI magnitude on
ATLAS** — nothing real or placebo clears +0.045 with 7 clusters. This is *underpowered / cannot-confirm*, not
*confirmed-absent*. On the non-overlap subset CPI is **not even estimable** (one-class CV folds; 3 destabilising
over 3 complexes) → **INDETERMINATE**.

**H2 — CPI(confidence | geometry) ≤ floor → confidence is BLIND: CONFIRMED (both models).**
MPNN −0.0085 [−0.027, −0.002] and ESM-IF1 −0.0008 [−0.009, +0.005], both ≤ their placebo floors (+0.046/+0.045),
over 50 positions. The decomposition's negative-control direction holds on TCR–pMHC.

## Reading against the pre-registered falsifier (verbatim, rule 1)
> *"If Spearman(L, ΔΔG) ≥ 0, **or** CPI(L | geometry) does not clear the placebo floor, the decomposition does not
> replicate on TCR–pMHC — reported verbatim as a bounded-generalization result."*

The **CPI(L | geometry) condition is not met in either model**, so the falsifier fires: **by the pre-registered
CPI criterion the decomposition does not replicate on TCR–pMHC.** This is reported as a **bounded generalization**,
qualified by two facts that were themselves pre-registered concerns:
1. **Power.** 7 complexes with interface mutations; the placebo floor (+0.045) engulfs the SKEMPI-magnitude effect.
   The CPI test is underpowered to detect ±0.05 here; the *more powered* rank test (Spearman) does replicate.
2. **Independence.** 85 % SKEMPI-overlap; the non-overlapping subset (16 substitutions, 3 complexes, one dominant)
   is indeterminate. ATLAS is substantially a SKEMPI subset and is treated as non-independent.

**Bottom line.** The leverage *rank-direction* (Spearman(L, ΔΔG) < 0) and the *confidence-blind* property (H2)
generalize to TCR–pMHC in both ProteinMPNN and ESM-IF1; the stronger geometry-controlled CPI add-on could **not**
be confirmed on this fixture, because ATLAS is too small and too SKEMPI-overlapping to power it. Per CLAUDE.md
rule 2 this **bounds the generality of the SKEMPI result; it does not refute it.** No clean large non-overlapping
natural ΔΔG_bind fixture exists (PREREG §9); an experimental TCR–pMHC leverage test would need a purpose-built
mutation panel, not a re-mined database.

## Honest scope
TCR–pMHC is germline-biased, μM-affinity, peptide-in-groove recognition — different physics from general PPI.
`L` here is an inverse-folding proxy, not measured binding energy. `n`=76 interface mutations / 7 complexes (full),
12 / 3 (non-overlap); WT-gate 91.5 %; SKEMPI-overlap 85 %. Numbers → `results/atlas_summary.csv`,
`results/atlas_leverage.csv`. Third-party ATLAS data (`data/atlas/`) is gitignored.

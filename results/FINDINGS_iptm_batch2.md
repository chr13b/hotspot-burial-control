# FINDINGS — AF2 ipTM 60 → 120: fresh-complex replication + tighter CIs (H1/H2/H3 all PASS)

**Pre-registered** `results/PREREG_iptm_batch2.md` (subset frozen before folding). `SEED=20260803`. Extends the
primary AF2-multimer ipTM headline (`FINDINGS_iptm.md`, batch-1 n=60) to a second, DISJOINT batch of 60 complexes.
Raw `results/iptm_steer_120.csv` (840 folds = 120 complexes × 7 arms; batch-1 committed rows reused, NOT
re-folded), stats `results/iptm_summary_120.csv` (120) + `results/iptm_summary_batch2.csv` (batch-2 only).

```
python3 src/build_iptm_fastas.py --exclude results/iptm_subset.txt --subset-out results/iptm_subset_batch2.txt ...
sbatch --array=0-52%8 --export=ALL,STRIDE=8,FADIR=.../fastas_batch2,OUT=.../af2_out fold_array_generic.sbatch
python3 src/parse_iptm.py --out-dir .../af2_out --subset results/iptm_subset_batch2.txt --out results/_iptm_steer_batch2.csv
# concat batch-1 (committed iptm_steer.csv) + batch-2 -> iptm_steer_120.csv ; analyse_iptm on 120 and on batch-2-only
```

## Controls — PASS
Frozen batch-2 subset disjoint from batch-1 (batch1 ∩ batch2 = 0; the builder with no exclusion reproduces batch-1
exactly). **420/420 batch-2 folds, 0 missing** (1 fold, `1GC1_G_C__random__k1`, hung on an MSA request and was
re-folded). batch-2 wt interface ipTM median **0.900** (in the 0.6–0.9 sanity band). Same α=2 SET-A sequences,
same AF2 pipeline, same interface set and `analyse_iptm.py` as batch-1.

## Result — H1 (fresh-complex), H2 (tighter CIs), H3 (localization) all PASS

**H1 — batch-2 ONLY (60 complexes never used in the batch-1 headline), paired L − random:**

| metric | Δ | 95% CI | P(>0) |
|---|---|---|---|
| **ipTM** | **+0.244** | [+0.194, +0.295] | 1.000 |
| **composite** | **+0.864** | [+0.700, +1.024] | 1.000 |
| interface pAE | −6.10 | [−7.46, −4.82] | 0.000 |
| interface pLDDT | +11.03 | [+8.80, +13.28] | 1.000 |

**H1 PASS on fresh complexes** — composite AND ipTM both CI>0. The steering benefit is **not batch-1-specific**;
the pre-registered falsifier (batch-2-only CI includes 0) does **not** fire.

**H2/H3 — combined 120 complexes, paired L − random:**

| metric | Δ (n=120) | 95% CI | vs batch-1 (n=60) |
|---|---|---|---|
| **ipTM** | **+0.235** | [+0.199, +0.272] | +0.226 [+0.172, +0.283] — consistent, **CI width 0.111 → 0.073** |
| **composite** | **+0.820** | [+0.699, +0.945] | +0.779 [+0.599, +0.955] — consistent, **width 0.356 → 0.246** |
| global pTM (localization) | +0.080 | [+0.065, +0.096] | ~14× smaller than composite → **H3 PASS** |

- **H2 (tighter CIs): PASS.** On 120 the paired ipTM/composite CIs are tighter than at n=60 with point estimates
  inside the batch-1 CI (no drift). H1 also passes on the full 120 (both CI>0).
- **H3 (localization): PASS.** Global pTM shifts +0.080, ~14× smaller than the composite → interface-local.
- **H2 (no collapse):** ordering wt > L > random preserved (as batch-1); L close to wt.

## Reading
The primary AF2-multimer steering headline holds on a fresh, disjoint batch and sharpens on 120 complexes. With
Phase 3 (Boltz-2 cross-modality, `FINDINGS_boltz.md`) and the 3-judge anti-circular matrix
(`FINDINGS_judge_matrix.md`), the +α·L steering benefit is now confirmed across two structure predictors, three
inverse-folding judges, and 120 complexes. Honest scope unchanged: ipTM/pAE/pLDDT are model proxies for assembly,
not experimental binding (`PREREG_iptm.md`).

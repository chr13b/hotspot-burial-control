# FINDINGS — steering a SECOND model (ESM-IF1) by +α·L also works (SET-B): judges AND AF2 fold confirm

**Pre-registered** `results/PREREG_esmif_steer.md`. `SEED=20260803`. The converse of the standing CFG result:
steer **frozen ESM-IF1** by `+α·L` (its OWN interface leverage — the symmetric analog of `cfg_steer.py`, which
steers ProteinMPNN by ProteinMPNN's leverage) and measure with INDEPENDENT models. **Both legs pass** — the
non-self judges rate the ESM-IF1-steered residues higher, AND an independent structure predictor (AF2-multimer)
folds them to better interfaces. Drivers `src/cfg_steer_esmif.py` (one-shot biased native-conditional sampler
over `leverage_pq_skempi_esmif.csv`); folds reuse the AF2 pipeline. Raw: `results/cfg_steer_esmif_seqs.csv`,
`results/iptm_steer_esmif.csv`; stats `results/iptm_summary_esmif.csv` + `results/cfg_judge_matrix.csv`.

## H1a — judge-leverage (anti-circular), paired L − random (from `FINDINGS_judge_matrix.md`)
SET-B (ESM-IF1-steered) sampled interface residues rated by NON-self judges (ESM-IF1 excluded = circular):

| judge | paired L − random | 95% CI | P(>0) |
|---|---|---|---|
| **ProteinMPNN** | **+0.437** | [+0.383, +0.492] | 1.000 |
| **MIF** | **+0.568** | [+0.478, +0.662] | 1.000 |

Both anti-circular judges PASS — the steered residues are more binding-favorable to models that did not do the
steering. (Self/circular ESM-IF1→ESM-IF1 +1.15, reference only.)

## H1b — AF2 fold transfer, paired L − random (n=60, complex-clustered 95% CI)
The α=2 ESM-IF1-steered sequences folded with AF2-multimer (wt reused from the batch-1 ipTM run; 420 folds,
0 missing; wt interface ipTM median **0.880**):

| metric | Δ (L − random) | 95% CI | P(>0) |
|---|---|---|---|
| **ipTM** | **+0.158** | [+0.102, +0.215] | 1.000 |
| **composite** | **+0.525** | [+0.326, +0.714] | 1.000 |
| interface pAE | −4.16 | [−5.69, −2.68] | 0.000 |
| interface pLDDT | +5.05 | [+2.56, +7.43] | 1.000 |
| global pTM (localization) | +0.064 | [+0.040, +0.088] | 1.000 |

**H1b (fold transfer): PASS** — composite AND ipTM both CI>0, all three interface metrics agree. The pre-registered
**falsifier (primary judge OR fold paired L−random ≤ 0) does NOT fire.**

## H2 / H3
- **H3 (localization): PASS.** Global pTM shifts +0.064, ~8× smaller than the composite → interface-local.
- **H2 (no collapse): satisfied.** L−wt ipTM −0.070, composite −0.29 → ordering **wt 0.88 > L ≈ 0.81 > random**;
  L close to wt, no interface collapse. (Per-arm interface native recovery recorded in `cfg_steer_esmif.csv`.)

## Reading
Steering is **not one model's quirk and works in reverse.** ProteinMPNN steered by `+α·L` was already confirmed
(ESM-IF1/MIF judges + AF2/Boltz folds); here the SECOND model, ESM-IF1, steered by `+α·L`, is confirmed the same
two ways — ProteinMPNN + MIF judges (+0.44 / +0.57) and an AF2 fold (composite +0.525, ipTM +0.158). The ESM-IF1
steering effect is somewhat smaller than the ProteinMPNN-steered AF2 headline (ipTM +0.226, composite +0.779) but
the same direction with CIs excluding 0. Honest scope: the SET-B steerer is a one-shot biased native-conditional
sampler (disclosed in `PREREG_esmif_steer.md`), `L` is an inverse-folding proxy, and ipTM/pAE/pLDDT are model
assembly proxies — the effect is a paired L−random contrast, z-scored per metric.

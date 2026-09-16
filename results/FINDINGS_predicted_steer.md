# FINDINGS — Predicted-backbone steering: the +α·L benefit SURVIVES on the backbones designers actually use

Pre-registered `results/PREREG_predicted_steer.md`. `SEED=20260803`, `NBOOT=5000`, complex-clustered. Closes the
softest reviewer attack on the steering result: it was measured on **crystal** backbones; does it survive when the
leverage is **computed and applied on the predicted backbone** (the staged-design regime)? **Yes — essentially
undiminished.**

## Phase 1 — DECISIVE (CPU/GPU): steer on the predicted backbone, judge anti-circularly
`src/cfg_steer_predicted.py` tilts a frozen ProteinMPNN by `+α·L` (α=2, K=64) at the interface of the **predicted**
structure, using **predicted-backbone leverage**; steered residues are scored by an INDEPENDENT model's leverage
(ESM-IF1, MIF) **also computed on the predicted backbone** (`src/leverage_judge_predicted.py`). Paired
complex-clustered bootstrap, n=106 complexes (of 140 predicted; 34 dropped for <3 usable interface positions / size).

**Judge-level L − random (primary) and L − naive (secondary), per backbone × judge:**

| backbone | judge | **L − random** | 95% CI | **L − naive** | 95% CI | P(>0) |
|---|---|---|---|---|---|---|
| **OpenFold3** | ESM-IF1 | **+0.733** | [+0.645, +0.827] | +0.167 | [+0.137, +0.198] | 1.00 |
| **OpenFold3** | MIF | **+0.704** | [+0.624, +0.789] | +0.221 | [+0.192, +0.250] | 1.00 |
| **AF2-multimer** | ESM-IF1 | **+0.760** | [+0.678, +0.846] | +0.174 | [+0.146, +0.204] | 1.00 |
| **AF2-multimer** | MIF | **+0.737** | [+0.659, +0.818] | +0.213 | [+0.185, +0.241] | 1.00 |
| crystal (control) | ESM-IF1 | +0.751 | [+0.679, +0.824] | +0.187 | [+0.158, +0.216] | 1.00 |
| crystal (control) | MIF | +0.677 | [+0.612, +0.744] | +0.217 | [+0.191, +0.245] | 1.00 |

**The pre-registered falsifier does NOT fire.** `L − random > 0` at the judge level across **both** anti-circular
judges (ESM-IF1, MIF) on **both** predicted backbones (OF3, AF2), every CI excluding zero, every P(>0)=1.0. The
steering benefit is **not a crystal-backbone artifact** — it transfers to the design regime.

**Barely attenuated — stronger than pre-registered.** We expected `L − random > 0 but attenuated` vs crystal.
Instead the predicted-backbone effect (OF3 +0.73/+0.70, AF2 +0.76/+0.74) is **essentially equal to crystal**
(+0.75/+0.68) — AF2 even edges it. This coheres with the detection side (§8: the mixed derivative on OF3/AF2
backbones lands on the *surviving* part of the dose law), now shown for the **intervention** side too.

**Specificity holds on predicted backbones.** `L − naive > 0` everywhere (OF3 +0.17/+0.22, AF2 +0.17/+0.21, matching
crystal +0.19/+0.22): the gain needs the *binding* direction, not merely a confident one — even when both are read
off the predicted structure.

**Positive control (before trusting the predicted contrast).** The SAME pipeline on the **crystal** backbone over
the matched 106-complex subset reproduces the committed crystal judge numbers (ESM-IF1 L−random **+0.751** vs
committed +0.773; MIF **+0.677** vs +0.710; L−naive +0.187/+0.217 vs +0.159/+0.242) — the new scripts + predicted
judge caches are validated. → `results/cfg_steer_predicted.csv` (crystal block).

**Native recovery preserved** (`results/cfg_steer_predicted_recovery.csv`): interface recovery for the L arm
(OF3 0.288, AF2 0.280) ≈ crystal (0.290) and well above random (0.21–0.22); naive is highest (0.34–0.37, the
confidence direction). Steering does not wreck recovery on predicted backbones.

## Method notes / honest scope
- **Predicted-backbone judge leverage had to be built** — `leverage_predicted.py` scores ProteinMPNN only. We
  computed ESM-IF1 + MIF leverage on the 140 OF3/AF2 PDBs (`leverage_judge_predicted.py`, reusing the exact
  leverage_esmif/leverage_extra_models scorers; complex top-1 recovery 0.62–0.65 ESM-IF1, 0.55–0.57 MIF — healthy).
- **PiFold judge unavailable** (no weights on disk, no committed crystal cache — it was a local-only run). Reported,
  not faked: ESM-IF1 + MIF are the two anti-circular judges. Both agree decisively, on both backbones.
- **Coverage:** predicted-steer set = 140 shared complexes; 106 carry ≥3 usable interface positions on the
  predicted structure (the honest design-time set); 2 oversized complexes (2NYY/2NZ9, 1700 res) dropped from the
  judge caches. Overlap with the 60-complex ipTM fold set = 21 (the Phase-2 folding set).
- Predicted PDBs carry the crystal chain IDs + native sequence (align position-for-position), so the interface keys
  transfer directly; the crystal source is a true matched control.

## Phase 2 — CONFIRMATION (GPU): AF2-multimer on the predicted-backbone steered sequences → H1 PASS (n=93)
Pre-authorized because Phase 1 cleared zero. Folded the OF3-steered sequences (wt / L / random, k0-2) with
**AF2-multimer** (ColabFold, templates off, 1 model, 3 recycles — identical to the crystal ipTM pipeline). Extended
from the original 21 overlap complexes to the **full n=93** predicted-steer set (651/651 folds; 13 of the 106
steered complexes excluded at max-res 600, matching the filter; wt interface ipTM median ~0.84, healthy). Paired
complex-clustered bootstrap, mean-over-k primary (best-of-k in the CSV):

| metric | **L − random** (n=93) | 95% CI | P(>0) | crystal anchor |
|---|---|---|---|---|
| **ipTM** | **+0.181** | [+0.144, +0.220] | 1.00 | +0.235 |
| interface pAE (↓ better) | −5.13 | [−6.26, −4.08] | 0.00 | — |
| interface pLDDT | +8.28 | [+6.56, +10.01] | 1.00 | — |
| **composite** (z-mean ipTM, −pAE, pLDDT) | **+0.681** | [+0.553, +0.815] | 1.00 | +0.82 |
| global pTM (localization control) | +0.076 | [+0.059, +0.094] | 1.00 | — |

**H1 PASS** — composite(L) > random AND ipTM(L) > random, both CI>0. **H3 PASS** — localized: |ΔpTM| 0.076 ≪
Δcomposite 0.681, so the effect is at the interface, not a global foldability change. **H2** — L−wt ipTM −0.110,
composite −0.450: ordering **wt > L > random**, no catastrophic collapse (same pattern as the crystal run).
**Attenuated vs crystal (~77–83%: ipTM +0.18 vs +0.24, composite +0.68 vs +0.82)** — unlike the judge level (barely
attenuated), the *fold* level attenuates, but stays decisively CI>0. Standing caveat (crystal/naive work): the ipTM
gain is partly a foldability effect, so Phase 2 confirms the effect **transfers** to predicted backbones; the
binding-*specific* isolation is the judge level (Phase 1, L−naive>0) + the FoldX physics result. →
`results/iptm_predicted{,_steer}.csv`.

**Before/after (auditable record):** the original **n=21** overlap subset gave ipTM L−random **+0.188** [+0.099,
+0.279], composite **+0.662** [+0.368, +0.955] (H1 PASS). Extending to **n=93** leaves the point estimates
essentially unchanged (ipTM +0.181, composite +0.681) and **tightens the CIs ~2×** — the fold-level confirmation is
not a small-n artifact.

Operational note: the ColabFold mmseqs2 MSA server intermittently strands folds in `PENDING` (the known hang); a
30-min per-fold timeout + idempotent resubmit rounds driven by a **connection-independent Slurm manager**
(`jobs_pred_fold_manager.sbatch`) recovered all 651/651 (`jobs_iptm_fold_pred.sbatch`).

## Bottom line
On **predicted (OF3 and AF2) backbones** — the staged-design regime — a `+α·L` tilt on a frozen ProteinMPNN raises
an independent model's binding-leverage **essentially as much as on crystals** (L−random +0.70 to +0.76 across two
judges × two predictors, all CI>0), beats the confidence tilt (L−naive > 0), and preserves recovery. An independent
**structure predictor** (AF2-multimer, n=93) confirms it (Phase 2: ipTM L−random +0.18 [+0.14,+0.22], composite
+0.68 [+0.55,+0.82], both CI>0, H1 PASS), attenuated to ~80% of the crystal fold-level effect but decisive. The steering knob is **not a crystal
artifact**; it works where designers actually use it.

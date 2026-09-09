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

## Phase 2 — CONFIRMATION (GPU): structure-predictor metrics on the predicted-backbone steered sequences
Pre-authorized because Phase 1 cleared zero. Folding the predicted-backbone steered sequences (wt / L / random) with
AF2-multimer (± Boltz-2) over the 21-complex overlap; H1/H2/H3 as in the crystal ipTM run. → `results/iptm_predicted.csv`
(status recorded there and appended to this file when it lands).

## Bottom line
On **predicted (OF3 and AF2) backbones** — the staged-design regime — a `+α·L` tilt on a frozen ProteinMPNN raises
an independent model's binding-leverage **essentially as much as on crystals** (L−random +0.70 to +0.76 across two
judges × two predictors, all CI>0), beats the confidence tilt (L−naive > 0), and preserves recovery. The steering
knob is **not a crystal artifact**; it works where designers actually use it.

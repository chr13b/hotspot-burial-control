# FINDINGS — the naive-guidance SPECIFICITY arm: L beats a matched confidence tilt (binding direction required)

**Pre-registered** `results/PREREG_naive.md` (committed before any naive number). `SEED=20260803`, NBOOT=5000,
n=60 complexes (frozen ipTM subset). The committed steering result shows `+α·L` beats a **random** matched-
magnitude tilt. This arm is a *stronger* control: **naive-guidance** — frozen ProteinMPNN tilted toward its OWN
complex-conditioned confidence (`conf_i = lP_i − mean_a(lP_i)`, direction only), scaled to `‖L_i‖` per position, so
it differs from `L` **only in direction** (matched magnitude, exactly as `random` permutes `L_i`). If a confident-
but-not-binding tilt matched `L`, the direction-specificity claim would be bounded. It does not. Drivers
`src/cfg_steer_naive.py` (generate), `src/judge_dumped.py` (judge by non-self models), `src/analyse_naive.py`
(paired bootstrap). Raw: `results/cfg_steer_naive_seqs.csv`, `results/cfg_naive_judge.csv`,
`results/cfg_steer_naive.csv`; stats `results/cfg_naive_summary.csv`.

## Result — paired contrasts, anti-circular judges (dumped3; ProteinMPNN is the steered model → self/reference)
Mean judge-leverage of the sampled interface residues; paired per complex, complex-clustered 95% CI, P(>0):

| judge | contrast | Δ | 95% CI | P(>0) |
|---|---|---|---|---|
| **ESM-IF1** (anti-circular) | **L − naive** | **+0.159** | [+0.110, +0.207] | 1.000 |
| | naive − random | +0.521 | [+0.414, +0.640] | 1.000 |
| | L − random (standing) | +0.680 | [+0.589, +0.783] | 1.000 |
| **MIF** (anti-circular) | **L − naive** | **+0.242** | [+0.191, +0.292] | 1.000 |
| | naive − random | +0.428 | [+0.306, +0.572] | 1.000 |
| | L − random (standing) | +0.669 | [+0.563, +0.799] | 1.000 |
| ProteinMPNN (self/ref) | L − naive | +0.300 | [+0.272, +0.328] | 1.000 |

K64 (mean over K=64, higher power) corroborates: ESM-IF1 **L − naive = +0.175** [+0.137, +0.212], naive − random
+0.510, L − random +0.685 (`cfg_naive_summary.csv`).

## Falsifier read-out — does NOT fire; specificity holds
Pre-registered falsifier: under an anti-circular judge, **naive − random ≥ L − random** → direction-specificity
bounded. Observed the opposite for both judges:
- ESM-IF1: naive − random **+0.521** < L − random **+0.680**, and **L − naive +0.159 (CI>0)**.
- MIF: naive − random **+0.428** < L − random **+0.669**, and **L − naive +0.242 (CI>0)**.

`L` beats the matched-magnitude confidence tilt with CIs excluding zero under **two independent judges** →
**the binding direction is required; a confident tilt is not sufficient.** The falsifier does not fire.

## The honest nuance (reality departed from the naive expectation — reported verbatim per PREREG rule)
The pre-registration *expected* `naive − random ≈ 0` ("a confident-but-not-binding tilt behaves like random").
**That expectation was wrong.** The confidence tilt is far from inert: it recovers **~64–77%** of the full
`L − random` effect (ESM-IF1 0.521/0.680 ≈ 77%; MIF 0.428/0.669 ≈ 64%). So ProteinMPNN's own confidence at the
interface is **substantially binding-correlated** — steering toward "what the model already prefers" is itself a
partial binding signal, not noise. What *survives* is the pre-registered specificity core: even against this much
stronger control, the explicit binding direction `L` adds a **robust further increment** (L − naive CI>0, both
judges, both methods). The claim is therefore sharpened, not weakened: `random ≪ naive < L`, with the last step
(naive→L) — the part that needs the *binding* direction specifically — small but decisive.

## Reading — and the connection to RedNet
This is the accessible half of the blocked RedNet head-to-head (`FINDINGS_rednet.md`). RedNet's contrastive decode
is the *binding/contrastive direction* (`α·(logP_complex − logP_unbound)` = our `+α·L`). Here we isolate that
direction's marginal value over a cheap alternative — the model's own confidence — on frozen ProteinMPNN: a
confidence tilt gets most of the way, but the explicit binding direction adds a further CI-excluding-zero
increment (L − naive +0.16/+0.24). That is exactly the quantity a retrained contrastive method (RedNet) is built
to capture, measured here without it.

**But the two readouts diverge — see Phase 4.** The `L > naive` advantage is a property of the inverse-folding
*judge* readout; at the AF2 *fold* level a confidence tilt does as well as or slightly better than `L`. That split
is itself evidence for the project's frustration thesis.

## Phase 4 — structure-level (AF2 fold): the divergence the frustration thesis predicts
Folded the naive arm (180 folds, 60×k0-2, **0 missing**; naive interface ipTM median 0.830), reusing the committed
wt/L/random folds. Paired, mean over k, complex-clustered 95% CI (n=60; pAE/composite n=58):

| metric | naive − random | **L − naive** | L − random |
|---|---|---|---|
| **ipTM** | +0.260 [+.205,+.317] | **−0.034** [−.066,−.006] | +0.226 [+.171,+.281] |
| **composite** | +0.933 [+.745,+1.120] | **−0.150** [−.258,−.052] | +0.782 [+.601,+.965] |
| interface pLDDT | +12.9 [+10.4,+15.3] | **−3.42** [−4.87,−2.15] | +9.47 |
| interface pAE (↓ better) | −5.41 | +0.14 [−.64,+.93] (ns) | −5.27 |
| global pTM (localization) | +0.089 | −0.009 [−.020,+.002] | +0.080 |

**At the fold level the ordering FLIPS to `random ≪ L ≲ naive`:** the confidence tilt folds to interfaces AF2 scores
as confidently as — slightly better than — the L-steered ones (L − naive negative on ipTM / composite / interface
pLDDT, CIs excluding 0; indistinguishable on interface pAE). The pre-registered Phase-4 expectation ("naive ≈
random") is **doubly wrong** — naive ≫ random AND naive ≳ L — so **at the ipTM level the specificity is BOUNDED**
(the structure-level falsifier fires).

**This is coherent, not contradictory — it is the frustration mechanism.** ipTM / interface pLDDT are assembly-
*confidence*/foldability proxies; the naive tilt steers toward exactly the residues the model is most confident
about, so it folds most confidently. `L` steers toward *binding-favorable* residues which — the central thesis of
this project — are frequently **frustrated** (the binding-optimal residue sits in the confidence tail), so they
fold slightly *less* confidently than the naive picks. The binding-specific advantage of `L` over `naive` is
therefore **invisible to ipTM** yet **visible to the inverse-folding leverage judges** (L − naive +0.16/+0.24,
CI>0). Two proxies, two answers, diverging exactly where frustration predicts.

**Honest caveat for the ipTM steering headline.** Because a pure confidence tilt reproduces (indeed exceeds) the
ipTM gain, **ipTM improvement from `+α·L` is largely a confidence/foldability effect and does not by itself isolate
a binding improvement** — the binding-specific increment of the L direction is carried by the inverse-folding-judge
readout. This bounds the *interpretation* of the ipTM steering result, not the standing `L > random` ipTM fact
(+0.226, CI>0), which holds. Raw: `results/iptm_steer_naive.csv`, `results/iptm_summary_naive.csv`.

## Honest scope
- `L` is an inverse-folding leverage proxy; the judges (ESM-IF1, MIF) are independent inverse-folding proxies, not
  experimental binding. The effect is a paired L−naive / naive−random contrast in judge-leverage of the sampled
  interface residues, matched-magnitude by construction (`‖naive_i‖ == ‖L_i‖`, asserted per position).
- PiFold is omitted as a judge here (no `leverage_pq_skempi_pifold.csv` P/Q cache); ProteinMPNN is self/circular
  (the steered model), reported for reference only. The two anti-circular judges (ESM-IF1, MIF) agree.
- Protocol identical to `cfg_steer.py` (K=64, order=None, temp=1.0, SEED) so decoding-order variance cancels in
  the paired contrasts.
- The judge-level and fold-level readouts **disagree by design** (Phase 4): `L > naive` in inverse-folding
  leverage but `naive ≳ L` in ipTM/composite. Neither is "the" answer — they measure binding-favorability vs
  assembly foldability, and the gap is the frustration signature. Report both; do not cite one alone.

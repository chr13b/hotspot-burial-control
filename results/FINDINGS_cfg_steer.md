# FINDINGS — the CFG direction is ACTIONABLE: steering ProteinMPNN by +α·L raises an independent model's binding-leverage

**Pre-registered** in `results/PREREG_cfg_steer.md` (frozen before any steering number). **Script**
`src/cfg_steer.py`; analysis `src/cfg_steer_analyse.py`. SEED=20260803. **Raw** `results/cfg_steer.csv`
(271 complexes × 2 directions × 5 α); **summary + CIs** `results/cfg_steer_summary.csv`.

Exact command:
```
python3 src/cfg_steer.py --alphas 0,0.25,0.5,1,2 --K 64 --out results/cfg_steer.csv
python3 src/cfg_steer_analyse.py --in results/cfg_steer.csv --out results/cfg_steer_summary.csv
```

## What was tested
The paper *names* the mixed derivative L as the classifier-free-guidance direction but never guides with it.
Here we add `bias_by_res[i,a] = α·L_i(a)` to a **frozen off-the-shelf ProteinMPNN**'s logits at interface
positions i (0 elsewhere), sample K=64 sequences per complex, and ask whether the sampled interface residues
bind better. **Anti-circularity (the whole point): the headline is measured by a DIFFERENT model** — the mean
**ESM-IF1** leverage of the sampled residues — not by MPNN's own L. **Specificity control:** a random direction
of matched per-position magnitude (a per-position permutation of the same L vector). Pre-registered grid
α ∈ {0, 0.25, 0.5, 1, 2}. n = 271 SKEMPI complexes present in both the MPNN and ESM-IF1 pq caches with ≥3
usable interface positions and ≤700 residues.

## Positive control (rule 6, ran first)
Smoke on 3 complexes: for `direction=L`, ESM-IF1 leverage rose with α; for `direction=random`, it did not.
Gate passed → full sweep run. (The device port used for GPU speed was separately gated: the ported code on
CPU reproduces the pre-port smoke **bit-identically**, so the port changes only placement, not the math.)

## Result — H1 and H2 both confirmed, falsifier does NOT fire

**Mean ESM-IF1 binding-leverage of the sampled interface residues** (complex-clustered bootstrap 95% CI,
5,000 resamples):

| α | **L direction** | random control | **L − random (paired)** | P(>0) |
|---|---|---|---|---|
| 0.00 | −0.2048 [−0.249, −0.161] | −0.2048 [−0.249, −0.161] | +0.000 (identical: B=0) | — |
| 0.25 | −0.0793 [−0.120, −0.038] | −0.2095 [−0.253, −0.166] | **+0.1302 [+0.123, +0.138]** | 1.000 |
| 0.50 | +0.0148 [−0.026, +0.056] | −0.2294 [−0.277, −0.185] | **+0.2441 [+0.230, +0.259]** | 1.000 |
| 1.00 | **+0.1360 [+0.096, +0.174]** | −0.3178 [−0.370, −0.267] | **+0.4538 [+0.427, +0.481]** | 1.000 |
| 2.00 | **+0.2650 [+0.226, +0.305]** | −0.5079 [−0.575, −0.442] | **+0.7729 [+0.726, +0.823]** | 1.000 |

- **H1 (actionable).** As α rises, the L-arm ESM-IF1 leverage increases monotonically, from −0.205 at α=0 to
  **+0.265 at α=2 — a +0.47 shift**, with the CI excluding zero from α=1 onward. Steering ProteinMPNN along its
  own guidance direction produces sequences a *second, architecturally-distinct* model scores as
  higher-binding.
- **H2 (specificity).** The random direction of matched magnitude does the **opposite** — ESM-IF1 leverage
  *falls* (−0.205 → −0.508). The paired L−random contrast is positive at every α>0 with **P(>0)=1.000** and CIs
  far from zero. The effect is specific to the L **direction**, not to perturbation magnitude. No bootstrap
  resamples were dropped (no nan means).

**Interface native recovery — the cost is not merely graceful, it is absent for the L arm.** The
pre-registration anticipated "a graceful cost to native interface recovery." Instead, steering by L slightly
**raises** interface recovery (0.276 → 0.286 → 0.292 → 0.297 → 0.297), while the random arm **degrades** it
(0.276 → 0.220). So the L direction concentrates probability on residues that are simultaneously more
native-consistent *and* higher binding-leverage; only the random perturbation pays the recovery cost. This is
stronger than H1 predicted.

**Localization control — passes.** Non-interface recovery is flat across α for both arms (0.3649 → 0.3648;
range < 0.001), confirming the tilt acts only where applied.

**Sweet spot.** With interface recovery preserved at all α (never below the α=0 baseline for the L arm), the
binding constraint is the only active one: ESM-IF1 leverage becomes significantly positive at **α = 1.0**
(+0.136 [+0.096, +0.174]) and is largest at **α = 2.0** (+0.265 [+0.226, +0.305]). So the actionable range is
**α ≈ 1–2**, with no recovery/binding tradeoff in the tested window.

## Reading and honest limits
1. **The named direction works as a knob.** This turns the paper's L-is-the-CFG-direction framing from an
   analogy into an operationalized result: a frozen inverse-folding model can be nudged toward higher-binding
   interface chemistry by biasing its logits with `+α·L`, no retraining.
2. **Anti-circular but not orthogonal.** ESM-IF1 is a different architecture and readout, and the random
   control makes the specificity non-trivial — but inverse-folding models share biases (indeed the
   leverage decomposition replicates across MPNN/ESM-IF1/PiFold/MIF), so this is "a second model agrees the
   steered residues are higher-binding-leverage," not "an independent binding oracle confirms it." The headline
   metric is a model proxy for binding, not experimental ΔΔG. Connecting the steered sequences to an
   experimental or physics binding readout is the natural next step and is not claimed here.
3. **Recovery is measured against the native, which is itself only an approximate binding optimum.** That
   interface recovery rises under L-steering says the L direction aligns with native chemistry on average; it
   is not a direct binding measurement.

## Reproduce
Inputs (regenerated deterministically from the committed scorers; pq caches are gitignored):
`leverage_pq_skempi.csv` (`leverage_decomposition.py --stage score-skempi`),
`leverage_pq_skempi_esmif.csv` (`leverage_esmif.py --stage score`), and the committed
`leverage_skempi_positions.csv`. Sweep on one V100 (~7 s/complex via the optional CUDA port; the CPU path is
bit-identical and needs no GPU, just longer). SEED=20260803, K=64.

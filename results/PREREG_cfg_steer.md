# PRE-REGISTRATION — is the CFG direction ACTIONABLE? (steer ProteinMPNN by +α·L)

**Committed BEFORE any steering number (CLAUDE.md rule 1).** SEED=20260803.

## Motivation
The paper *names* the mixed derivative L as the classifier-free-guidance direction but never guides with it.
The obvious reviewer question: "you named the direction — did you use it?" This tests whether tilting
ProteinMPNN's sampling by `+α·L` at interface positions actually produces sequences that bind better — turning
the framing from an analogy into an operationalized result, and connecting §4 to the design regime.

## Mechanism (no source change beyond one backward-compatible param)
`src/decoding/mpnn_steer.py` `sample()`/`sample_ptemp()` already add a per-position, per-aa `bias_by_res` to the
logits before the softmax. Setting `bias_by_res[i, a] = α · L_i(a) · T` at interface positions i (0 elsewhere)
makes the effective decoded logit `logit_i(a) + α·L_i(a)` — exactly the CFG tilt along the guidance direction.
L is ProteinMPNN's own leverage (from `leverage_pq_skempi.csv`).

## The anti-circularity move (the whole point)
Steering by L and then measuring L would be circular. So the **headline metric is measured by a DIFFERENT
model**: the mean **ESM-IF1** leverage of the sampled interface residues (`leverage_pq_skempi_esmif.csv`). If
tilting ProteinMPNN along *its* guidance direction produces sequences a *second, independent* model also scores
as higher-binding, the direction is genuinely actionable, not a self-fulfilling artifact.

## Hypotheses
- **H1 (actionable).** As α rises over a modest range, the mean ESM-IF1 binding-leverage of the sampled
  interface residues **increases monotonically**, at a graceful cost to native interface recovery.
- **H2 (specificity — the control).** Steering by a **random** direction of matched per-position magnitude
  degrades recovery similarly but does **not** raise ESM-IF1 leverage. So the effect is specific to the L
  *direction*, not to perturbation magnitude.
- **Falsifier:** if ESM-IF1 leverage does not rise with α, or rises no more than the random control, the CFG
  direction is **not** actionable — reported as such, verbatim.

## Metrics (per complex × α, K samples, seed fixed)
- **interface native recovery** (the fold-stability cost; falls with α).
- **non-interface recovery** (localization control; should be ~flat — we steer only the interface).
- **mean MPNN L of sampled** (by-construction check that the tilt did something).
- **mean ESM-IF1 L of sampled** (INDEPENDENT headline).
- **sweet spot:** the α where ESM-IF1 leverage is meaningfully up while interface recovery is still ≥ ~50% of
  the α=0 baseline.

α ∈ {0, 0.25, 0.5, 1.0, 2.0}; both the L-direction arm and the random-direction control; a subset of SKEMPI
complexes present in BOTH the MPNN and ESM-IF1 pq files. Report per-complex, complex-clustered bootstrap.

Output: `results/cfg_steer.csv`. Script: `src/cfg_steer.py`.

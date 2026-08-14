# PRE-REGISTRATION — Bennett hardening battery (fail-fast venue fork)

**Date:** 2026-08-14. **Seed:** 20260803. Written BEFORE any of these numbers exist.
**Trigger:** two blind adversarial reviews (research-strategist + idea-critic) converged: the paper's
ICLR viability hinges on whether the Bennett positive ("+0.025 energetics beyond geometry",
`bennett_occlusion_energetics.csv`) survives (a) a stronger *all-atom* occlusion baseline and (b) a
*non-parent* scorer — and on hardening the SKEMPI-side confidence claims. This file pre-registers the
tests and their decision rules so no threshold moves after a number is seen (CLAUDE.md rule #1).

## Venue fork
**T1 is the decider; T2 confirms it. T3/T4/T5 harden regardless of venue.**
- **Spine B (ICLR):** the model encodes per-substitution binding energetics beyond geometry — requires T1 SURVIVE.
- **Spine A (TMLR):** the burial-confound correction + confidence-diagnostic; the positive demotes to a bonus — if T1 COLLAPSES.

---

## T1 — All-atom occlusion baseline  (THE decider)
Build a **min-over-rotamer all-atom vdW-clash** feature for each (interface position on binder chain B,
substitution) against all heavy atoms of partner chain A, on the 73 Bennett design PDBs.

- **Validity gate (positive control) — checked and reported BEFORE any AUROC:**
  1. the rotamer builder reconstructs the **native** side chains present in the PDBs to **median heavy-atom RMSD < 1.0 Å**;
  2. clash ≈ 0 for Gly/Ala; elevated for large hydrophobics placed into buried pockets.
  If the gate fails, **T1 is INVALID → reported as could-not-run**, no venue decision drawn from it.
- **Geometry baseline** = {min-rotamer clash, partner-contact count, ΔSASA, native & substituted volume}.
- **Metric** = cross-validated logistic **ΔAUROC(add P over geometry)** at the interface layer,
  design-clustered bootstrap (3000 reps), seed 20260803.
- **DECISION (pre-registered):**
  - **SURVIVES → Spine B / ICLR:** ΔAUROC lower CI > 0 **AND** point estimate ≥ **+0.010**.
  - **COLLAPSES → Spine A / TMLR:** CI includes 0, **OR** point < +0.010.
  - Report the number either way.

## T2 — Circularity break (non-parent scorer)
Parents are ProteinMPNN outputs. Re-score Bennett interface substitutions with **ESM-IF1** (did not
generate the parents). Recompute interface AUROC(P), P−Q, and the T1 ΔAUROC under the non-parent scorer.
- **DECISION:** circularity **DEFUSED** if interface AUROC(P) > 0.5 (CI excludes 0.5), P−Q > 0 (CI excludes 0),
  and the T1 ΔAUROC sign holds. If signs flip → the positive is **parent-scorer-specific** (real weakness; report it).

## T3 — Scalar-confidence fork on Bennett (constraint-vs-leverage theory)
Per interface position compute scalar confidence: **log p_native** and **negentropy** of the complex-conditioned
distribution. Position hotspot label = **fraction of the position's measured substitutions that abolish binding
(kd_lb ≥ per-library cap) ≥ the across-interface median** (pre-set, balanced). AUROC vs SKEMPI's 0.538.
- **DECISION:** theory **FIRES** if Bennett confidence-AUROC > 0.5 (CI excludes 0.5) AND > 0.538 (de-novo is the
  maximally binding-dominated regime, so confidence should track leverage best here). **SELF-REFUTES** if ≈ chance
  → fall back to **P−Q gain** (+0.076, already measured) as the leverage signal. Either way the
  distribution-vs-confidence dissociation stands.

## T4 / R2 — Cross-model confidence nugget (cannot fail)
Per model in {ProteinMPNN, ESM-IF1, PiFold, MIF, ProBID}: interface **confidence-AUROC** for SKEMPI hotspots
(panels joined to `kl_detector_joined` labels; ProBID via top_prob), and CPI(confidence | geometry) where feasible.
- **No pass/fail — report the spread.** Expectation: ≈ chance / conditionally-independent across all 5 →
  "a property of inverse folding, not one network." A model whose confidence *does* predict is a reportable spread, not a failure.

## T5 — Burial-residualize cross-predictor agreement (kill-shot #5)
The OF3-vs-AF2 per-complex deficit agreement (ρ≈0.57): recompute as **partial correlation controlling for
interface burial**, and after **dropping the shared top-3 leverage complexes**.
- **DECISION:** agreement **SURVIVES** if partial-ρ CI excludes 0 AND ρ stays > ~0.3 after the top-3 drop.
  If it vanishes under burial-residualization → the "conditioning-set tax" is a **recursive burial confound**
  (report honestly; reframe §5).

---
**Scripts (one per test):** `src/p_bennett_occlusion_allatom.py` (T1), `src/p_bennett_nonparent.py` (T2),
`src/p_bennett_conf_fork.py` (T3), `src/xmodel_confidence.py` (T4), `src/deficit_burial_residualize.py` (T5).
Each takes `--out`, writes a CSV with the exact command, prints a one-line summary. Seed 20260803 fixed.

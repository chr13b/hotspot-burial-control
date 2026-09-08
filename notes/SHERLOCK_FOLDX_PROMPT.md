# Sherlock — FoldX physics ΔΔG_bind (two lanes) [paste-to-run]

`SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.
**Pre-register `results/PREREG_foldx.md` FIRST** (before any FoldX number). CLAUDE.md rules apply (never
fabricate; every number → committed CSV; positive controls; `git add` by name; two trailer lines; push). Plan +
rationale in `notes/FOLDX_PLAN.md`. FoldX is a **physics energy function — a binding readout from a modality
independent of inverse folding**; it is *not* ground truth (an approximate, fit energy function) — frame it as
"physics ΔΔG_bind (FoldX)", not truth.

## Phase 0 — FoldX + structures
- Obtain the FoldX binary (free academic download; no GPU, no gated weights) and its rotabase; put it on
  `$SCRATCH`, not in the repo. Smoke-test on one complex.
- `RepairPDB` each crystal structure used (SKEMPI fixture PDBs under `data/PDBs/`); cache the repaired PDBs on
  `$SCRATCH`. **Checkpoint/idempotent:** skip any (structure, mutation) whose FoldX output already exists;
  append results incrementally; a crash re-runs only the remainder. Shard across cores/nodes.

## Lane A — DETECTION: L vs FoldX on SKEMPI (accuracy, reviewer-anticipation)
Pre-registered metric: **Spearman(prediction, experimental ΔΔG)** on the shared single-mutation set.
- For each SKEMPI single mutant in our fixture, compute FoldX ΔΔG_bind (`BuildModel` the mutation →
  `AnalyseComplex` on wt and mutant → ΔΔG_bind = ΔG_mut − ΔG_wt across the interface).
- Report **Spearman(FoldX, experimental)** and, side by side on the *same* mutations, **Spearman(L, experimental)**
  (recompute L on that subset if needed). Same fixture, same metric. Disclose FoldX is a fit energy function (it
  may match/beat a zero-shot readout — expected, not our claim). → `results/foldx_detection.csv`.

## Lane B — STEERING SPECIFICITY (the key lane): L-steered vs naive vs random on physics ΔΔG_bind
This is the non-inverse-folding binding readout that answers "does L beat 'just be more confident' for *binding*",
independent of the inverse-folding judges.
- Complexes: the **60 ipTM complexes** (`results/iptm_subset.txt`). Arms + interface sequences:
  `results/cfg_steer_seqs.csv` (wt, L, random) and `results/cfg_steer_naive_seqs.csv` (naive), α=2, per k.
- For each (complex, arm, k): the interface residues where the arm's sequence differs from wt define the mutation
  set on that complex's crystal backbone. `BuildModel` those interface mutations → `AnalyseComplex` → **FoldX
  ΔΔG_bind vs wt**. (wt arm = 0 by construction; use it as a positive control that FoldX ΔΔG_bind(wt→wt)=0.)
- Aggregate k per (complex, arm) (mean; best-of-k secondary), then report the **paired, complex-clustered
  bootstrap** (95% CI, P): **L − naive**, **L − random**, **naive − random** on ΔΔG_bind (more-negative =
  more-favorable binding; orient signs so >0 = better). → `results/foldx_steer.csv`.
- **Decisive test — L − naive on ΔΔG_bind.** If L is more binding-favorable than naive (CI clear): L's advantage
  over the confidence tilt holds on a **physics, non-IF** readout — the strongest form of the steering claim.
- **Pre-registered falsifier (verbatim if it fires):** if **L − naive ΔΔG_bind ≤ 0**, the binding-specific
  steering advantage does *not* transfer to physics → reported plainly, bounding the steering claim to the
  inverse-folding judges. A null here does not touch L>random or the detection results.

## Positive controls / honesty
- FoldX ΔΔG_bind(wt) ≈ 0 (sanity); report FoldX's own run-to-run spread (it has a stochastic side-chain step) so
  an effect smaller than it is not over-read.
- Frame FoldX as an approximate physics energy function independent of the inverse-folding modality — a
  cross-modality check, not ground truth.
- (Optional accuracy check, subset only) **Rosetta `flex_ddG`** on ~10–20 complexes as a gold-standard spot-check
  — heavier/slower, so subset only; skip unless the FoldX result is borderline or a reviewer insists.

## Deliverables (commit by name, two trailer lines, push)
`PREREG_foldx.md`, `foldx_detection.csv`, `foldx_steer.csv`, `FINDINGS_foldx.md`, any `src/` driver. Message me
the Lane-A Spearmans and the Lane-B paired **L − naive** ΔΔG_bind.

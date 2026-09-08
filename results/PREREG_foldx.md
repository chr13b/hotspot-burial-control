# PRE-REGISTRATION — FoldX physics ΔΔG_bind, two lanes (a non-inverse-folding binding readout)

**Committed BEFORE any FoldX number (CLAUDE.md rule 1).** `SEED=20260803`. Plan/rationale: `notes/FOLDX_PLAN.md`.

## Framing (honesty)
FoldX is an **approximate, hand-fit physics energy function** — **not ground truth**. Its value here is that it is
a binding readout from a **modality independent of inverse folding**: the steering result is currently carried by
inverse-folding *judges* (ESM-IF1/MIF/PiFold leverage) and by ipTM (a foldability/confidence metric the naive arm
wins). FoldX ΔΔG_bind is the physics cross-check. Reported as "physics ΔΔG_bind (FoldX)", never as truth.

**Binary dependency (disclosed):** FoldX 5 (academic binary + `rotabase.txt`) is license-gated and must be
obtained by the operator; the exact version is recorded here once staged at `$SCRATCH/ftax/foldx/`. No FoldX
number is produced until then.

## Method (fixed)
Per structure: `RepairPDB` the crystal once (cache on `$SCRATCH`). Per mutation set: `BuildModel` (individual_list)
→ `AnalyseComplex` on wt and mutant → **ΔΔG_bind = ΔG_bind(mut) − ΔG_bind(wt)** (interaction energy across the
interface). FoldX has a stochastic side-chain step → `numberOfRuns=5`; report the **mean and the run-to-run
spread** so an effect smaller than FoldX's own noise is not over-read. Idempotent/sharded: skip any (structure,
mutation-set) whose output exists; append incrementally.

## Lane A — DETECTION (L vs a physics SOTA; reviewer-anticipation)
Shared **SKEMPI single-mutation interface fixture** (`results/leverage_skempi_mutations.csv`, rows with
experimental `ddG` and `L`, `is_interface==1`). Pre-registered metric: **Spearman(prediction, experimental ΔΔG)**.
- Report **Spearman(FoldX ΔΔG_bind, experimental ΔΔG)** and, on the *same* mutations, **Spearman(L, experimental
  ΔΔG)** — same fixture, same metric.
- **Disclosure:** FoldX is a *fit* energy function; it may match or beat a zero-shot readout — that is **expected
  and not our claim.** The claim Lane A supports is only that `L` is *a zero-shot readout competitive with the
  standard physics tool.* → `results/foldx_detection.csv`. (The `L` side is committed-data and reported now as
  `results/foldx_laneA_Lbaseline.csv`; the FoldX side is added when the binary lands.)

## Lane B — STEERING SPECIFICITY (the decisive lane)
The 60 ipTM complexes (`results/iptm_subset.txt`), arms **wt / L / naive / random** (interface sequences:
`results/cfg_steer_seqs.csv` + `results/cfg_steer_naive_seqs.csv`, α=2, per k∈{0,1,2}). For each (complex, arm, k)
the interface residues where the arm's sequence differs from wt define the mutation set on that complex's crystal
backbone → `BuildModel` → `AnalyseComplex` → **FoldX ΔΔG_bind vs wt**. Aggregate k per (complex, arm) (mean;
best-of-k secondary), then the **paired, complex-clustered bootstrap** (95% CI, P; NBOOT=5000): **L − naive**,
**L − random**, **naive − random** on ΔΔG_bind. → `results/foldx_steer.csv`.

**Sign convention (fixed):** ΔΔG_bind more negative = more favorable binding. We report a **favorability** contrast
oriented so **>0 = better binding** (i.e. `favorability = −ΔΔG_bind`; the paired arm-A − arm-B is on favorability).

**Decisive test:** **L − naive** on ΔΔG_bind favorability. If L is more binding-favorable than the confidence
tilt with a CI clear of 0, L's advantage over "just be more confident" holds on a **physics, non-inverse-folding**
readout — the strongest form of the steering claim (it closes the gap the judge/ipTM divergence exposed).

**Pre-registered falsifier (reported verbatim if it fires):** if **L − naive ΔΔG_bind favorability ≤ 0** (CI
includes or is below 0), the binding-specific steering advantage does **not** transfer to physics → the steering
claim is bounded to the inverse-folding judges. This does not touch the standing **L > random** result or Lane A.

## Positive controls
- **FoldX ΔΔG_bind(wt→wt) ≈ 0** (the wt arm is 0 by construction; verify the pipeline returns ~0 ± FoldX noise).
- Report FoldX's **run-to-run spread** (numberOfRuns=5) as the noise floor.
- Optional gold-standard spot-check: **Rosetta `flex_ddG`** on ~10–20 complexes (heavier; only if the FoldX result
  is borderline or a reviewer insists).

## Outputs / scripts
`results/foldx_worklist.csv` (the exact per-mutation build list, both lanes) + `results/foldx_laneA_Lbaseline.csv`
from `src/foldx_prep.py` (FoldX-independent). `results/foldx_detection.csv`, `results/foldx_steer.csv`,
`results/FINDINGS_foldx.md` from `src/foldx_ddg.py` (RepairPDB/BuildModel/AnalyseComplex driver, idempotent) +
`src/foldx_analyse.py`, produced once the binary is staged.

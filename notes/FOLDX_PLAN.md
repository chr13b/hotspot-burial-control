# FoldX plan — a physics (non-inverse-folding) binding readout, two lanes

*Elevated from "optional" after the naive-guidance result. FoldX is downloadable (academic, free binary), no
weights, no gated deps — the accessible physics comparator. `SEED=20260803`. Pre-register `PREREG_foldx.md`
before any number.*

## Why this matters now
The steering result is currently validated by inverse-folding **judges** (ESM-IF1/MIF/PiFold leverage) and by
**structure-predictor ipTM**. The naive-guidance arm exposed the gap a sharp reviewer will press: L beats the
confidence tilt **only at the judge level**, and the judges are *inverse-folding proxies* — they could share a
blind spot with the steered ProteinMPNN, and ipTM is a foldability/confidence metric (naive wins it). **FoldX is
a physics energy function — a binding readout from a completely different modality**, independent of the
inverse-folding family. If L beats naive on FoldX ΔΔG_bind, the binding-specific advantage of L over "just be
more confident" is shown on a **non-IF, physics** readout — closing the gap.

## What "ΔΔG accuracy" means / what FoldX measures
FoldX predicts **ΔΔG_bind** (change in binding free energy on mutation) from a hand-crafted energy function.
"Accuracy" = correlation of predicted ΔΔG with **experimental** ΔΔG (Spearman/RMSE). FoldX is a *fit* energy
function, not ground truth — but it is a standard, widely-used physics baseline and, crucially, **not** an
inverse-folding model.

## Lane A — DETECTION (L vs physics SOTA, reviewer-anticipation)
On the SKEMPI single-mutation fixture: run FoldX (`RepairPDB` → `BuildModel`/`AnalyseComplex`) to get ΔΔG_bind
per mutation; report **Spearman(FoldX, experimental ΔΔG)** vs **Spearman(L, experimental ΔΔG)** on the *same*
mutations, same metric. Fair, same fixture; disclose FoldX is a fit energy function (it may match/beat a
zero-shot readout — that is expected and not our claim). Positions L as "a zero-shot readout competitive with
the standard physics tool."

## Lane B — STEERING SPECIFICITY (the key one; your addition)
On the 60 ipTM complexes, for each arm — **wt / L-steered / naive-steered / random** (interface sequences from
`cfg_steer_seqs.csv` + `cfg_steer_naive_seqs.csv`) — build the mutant complex and compute **FoldX ΔΔG_bind vs
wt**. Report paired per-complex: **L − naive**, **L − random**, **naive − random** (complex-clustered bootstrap
95% CI). The decisive test: **L − naive on physics ΔΔG_bind**.
- If **L < naive** in ΔΔG_bind (more favorable binding energy, CI clear): L's binding advantage over the
  confidence tilt holds on a **physics, non-IF** readout — the strongest form of "L-steering is binding-specific,
  not foldability." This is what shows the steering's relevance beyond inverse-folding proxies.
- If not: reported verbatim — the binding-specific steering advantage is bounded to the inverse-folding judges;
  we do not overclaim.

## Pre-registration + honesty
Write `PREREG_foldx.md` first (Lane A metric; Lane B falsifier above). FoldX is an approximate energy function —
frame as "physics ΔΔG_bind (FoldX), independent of the inverse-folding modality," not ground truth. This does not
touch the standing L>random / L>naive(judge) facts; it *extends* the readout to physics.

## Feasibility
FoldX binary is a free academic download (no GPU); BuildModel/AnalyseComplex is fast per mutation. Runnable on
Sherlock or locally once the binary is fetched. Rosetta `flex_ddG` is the heavier alternative (skip unless a
reviewer insists). Deliverables: `PREREG_foldx.md`, `foldx_detection.csv`, `foldx_steer.csv`, `FINDINGS_foldx.md`.

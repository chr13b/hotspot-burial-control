# PRE-REGISTRATION — the constraint-vs-leverage gradient as a multi-point law

**Committed BEFORE computing any per-class confidence-AUROC (CLAUDE.md rule 1).** SEED=20260803.

## Motivation
The paper's constraint-vs-leverage fork (§4, T3) currently rests on a **2-point** comparison: interface-hotspot
*confidence*-AUROC is ≈0.538 on natural SKEMPI and ≈0.60 on de-novo Bennett binders. §9 concedes this is
"suggestive, not formal" — two fixtures with *different* hotspot-label constructions. This experiment upgrades it
to a gradient measured **within one pipeline and one label construction**, by stratifying SKEMPI interfaces on
its own pre-existing `Hold_out_type` field (not a proxy we invent), with de-novo as the anchor top point.

## Theory
Confidence (a functional of the bound distribution) reads *fold-stability constraint*. It predicts a hotspot
only insofar as the hotspot's identity is fold-stability-determined. Where binding is **decoupled** from each
partner's independent fold — transient molecular recognition of a non-self surface — hotspots are frequently
frustrated, so confidence should be **blind** (AUROC ≈ 0.5). Where the interface is rigid/pre-organized or the
scaffold is idealized under a binding-dominated objective, confidence should see more.

## Strata (ordered a priori by binding/fold *decoupling*, most-decoupled first)
1. **TCR/pMHC** — lowest-affinity, most transient germline-biased recognition. Predict LOWEST (≈ chance/below).
2. **AB/AG** — transient, affinity-matured recognition of non-self surfaces. Predict low (≈ chance).
3. **Pr/PI** — protease–inhibitor: rigid, pre-organized canonical-inhibitor loops that *are* the inhibitor's
   stable fold. Predict HIGHER than the recognition classes.
4. **de-novo (Bennett)** — binding-dominated selection on idealized rigid scaffolds. Predict HIGHEST (anchor;
   from `bennett_conf_fork.csv`, different pipeline — reported as the anchor, not a within-pipeline stratum).

## Hypotheses
- **H1 (confirmatory, the one we are confident in).** de-novo confidence-AUROC exceeds **every** natural SKEMPI
  class, and every natural class is at or near chance (CI touches 0.5). This replicates and *strengthens* the
  2-point: the blindness is a property of natural interfaces of every type, not a single-fixture artifact.
- **H2 (exploratory, weaker).** Among the three natural classes, confidence-AUROC rises monotonically along the
  a-priori rank TCR/pMHC ≤ AB/AG ≤ Pr/PI (Spearman(rank, AUROC) > 0). **A non-monotonic result is a valid,
  reported null** — it would mean confidence is uniformly blind across natural interface types, which is itself
  a clean statement. We will NOT re-order the strata after seeing the numbers.

## Primary statistic and confound control
- **Primary:** per stratum, confidence-AUROC = AUROC(is_hot ~ log p(native)) over interface positions,
  **complex-clustered bootstrap** (resample complexes), 2000 replicates, seed 20260803.
- **Circularity control (load-bearing — this is a burial project).** Report alongside each class:
  (a) burial-AUROC = AUROC(is_hot ~ burial), and (b) burial-residualized confidence-AUROC (AUROC of the
  Pearson residual of conf on burial). If the confidence signal is just burial, (b) collapses to 0.5 in every
  class; any gradient must survive (b) to count.
- n_complexes, n_interface, n_hot reported per class. Classes with <5 complexes or <15 hotspots are reported
  but flagged underpowered and excluded from the H2 trend test.

## Falsifier
If de-novo does NOT exceed the natural classes (H1 fails) after this within-pipeline breakout, the
constraint-vs-leverage fork is weaker than the 2-point suggested and we say so. H2 is exploratory and either
outcome is reported verbatim.

Output: `results/threepoint_law.csv`. Script: `src/p_3point_law.py`.

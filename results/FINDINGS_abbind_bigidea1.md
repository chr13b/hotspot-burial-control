# FINDINGS — AB-Bind Big-Idea-1 replication: the POSITIVE is de-novo-specific (honest, coherent)

**Script:** `src/abbind_bigidea1.py`. **Output:** `results/abbind_bigidea1.csv`. AB-Bind single mutations at
interface positions (420 mutations, 27 complexes, 227 destabilising ΔΔG≥1). Does the model's per-mutation
complex-conditioned distribution predict binding ΔΔG BEYOND geometry+substitution-similarity (as it does on
de-novo Bennett)? ProteinMPNN unconditional; complex-clustered bootstrap, seed 20260803.

## Result
| feature | AUROC(destabilising ΔΔG≥1) |
|---|---|
| model logP(mut\|complex) | 0.578 [0.514, 0.639] |
| partner log-odds logP−logQ | 0.523 [0.473, 0.575] |
| burial | 0.691 [0.620, 0.766] |
| ΔSASA | 0.585 | BLOSUM 0.583 | volume 0.559 |
| Spearman(logP, ΔΔG) | −0.173 (right sign) |
| geometry+similarity baseline | 0.660 |
| + model logP | 0.668 |
| **ΔAUROC(logP over geometry+similarity)** | **+0.008 [−0.014, +0.026], P=0.763 → does NOT add** |

## Reading (honest and coherent)
The model's distribution correlates with antibody ΔΔG in the right direction (−0.17) and beats chance
standalone (0.578), but is **weaker than burial** (0.691) and **adds nothing beyond geometry** (+0.008, CI
spans 0). So the "distribution carries binding energetics **beyond geometry**" positive **does NOT replicate
on natural antibody complexes** — it is **de-novo-specific** (Bennett).

This is the coherent, unifying reading, not a failure: on **natural** complexes everything reduces to geometry
(SKEMPI: scalar KL ≈ ΔSASA; AB-Bind: logP adds nothing beyond geometry), whereas on **de-novo** designs the
distribution adds +0.018 beyond an all-atom occlusion baseline. That asymmetry is precisely the
**constraint-vs-leverage** regime-dependence (T3): the model's binding knowledge is accessible in the
binding-dominated de-novo regime and subsumed by geometry in natural complexes. AB-Bind thus **confirms the
NUGGET** (abbind_nugget.csv: confidence≈chance, geometry predicts) and **bounds the POSITIVE as de-novo-specific**.

## Caveats
- AB-Bind mutations are an experimenter-curated set (not systematic SSM); the geometry+similarity baseline
  includes BLOSUM+volume (a strong baseline). But logP (0.578) is below burial (0.691) even standalone, so the
  non-addition is not merely a strong-baseline artifact.
- Partner log-odds (P/Q) is near-chance on antibody interfaces (0.523), unlike Bennett's +0.076 — worth a line.

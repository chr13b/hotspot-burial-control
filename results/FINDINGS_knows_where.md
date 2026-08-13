# FINDINGS — "knows WHERE / knows WHAT" (Big Idea 1, pre-registered)

**Pre-registered:** `results/PREREG_knows_where.md` (committed 287f884, before the analysis existed).
**Script:** `src/bennett_knows_where.py`. **Outputs:** `results/bennett_knows_where.csv` (+ `_pairs.csv`).
Bennett de-novo SSM, 73 designs, 60,971 (position, substitution) pairs; ProteinMPNN unconditional,
sequence-free; design-clustered bootstrap, seed 20260803.

## Question
Does an inverse-folding model rank the 19 measured substitutions at a position by whether they **retain
binding** (kd_lb < cap) — and does that ability differ between positions governed by fold **stability**
(core) and by **binding** (interface), and between complex- and binder-conditioned distributions?

## Result

| layer | AUROC **P** (complex) | AUROC **Q** (binder-alone) | BLOSUM62 | volume | hydropathy | n pairs |
|---|---|---|---|---|---|---|
| **core** (stability, +control) | **0.721** [0.700,0.743] | 0.726 [0.704,0.748] | 0.620 | 0.608 | 0.596 | 15,162 |
| surface (neutral) | 0.680 [0.646,0.721] | 0.684 [0.650,0.727] | 0.570 | 0.580 | 0.505 | 18,639 |
| **interface** (binding) | **0.615** [0.601,0.628] | **0.539** [0.527,0.550] | 0.589 | 0.539 | 0.579 | 27,170 |

**Pre-registered tests — all three fired:**
- **P1 ✓** interface AUROC(P) = 0.615 [0.601,0.628] > 0.5. The model ranks interface substitutions by
  measured binding above chance, and **beats every sequence baseline** at the interface (BLOSUM 0.589,
  hydro 0.579, volume 0.539) — it is not merely a substitution-similarity matrix.
- **P2 ✓ (dissociation)** core AUROC(P) 0.721 vs interface 0.615, **Δ = +0.107**, CIs non-overlapping. The
  model predicts the *stability* question markedly better than the *binding* question — exactly the
  built-in positive control the design demands.
- **P3 ✓ (the surprise — optimistic branch)** conditioning on the partner **adds** interface-binding
  information: interface AUROC(P) − AUROC(Q) = **+0.076 [+0.068,+0.084], P(>0)=1.000**, and it is
  **interface-specific** (P ≈ Q at core and surface, where the partner is irrelevant: 0.721/0.726 and
  0.680/0.684).

## Reading (honest)
The model's partner-conditioning carries **genuine interface-binding information**: it down-weights
substitutions that clash with the partner, and those substitutions also abolish binding — steric
complementarity. This is a **positive result on de-novo designs with experimental per-substitution
labels**, and it **softens (does not contradict) the crystal KL≈ΔSASA scalar finding**: the *full*
conditional distribution carries more per-substitution binding signal (P3 +0.076, interface-specific) than
the *scalar* KL summary, which on crystal is well-approximated by ΔSASA. Two true things coexist: (i) the
model is relatively **worse** at binding than at fold-stability (P2, −0.107); (ii) but its partner-
conditioning **does** add real, baseline-beating interface-binding signal (P1, P3).

## Caveats and the decisive follow-up
- **Occlusion vs energetics is not yet separated.** P3's advantage is consistent with the model encoding
  steric occlusion (a clashing substitution is disfavored *and* abolishes binding) — a form of energetics,
  but a geometric one. The sharp follow-up: does interface AUROC(P) beat AUROC(Q) **after controlling for
  ΔSASA / a clash model** (per-substitution volume increase × contact area)? If yes → the model encodes
  binding beyond geometry; if no → it is occlusion. This is the per-substitution analogue of the KL≈ΔSASA
  test and is the natural next experiment (CPU, same data).
- Labels convolve display/fold-stability with binding — the three-way stratification is the control (core
  = stability). 4 targets / one epitope each; design-clustered bootstrap; report per-target signs.
- Parent sequences are ProteinMPNN outputs (native-biased) — but the native is excluded and we rank the 19
  alternatives; the bias is identical across strata.
- Pooled AUROC mixes positions of differing base rate within a layer; per-position AUROC (≥3 of each class)
  is the robustness check (to add).

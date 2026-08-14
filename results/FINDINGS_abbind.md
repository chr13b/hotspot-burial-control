# FINDINGS — AB-Bind second fixture: the nugget replicates (single-fixture objection defused)

**Script:** `src/abbind_nugget.py`. **Outputs:** `results/abbind_nugget.csv` (+ `abbind_positions.csv`).
**Data:** AB-Bind (Sirin et al. 2016; `~/ftax/data/ab-bind`, github.com/sarahsirin/AB-Bind-Database), 1101
mutants / 32 antibody–antigen (and related) complexes with experimental ΔΔG(kcal/mol). ProteinMPNN
unconditional confidence; complex-clustered bootstrap, seed 20260803.

## Question
Does the SKEMPI nugget — per-residue **confidence is at chance for interface hotspots while free geometry
predicts, and confidence adds ~0 over geometry** — replicate on a *second, biophysically-distinct* fixture?
This is the direct answer to both critics' #1 structural weakness (single main fixture).

## Result
Per interface position (single-mutation ΔΔG; hotspot = ΔΔG ≥ 1 kcal/mol, destabilising binding). Positive
control: mutation-WT == structure residue = **0.932** (491 mapped positions; one complex 3NPS dropped on a
parser error). 324 interface measured positions, 27 complexes, 173 hotspots.

| feature | AUROC (interface hotspots) | verdict | SKEMPI analogue |
|---|---|---|---|
| **confidence** (log p native) | **0.560 [0.497, 0.624]** | **~chance** (CI spans 0.5) | 0.538 |
| burial | 0.728 [0.659, 0.796] | predicts | 0.689 |
| ΔSASA | 0.604 [0.549, 0.666] | predicts | 0.585 |
| nbr | 0.658 [0.595, 0.728] | predicts | 0.673 |
| full geometry (burial+nbr+ΔSASA) | 0.677 | — | 0.734 |
| geometry + confidence | 0.685 | — | — |
| **ΔAUROC(confidence over geometry)** | **+0.008 [−0.005, +0.021], P=0.891** | adds ~nothing | ~0 |

## Reading
**The nugget replicates.** On antibody–antigen ΔΔG, confidence ranks interface hotspots at chance (0.560, CI
includes 0.5), far below trivial burial (0.728), and adds essentially nothing over the free-geometry baseline
(+0.008, CI spans 0). Identical shape to SKEMPI. "Confidence is not competence, and free geometry is what
predicts" is therefore **not a SKEMPI artifact** — it holds across two SKEMPI-class fixtures of different
biophysics, plus the de-novo Bennett fixture for the distribution-knows-binding positive. The single-main-
fixture objection is defused for the core claim.

## Caveats
- Confidence point estimate (0.560) is marginally above SKEMPI's (0.538) but the CI includes chance; the
  operative fact is confidence ≪ burial and adds ~0 over geometry.
- One complex (3NPS) dropped on a PDB-parser error; 27/32 contribute interface positions. Not hotspot-selective.
- AB-Bind mixes single- and multi-mutants; we use single mutations only for clean per-position labels.
- This replicates the NUGGET (confidence≠competence); the full Big-Idea-1 positive remains a de-novo (Bennett)
  result. A per-substitution distribution test on AB-Bind (Big-Idea-1 analogue) is a possible extension.

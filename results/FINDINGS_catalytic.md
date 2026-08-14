# FINDINGS — catalytic dissociation: real raw effect, but MOSTLY COMPOSITION (honest partial-null)

**Script:** `src/catalytic_dissociation.py`. **Output:** `results/catalytic_dissociation.csv`
(+ `catalytic_positions.csv`). M-CSA catalytic residues (130 enzymes selected → 114 scored, 40,951 positions,
391 catalytic = 1.0%; data `~/ftax/data/m-csa`). ProteinMPNN unconditional confidence vs ESM-2 (150M)
per-position negentropy; enzyme-clustered bootstrap, seed 20260803.

## Question
Does the nugget generalize beyond binding hotspots — is IF confidence blind to CATALYTIC residues *even
though* a sequence PLM (ESM-2) finds them (a constraint-vs-function dissociation)? And critically, does any
such effect survive controlling for **amino-acid composition** (catalytic residues are enriched in
His/Asp/Glu/Cys/Ser — ProteinMPNN's worst-recovered types)?

## Result
| | AUROC (catalytic) |
|---|---|
| MPNN confidence | **0.398 [0.360, 0.439]** — anti-predictive |
| ESM-2 negentropy | 0.755 [0.704, 0.807] — predicts |
| ESM-2 logp(native) | 0.760 [0.712, 0.808] — predicts |
| raw dissociation (ESM-2 − MPNN) | +0.357 [+0.306, +0.405], P=1.000 |

**Composition control (the decisive test):**
| | AUROC | ΔAUROC over aa-identity |
|---|---|---|
| amino-acid identity alone | **0.853** | — |
| + MPNN confidence | 0.853 | **−0.0004 [−0.0026, +0.0017] → VANISHES** |
| + ESM-2 negentropy | 0.885 | **+0.0317 [+0.0165, +0.0472] → survives (small)** |

## Honest reading
The striking raw dissociation is **mostly a composition artifact.** Amino-acid identity alone predicts
catalytic residues at 0.853; once controlled, **IF confidence adds nothing** (ΔAUROC ≈ 0) — the "anti-
prediction" (0.398) is entirely that catalytic residues are the amino-acid types ProteinMPNN recovers worst,
**not** a frustration/constraint signal beyond composition. ESM-2 conservation retains a small genuine
signal beyond composition (+0.032). So the *beyond-composition* dissociation is real but ~10× smaller than the
raw one, and the clean "IF is blind to catalytic sites because they are frustrated" mechanism **does not
survive**. Same confound that bit ProBID-Net (composition), caught by the same control.

## Verdict for the paper
**NOT a headline.** This is an honest partial-null: the field-level generalization to catalytic sites does
NOT cleanly hold (the mechanistic claim is composition, not frustration). It is weaker than the binding-
hotspot nugget, which *does* survive its geometry control (CPI 0.000). At most a one-paragraph cautionary
note ("IF confidence's apparent blindness to catalytic residues is amino-acid composition; the clean
constraint-vs-function dissociation we find for *binding* hotspots does not transfer to catalytic sites once
composition is controlled"). Does NOT raise the ICLR ceiling; reported for the record and the discipline.

## Caveats / possible (lower-priority) rescue
- The comparison controls amino-acid IDENTITY; a subtler test (per-type frustration: within His residues, do
  catalytic His have lower IF confidence than non-catalytic His?) could still show a within-type effect, but
  the aa-controlled ΔAUROC≈0 already bounds it as small.
- ESM-2 150M; a larger PLM (650M) would likely raise the ESM side but not change the MPNN-is-composition verdict.
- Catalytic label = M-CSA mechanistic roles (proton shuttle / covalent / electron shuttle / reactant).

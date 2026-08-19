# Paste-to-start prompt for Project #3 (AlphaFold-Multimer)

Copy everything in the code block below into a fresh Claude Code session (on Sherlock, or wherever the GPUs
are) to kick this project off. It is written to make the agent read the handoff, freeze the pre-registration
before touching a GPU, and run the powered pilot before committing compute.

```
We are starting the follow-up paper to "Confidence is not competence" (the inverse-folding hotspot paper).
This project asks: is confidence the wrong derivative across structural generative models too? — specifically
for AlphaFold-Multimer. Read handoff_afmultimer/README.md in full first; it has the two-tier design, the
pre-registered discriminator, the reusable code, the 344-complex list, and the decision points.

Before any GPU run, do this in order and STOP for my go after step 3:

1. Write handoff_afmultimer/PREREG.md by copying the ground rules from ../CLAUDE.md and ../BRIEF.md §4
   (pre-register the falsifiers: burial-matched pairs, CPI + within-stratum-AUROC readouts, complex-clustered
   bootstrap, fixed seeds, positive control before trusting any zero). Freeze it — do not move a falsifier
   after seeing a number. Commit it.

2. Decide and record two things in PREREG.md: (a) ColabFold vs full AlphaFold-Multimer (default ColabFold with
   MSA caching); (b) the memorization control — a recent/held-out-PDB subset or template-off ablation, mirroring
   the main paper's Exp A/D leakage checks (AF has seen many of these PDBs).

3. Set up the AF stack + MSA cache on Sherlock and smoke-test on 1BRS (barnase–barstar), predicting the BOUND
   complex and each UNBOUND monomer. Confirm handoff_afmultimer/starter_afm_mixed_derivative.py's pLDDT/PAE
   parser aligns to ../results/leverage_skempi_positions.csv (same positions => the two papers are directly
   comparable). Report the smoke-test result and the estimated GPU-hours for the full Tier-1 run, then stop
   for my go.

4. (After my go) Run the Tier-1 powered pilot on ~30 complexes: confidence-blindness (is pLDDT/PAE blind at
   hotspots within burial-matched pairs?) + the partner-ablation mixed-derivative CPI. Go/no-go on the full
   344 and on Tier 2 (per-mutation predictions) from the pilot.

Discipline: every number traces to a committed CSV with its exact command; honest nulls are valid outcomes
(both "AF confidence is blind, the ablation derivative carries binding" AND "AF is partner-sensitive, so the
thesis is IF-specific" are publishable); never fabricate or extrapolate a measurement. Keep me updated on
GPU-hours and the go/no-go at each gate.
```

Companion note for you (not for the agent): the handoff is self-contained, but the one thing only *you* can
supply at kickoff is the Sherlock allocation + which AF stack is already installed there. Everything else the
agent can drive from the README + starter skeleton.

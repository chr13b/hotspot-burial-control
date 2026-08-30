# Sherlock kickoff — R2: leverage on REAL predicted backbones (closes the design-regime gap)

**Why this needs Sherlock:** the OpenFold3 / AF2-multimer predicted structures live at
`/scratch/users/cbertsch/ftax/predicted` — not on the laptop. Everything else (ProteinMPNN, the leverage
scorer, SKEMPI ΔΔG) is already in the repo. This is the paper's #2 reviewer objection (R2): the mixed
derivative has only ever been computed on CRYSTAL backbones (natural complexes with known answers) or under
SYNTHETIC jitter — never on the predicted backbones a designer actually uses. Either outcome is publishable.

## PASTE FROM HERE

You are continuing `hotspot-burial-control` on Sherlock. Read `CLAUDE.md` + `BRIEF.md` (binding rules:
pre-register before any number; never fabricate/extrapolate; write CSVs with the exact command; run a positive
control before trusting a zero).

**Goal.** Compute `CPI(L | geometry)` and the `geometry + |L|` hotspot ranker on the OpenFold3 and
AF2-multimer PREDICTED backbones, for the ~127 complexes shared with the SKEMPI ΔΔG fixture (the set already
used in `expA_gap_summary.csv` / `expD_gap_summary.csv`). This tests whether the binding signal in the mixed
derivative SURVIVES on the structures designers actually get from a folding model.

**Freeze this pre-registration (commit BEFORE any σ>0 / any CPI number):**
- **Prediction:** on predicted backbones, `CPI(L | geometry)` is POSITIVE but ATTENUATED vs the crystal value
  (+0.059 mutation-level), consistent with the dose law — OpenFold3/AF2 interface RMSD is typically ~0.5–1.5 Å,
  the regime where the crystal ladder shows partial-to-full survival. Predict roughly the σ≈0.5–1.0 Å rung:
  CPI in ~[+0.01, +0.04].
- **Positive control (gate before trusting):** re-score leverage on the CRYSTAL backbones for these SAME 127
  complexes through this pipeline; it must reproduce the committed crystal CPI(L|geom) (≈+0.059 mutation-level,
  ≈+0.0048 position-level). If it does not, the predicted-backbone number is not interpretable — stop and debug.
- **Falsifier (either way strengthens the paper):** if CPI(L|geom) COLLAPSES to the placebo floor on predicted
  backbones, leverage does NOT generalize to the design regime → report it as a measured factorization tax on
  the real pipeline (a STRONGER version of the thesis). If it SURVIVES → the actionable claim generalizes and
  the dose law is conservative. Do not soften either result.

**Method (mirror the crystal leverage pipeline, swap the PDB source):**
1. For each of the 127 complexes, take the OF3 and AF2 predicted PDB from `$SCRATCH/ftax/predicted/...` (the
   expA pipeline already aligned these to SKEMPI chain/residue numbering — reuse `expA_p0_positions.csv` /
   `expA_kl_joined.csv` for the mapping; do NOT re-align from scratch).
2. Run the committed leverage scorer (`src/leverage_decomposition.py` score stage, ProteinMPNN, sequence-free
   unconditional marginal) with `DATA/PDBs` pointed at the predicted-PDB dir instead of the crystal dir. The
   monomer construction (delete partner chain) is identical.
3. Join to SKEMPI single-mutation ΔΔG; compute `CPI(L | burial+nbr+ΔSASA)` (mutation- and position-level) and
   the `geometry + |L|_rms` AUROC lift, using the SAME `LD.cpi` estimator (complex-clustered bootstrap, z-scored
   features, seed 20260803). Geometry (burial/nbr/ΔSASA) should be recomputed FROM THE PREDICTED structure
   (that is the honest baseline a designer has), not the crystal — state this.
4. Report per predictor (OF3, AF2) and pooled, with CI, alongside the crystal positive-control value.

**Compute:** ProteinMPNN is CPU-fast (~seconds/complex); 127 complexes × 2 predictors × (complex+monomer) is
minutes-to-an-hour on one node. No GPU strictly needed for ProteinMPNN, but a node with the predicted PDBs
mounted is the point.

**On success:** commit `results/leverage_predicted_{of3,af2}.csv` (+ the crystal control) with exact commands;
update paper §6 — merge the predicted-backbone leverage INTO the dose-law section (per `MANUSCRIPT_PREP.md`
Phase C: §6 is the empirical anchor for "does this matter in the design regime", NOT a separate appendix
section); update `verdict-state.md`. `git push` to main. Do NOT touch Phase 2 (MultiFlow) or the figures.

## PASTE TO HERE

**For Chris (not the agent):** this is the last reject-level item (R2). With R1 already killed (R²(L|P)=~70%
irreducible) and this closed, both of the weaknesses-audit's reject-level objections are answered. See
`notes/MANUSCRIPT_PREP.md` Phase A.

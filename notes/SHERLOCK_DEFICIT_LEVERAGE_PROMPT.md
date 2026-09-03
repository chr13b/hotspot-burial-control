# Sherlock task — #7: does confidence degrade where leverage is stressed? (per-complex correlation)

**Paste to the Sherlock session.** Exploratory, honest-null-is-fine. Test whether the per-complex
*confidence-recovery deficit* on predicted backbones (§6) coincides with per-complex *leverage stress* — i.e.
do the two readouts of the decomposition move together across complexes? This can only run on Sherlock: the raw
per-position predicted-backbone leverage lives in `results/expD_scored_positions.csv`, which is a git-LFS pointer
locally. `SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.

## 0. Sync
```bash
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
git pull --no-edit origin main          # HEAD >= b2386c6
srun -c 4 --mem=8G -t 1:00:00 --pty bash    # light; CPU fine
```

## 1. Two per-complex quantities
- **Confidence deficit (already committed):** `results/expD_af2_of3_corr_percomplex.csv` has per-complex
  `d_of3`, `d_af2`, `d_crystal` (the burial-matched recovery deficit under each predictor). Use `d_of3` (and
  `d_af2` as a check) — more-negative = larger confidence deficit.
- **Leverage retention (compute here):** from the R2 predicted-backbone scored positions
  (`results/expD_scored_positions.csv` — the LFS blob's real content, or the source table on `$SCRATCH` from the
  R2 run; see `results/expD_backbone_manifest.csv` / the `leverage_predicted.py` command string). For each
  complex, compute a per-complex leverage metric on the **predicted** backbone — e.g. mean |L|_rms over its
  interface positions, or (better) CPI-style leverage kept vs the crystal value. If a per-complex predicted vs
  crystal leverage ratio is easy, that "retention" is the ideal x; otherwise per-complex mean |L| on predicted
  is an acceptable proxy (state which you used).

## 2. Correlate
Merge the two per-complex tables on `complex_id` (~140 shared). Report **Spearman(d_of3, leverage-retention)**
and Spearman(d_af2, ·), with a complex-bootstrap 95% CI. Interpretation:
- a **negative** correlation (larger deficit ↔ lower leverage retention) = the two readouts degrade *together*,
  a within-complex confirmation that the confidence deficit and leverage stress are the same phenomenon;
- a **null** = they are *separable* (confidence degrades independently of where leverage survives) — an honest
  dissociation, equally reportable.
Do **not** spin whichever way it lands; ~140 complexes with sparse hotspots is likely underpowered, so frame it
as exploratory and report the CI.

## 3. Positive control (rule 6)
Before trusting the correlation, confirm the merge is real: the per-complex leverage metric must reproduce the
committed pooled R2 number when aggregated (CPI(L|geom) ≈ +0.039 OF3), and `d_crystal` must be ~0 (the crystal
control had no deficit). If either fails, the join/aggregation is wrong — stop and report.

## 4. Deliverables
Commit **only** `results/deficit_vs_leverage_percomplex.csv` (columns: `complex_id, d_of3, d_af2,
leverage_metric, metric_name`) + a short `results/FINDINGS_deficit_leverage.md` (the Spearman(s) + CI, which
metric you used, the positive-control result, `SEED`, exact command). `git add` by name, commit with the repo's
two trailer lines, push, and message me the correlation.

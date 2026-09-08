# Sherlock — Lane A deepening: reproduce + AB-Bind physics + L→kcal/mol calibration [paste-to-run]

`SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.
`git pull --no-edit origin main` FIRST. CLAUDE.md rules apply (never fabricate; every number → committed CSV;
positive controls; `git add` by name; two trailer lines; push). **Pre-register `results/PREREG_laneA_deepen.md`
before any new number.** Three tasks, independent; do #1 and #3 even if #2's fixture is missing.

## Task 1 — REPRODUCE the Lane A deepening (CPU, no FoldX binary; a check on the local run)
`python3 src/foldx_laneA_deepen.py` and confirm it reproduces `results/foldx_laneA_deepen.csv`:
Spearman(FoldX,exp)=+0.4344 [+0.38,+0.49]; Spearman(L,exp)=−0.3009 [−0.35,−0.24]; **partial(L,exp|FoldX)=−0.173
[−0.229,−0.117] P(<0)=1.0**; learned OOF ensemble uplift +0.014 [−0.001,+0.031]. If any number drifts, STOP and
report (it should be deterministic at SEED). This is the recovered "L adds beyond physics" result — a check, not
a new claim.

## Task 2 — AB-Bind physics detection (#2: does L's detection + beyond-physics hold on antibody–antigen?)
**New FoldX runs** (needs the licensed binary at `$SCRATCH/ftax/foldx/`, as in `PREREG_foldx.md`). Pre-registered
metric identical to SKEMPI Lane A: **Spearman(FoldX ΔΔG_bind, exp ΔΔG)**, **Spearman(L, exp)**, **partial(L,exp
|FoldX)**, complex-clustered 95% CI.
- Fixture: the AB-Bind single-mutation interface set with experimental ΔΔG **and** `L` already computed (look for
  `results/leverage_abbind_mutations.csv` or the AB-Bind rows in the leverage fixture used for `abbind_cpi.csv`).
  If `L` is not yet computed on AB-Bind, compute it with the **same** partner-ablation code used for SKEMPI
  (`src/leverage_decomposition.py`) BEFORE any FoldX — no new modeling choices.
- Reuse `src/foldx_prep.py` / `src/foldx_ddg.py` / `src/foldx_analyse.py` unchanged, pointing the worklist at the
  AB-Bind fixture (`--lane A` path). RepairPDB → BuildModel → AnalyseComplex, numberOfRuns=5.
- **Honest power caveat (state upfront):** AB-Bind is ~27 complexes — likely **underpowered**; report the CI and
  say "indeterminate" if it spans zero rather than over-reading. A replicated *direction* (negative L–exp, positive
  FoldX–exp, partial(L|FoldX)<0) is the win; magnitude is secondary. → `results/foldx_detection_abbind.csv`.
- **Positive control:** confirm the AB-Bind wt/self mappings resolve (WT-identity gate, as ATLAS) before trusting
  a null; report how many mutations mapped cleanly.

## Task 3 — Calibrate L → kcal/mol (#3, CPU, no binary)
On the SKEMPI Lane A mutations (paired `L`, FoldX ΔΔG_bind, experimental ΔΔG), fit and **cross-validate**
(complex-clustered K-fold, no complex in both train and test):
- Linear `ΔΔG_exp ≈ a + b·(−L)` → slope `b` (kcal/mol per unit leverage) with complex-clustered 95% CI; report
  out-of-fold **RMSE** and Pearson/Spearman of the calibrated `−b·L` against experimental ΔΔG.
- Same for FoldX as the reference calibration; report both so the reader sees where zero-shot `L` sits vs the
  fitted tool in **absolute kcal/mol**, not just rank.
- **Guard against the tuning objection:** the calibration is a *single global* slope+intercept (2 params), fit
  out-of-fold; do **not** fit per-position temperatures (that edges into the "you fit it" critique). Report it as
  a *reading*, not a re-trained model. → `results/laneA_calibration.csv`, one figure optional.

## Deliverables (commit by name, two trailer lines, push)
`PREREG_laneA_deepen.md`, `foldx_detection_abbind.csv` (if run), `laneA_calibration.csv`,
`FINDINGS_laneA_deepen.md` (what replicated, what was indeterminate, the calibration slope). Message me the AB-Bind
partial(L|FoldX) and the calibration slope+RMSE. Note: Task 1 & 3 are CPU-only; only Task 2 needs the FoldX binary.

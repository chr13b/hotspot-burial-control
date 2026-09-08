# PRE-REGISTRATION — Lane A deepening (reproduce + AB-Bind physics + L→kcal/mol calibration)

Registered **before any new number**. `SEED=20260803`, `NBOOT=5000`, `KFOLD=10` (complex-clustered throughout —
the unit of resampling is the complex, matching Lane B and `foldx_laneA_deepen.py`). CLAUDE.md rules apply: never
fabricate; every number → committed CSV; positive controls before trusting a zero/null; falsifiers below are fixed
and will not be moved after seeing a value.

Three independent tasks. #1 and #3 are CPU-only (no FoldX binary) and are the guaranteed deliverables; #2 needs the
licensed FoldX binary **and** a fixture that may be missing (see the contingency in §2).

---

## Task 1 — REPRODUCE the committed Lane-A deepening (determinism check, NOT a new claim)
`python3 src/foldx_laneA_deepen.py` must reproduce `results/foldx_laneA_deepen.csv` **exactly** (it is deterministic
at `SEED`). Pre-registered expected values (the committed CSV):

| metric | value | 95% CI |
|---|---|---|
| `spearman_FoldX_vs_exp` | +0.4344 | [+0.3804, +0.4900] |
| `spearman_L_vs_exp` | −0.3009 | [−0.3548, −0.2443] |
| `partial_L_given_FoldX` | −0.1727 | [−0.2288, −0.1170], P(<0)=1.0 |
| `ensemble_uplift_over_FoldX` | +0.0142 | [−0.0011, +0.0305], P(>0)=0.967 |

**Decision rule:** any drift in any cell → **STOP and report** (would indicate non-determinism / a regression). No
new number is produced; this is a check on the recovered "L adds beyond a fitted physics ΔΔG function" result.

## Task 2 — AB-Bind physics detection (does L's detection + beyond-physics generalise to antibody–antigen?)
**Pre-registered metrics** (identical to SKEMPI Lane A), computed on the AB-Bind single-mutation interface set that
already carries experimental ΔΔG and `L` (`results/leverage_abbind_mutations.csv`), with **new FoldX runs** for
ΔΔG_bind:
- `Spearman(FoldX ΔΔG_bind, exp ΔΔG)` — directional expectation **> 0** (physics detects destabilisation).
- `Spearman(L, exp ΔΔG)` — directional expectation **< 0** (replicate the SKEMPI leverage sign).
- `partial(L, exp | FoldX)` — directional expectation **< 0** (the key test: L carries binding rank-signal
  **beyond** the fitted physics energy on antibody–antigen too).
- All three with **complex-clustered 95% CI** (`NBOOT`, `SEED`).

**Honest power caveat (stated up front, not after the fact):** AB-Bind here is ≤ ~23 real-crystal complexes —
**likely underpowered**. The pre-registered **win is a replicated *direction*** (`Spearman(L,exp)<0`,
`Spearman(FoldX,exp)>0`, `partial(L|FoldX)<0`); **magnitude is secondary**.

**Falsifier (fixed):**
- `partial(L | FoldX) ≥ 0` with the 95% CI **excluding the negative side** → FoldX fully explains L on
  antibody–antigen → the "L beyond physics" result is **bounded to SKEMPI-type complexes** (a real limitation, to be
  reported as such).
- `partial(L | FoldX)` CI **spanning zero** → **INDETERMINATE (underpowered)** — reported as such, **not** a
  refutation and **not** a confirmation. No over-reading either way.

**Positive controls before trusting any null (CLAUDE.md rule 6):**
- **WT-identity gate** (as ATLAS): after RepairPDB, the wt residue at each (chain, resnum) must match the AB-Bind wt;
  report cleanly-mapped / total. A low mapping rate invalidates the null (would mean structure/numbering mismatch),
  not a real absence of signal.
- `Spearman(L, exp)` on the AB-Bind set must reproduce the standing committed AB-Bind leverage direction (sanity that
  the `L`/ddG columns are read correctly).

**Fixture-availability contingency (a setup fact, discovered before any number):** the AB-Bind raw source
(`~/ftax/data/ab-bind/AB-Bind_experimental_data.csv`, holding the authoritative `Partners(A_B)` chain-group
partition, and the AB-Bind-specific PDBs) has been **purged from `$SCRATCH`** (90-day inactivity; the committed
`L`/ddG fixture survives, the raw partition does not). Task 2 therefore runs **only if both gates pass**:
  1. the `Partners(A_B)` partition is restored from the **authoritative public AB-Bind release**, and a
     **positive control** confirms the re-fetched single-mutation (PDB, mutation, ddG) rows **match the committed
     `leverage_abbind_mutations.csv`** (same ddG values) — so the recovered partition is provably the one used; **and**
  2. the FoldX **WT-identity gate** on the available generic crystal PDBs maps a usable fraction of mutations.
Homology-model entries (`HM_*`, no crystal structure) are **excluded** — reported as a coverage cut, not silently
dropped. If either gate fails, Task 2 is reported **BLOCKED (fixture missing)** — per the handoff's explicit
"do #1 and #3 even if #2's fixture is missing"; **no chain-group partition will be reconstructed by any new modeling
choice.**

## Task 3 — Calibrate L → kcal/mol (CPU, no binary)
On the SKEMPI Lane-A mutations (paired `L`, FoldX ΔΔG_bind, experimental ΔΔG; the exact merge of
`foldx_laneA_deepen.py`, n≈2948 / 284 complexes), a **single global** 2-parameter calibration, cross-validated with
**complex-clustered K-fold** (K=`KFOLD`, no complex in both train and test):
- `ΔΔG_exp ≈ a + b·(−L)` → slope `b` (kcal/mol per unit leverage) with **complex-clustered 95% CI** (global fit +
  bootstrap); out-of-fold **RMSE**; out-of-fold **Pearson** and **Spearman** of the calibrated `a + b·(−L)` vs
  experimental ΔΔG.
- **Same for FoldX** as the reference calibration (`ΔΔG_exp ≈ a + b·FoldX`) — so the reader sees where zero-shot `L`
  sits vs the fitted physics tool in **absolute kcal/mol**, not just rank.
- Report an **intercept-only** (predict-the-train-mean) OOF RMSE as the floor.

**Guard against the tuning objection (fixed):** the calibration is a **single global slope + intercept (2 params)**,
fit out-of-fold — **no per-position temperatures, no per-complex parameters** (that would edge into the "you fit it"
critique). It is reported as a **reading** (where L lands in kcal/mol), **not** a re-trained model. No falsifier —
this is descriptive calibration; the number reported is the slope and the honest OOF error.

---

### Deliverables
`results/PREREG_laneA_deepen.md` (this file), `results/foldx_detection_abbind.csv` (if Task 2 runs),
`results/laneA_calibration.csv`, `results/FINDINGS_laneA_deepen.md`. Every value traces to a committed CSV with the
exact command and SEED.

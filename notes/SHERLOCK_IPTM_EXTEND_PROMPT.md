# Sherlock — extend predicted-backbone ipTM confirmation from n=21 to n≈106 [paste-to-run]

`SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.
`git pull --no-edit origin main` FIRST. CLAUDE.md rules apply (never fabricate; every number → committed CSV;
positive controls; `git add` by name; two trailer lines; push). This does **NOT** change the pre-registered
falsifier or the crystal anchors — it only raises the sample size of the *confirmation* lane.

## Why (a robustness upgrade, not a new claim)
The predicted-backbone steering result is decisive at the **judge level, n=106** (`cfg_steer_predicted.csv`,
L−random +0.70–0.76). The structure-predictor **confirmation** (AF2-multimer ipTM) was folded only on the **21**
complexes overlapping the original 60-complex ipTM set (`iptm_predicted.csv`, ipTM L−random +0.19). This run folds
the **remaining ~85** predicted-backbone steered complexes so the fold-level confirmation also reaches **n≈106**,
matching the judge lane and removing any "small-n" objection. Expected: ipTM L−random stays **> 0, ~+0.19**
(attenuated vs the crystal +0.235, as already reported); report the new value honestly whatever it is.

## Phase 0 — inputs (reuse; do not re-steer if avoidable)
- The OF3-steered sequences are the fold inputs (steer on the predicted OF3 backbone, fold with AF2-multimer =
  independent predictor). Reuse the committed Phase-1 steered sequences (the `--seqs-out` dump from
  `src/cfg_steer_predicted.py`, arms wt / L / random, k dumped). **If the steered-sequence dump is missing for the
  non-overlap complexes**, regenerate it CPU-only with `src/cfg_steer_predicted.py --source of3 --alphas 0,2 --K 64`
  restricted to the missing complexes (same command that produced Phase 1; deterministic at SEED).
- Fold set = the 106 predicted-steer complexes minus the 21 already in `iptm_predicted.csv` (≈85 new). Confirm the
  count and report it.

## Phase 1 — fold + metrics (EXACT crystal/Phase-2 pipeline, unchanged)
Fold the new complexes' steered sequences (wt / L / random, same k as Phase 2) with **AF2-multimer** (ColabFold,
templates off, 1 model, 3 recycles — identical to `iptm_predicted`), reusing `src/build_iptm_fastas.py` →
`src/parse_iptm.py` → `src/analyse_iptm.py`. Do **not** invent new metrics: per fold record ipTM, global pTM,
interface pLDDT, interface pAE; the pre-registered composite = z-mean(ipTM, −interface pAE, interface pLDDT);
global pTM = localization control. Idempotent: skip any (complex, arm, k) already folded; append; a crash re-runs
only the remainder. wt interface-ipTM median must be healthy (~0.84) as before — a positive control.

## Phase 2 — combined analysis
Concatenate the new folds with the committed 21 and recompute the paired, complex-clustered bootstrap
(NBOOT=5000) **L − random** (and L − naive if naive was folded; L − wt) for ipTM, interface pAE, interface pLDDT,
global pTM, and the composite, **mean-over-k primary + best-of-k secondary**, on the full n≈106. Restate H1
(composite(L)>random AND ipTM(L)>random, both CI>0), H2 (no collapse), H3 (|ΔpTM| ≪ Δcomposite). Anchor each number
to the crystal value (ipTM +0.235 AF2, composite +0.82) so the attenuation stays explicit. → overwrite
`results/iptm_predicted.csv` with the full-n result (keep the n=21 numbers in `FINDINGS_predicted_steer.md` for the
record so the before/after is auditable).

## Compute estimate (what you asked for)
- Current Phase 2 = **147 folds over 21 complexes** (≈7 folds/complex: wt + L×3k + random×3k).
- New: ≈85 complexes × ≈7 = **≈595 AF2-multimer folds**.
- AF2-multimer here (1 model, 3 recycles, templates off) ≈ **2–5 min GPU/fold** for 200–500-residue complexes
  (plus mmseqs2 MSA server latency, partly cached) → **≈30–50 GPU-hours total** (point estimate ~40).
- As a **job array** across G GPUs the wall-clock is ~40/G h (e.g. 10 GPUs → ~4 h; 20 → ~2 h). The known MSA-server
  `PENDING` hang needs the same 30-min per-fold timeout + idempotent resubmit used in Phase 2.
- **Bottom line: ~40 GPU-hours (range 30–60), a few hours wall-clock on a modest array.** CPU cost (re-steer if
  needed) is negligible (~1 CPU-hour).

## Deliverables (commit by name, two trailer lines, push)
Updated `results/iptm_predicted.csv` (full n≈106), updated `results/FINDINGS_predicted_steer.md` (n=21 → n≈106,
before/after, H1/H2/H3 verdicts, attenuation vs crystal), the per-fold raw. Message me the full-n **ipTM and
composite L − random** (+ n). Leave the paper untouched — I fold in the n update myself.

# Sherlock bundle — paste-to-run (judge matrix · ESM-IF1 steering · Boltz-2 · 60→120 · PyMOL hero)

`SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.
**CLAUDE.md rules apply:** pre-register every falsifier BEFORE seeing any number; never fabricate/simulate a
measurement; every reported number traces to a committed CSV with the exact command; run positive controls
before trusting a zero; `git add` by name (never `-A`); commit with the repo's two trailer lines; push.
Pre-registered already (frozen — do not edit): `results/PREREG_iptm.md`, `results/PREREG_cfg_steer.md`.

## Cross-cutting rules — apply to EVERY fold job
- **Sharding.** Split each fold job's complex list into N balanced shards, one GPU each (sbatch `--array`, or N
  `srun` sessions). This is the same total GPU-hours, just faster wall-clock — do not oversubscribe unevenly.
- **Checkpoint / idempotent skip.** Before folding each `(complex, arm, k)`, check whether its output model
  already exists on `$SCRATCH`; if so, **skip**. Append each fold's scores to its per-shard CSV **immediately**
  (never buffer a whole shard in memory). A crash then loses only the in-flight fold, and a re-run does **only
  the unrun remainder** — never re-fold an existing output.
- **Positive controls, every fold job.** wt-sanity gate (median wt interface ipTM in 0.6–0.9; if systematically
  low, the chain-order/MSA setup is wrong — STOP and fix) and a determinism spread (same seq, 2–3 seeds) kept
  visible next to every effect.
- **Budget.** ~1,200 new folds total (Boltz-2 420 + AF2 batch2 420 + AF2 ESM-IF1-steered 360) ≈ ~100 GPU-hours
  ≈ ~1 day on 4 GPUs sharded. Judges + steering are CPU (hours).

## Phase 0 — sync + regenerate caches
```
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
git pull --no-edit origin main
# regenerate the git-ignored pq caches (leverage_pq_skempi*.csv) as in the CFG handoff, then SET-A sequences:
python3 src/cfg_steer.py --alphas 0,2 --K 64 --dump-seqs \
        --seqs-out results/cfg_steer_seqs.csv --out results/_cfg_forfold.csv
```
`cfg_steer_seqs.csv` = SET-A (ProteinMPNN-steered) wt/L/random arms — the input for Phases 1 and 3.

## Phase 1 — judge matrix on SET-A (CPU, ~free; anti-circular)
Score SET-A's α=2 steered interface residues with the leverage of each **non-self** judge — **ESM-IF1, PiFold,
MIF** (NOT ProteinMPNN: it's the steered model → circular). Reuse the committed scorers (`src/leverage_esmif.py`,
`src/models/ftax_pifold.py`, and the MIF scorer used for `leverage_mif.csv`). For each judge report the **paired
L-arm vs random-arm** judge-leverage (complex-clustered bootstrap 95% CI, P>0), mirroring the standing ESM-IF1
result (α: −0.20→+0.27; L−random +0.77). → `results/cfg_judge_matrix.csv` (cols: `steered_model, judge, arm,
alpha, mean_leverage, …` + a paired L−random summary block). **Falsifier (report verbatim if it fires):** a
judge whose paired L−random ≤ 0 does not corroborate.

## Phase 2 — ESM-IF1 as a SECOND steered model (SET-B)
**Write `results/PREREG_esmif_steer.md` first** (H1: steering *frozen ESM-IF1* by `+α·L` raises the
**ProteinMPNN**-leverage of the sampled interface residues, paired vs matched-random; H2: native recovery
preserved; H3 localization on the fold; falsifier: paired L−random ≤ 0). Then:
- Steer frozen ESM-IF1 by `+α·L` at the interface positions (same α grid {0,2}, K, and committed interface set
  as `cfg_steer.py`; adapt it to the ESM-IF1 autoregressive decoder, or add `src/cfg_steer_esmif.py`). Dump SET-B
  → `results/cfg_steer_esmif_seqs.csv` (wt/L/random, k=0..2), on the **batch-1 60 complexes**.
- **Judge** SET-B by ProteinMPNN + PiFold + MIF (NOT ESM-IF1) → append to `cfg_judge_matrix.csv`
  (`steered_model=ESM-IF1`).
- **Fold** SET-B with the SAME AF2-multimer pipeline as the original ipTM run: **wt is the identical crystal
  sequence already folded in that run — REUSE those wt models, do not re-fold.** Fold only L(k0..2)+random(k0..2)
  = **360 new folds**. Parse with `src/parse_iptm.py` (crystal interface set) → `results/iptm_steer_esmif.csv`;
  analyse with `src/analyse_iptm.py` → `results/iptm_summary_esmif.csv`.

## Phase 3 — Boltz-2 cross-folder on SET-A batch1 (decorrelated structure predictor)
**Write `results/PREREG_boltz.md` first** (H1: paired composite(L) > composite(random) AND ipTM(L) >
ipTM(random) under Boltz-2; localization: global pTM shifts far less; falsifier: composite or ipTM paired
L−random ≤ 0 → the ipTM gain is AF2-specific, reported verbatim). Then:
- Install/verify **Boltz-2** (permissive open license). **wt-sanity gate FIRST** on ~5 wt complexes; if median
  interface ipTM is not ~0.6–0.9, STOP and fix chain order / MSA before trusting any contrast.
- Fold SET-A batch1 (60 × 7 = **420 folds**: wt + L k0..2 + random k0..2) with Boltz-2. Record per fold:
  `complex_id, direction, k, iptm, ptm, interface_pae, interface_plddt` (Boltz-2's native ipTM; `interface_pae`
  on the SAME crystal interface set `parse_iptm.py` uses). → `results/iptm_steer_boltz.csv`; analyse with
  `analyse_iptm.py` → `results/iptm_summary_boltz.csv` (the pre-registered composite + all four metrics + the
  localization control). If Boltz-2 setup fights us, fall back to **Boltz-1** and note it in FINDINGS.

## Phase 4 — 60 → 120 (tighter CIs on the primary AF2 headline)
**Write `results/PREREG_iptm_batch2.md` and `results/iptm_subset_batch2.txt` BEFORE folding:** SEED-shuffle the
eligible SKEMPI complexes NOT in `results/iptm_subset.txt` (all three arms present in `cfg_steer_seqs.csv`,
≤~600 residues), take the next 60, freeze the list. Then dump their steered sequences (`cfg_steer.py
--dump-seqs` restricted to batch2), fold wt+L+random (**420 folds**) with the SAME AF2 pipeline, parse, and
**append to the committed 60** → `results/iptm_steer_120.csv` (original committed rows reused, NOT re-folded).
Re-run `analyse_iptm.py` on the 120 → `results/iptm_summary_120.csv`.

## Phase 5 — PyMOL publication cartoon of the hero complex
For the pre-registered median-effect complex **1GL0_E_I** (`results/hero_pdbs/*.pdb` + `1GL0_E_I_meta.csv`),
render a PyMOL cartoon for wt / L / random: cartoon + the interface shown as a surface/sticks patch, **color by
B-factor (pLDDT)**, ray-traced, with a **consistent orientation across the three** (align on the receptor
chain). Annotate each with its ipTM + interface_plddt + **interface_pae** (from meta.csv). →
`results/figures/fig_hero_pymol.png` (+ the script `src/render_hero_pymol.pml`). Keep the biotite `fig_hero` as
fallback; this is the publication version.

## Deliverables (commit by name; two trailer lines; push)
`cfg_judge_matrix.csv`; `cfg_steer_esmif_seqs.csv`, `iptm_steer_esmif.csv`, `iptm_summary_esmif.csv`;
`iptm_steer_boltz.csv`, `iptm_summary_boltz.csv`; `iptm_subset_batch2.txt`, `iptm_steer_120.csv`,
`iptm_summary_120.csv`; `results/figures/fig_hero_pymol.png` (+ `.pml`); the new `PREREG_*.md` +
`FINDINGS_*.md`; any new `src/` drivers. Message me the paired **L−random headline** for each phase.

## What each phase buys (and the honest scope)
- Judge matrix + ESM-IF1 steering = **within-modality robustness** ("not one model's quirk; the reverse steering
  also works").
- Boltz-2 = the **cross-modality / decorrelated** confirmation (guards the "two inverse-folding models share a
  blind spot" risk).
- 60→120 = tighter CIs on the primary AF2 headline.
- A null in any phase **bounds** the corresponding claim; it does not erase the standing AF2 + ESM-IF1 result.
Report every falsifier verbatim; keep determinism spreads visible next to every effect.

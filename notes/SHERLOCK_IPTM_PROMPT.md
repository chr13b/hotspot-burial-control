# Sherlock task — fold the CFG-steered sequences, read AF2-multimer ipTM (paste-to-run)

**Pre-registered:** `results/PREREG_iptm.md` (frozen — do not edit). Does steering a frozen ProteinMPNN by
`+α·L` at interface positions produce sequences that an **independent structure predictor (AF2-multimer)** is
more confident assemble (higher interface ipTM) than a matched-magnitude **random** direction? Confirmation of
the CFG-steering result, not a load-bearing pillar. `SEED=20260803`. One model (AF2-multimer); OpenFold3 kept
optional. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.

## Phase 0 — sync + GPU node
```bash
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
git pull --no-edit origin main               # HEAD >= this commit; LFS filter already passthrough
# request the SAME GPU node you used for Exp D / OF3 / AF2-multimer folding
srun --partition gpu --gres=gpu:1 -c 8 --mem=32G -t 12:00:00 --pty bash
cd /scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control
```
Ensure the two pq caches exist (regenerate as in the CFG handoff if not): `leverage_pq_skempi.csv`,
`leverage_pq_skempi_esmif.csv`; the committed `leverage_skempi_positions.csv` is in git.

## Phase 1 — regenerate the steered sequences (deterministic)
```bash
python3 src/cfg_steer.py --alphas 0,2 --K 64 --dump-seqs \
        --seqs-out results/cfg_steer_seqs.csv --out results/_cfg_forfold.csv
```
`cfg_steer_seqs.csv` carries, per complex: `direction` ∈ {wt, L, random}, `alpha` (2 for steered, NaN for wt),
`k` (0..2 for steered; −1 for wt), `chains` = `"A:SEQ|B:SEQ|…"` (wt background with the interface positions
replaced by the α=2 steered ProteinMPNN samples). This is the exact set to fold.

## Phase 2 — pre-register the subset, then fold with AF2-multimer
- **Fix the subset BEFORE folding** (rule 1): take a `SEED`-shuffled first **60 complexes** that have all three
  conditions present and ≤ ~600 residues (to fit the GPU budget); write the chosen `complex_id` list to
  `results/iptm_subset.txt` and do not change it after seeing any ipTM.
- Fold each sequence with the **same AF2-multimer pipeline Exp D used** (the one that produced
  `expD_backbone_manifest.csv`'s `iptm`/`interface_plddt`). Build the multimer input from the `chains` field
  (each `C:SEQ` is one chain; preserve chain order). Fold: `wt` (k=−1), `L` (k=0..2), `random` (k=0..2) per
  complex — ~60 × 7 = ~420 folds; at a few min/fold on a V100 this is ~0.5–1.5 GPU-days.
- Record per fold: `complex_id, direction, k, iptm, ptm, interface_pae, interface_plddt` → `results/iptm_steer.csv`.

## Phase 3 — positive controls (rule 6) then the test
- **wt sanity:** median wt interface ipTM must be in the normal range (~0.6–0.9). If wt ipTMs are systematically
  low, the chain order / MSA setup is wrong — STOP and fix before trusting any contrast.
- **Determinism/spread:** fold one wt sequence under 2–3 AF2 seeds; report the ipTM spread so an effect smaller
  than it is not over-read.
- **The test (paired, per complex) — MULTI-METRIC, do not rely on ipTM alone:** aggregate k=0..2 (mean; best-of-k
  secondary) per condition, then report the paired **L − random** contrast (complex-clustered bootstrap 95% CI,
  `P(>0)`) for **all four** metrics:
  - **ipTM** (higher better), **interface pAE** (lower better → sign flips), **interface pLDDT** (higher better),
    and **global pTM** as the **localization control** (should move far less than the interface metrics).
  - **Composite (robust to per-metric noise — the pre-registered rule):** z-score each interface metric across
    ALL folds, sign-orient so higher = better (ipTM, −interface pAE, interface pLDDT), average into one composite
    per fold. **H1 passes iff paired composite(L) > composite(random) AND ipTM(L) > ipTM(random)** (both CIs,
    P>0). A single interface metric flipping while the composite + ipTM hold is expected noise, NOT a refutation;
    a single metric agreeing is NOT sufficient. **H3 (localization):** global pTM shift ≪ the composite shift.
  - Also report **H2:** paired `interface-metric(L) − (wt)` (should not be strongly negative).
  - Interface pAE = mean predicted aligned error over TCR↔partner interface residue *pairs* (use the interface
    residue set; if the folder outputs a full PAE matrix, average the cross-interface block both ways).

## Phase 4 — deliverables
Commit **only** `results/iptm_steer.csv`, `results/iptm_subset.txt`, `results/FINDINGS_iptm.md`, and any small
fold-driver script you add (`src/fold_iptm.py`). `git add` by name (not `-A`); commit with the repo's two
trailer lines; push; message me the paired L−random ipTM.

`FINDINGS_iptm.md`: the paired L−random contrast for **all four metrics** (ipTM, interface pAE, interface pLDDT,
global pTM) with CI + P>0, whether the three interface metrics agree (H1), the localization check (H3), L−wt
(H2), the wt-sanity + determinism controls, `n` complexes, the subset definition, `SEED`, exact commands.

## Guardrails
- **Report the falsifier verbatim if it fires** ("ipTM(L) ≤ ipTM(random) paired → steering does not raise the
  structure predictor's interface confidence"). An honest null bounds the claim; it does not refute the
  ESM-IF1-leverage steering result. Do not massage the subset or k-aggregation after seeing numbers.
- **Headline = the paired L vs random contrast** (specificity), not L vs wt alone.
- ipTM is a model proxy, metric-noisy — keep the determinism spread visible next to the effect.

## Optional (door open) — OpenFold3 second predictor
If the AF2 result is worth hardening, repeat Phases 2–3 under **OpenFold3** on the same subset + sequences,
writing `results/iptm_steer_of3.csv`. ipTM is a single-predictor metric, so this is confirmation only — not
required for the headline.

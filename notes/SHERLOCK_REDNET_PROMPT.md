# Sherlock — RedNet head-to-head (frozen +α·L vs retrained decoder) [PREPARED; run on decision]

`SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`.
**Pre-register `results/PREREG_rednet.md` FIRST.** CLAUDE.md rules apply (pre-register; never fabricate; every
number → committed CSV; positive controls; `git add` by name; two trailer lines; push). Sharding + idempotent
per-fold checkpoint as in the main bundle.

## The claim — honest framing (NOT "we beat RedNet")
Does a **frozen, off-the-shelf `+α·L` tilt** recover **most of what RedNet buys by *retraining*** a decoder
around the same contrastive direction? RedNet is retrained (a disclosed advantage); the interesting, publishable
result is *"no retraining is needed for most of the benefit."* If RedNet ≫ frozen, we report that honestly.

## Phase 0 — get RedNet
Clone `zw2x/rednet_public` (verified in §8: α-tilt in `sampling_utils.py`, apo contrast in
`infer_pipeline.py`); install; obtain weights; reproduce one of their examples as a sanity check.

## Phase 1 — shared sequence sets (SAME complexes, SAME interface positions, SAME K)
On the ipTM 60-complex set (or a pre-registered subset), per complex generate:
- **wt** (crystal) and **random** (matched-magnitude) — already committed (`cfg_steer_seqs.csv`).
- **ours** — frozen ProteinMPNN `+α·L` — already committed.
- **RedNet** — run their pipeline on the identical targets + interface positions.
- **naive-guidance** — frozen ProteinMPNN tilted toward its own **confidence** (matched magnitude): shows the
  gain needs the *binding* direction, not any tilt (complements the random control).

## Phase 2 — judge (CPU, ~free)
Score every arm with the independent judge matrix {ESM-IF1, ProteinMPNN, MIF} leverage (non-self only).

## Phase 3 — fold (GPU, sharded + checkpointed)
Fold **only the NEW arms (RedNet, naive-guidance)** with AF2-multimer — **REUSE the committed wt / ours / random
AF2 folds** from `results/iptm_steer.csv` (do NOT re-fold them: that is the compute saving, and it keeps "ours"
on the exact same folds as the standing result). Record **all four metrics** — ipTM, global pTM, interface pAE,
interface pLDDT — via `src/parse_iptm.py` on the committed **crystal** interface set (unbiased), and analyse with
`src/analyse_iptm.py` so the **pre-registered composite** (z-mean of the three interface metrics) is computed
identically to the main result (coherent with `iptm_summary*.csv`). Optionally also fold with Boltz-2 for a
cross-folder check — reporting Boltz as a *within-folder* paired effect (not a magnitude match to AF2).

## Phase 4 — compare (paired, per complex)
Report paired contrasts on judges + ipTM: **ours − random** (specificity, our headline), **ours − naive**
(needs the binding direction), and **ours vs RedNet** — the fraction of RedNet's gain that the frozen tilt
recovers. Fairness: same complexes/positions/judges/folder/budget; RedNet is retrained (disclosed). →
`results/rednet_compare.csv`, `PREREG_rednet.md`, `FINDINGS_rednet.md`.

## Falsifiers / honesty
Report verbatim. RedNet ≫ frozen → retraining helps (state the gap). naive ≈ ours → our binding-direction
specificity is weaker than claimed (disclose). This is a *positioning* comparison, not a leaderboard; a null
bounds the "no-retraining-needed" claim, it does not touch the standing steering result.

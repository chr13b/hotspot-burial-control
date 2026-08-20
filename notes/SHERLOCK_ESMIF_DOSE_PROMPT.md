# Sherlock kickoff — ESM-IF1 backbone-error dose law (#9)

Paste the block below into a fresh agent on Sherlock (after `git pull` and a GPU allocation).
This closes the one empirical gap in the dose-law class claim: we show it under ProteinMPNN; this
shows it under a **second, architecturally different** inverse-folding model (ESM-IF1, a
GVP-transformer). On this CPU box the run was underpowered (20 complexes → CPI −0.0097, and the σ=0
Spearman −0.16 *matched* the committed ESM-IF1 subset, proving the scorer is correct and the problem
is power, not a bug). A GPU fixes the power problem.

---

## PASTE FROM HERE

You are continuing the `hotspot-burial-control` project on Sherlock. Read `CLAUDE.md` and
`BRIEF.md` first — the pre-registration and false-positive rules are binding, especially: **never
fabricate, extrapolate or simulate a measurement; if it cannot be run, say exactly why and stop.**

**Goal.** Replicate the backbone-error dose law under ESM-IF1. The pre-registered class claim (paper
§4) is: *CPI(leverage | geometry) survives sub-Ångström backbone jitter (σ ≤ 0.5 Å) and collapses as
the jitter approaches ~1 Å, because the sensitivity is to the backbone the derivative is read from,
not to the network.* ProteinMPNN already shows this (`results/leverage_noise_ladder_075full.csv` and
the committed ladder). Your job is the ESM-IF1 curve.

**Freeze this pre-registration BEFORE running anything (write it into the commit message):**
- **Prediction:** at σ=0, ESM-IF1 CPI(L|geometry) is positive and clears the placebo floor
  (+0.0007); it holds at σ=0.5 Å and collapses toward 0 by σ=1.0 Å — the same shape as ProteinMPNN.
- **Sanity anchor (already verified):** ESM-IF1 σ=0 Spearman(L, ddG) must land near the committed
  full-run value **−0.176** (`results/leverage_esmif.csv`). If it does, the scorer is sound.
- **Falsifier:** if, on a **powered** sample (≥100 complexes), σ=0 does NOT clear the floor, the
  "second model family" empirical is a null — **report it as a null**, keep the mechanism-first claim,
  and do not soften the language to rescue it.

**The one code change required — device.** The script is CPU-hardcoded in three places; a naïve
"model.to('cuda')" will crash with a device-mismatch because the batch converter stays on CPU. Thread
a single `device` through all of:
1. `src/leverage_noise_ladder_esmif.py` → `fe.load_esmif(device=...)` (currently `"cpu"`).
2. `src/leverage_esmif.py:60` → `fe.esmif_conditional_logprobs(..., device=...)` inside `esmif_lp`
   (currently hard-coded `"cpu"` — **this is the load-bearing one**; the noise ladder calls `esmif_lp`).
3. Confirm `esmif_conditional_logprobs` and `whole_complex_logprobs` in `src/models/ftax_esmif.py`
   already take `device=` (they do) and that the coord/token tensors are built on that device.
Add a `--device cuda` arg (default `cpu`) rather than editing constants, so the change is reviewable
and the CPU path still reproduces. Run the existing CPU path once on 3 complexes first to prove you
didn't break it.

**Data to stage on Sherlock** (do NOT commit any of it):
- `results/leverage_skempi_mutations.csv` — committed, already in the repo.
- `~/ftax/data/PDBs/*.pdb` and `~/ftax/data/skempi_v2.csv` — the SKEMPI structures + table (same as
  the ProteinMPNN runs; `DATA = ~/ftax/data` in `src/leverage_decomposition.py:65`).
- The ESM-IF1 checkpoint (`esm_if1_gvp4_t16_142M_UR50`) — `fe.load_esmif` downloads/uses `DEFAULT_CKPT`.

**Powered pilot first (do NOT spend the full budget blind):**
```
python3 src/leverage_noise_ladder_esmif.py --sigmas 0.0 --limit 100 --device cuda \
    --out results/_pilot_esmif_sigma0.csv
```
Check the printed line: `Spearman(L,ddG)` should be ≈ −0.18 and `CPI(L|geom)` should be positive with
P(>0) high. If Spearman matches but CPI is noisy, raise `--limit` (more complexes = more power), not
your expectations. If Spearman is far from −0.18, the device port is wrong — stop and debug.

**Full run once the pilot clears:**
```
python3 src/leverage_noise_ladder_esmif.py --sigmas 0.0,0.25,0.5,0.75,1.0 --limit 200 --device cuda \
    --out results/leverage_noise_ladder_esmif.csv
```
Budget: ESM-IF1 is ~15× ProteinMPNN per complex; 200 complexes × 5 σ is the expensive part. If wall
time is tight, drop to `--sigmas 0.0,0.5,1.0` (the three points that carry the claim) at the highest
`--limit` you can afford. `--limit` selects the first-N complexes deterministically (sorted ids), so
the pilot's complexes are a subset of the full run — consistent, not cherry-picked.

**On success:**
1. Commit `results/leverage_noise_ladder_esmif.csv` with the exact command + seed (SEED=20260803).
2. Update `results/FINDINGS_conservation.md`'s "beyond a second IF model family (ESM-IF1)" line and
   paper §4: replace the "GPU-scale follow-up, scorer validated" signpost with the actual curve
   (σ → CPI, CI, Spearman). State the shape plainly (survives ≤0.5, collapses ~1.0) or report the null.
3. Update memory `verdict-state.md`: ESM-IF1 dose law is now empirical, not signposted.
4. `git push` to main.

**Do not** touch Phase 2 (MultiFlow) or AF-Multimer — those are separate. This is one curve.

## PASTE TO HERE

---

**Why this is worth the GPU-hours (for Chris, not the agent):** it converts the dose-law class claim
from "mechanism + one model" to "mechanism + two architecturally distinct models," which is the
single cheapest way to kill the reviewer's easiest attack — *"this is a ProteinMPNN quirk."* See
`notes/PAPER_DRAFT.md` §4 and the 0.8→0.9 analysis.

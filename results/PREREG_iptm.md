# PRE-REGISTRATION — does steering by L improve an independent *structure predictor*'s interface confidence?

**Committed BEFORE any ipTM number (CLAUDE.md rule 1).** `SEED=20260803`. This extends the CFG-steering result
(`FINDINGS_cfg_steer.md`): steering a frozen ProteinMPNN by `+α·L` at interface positions raised the ESM-IF1
*leverage* of the sampled residues. Here we ask a harder, physics-adjacent question: do the steered sequences
also fold — under an independent **structure predictor (AF2-multimer)** — to complexes the predictor is *more
confident* assemble? This is confirmation, not a load-bearing pillar (the ESM-IF1 validation already stands).

## Design
For a pre-registered subset of SKEMPI complexes (target ~50–80, to fit ~0.5–1.5 GPU-days on one model), take the
sequences produced at **α = 2** (the strongest steering, where the ESM-IF1 leverage gain was largest, +0.77):
- **wt** — the crystal sequence (reference);
- **L-steered** — wt background with the interface positions replaced by the `+α·L`-steered ProteinMPNN samples
  (`direction=L`, k = 0..2), from `src/cfg_steer.py --dump-seqs`;
- **random-steered** — the matched-magnitude random-direction control (`direction=random`, k = 0..2).
Fold each with **AF2-multimer** (one model; reports ipTM natively; the pipeline the project's Exp D used) and
record **interface ipTM** (primary) and **interface pAE** (secondary), per sequence.

## Hypotheses
- **H1 (specificity — the load-bearing test).** Paired per complex, the **interface** metrics favour L over the
  matched-magnitude random direction — **ipTM(L) > ipTM(random)**, and *consistently* **interface pAE(L) <
  interface pAE(random)** and **interface pLDDT(L) > interface pLDDT(random)**. Corroboration across all three is
  the claim; a lift in one metric alone is not.
- **H2 (no collapse).** The interface metrics for L-steered are **not below** wt by more than a small margin —
  steering does not destroy a foldable interface (consistent with the CFG result, where native recovery *rose*).
- **H3 (localization).** The improvement is **interface-specific**: global pTM shifts far less than the interface
  metrics (mirroring the CFG localization control). If global pTM moves as much as ipTM, the effect is not
  interface-localized — disclose it.
- **Falsifier.** If ipTM(L) ≤ ipTM(random) paired, or the interface metrics **disagree** in direction, the
  steering benefit does **not** cleanly transfer to a structure predictor — reported verbatim. A null bounds the
  claim; it does not refute the ESM-IF1-leverage steering result.

## Positive controls (rule 6, before trusting any comparison)
1. **wt sanity:** the wt folds' interface ipTM must be in the normal range for real complexes (~0.6–0.9); if wt
   ipTM is systematically low, the folding setup (chain order / MSA) is wrong — stop and fix.
2. **Determinism:** the same sequence folded twice (or the AF2 seed sweep) gives ipTM within the model's own
   spread; report that spread so an effect smaller than it is not over-read.
3. **Pairing:** every L-vs-random contrast is *within the same complex* (paired), and only complexes with all
   three conditions folded enter the test.

## Metrics — a MULTI-METRIC hedge (do not rely on ipTM alone), with a localization control
ipTM is a model proxy and metric-noisy, so we pre-register **four complementary readouts** and require the effect
to be *consistent* across the interface ones — an effect that shows in only one metric is not trusted:
- **ipTM** (interface pTM) — primary interface-assembly confidence. *Higher = better.*
- **interface pAE** (mean predicted aligned error across the TCR↔partner interface residue pairs) — a *different*
  and largely complementary failure mode from pTM. *Lower = better.*
- **interface pLDDT** (mean pLDDT over interface residues) — local per-residue confidence. *Higher = better.*
- **global pTM** — the **localization control**: it should move *much less* than the interface metrics if the
  effect is genuinely interface-specific (mirroring the CFG result's flat non-interface recovery). A large global
  pTM shift would mean the fold changed globally, not at the interface — a red flag to disclose.

For each: report the paired **L − random** contrast (complex-clustered bootstrap 95% CI, `P(>0)`), the **L − wt**
difference, `n` complexes, `SEED`, and folded-sequence provenance. Aggregate k=0..2 per condition (mean; best-of-k
secondary). **The headline is corroboration across ipTM AND interface pAE AND interface pLDDT** with global pTM
flat; report each even if some disagree (an inconsistent signal is itself informative and must not be hidden).

## Model
**AF2-multimer** (primary, one model). **OpenFold3 kept as an OPTIONAL second predictor** (door open): if the
AF2 result is interesting, repeat the identical pipeline under OF3 to show the confirmation is not AF2-specific —
but ipTM is inherently a single-predictor metric, so one model is sufficient for the headline.

## Honest scope
ipTM is a *model proxy* for "does this assemble," and is metric-noisy; this is a structure-predictor
confirmation of the steering direction, not an experimental binding measurement. The steering result's primary
validation remains the anti-circular ESM-IF1 leverage; a positive ipTM result strengthens it, a null bounds it.

Output: `results/iptm_steer.csv` (+ `FINDINGS_iptm.md`). Sequences from `results/cfg_steer_seqs.csv`
(`cfg_steer.py --dump-seqs`). Fold script + handoff: `notes/SHERLOCK_IPTM_PROMPT.md`.

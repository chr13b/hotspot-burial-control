# In-pocket options & scooping/comparison notes (not committed to the current run)

*Captured 2026-09-05. These are options we deliberately keep available but are NOT in the locked
Sherlock bundle (PiFold+ESM-IF1+MIF judges · ESM-IF1 steer · Boltz-2 · 60→120).*

## Steered models beyond ProteinMPNN + ESM-IF1 (the locked pair)
Criteria for a good *steer* target: logit access at interface positions; per-position/AR factorisation so a
`+α·L` tilt+sample is well-defined; `L` computable (partner-ablation pass); architecturally distinct from
models already steered; frozen public weights; CPU-feasible.
- **MIF-steered** (masked — bias masked-position logits) and **PiFold-steered** (one-shot — bias logits then
  sample): both are non-autoregressive, so the "guidance" semantics are weaker than the AR pair, but they'd show
  steering generalises even to non-AR paradigms. In-pocket third; symmetric combos are fine
  (PiFold-steered→MIF-judged and MIF-steered→PiFold-judged both non-circular). Not needed now.
- **LigandMPNN**: rejected — MPNN-family, correlated with the model we already steer.

## Judges (free — CPU leverage on existing sequences; run the full NON-SELF matrix)
Locked: ESM-IF1, ProteinMPNN, PiFold. **Add MIF** (leverage already implemented → free) for a 4th
architecture. Possible later: Frame2seq. Rule: judge ≠ steered model (non-circular).

## Optional extra folds (GPU — only if budget is plentiful)
Boltz-2 on batch2; Boltz-2 on SET-B (ESM-IF1-steered). Not required — each claim already has its cheapest
sufficient fold.

## Scooping check (2026-09-05, light literature pass) — NOT scooped
- **BA-Cycle** (arXiv 2410.09543) = the `L` operator; already cited/credited.
- **RedNet** (bioRxiv 2026.05.09.722041) = the design-time `+α·(logit_bound − logit_apo)` tilt; already
  cited/credited (they retrain a decoder; we steer frozen + do the decomposition/control).
- No paper has our specific contribution: the confidence(=φ(P) diagonal) vs leverage(=mixed derivative)
  **decomposition + non-identifiability no-go**, the **first beyond-geometry(+conservation) control** on an
  inverse-folding binding signal, and the **feature-class law**. The field around us is *property-driven /
  guidance* design (optimise a property), not the *diagnostic* we build.

## Citations to VERIFY (verify-references pass) and likely add to §8's "conditioning-aware methods" list
Do NOT add from search snippets — fetch + verify first.
- **EnerBridge-DPO** (arXiv 2506.09496) — energy-guided inverse folding (Markov bridge + DPO).
- **Property-driven inverse folding / MoMPNN** (arXiv 2603.06748; OpenReview m826DekCpp) — guidance on ProteinMPNN.
- **Plug-and-Play guidance for discrete diffusion** (arXiv 2606.06303) — general discrete-diffusion guidance.
These *presuppose* the conditioning-set problem we *measure* — same framing as our existing DeSAE/UMA-Inverse cites.

## Comparison stance (do we benchmark vs another method?)
We already compare `L` to: every scalar of `P` (confidence/KL — at the placebo floor), cheap geometry, masked
conservation, and a *supervised* geometry+substitution baseline (L beats it zero-shot). On the shared L→ΔΔG task
we are comparable to **BA-Cycle** (cited). A head-to-head vs a supervised SOTA ΔΔG predictor (FoldX/Rosetta/GNN)
is a *different* claim (predictive accuracy) than ours (confidence-blind / mixed-derivative-sees-it / beyond
geometry) — not needed for the thesis; a §8 sentence positioning L (zero-shot, comparable to BA-Cycle,
orthogonal contribution) suffices. No new benchmark scheduled.

## Python package (impact play — POST-submission, not for the 25 Sept deadline)
Deliver a small pip package: model-agnostic CORE (`leverage`: partner-ablation mixed derivative, hotspot
ranking, the `+α·L` steering knob) + **validated adapters** for the models we benchmarked (ProteinMPNN,
ESM-IF1) behind a `Model` protocol (score p(seq|structure) for complex AND monomer). Model-agnostic interface,
but only *recommend* the benchmarked adapters; others work via the protocol without us claiming they're tested.
Scope: for the deadline ship **reproducible code + Zenodo** (anonymised for double-blind); build the polished
PyPI package for camera-ready / post-acceptance so it doesn't eat the 20-day runway.

## Comparison plan — fair head-to-heads (2026-09-06)
**We are ZERO-SHOT / unsupervised for binding:** `L` is read off a model never trained on binding labels; no
ΔΔG labels enter its computation. (Labels are used only to *evaluate*.) So the fairest comparators are other
zero-shot / physics scores, with any *supervised* comparator's training-data advantage disclosed.

**Detection claim** ("scalars blind; the mixed derivative sees binding beyond geometry+conservation+one-pass"):
- Already in-paper: `L` (zero-shot) vs a **supervised geometry+substitution baseline** *trained on the binding
  labels* — `L` adds **+0.030 AUROC** on top and reaches **0.647** alone (`leverage_effect_size.csv`). That is
  the "supervised baseline we beat zero-shot."
- Optional SOTA add (reviewer-anticipation, moderate effort): a **physics ΔΔG tool** (FoldX or Rosetta ddG) on
  the SAME SKEMPI fixture/mutation set, same metric (Spearman, hotspot-AUROC). Fair (non-learned). Report
  BA-Cycle's published SKEMPI correlation as the same-`L`-operator reference (already cited). Not load-bearing.

**Steering / actionable claim** — the higher-value comparison, since we *intervene*:
- **Frozen `+α·L` tilt vs RedNet (retrained decoder)**, same complexes, same interface positions, same
  independent judges (ESM-IF1/PiFold/MIF leverage + AF2/Boltz-2 ipTM), same budget. Framing: *"a frozen,
  off-the-shelf tilt recovers most of what RedNet buys by retraining"* — if frozen ≈ RedNet that is a strong
  "no retraining needed"; if RedNet > frozen, disclose honestly (retraining helps, direction alone is most of
  it). RedNet code is public (`zw2x/rednet_public`, already verified in §8) → **feasible**. This is the
  comparison that credibly positions us against another *method*.
- **Non-SOTA / naive-guidance baseline:** a tilt toward the model's own *confidence* (or BLOSUM-favorable),
  same judges — shows the gain needs the *binding* direction, not any guidance (complements the random control).
- Fairness rules for all: same fixture/complexes, same positions, same metric, same judges; disclose
  training-data leakage for supervised comparators.

**Recommendation:** the RedNet frozen-vs-retrained head-to-head is the one worth doing (feasible + directly on
our actionable claim) — a candidate to add to the Sherlock bundle *or* keep rebuttal-ready; the FoldX detection
baseline is optional reviewer-anticipation; a full SOTA-ΔΔG accuracy leaderboard is a different claim and is
skipped.

**UPDATE 2026-09-07 — RedNet head-to-head BLOCKED → rescued.** RedNet's pipeline needs private packages
(`faust`, `atomtools`; not public, not in the author's repos) and its Zenodo weights (record 20113403) are
access-restricted, so the true head-to-head cannot be run and we will not fake it (rule 3). **Two rescues, both
accessible and arguably stronger:** (1) a **code-level positioning result** — RedNet's released decode is
`logits=(1+α)·logP_complex − α·logP_contrast` = our `+α·L` (contrast = unbound monomer; scorer `cd_ll=ll−ub_ll`),
so a published *retrained* method converges on our exact direction (edge = retrained decoder + β-mask); (2) a
**naive-guidance specificity arm** — frozen ProteinMPNN tilted toward its own confidence, matched magnitude, run
via `notes/SHERLOCK_REDNET_PROMPT.md` (rescued). Do NOT chase a different *retrained* SOTA (same access risk, no
added claim). For a runnable "vs SOTA" on the *detection* claim, **FoldX/Rosetta** (physics, downloadable, no
weights) is the accessible comparator — optional, different claim.


# Sherlock — RESCUED "vs another method": naive-guidance specificity + RedNet code-positioning

**The RedNet head-to-head is BLOCKED and we are NOT faking it (CLAUDE.md rule 3).** RedNet's pipeline needs
private companion packages (`faust`, `atomtools` — imported at module top in `cli/infer_pipeline.py` /
`cli/make_select_data.py`, not in `pyproject.toml`, not public, not among the author's 6 public repos), and its
Zenodo weights (record 20113403) are access-restricted. So we cannot run their sampler or format our complexes
into their pipeline. **We rescue the comparison two ways, both honest and fully accessible.**

`SEED=20260803`. Repo: `/scratch/users/cbertsch/project/factorization-tax/hotspot-burial-control`. Pre-register
first; every number → committed CSV; positive controls; `git add` by name; two trailer lines; push.

## The gift the recon produced (no run needed — a code-level positioning result)
RedNet's released contrastive decode is, in code, `logits = (1+α)·logP_complex − α·logP_contrast`, i.e. exactly
our `+α·L` (`= logP_complex + α·(logP_complex − logP_contrast)`); `contrast` = the **unbound monomer** for
`run_hdimer` (off-target for `run_sel`), and the scorer computes `cd_ll = ll − ub_ll` (bound minus unbound).
**The field's retrained binder-design method converges on our exact direction; its only edge is a retrained
decoder + a β plausibility mask.** Write this up (see Phase 3) — do NOT claim a head-to-head measurement.

## Phase 1 — the naive-guidance specificity arm (frozen ProteinMPNN, matched magnitude)
Pre-register `results/PREREG_naive.md` FIRST (H1: a matched non-binding tilt does NOT raise binding-favorability
the way `L` does; falsifier: naive − random ≥ L − random under an anti-circular judge → the specificity claim is
bounded, reported verbatim). Then generate the **naive** arm on the ipTM 60-complex set, same committed interface
set / α grid {0,2} / K as `cfg_steer.py`: bias the interface logits toward the model's **own confident residues**
— its complex-conditioned logit direction (amplify what it already prefers) — scaled to the **same per-position
magnitude** as the `+α·L` tilt (so it differs from L only in *direction*, like the random arm but pointed along
confidence rather than at random). → `results/cfg_steer_naive_seqs.csv`. (This is a third arm alongside the
committed wt / L / random.)

## Phase 2 — judge the naive arm (CPU, ~free — the core result)
Score the naive arm's interface residues with the **non-ProteinMPNN** judges {ESM-IF1, MIF, PiFold} leverage
(ProteinMPNN is the steered model → circular), reusing `judge_dumped.py` + `cfg_judge_matrix.py`. Report paired
**naive − random** and **L − naive** per complex (complex-clustered bootstrap 95% CI). Expected: naive ≈ random
≪ L (the confidence tilt does not carry binding), which is the specificity result. → append to a
`results/cfg_naive_judge.csv`.

## Phase 3 — the RedNet code-positioning note (no run)
Write `results/FINDINGS_rednet.md`: the mechanism-identity (exact formula above, with the `sampling_utils.py` /
`infer_pipeline.py` line refs the recon found), RedNet's genuine additions (retrained decoder + β-mask), and the
**blocker stated plainly** (private `faust`/`atomtools`, access-restricted Zenodo 20113403) — so the record shows
the head-to-head is *measurement-blocked*, positioned at the code level, never faked. Add the blocker + the
mechanism-identity to memory (`bundle-run.md` or a new note) for reproducibility.

## Phase 4 — OPTIONAL GPU: fold the naive arm
Only if GPU is free: fold the naive arm with AF2-multimer (k=0..2), **reusing the committed wt / L / random
folds** from `iptm_steer.csv` (fold ONLY naive ≈ 60×3 = 180 folds); parse with `parse_iptm.py` (crystal
interface set), analyse with `analyse_iptm.py`; report ipTM **naive − random** and **L − naive** + the composite.
Expected: naive ≈ random, confirming at the structure level that only the binding direction transfers.

## Deliverables / guardrails
`cfg_steer_naive_seqs.csv`, `cfg_naive_judge.csv`, `PREREG_naive.md`, `FINDINGS_rednet.md` (+ optional
`iptm_steer_naive.csv`, `iptm_summary_naive.csv`), any small driver. Report the paired L − naive headline. If the
naive tilt matches L (falsifier), report verbatim — that would bound our direction-specificity claim, and is a
finding, not a failure.

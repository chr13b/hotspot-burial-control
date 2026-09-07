# PRE-REGISTRATION — the naive-guidance SPECIFICITY arm (does the *binding* direction matter, or any tilt?)

**Committed BEFORE any naive-arm number (CLAUDE.md rule 1).** SEED=20260803.

## Motivation
The intended "vs another method" comparison was a head-to-head against **RedNet** (`zw2x/rednet_public`). That
measurement is **BLOCKED and not faked** (CLAUDE.md rule 3): RedNet's pipeline imports private companion packages
(`faust`, `atomtools`) that are not in `pyproject.toml`, not in the repo, and not among the author's public repos,
and its weights (Zenodo record 20113403) are access-restricted. The recon did, however, verify at the code level
that RedNet's released contrastive decode is `logits = (1+α)·logP_complex − α·logP_contrast` — **exactly our
`+α·L`** (`= logP_complex + α·(logP_complex − logP_contrast)`; contrast = unbound monomer for `run_hdimer`,
off-target for `run_sel`; scorer computes `cd_ll = ll − ub_ll`). That is written up as a code-level positioning
note in `results/FINDINGS_rednet.md`, **not** claimed as a head-to-head. See also memory `bundle-run.md`.

This pre-registration is for the accessible half we *can* run: a **third steering arm** that isolates whether the
benefit of `+α·L` is specific to the **binding direction**, or whether *any* matched-magnitude tilt would do. The
committed `random` arm already controls for a *random* direction of matched magnitude. The **naive-guidance** arm
is a stronger, non-random control: a real, directional tilt toward **what the model already prefers** (its own
complex-conditioned confidence), matched to L's per-position magnitude — the tilt a naive practitioner would reach
for. If `L` beats it, the gain requires the *binding* direction, not merely a confident one.

## Mechanism — the naive tilt (differs from L ONLY in direction; matched per-position magnitude)
For each interface position *i* (same committed interface set as `cfg_steer.py`,
`results/leverage_skempi_positions.csv` `is_interface`), with ProteinMPNN's own leverage `L_i` (20-vector, from
`results/leverage_pq_skempi.csv`) and complex-conditioned log-probs `lP_i` (same file, `lP_*` columns):

- **confidence direction** `conf_i = lP_i − mean_a(lP_i)` (centered; the direction of what the model prefers).
- **naive vector** `naive_i = ‖L_i‖₂ · (conf_i / ‖conf_i‖₂)`.
- applied bias at position *i*: `B[i, :20] = α · naive_i` (0 off-interface), fed to `mpnn_steer.draw(bias_by_res=…)`
  exactly as the L and random arms are.

By construction `‖α·naive_i‖₂ = α·‖L_i‖₂ = ‖α·L_i‖₂` — **identical per-position magnitude to the L tilt and to the
random tilt** (the random arm is `rng.permutation(L_i)`, which also preserves `‖L_i‖₂`). The three arms — **L**
(binding-leverage direction), **random** (scrambled direction), **naive** (confidence direction) — differ *only*
in direction. The script asserts `‖naive_i‖ == ‖L_i‖` per position. Protocol identical to `cfg_steer.py`:
α ∈ {0, 2}, K=64, temp=1.0, `order=None`, SEED — so decoding-order variance affects all arms equally and cancels
in the paired contrasts.

## Anti-circularity (the whole point)
ProteinMPNN is the steered model, so judging by ProteinMPNN is circular. The naive arm's sampled interface
residues are scored by the **non-self judges** with a per-position P/Q leverage cache: **ESM-IF1**
(`leverage_pq_skempi_esmif.csv`) and **MIF** (`leverage_pq_skempi_mif.csv`). (PiFold is omitted: no
`leverage_pq_skempi_pifold.csv` P/Q cache exists; ProteinMPNN reported self/reference only.) Same
`judge_dumped.py` path used for the committed L/random arms → directly comparable.

## Hypotheses
- **H1 (specificity).** A matched-magnitude **non-binding** tilt does not raise binding-favorability the way `L`
  does. Concretely, under an **anti-circular** judge (ESM-IF1 and/or MIF), paired per-complex:
  - **naive − random ≈ 0** (a confident-but-not-binding tilt behaves like the random control), AND
  - **L − naive > 0** (the binding direction still wins).
- **Expected ordering:** `L  ≫  naive ≈ random` in judge-leverage of the sampled interface residues.

## Falsifier (pre-registered; reported verbatim if it fires)
Under an anti-circular judge, **naive − random ≥ L − random** (i.e. the confidence tilt carries as much
binding-favorability as L). That would mean the effect is not specific to the binding direction — the
direction-specificity claim is **bounded**, and we report it as a finding, not a failure. Equivalently, if
**L − naive** has a 95% CI that includes or falls below 0, L does not beat the naive tilt.

## Metrics & statistics
Per complex, per anti-circular judge, the **dumped3** measure (mean judge-leverage over the ≤3 folded interface
samples at α=2, via `judge_dumped.py` — the same uniform measure used for the committed matrix) is primary; the
**K64** meanL (mean over K=64 sampled interface residues, higher power) is secondary. Paired contrasts
**naive − random**, **L − naive**, and **L − random** (context) with a **complex-clustered bootstrap** (NBOOT=5000,
SEED=20260803), 95% CI and P(>0) — identical machinery to `cfg_judge_matrix.py::paired`.

Complex set: the frozen **60-complex ipTM subset** (`results/iptm_subset.txt`) — this aligns the judged
specificity test and the optional Phase-4 fold on the *same* complexes. The L/random arms are the committed SET-A
dumped rows restricted to these 60 (all contrasts, incl. L−random, are computed on the naive complex set for a
fair falsifier comparison). α ∈ {0, 2}, K=64.

## Optional structural confirmation (Phase 4, GPU, only if free)
Fold the naive arm (60-complex ipTM subset, k=0..2) with AF2-multimer, **reusing** the committed wt/L/random folds
in `results/iptm_steer.csv` (fold only naive ≈ 180 folds); parse on the committed **crystal** interface set with
`parse_iptm.py`; analyse with `analyse_iptm.py` (the pre-registered composite). Expected: **naive ≈ random** on
ipTM/composite, confirming at the structure level that only the binding direction transfers.

## Outputs / scripts
`results/cfg_steer_naive_seqs.csv` (dumped k0-2 @ α=2) and `results/cfg_steer_naive.csv` (K64 meanL) from
`src/cfg_steer_naive.py`; `results/cfg_naive_judge.csv` from `src/judge_dumped.py`; paired stats
`results/cfg_naive_summary.csv` + `results/FINDINGS_naive.md` from `src/analyse_naive.py`. Optional
`results/iptm_steer_naive.csv`, `results/iptm_summary_naive.csv`.

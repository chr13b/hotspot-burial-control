# FINDINGS — the judge matrix: steering is NOT one model's quirk (both directions, anti-circular)

**Pre-registered** bundle Phase 1 (falsifier: an anti-circular judge with paired L−random ≤ 0 does not
corroborate). `SEED=20260803`. Driver `src/cfg_judge_matrix.py`; raw `results/cfg_judge_matrix.csv`. The
judge-leverage of the sampled α=2 interface residues (mean over K=64 samples × interface positions of the judge's
own leverage `L(a)` at the sampled residue), paired **L-arm − random-arm** per complex, complex-clustered
bootstrap 95% CI (5000 resamples).

## Result — both anti-circular directions PASS; falsifier does NOT fire

| steered model | judge | circular? | paired L − random | 95% CI | P(>0) | n cx |
|---|---|---|---|---|---|---|
| ProteinMPNN | **ESM-IF1** | anti-circular | **+0.773** | [+0.727, +0.822] | 1.000 | 271 |
| ESM-IF1 | **ProteinMPNN** | anti-circular | **+0.437** | [+0.383, +0.492] | 1.000 | 60 |
| ProteinMPNN | ProteinMPNN | self (ref) | +0.801 | [+0.775, +0.829] | 1.000 | 271 |
| ESM-IF1 | ESM-IF1 | self (ref) | +1.147 | [+1.048, +1.251] | 1.000 | 60 |

- **Anti-circular, both directions.** Steering ProteinMPNN by `+α·L` produces interface residues **ESM-IF1**
  rates as more binding-favorable than a matched-magnitude random control (+0.773, the standing SET-A result);
  and — the new leg — steering **ESM-IF1** by `+α·L` (SET-B, `cfg_steer_esmif.py`, its own leverage) produces
  residues **ProteinMPNN** rates higher (+0.437). Both CIs exclude 0 decisively. The effect is **not one model's
  quirk**: the reverse steering also works, and a second inverse-folding model corroborates each direction.
- **Self/circular rows** (steer-by-L, judge-by-same-L) are strongly positive by construction (+0.80 / +1.15) and
  are reported for reference only — they are the trivial direction and carry no corroborative weight.
- **α=0 baseline** (`results/cfg_judge_matrix.csv`): paired L−random ≈ 0 at every judge (no steering, no effect),
  the expected no-op control.

## Method / honest scope
SET-A = ProteinMPNN-steered (`cfg_steer.py`, 271 complexes); SET-B = ESM-IF1-steered (`cfg_steer_esmif.py`, the
frozen batch-1 60). The SET-B steerer is a **one-shot biased native-conditional** sampler (bias ESM-IF1's cached
per-position conditional log-probs by `+α·L_esmif`, sample interface positions independently, wt elsewhere) —
disclosed in `PREREG_esmif_steer.md`; ESM-IF1's conditional logits and leverage both come from the committed
`leverage_pq_skempi_esmif.csv`, so no live ESM-IF1 pass is needed (and none can drift out of alignment). Judges
are the committed per-position leverage caches (`leverage_pq_skempi{,_esmif}.csv`). **`L` here is an
inverse-folding leverage proxy, not experimental ΔΔG** — the AF2/Boltz fold legs (Phases 2/3) test the
downstream structural consequence. PiFold + MIF are **additional non-self judges** planned for this matrix; they
are not yet set up on Sherlock (laptop-only caches) — enrichment attempted separately and appended here if it
lands. The two-model cross-judge result above already establishes the core "not one model's quirk" claim.

# FINDINGS — the judge matrix: steering is NOT one model's quirk (3 judges, both directions, anti-circular)

**Pre-registered** bundle Phase 1 (falsifier: an anti-circular judge with paired L−random ≤ 0 does not
corroborate). `SEED=20260803`. Drivers `src/cfg_judge_matrix.py` + `src/judge_dumped.py`; raw
`results/cfg_judge_matrix.csv` (+ `cfg_judge_dumped_set{A,B}.csv`). The judge-leverage of the sampled α=2
interface residues (mean over samples × interface positions of the judge's own leverage `L(a)` at the sampled
residue), paired **L-arm − random-arm** per complex, complex-clustered bootstrap 95% CI (5000 resamples).
Two methods: **K64** = over all K=64 samples (ProteinMPNN/ESM-IF1 judges, high power); **dumped3** = over the 3
actually-folded samples, applied UNIFORMLY across all judges incl. **MIF** (the multi-judge extension) — dumped3
reproduces K64 for MPNN/ESM-IF1 (e.g. ESM-IF1 judge +0.770 vs +0.773), validating it.

## Result — every anti-circular judge PASSES; falsifier does NOT fire

**Paired L − random judge-leverage (α=2), by (steered model × judge):**

| steered model | ESM-IF1 judge | ProteinMPNN judge | MIF judge |
|---|---|---|---|
| **ProteinMPNN** (SET-A, n=271) | **+0.773** [+0.727,+0.822] | *self +0.801* | **+0.710** [+0.663,+0.759] |
| **ESM-IF1** (SET-B, n=60) | *self +1.147* | **+0.437** [+0.383,+0.492] | **+0.568** [+0.478,+0.662] |

All six anti-circular cells have P(>0)=1.000 and CIs excluding 0.

- **Not one model's quirk — three ways.** Steering **ProteinMPNN** by `+α·L` yields interface residues that a
  DIFFERENT model rates as more binding-favorable — under **both ESM-IF1 (+0.773)** and **MIF (+0.710)**. And the
  reverse: steering **ESM-IF1** by `+α·L` (SET-B, its own leverage) yields residues rated higher by **ProteinMPNN
  (+0.437)** and **MIF (+0.568)**. Three independent inverse-folding models, both steering directions, all agree.
- **Self/circular rows** (steer-by-L, judge-by-same-L: MPNN→MPNN +0.80, ESM-IF1→ESM-IF1 +1.15) are strongly
  positive by construction and carry no corroborative weight — reported for reference only.
- **α=0 baseline** (`cfg_judge_matrix.csv`): paired L−random ≈ 0 at every judge — the no-steering no-op control.

## Positive controls (rule 6)
- **MIF scorer sanity:** MIF native recovery **0.506** on the regenerated cache (`leverage_pq_skempi_mif.csv`),
  in the healthy ~0.35–0.55 band (a broken alphabet map would read ~0.05).
- **Method cross-check:** dumped3 reproduces K64 for the two shared judges (ESM-IF1 +0.770 vs +0.773; ProteinMPNN
  +0.432 vs +0.437), so adding MIF via dumped3 is on the same footing.

## Method / honest scope
SET-A = ProteinMPNN-steered (`cfg_steer.py`, 271 complexes); SET-B = ESM-IF1-steered (`cfg_steer_esmif.py`, batch-1
60). The SET-B steerer is a one-shot biased native-conditional sampler over the committed
`leverage_pq_skempi_esmif.csv` (pre-reg `PREREG_esmif_steer.md`). Judges are the committed per-position leverage
caches; **MIF's cache was regenerated on Sherlock** (`leverage_extra_models.py --model mif --stage score`, after
adding an `FTAX_MIF_HUB` env override to `ftax_mif.py` — the committed default path was laptop-only). **PiFold**,
the other requested non-self judge, is **not set up on Sherlock** (no repo/checkpoint; laptop-only) and is not
included — MIF stands as the third independent judge. `L` here is an inverse-folding leverage proxy, not
experimental ΔΔG — the AF2/Boltz fold legs (Phases 2/3) test the downstream structural consequence.

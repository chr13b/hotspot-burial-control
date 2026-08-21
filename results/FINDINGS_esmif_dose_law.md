# FINDINGS — the backbone-error DOSE LAW under a second model family (ESM-IF1)

**Pre-registered before the ladder ran** (commit `90914b4`, frozen with the device port; BRIEF.md ground
rule 1). **Prediction:** at σ=0, ESM-IF1 CPI(L | geometry) is positive and clears the placebo floor
(**+0.00072**, the largest placebo in `w_placebo_ladder.csv`); it holds at σ=0.5 Å and collapses toward 0 by
σ=1.0 Å — the same shape as ProteinMPNN. **Falsifier:** if on a powered sample (≥100 complexes) σ=0 does not
clear the floor, the second-model-family empirical is a null; report it as a null and keep the mechanism-first
claim.

**Model:** `esm_if1_gvp4_t16_142M_UR50` (GVP-transformer, 142M, causal, native-teacher-forced conditional
readout). **Scorer:** `src/leverage_noise_ladder_esmif.py` (pure device port of the committed CPU path — see
below). **Seed 20260803.** Jitter is per-coordinate Gaussian with per-atom RMSD ≈ σ, applied to N/CA/C/O; the
monomer inherits the complex's jitter so the partner ablation stays clean. Machine: Sherlock, 1× A30.

---

## The curve

**Headline — the pre-registered command** (`results/leverage_noise_ladder_esmif.csv`):

```
python3 src/leverage_noise_ladder_esmif.py --sigmas 0.0,0.25,0.5,0.75,1.0 --limit 200 \
    --max-residues 1000 --device cuda --out results/leverage_noise_ladder_esmif.csv
```

| σ (Å) | CPI(L \| geom) | 95% CI (complex bootstrap) | P(>0) | Spearman(L, ΔΔG) | × placebo floor |
|---|---|---|---|---|---|
| 0.00 | **+0.0330** | [+0.0222, +0.0429] | 1.000 | −0.237 | 46× |
| 0.25 | **+0.0358** | [+0.0230, +0.0479] | 1.000 | −0.163 | 50× |
| 0.50 | **+0.0196** | [+0.0096, +0.0286] | 1.000 | −0.159 | 27× |
| 0.75 | +0.0032 | [−0.0030, +0.0085] | 0.850 | −0.080 | 4× |
| 1.00 | +0.0114 | [+0.0023, +0.0187] | 0.992 | −0.134 | 16× |

n = 2,045 mutations at every rung (same sample throughout — the rungs are paired, not independent samples).

**Power robustness — declared in advance, before any σ>0 number was seen**, same ladder over *every* complex
the fixture has (`results/leverage_noise_ladder_esmif_all285.csv`, `--limit 285`, n = 2,809; the 1.5/2.0 rungs
are `results/leverage_noise_ladder_esmif_tail.csv`, same `--limit 285`):

| σ (Å) | CPI(L \| geom) | 95% CI | P(>0) | Spearman(L, ΔΔG) | % of σ=0 |
|---|---|---|---|---|---|
| 0.00 | **+0.0362** | [+0.0273, +0.0452] | 1.000 | −0.252 | 100% |
| 0.25 | **+0.0350** | [+0.0256, +0.0444] | 1.000 | −0.170 | 97% |
| 0.50 | **+0.0266** | [+0.0178, +0.0353] | 1.000 | −0.172 | 73% |
| 0.75 | +0.0115 | [+0.0046, +0.0185] | 1.000 | −0.118 | 32% |
| 1.00 | +0.0177 | [+0.0103, +0.0253] | 1.000 | −0.169 | 49% |
| 1.50 | +0.0020 | [−0.0013, +0.0053] | 0.881 | −0.096 | 6% |
| 2.00 | +0.0011 | [−0.0026, +0.0042] | 0.731 | −0.103 | 3% |

**Noise-realization control** (`results/leverage_noise_ladder_esmif_redraw.csv`, `--limit 200`). The rung seed
is `SEED + round(σ·100)`, so σ = 0.74/0.76/0.99/1.01 are the *same jitter magnitude with an independent draw* —
the trick the committed ProteinMPNN run already used (`leverage_noise_ladder_extra.csv` has σ=1.01 for exactly
this reason). Three draws per magnitude:

| magnitude | draws (CPI) | mean | mean as % of σ=0 |
|---|---|---|---|
| ~0.75 Å | +0.0032 (0.75), +0.0027 (0.74), **+0.0146** (0.76) | +0.0068 | 21% |
| ~1.00 Å | +0.0114 (1.00), +0.0019 (0.99), **−0.0002** (1.01) | +0.0044 | 13% |

**The draw-to-draw spread at a fixed magnitude is ~0.012 — the size of the point estimates in that region.**
The complex-level bootstrap CI does *not* include this source of variance, so at σ ≥ 0.75 Å the per-rung CIs
are optimistic and the 0.75-vs-1.0 non-monotonicity in both ladders is **inside realization variance and is not
a difference**. (Project rule: a result inside the sampling variance of the procedure is not a result.)

---

## Verdict — replicates in shape; the cliff is SHIFTED OUT, not reproduced at 1 Å

**The falsifier does not fire, and the load-bearing half of the class claim replicates.**

1. **σ=0 clears the placebo floor decisively.** +0.0330 [+0.0222, +0.0429] (200 complexes) and +0.0362
   [+0.0273, +0.0452] (all 285), P(>0)=1.000 — **46–50× the +0.00072 floor**. The "second model family"
   empirical is *not* a null.
2. **It survives sub-Ångström jitter.** At σ=0.5 Å, +0.0196 / +0.0266, CI excludes zero, 27–37× the floor,
   ~60–73% of the crystal value retained. Exactly as pre-registered.
3. **It decays with backbone error.** By ~0.75–1.0 Å the estimate is down to 13–32% of σ=0 and individual noise
   draws straddle the floor (−0.0002 to +0.0177).

**Where it differs from ProteinMPNN, stated plainly rather than smoothed:**

| σ (Å) | ProteinMPNN CPI | ESM-IF1 CPI (285) |
|---|---|---|
| 0.00 | +0.0575 | +0.0362 |
| 0.25 | +0.0588 | +0.0350 |
| 0.50 | +0.0474 | +0.0266 |
| 0.75 | +0.0321 | +0.0115 |
| **1.00** | **+0.0024** [−0.00003, +0.0048] | **+0.0177** [+0.0103, +0.0253] |
| 1.50 | −0.0012 | +0.0020 [−0.0013, +0.0053] |

ProteinMPNN is at the floor by 1.0 Å (4% retained, CI touching zero). **ESM-IF1 is not** — on a single
full-power draw it still has +0.0177 with the CI excluding zero, and only reaches the floor by 1.5–2.0 Å
(+0.0020 and +0.0011, both CIs containing zero). Averaged over three draws at ~1 Å the retention is 13% —
attenuated but not extinguished. **So the ~1 Å cliff location is ProteinMPNN-specific; under ESM-IF1 the same
qualitative dose law has its collapse at ~1.5 Å.** The pre-registered wording "collapses toward 0 by σ=1.0 Å"
is therefore **confirmed for the direction and the sub-Ångström survival, and NOT confirmed for the 1.0 Å
cliff location** under this second model.

Two further honest differences:

- **ESM-IF1's rank correlation is far more jitter-robust than its CPI.** Spearman(L, ΔΔG) goes −0.252 → −0.169
  at 1.0 Å (33% loss) and is still −0.096 / −0.103 at 1.5 / 2.0 Å, whereas ProteinMPNN's goes −0.301 → −0.077
  (74% loss). A *candidate* explanation — **untested here, stated as a hypothesis, not a measurement**: the
  ESM-IF1 readout is teacher-forced on the native sequence context, which backbone jitter does not touch, so
  part of its L survives on sequence information alone. Testing that would need a native-context ablation.
- **Absolute magnitudes are smaller under ESM-IF1 at every rung** (+0.036 vs +0.058 at σ=0), consistent with
  the already-committed crystal-backbone comparison in `FINDINGS_leverage_esmif.md` (+0.0350 vs +0.0588).

**What this does and does not license.** It licenses: *the mixed derivative's binding signal is fragile to the
backbone it is read from, in both model families, and survives accurate (sub-Ångström) reconstruction in
both.* It does not license: *a universal ~1 Å cliff.* The cliff's **location is model-dependent** (≈1.0 Å for
ProteinMPNN, ≈1.5 Å for ESM-IF1) even though its **existence** is not. The paper's mechanism-first framing —
the sensitivity is to the input backbone, shared by any reader of the derivative — is what survives; the
specific threshold must be quoted per model.

---

## Positive controls run (rule 6 — never trust a number without one)

- **Alphabet round-trip** (an ESM→MPNN column slip silently drops recovery to ~0.05): the ladder now asserts
  it every run and it round-trips to `'ACDEFGHIKLMNPQRSTVWYX'`.
- **CPU path unbroken by the port**, 3 complexes: CPI +0.00082, Spearman −0.2840.
- **Same 3 complexes on GPU**: CPI +0.00082, Spearman −0.2840 — the device port is numerically a no-op.
- **σ=0 anchored to the committed crystal run.** Recomputing the committed ESM-IF1 L
  (`leverage_esmif_mutations.csv`) on exactly the pilot's row filter (interface-only, first 100 complex_ids)
  gives Spearman −0.2750; the σ=0 pilot returned **−0.2752** (`results/_pilot_esmif_sigma0.csv`, n=1,494,
  CPI +0.0462 [+0.0333, +0.0571]). At `--limit 285` the ladder's σ=0 Spearman is −0.252 against the committed
  full-run **−0.2554**.
- **Anchor correction.** The kickoff prompt (`notes/SHERLOCK_ESMIF_DOSE_PROMPT.md`) quotes the committed σ=0
  Spearman as −0.176. That value is not in `results/leverage_esmif.csv`; the committed value there is
  **−0.2554** [−0.3156, −0.1902]. The anchor was taken from the committed CSV, and matching −0.176 would have
  been a *failure* to reproduce.

## Reproducing

Environment is not committed (Sherlock, `$SCRATCH`): fair-esm 2.0.0 + torch_geometric 2.5.3 + torch_scatter
2.1.2+pt22cu121 + **biotite 0.41.2** (fair-esm 2.0.0 calls the pre-1.0 `biotite.structure.filter_backbone`),
installed `--no-deps` into a side PYTHONPATH dir so the shared venv is never mutated; checkpoint slimmed to
`{'args','model'}` (567 MB). Env script `$SCRATCH/ftax/env_esmif.sh`; `FTAX_ESMIF_CKPT` overrides the
checkpoint path. Cost: **~1.9 s/complex on one A30**, ≈6 min per rung at 200 complexes — the "~15× ProteinMPNN,
GPU-scale" estimate in the prior signpost was right about the ratio and the whole ladder still costs under an
hour.

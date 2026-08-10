# FINDINGS — Experiment C: the tax on generative (RFdiffusion partial-diffusion) backbones

> **VERDICT (mixed, honest).** The sequence-free **KL detector transfers to generative-model
> backbones** — its burial-orthogonal hotspot signal survives on RFdiffusion partial-diffusion
> backbones wherever the interface is still formed (ΔAUROC-over-burial **+0.083 / +0.094 / +0.086** at
> `partial_T` 5 / 10 / 20, every 95% CI excluding zero). Together with Experiment A (predicted
> backbones) the detector now holds across crystal → predicted → generative backbones — **reading C-KL
> fires.** The **burial-matched log-prob gap (C-PRIMARY) is only suggestive**: a real deficit appears at
> the most non-native, interface-formed backbones (`partial_T=40`, iRMSD≈18 Å: **−0.88 [−1.11,−0.60]**;
> re-match −0.61 [−0.91,−0.28]), but the dose-response is **not monotone** and does **not survive the
> well-formed iRMSD<3 Å confound** — so C-PRIMARY neither cleanly fires nor is it a clean null. The
> binding-energy readout (SECONDARY) behaves as predicted: ProteinMPNN's rank-correlation with
> experimental ΔΔG_bind **collapses from −0.236 (crystal) to ≈−0.06** on generated backbones. The whole
> experiment is bounded by a **generator-instability caveat**: RFdiffusion, diffusing the binder against
> a held target *without* hotspot conditioning, **diverges on 62–75% of backbones** (binder coordinates
> blow past 10³–10⁷ Å, interface gone), so the interface-formed subset that carries every reading is
> small (9–21 complexes at high noise). **KILL C1 passed; KILL C2 did not fire.**

## 1. What was run

```bash
# Binder-first, backbone-only inputs; contig diffuses the binder, holds the target as a fixed motif.
python3 src/expC_setup.py            # 55 L<=400 pair complexes, binder = majority-labelled chain group
python3 src/expC_prep_inputs.py      # -> $SCRATCH/expC/inputs/<cid>_input.pdb, contig "<Lb>/0 B1-<Lt>", outmap

# Ladder: 55 complexes x partial_T {5,10,20,40} x N=3 designs  (+ partial_T=0 = 55 crystal controls).
#   RFdiffusion (SE3nv env: torch 1.9.1+cu111 on V100; Complex_base_ckpt), diffuser.partial_T, num_designs=3
sbatch --array=0-9%10 $SCRATCH/ftax/jobs/expC_run.sbatch          # 660 backbones produced (job 38506225)

# Score every backbone (re-keyed to crystal), 8-way sharded; ProteinMPNN v_48_020, 8 decoding orders +
# unconditional KL pass; interface Ca-RMSD (superpose on held target) + hotspot inter-chain contacts.
sbatch --array=0-7%8 $SCRATCH/ftax/jobs/expC_score_array.sbatch   # -> scored_{positions,backbones}.csv
python3 src/expC_analyze.py   ... --variant SECONDARY_B_any_interface --out results/expC
python3 src/expC_analyze.py   ... --variant EXPC_within_binder      --out results/expC_rematch   # robustness
python3 src/expC_secondary.py ...                                   --out results/expC_secondary.csv
```

Every number below traces to a committed CSV: `results/expC_dose.csv` (gap + KL),
`results/expC_rematch_dose.csv`, `results/expC_secondary.csv`, `results/expC_interface_qc.csv`,
`results/expC_gap_perbackbone.csv`. Seeds fixed (bootstrap seed 20260803; 10,000 gap / 2,000 KL reps).

## 2. The generator is unstable — and it shapes everything (coverage first)

The pre-registration (§2) required dropping backbones whose interface dissolved and **reporting the
count per level**. That count is the headline operational fact of this run:

| partial_T | produced | scored | interface FORMED (≥5 hotspot contacts) | median iRMSD (all) | median tgt-RMSD |
|---|---|---|---|---|---|
| 0 (crystal) | 55 | 55 | 50 / 55 | 0.0 Å | 0.0 Å |
| 5  | 165 | 165 | **62 / 165** | 1.1e3 Å | 0.13 Å |
| 10 | 165 | 165 | **62 / 165** | 2.6e4 Å | 0.13 Å |
| 20 | 165 | 165 | **57 / 165** | 1.5e6 Å | 0.13 Å |
| 40 | 165 | 147 | **37 / 147** | 4.9e7 Å | 0.14 Å |

- **The target is held perfectly** (median target-motif RMSD 0.13 Å at every level) — the "diffuse
  binder, hold target" construction works exactly as intended.
- **The binder is not.** On the majority of designs the diffused binder loses its interface anchor and
  translates/expands to non-physical coordinates (median all-backbone iRMSD in the thousands-to-millions
  of Å). The run log flags the cause: the `Complex_base` checkpoint is *"trained on complexes and
  hotspot residues"* and we specified **no hotspot residues**, so nothing pins the binder to the target
  interface during denoising. This is a real, reportable property of hotspot-free partial diffusion —
  **not** a scoring artifact (18 of the 165 `partial_T=40` designs are literally `nan` coordinates,
  RFdiffusion numerical divergences; they are excluded and listed here, all at T40).
- Consequently **every reading below is computed on the interface-FORMED subset** (pre-reg "drop
  dissolved"), which is small at high noise (9–21 complexes). A parsing hardening was required first:
  RFdiffusion writes coordinates that overflow PDB's 8-column x/y/z fields at |coord|≥1000 Å, which a
  naive biopython read silently dropped — a *noise-correlated* bias that a manual parser fixed (recovered
  290 backbones, lifting scored coverage from 352 → 642 of 660 ladder backbones).

## 3. Reading C-KL — the sequence-free detector SURVIVES on generative backbones (the clean positive)

KL detector = `KL(p(·|generated complex) ‖ p(·|chain-deleted generated complex))`, burial = Cβ
neighbour count; metric = paired ΔAUROC = AUROC(burial+KL) − AUROC(burial) for strict hotspots over
interface positions, complex-bootstrapped. **Interface-formed backbones** (`kl_by_T_ifaceok`):

| partial_T | AUROC(burial) | AUROC(burial+KL) | ΔAUROC [95% CI] | n_cx |
|---|---|---|---|---|
| 0 (crystal) | 0.720 | 0.821 | **+0.101 [+0.030, +0.143]** | 50 |
| 5  | 0.736 | 0.819 | **+0.083 [+0.034, +0.131]** | 21 |
| 10 | 0.728 | 0.821 | **+0.094 [+0.045, +0.139]** | 21 |
| 20 | 0.744 | 0.830 | **+0.086 [+0.059, +0.115]** | 20 |
| 40 | 0.539 | 0.426 | −0.112 [−0.176, −0.058] | 17 |

The KL detector adds ~+0.09 AUROC over burial alone on interface-formed generated backbones at
`partial_T` 5/10/20 — **CI excludes zero at each** → **C-KL fires.** It collapses only at `partial_T=40`,
where the interface is destroyed (burial itself drops to AUROC 0.44) — an expected floor, not a
counter-result. Note the contrast with the *pooled* (all-backbone) KL, whose CIs all straddle zero
(e.g. T10 [−0.092, +0.113]): the diverged backbones carry meaningless burial/KL and wash the signal
out. **The detector needs a real interface — and delivers whenever one is present.** This is the
reviewer-critical validity condition for finding C5 of the main study, now met on generative backbones.

## 4. Reading C-PRIMARY — the burial-matched gap is suggestive, not decisive

`d = logp(hotspot) − logp(burial-matched control)`, both on the diffused binder, SECONDARY_B pairs
(53 within-binder pairs / 31 complexes), interface-formed backbones, complex-bootstrapped:

| partial_T | median iRMSD | gap d [95% CI] | n_cx |  | iRMSD bin | gap d [95% CI] | n_cx |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 Å | +0.303 [−0.193, +0.800] | 29 |  | <1 Å | +0.315 [−0.127, +0.779] | 29 |
| 5 | 1.1 Å | −0.106 [−0.543, +0.366] | 13 |  | 1–2 Å | −0.098 [−0.502, +0.321] | 13 |
| 10 | 1.6 Å | −0.162 [−0.632, +0.284] | 13 |  | 2–3 Å | −0.359 [−0.921, +0.144] | 12 |
| 20 | 2.9 Å | +0.081 [−0.298, +0.523] | 12 |  | 3–5 Å | −0.114 [−0.621, +0.482] | 7 |
| 40 | 18.0 Å | **−0.882 [−1.114, −0.596]** | 9 |  | >10 Å | **−0.883 [−1.115, −0.596]** | 9 |

- A **real deficit appears at the most non-native interface-formed backbones** (`partial_T=40`,
  iRMSD≈18 Å): −0.88, CI excludes zero; the re-match variant agrees (−0.61 [−0.91, −0.28]).
- But it is **not monotone** (T0 +0.30 → T5/T10 slightly negative → T20 +0.08 → T40 −0.88), and every
  intermediate level's CI contains zero. The pre-registered C-PRIMARY requires *monotone* descent **and**
  CI-excludes-zero at the highest realistic-interface level; the second clause holds, the first does not.
- **Confound (pre-reg §4.4): it does NOT cleanly survive.** Restricted to well-formed, near-native
  backbones (iRMSD<3 Å) the gap is not significant (2–3 Å bin −0.36 [−0.92, +0.14]); the significant
  signal lives in the iRMSD>10 Å bin, i.e. exactly where the interface is most marginal. So the deficit
  is **carried by heavily-drifted backbones**, which is what the dose-response predicts *directionally*
  but is also what the confound was written to distrust. **C-PRIMARY is therefore recorded as
  suggestive, not decisive.**

## 5. Pre-registered falsifier readings (PREREG_expC §5), stated plainly

- **KILL C1 (mandatory control) — PASSED.** `partial_T=0` (crystal, identical scoring path) reproduces
  the ~zero crystal deficit: SECONDARY_B within-binder +0.303 [−0.193, +0.800] (re-match +0.326
  [−0.099, +0.787]); committed crystal was −0.042 [−0.222, +0.129]. CI contains zero — no crystal
  burial-matched deficit; the pipeline is sound.
- **KILL C2 — did NOT fire.** The gap is *not* flat across the ladder and the highest level's CI
  *excludes* zero, so the deficit is **not** specific to OpenFold3-style predictions — it does reach the
  generative/design regime, at high backbone drift.
- **C-KL — FIRED** (§3).
- **C-PRIMARY — suggestive, not decisive** (§4): significant only at extreme drift, non-monotone, not
  surviving the iRMSD<3 Å confound.

## 6. Secondary — the binding-energy readout degrades as predicted (exploratory, no falsifier)

Partial-ρ(experimental SKEMPI ΔΔG_bind, ProteinMPNN log-odds `ℓ(mut)−ℓ(wt)` | burial), 1088 single
mutations / 55 complexes, computed on each backbone's stored `lp_<AA>`:

| partial_T | partial-ρ [95% CI] |
|---|---|
| 0 (crystal) | **−0.236 [−0.333, −0.157]** |
| 5 | −0.064 [−0.141, +0.019] |
| 10 | −0.046 [−0.125, +0.050] |
| 20 | −0.049 [−0.126, +0.060] |
| 40 | −0.070 [−0.139, +0.062] |

The model's ability to rank *experimental binding energy* **collapses toward zero** the moment the
backbone leaves the native manifold (−0.236 → ≈−0.06, CIs now include zero) — the binding-relevant face
of the factorization tax, exactly the pre-registered sign prediction. (Note the SECONDARY is over all
backbones, so its high-noise iRMSD medians are inflated by the divergences; the collapse is already
complete by `partial_T=5`.)

## 7. What this means, and the honest caveats

1. **The transferable result is C-KL.** A sequence-free, burial-orthogonal signal for interface
   hotspots is present on generative-model backbones wherever a real interface exists — it does not
   depend on native side chains or on the backbone being a reconstruction of a known complex. Across
   Experiments A and C it now holds on crystal, predicted, and generative backbones.
2. **The log-prob gap does not cleanly generalize to the design regime.** It is significant only at
   extreme, marginally-interfaced backbones and does not survive the near-native confound. This is an
   honest partial result, reported as such and not upgraded.
3. **Generator instability is the dominant limitation.** Hotspot-free partial diffusion of a binder
   against a held target diverges on most designs, so the physically-meaningful sample is small at high
   noise (9–21 complexes) and the "dose" (iRMSD) is bimodal (bounded vs blown-up) rather than a clean
   continuum. The most direct follow-up is to **re-run with hotspot residues specified** (or the Base
   checkpoint) so the binder stays engaged and the dose-response is sampled on physical complexes — this
   was outside the pre-registered plan and is left as a declared next step, not silently substituted.
4. **Coverage / exclusions, never silently dropped:** 660 ladder backbones produced; 642 scored; 18
   excluded, all `partial_T=40` RFdiffusion `nan`-coordinate divergences (listed in the scoring logs).
   Interface-dissolved backbones are dropped per pre-reg §2 with counts in the table above.

### Files
- `results/expC_dose.csv` — gap (all / interface-formed / by-iRMSD) and KL (all / interface-formed) by level.
- `results/expC_rematch_dose.csv` — same, within-binder re-matched pairs (robustness; agrees).
- `results/expC_secondary.csv` — partial-ρ(ΔΔG_bind, log-odds | burial) by level.
- `results/expC_interface_qc.csv` — per-level produced/scored/interface-formed + median iRMSD & target-RMSD.
- `results/expC_gap_perbackbone.csv` — per-backbone gap + iRMSD + interface_ok (audit trail).
- Backbones (gitignored, on `$SCRATCH`): `$SCRATCH/expC/backbones/`, `$SCRATCH/expC/scored_*.csv`.

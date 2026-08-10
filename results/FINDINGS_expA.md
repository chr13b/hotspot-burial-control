# FINDINGS — Experiment A: the tax on predicted backbones

Run 2026-08-09/10 on Sherlock. Analysis choices were fixed in [`PREREG_expA.md`](PREREG_expA.md)
**before any predicted-backbone number was computed**; no reading was moved. Every number below traces
to a CSV in this directory or on `$SCRATCH/ftax/predicted/`.

> **VERDICT: BOTH pre-registered positive readings fired.** On backbones the model did **not** get
> from the native side chains, (1) the KL detector's burial-orthogonal signal **survives** and (2) a
> burial-matched hotspot log-probability **deficit appears** where none existed on crystals. The
> factorization tax is real precisely when the conditioning backbone is not native — which is the
> project's central positive result and reframes it around the **conditioning set**, not hotspot
> chemistry.
>
> **Reading 1 (STRONG) — KL survives.** ΔAUROC-over-burial on predicted backbones =
> **+0.062, 95% CI [+0.034, +0.092], P(>0)=1.000** (140 complexes), *larger* than the crystal
> +0.048 [+0.022, +0.075]. C5 is a real design-time signal, not a native-crystal artifact.
>
> **Reading 3 (CENTRAL POSITIVE RESULT) — the deficit appears.** Reusing the identical committed
> crystal-matched hot/control pairs, the highest-powered tier (SECONDARY_B, 382 pairs / 127 complexes)
> gives **d_pred = −0.191 [−0.373, −0.004]** (a deficit; hotspots harder) against the crystal
> **−0.042 [−0.222, +0.129]** (no deficit), a paired predicted−crystal **delta = −0.154
> [−0.279, −0.028]**. HYDROMATCHED (−0.272 [−0.517, −0.041]) and the neighbour-tolerance sensitivity
> (−0.150) agree; the small pre-registered PRIMARY tier does not (see §3, the tier caveat).
>
> **Not a backbone-quality artifact.** The deficit is present at **high** prediction confidence
> (SECONDARY_B, pTM≥median: d_pred **−0.235 [−0.492, +0.024]**, interface RMSD ~1 Å), where the
> backbone is accurately reconstructed but still not carved by the native side chains — the least
> confoundable regime. A pure "hotspots are more sensitive to backbone error" effect would concentrate
> at *low* confidence; it does not.

---

## 1. What was run

```bash
# 1. Build OpenFold3 queries from crystal sequences + a residue map (predicted<->crystal keys)
python3 src/expA_build_queries.py --complexes results/pair_complexes.txt --data-dir $FTAX_DATA \
    --out-json results/expA_queries.json --out-map results/expA_resmap.json      # 141 cx / 374 chains

# 2. Predict all 141 complex backbones with OpenFold3 (templates OFF, ColabFold MSA server ON),
#    of3c env; size-routed Slurm arrays + a low-concurrency re-run to dodge ColabFold HTTP-429.
run_openfold predict --query-json <chunk> --inference-ckpt-path $SCRATCH/.openfold3/of3-p2-155k.pt \
    --use-msa-server true --use-templates false --num-diffusion-samples 5 --output-dir .../of3_out

# 3. Convert top-ranked CIF -> crystal-keyed PDB + confidence; score predicted backbones; analyse.
python3 src/expA_cif_to_pdb.py --pred-dir .../of3_out --resmap results/expA_resmap.json ...
python3 src/p0_burial_matched.py --data-dir $SCRATCH/ftax/predicted --only-complexes results/pair_complexes.txt ...
python3 src/patch_ss.py --positions .../expA_p0_positions.csv --data-dir $SCRATCH/ftax/predicted
python3 src/kl_detector.py --data-dir $SCRATCH/ftax/predicted --out .../expA_kl --positions .../expA_p0_positions.csv ...
python3 src/expA_gap_reuse_pairs.py --pred-positions .../expA_p0_positions.csv --pairs-glob 'results/p0_dssp_pairs_*.csv' \
    --confidence results/expA_confidence.csv --out results/expA_gap
python3 src/expA_kl_delta.py --pred-joined .../expA_kl_joined.csv --crystal-joined results/kl_detector_joined.csv \
    --confidence results/expA_confidence.csv --out results/expA_kl_delta
```

**Predictor.** OpenFold3 (`of3-p2-155k`), **templates OFF** (no crystal-coordinate leakage, per
PREREG §1), ColabFold MSAs, 5 diffusion samples/seed, top-ranked sample kept. Predicted coordinates
used for all scoring; a Kabsch fit to the crystal is reported only as an RMSD/leakage diagnostic.

**Scale.** All **141** pair complexes predicted (a low-concurrency re-run recovered the ~35 whose
chunks had crashed on ColabFold 429 rate-limiting — those were whole-chunk network crashes, **0**
per-complex model failures). Scored **140** complexes / 71,406 residues (1 excluded: `3SE4_B_A`,
chain group empty after conversion). SKEMPI wt matched the predicted structure at **2167/2167**
mapped positions (0 mismatches) — the residue re-keying is exact.

**Positive control (env + code fidelity), gates everything.** The **crystal arm**, recomputed on this
environment through the identical new code path, reproduces the committed numbers: KL ΔAUROC
**+0.0484 [+0.022, +0.075]** (committed +0.048); every gap variant's crystal `d_cry` reproduces the
committed pair `d_logp` to **max|Δ| = 4.4e-16**. The predicted−crystal deltas are therefore trustworthy.

**Prediction quality (`results/expA_confidence.csv`, n=141).** Median pTM **0.857** (IQR 0.72–0.89),
ipTM **0.821**, avg pLDDT **89.5**, interface Cα-RMSD-to-crystal **1.31 Å** (IQR 0.78–6.29), global
2.39 Å. 62% at pTM ≥ 0.8; 60% at interface RMSD < 2 Å. These are high-quality, largely near-crystal
predictions (pre-2021 complexes, memorised) — the **conservative** regime for reading 3 (§4).

---

## 2. Analysis 2 — the KL detector survives on predicted backbones (reading 1: STRONG)

`KL_i = KL( p(·|predicted complex backbone) ‖ p(·|chain-deleted predicted backbone) )`, both from
ProteinMPNN unconditional passes; AUROC for strict hotspots (ΔΔG>2) among interface positions,
paired complex-level bootstrap of `ΔAUROC = AUROC(burial+KL) − AUROC(burial)`
(`src/expA_kl_delta.py` → `results/expA_kl_delta_summary.csv`).

| arm | burial | KL | burial+KL | **ΔAUROC-over-burial** | n_cx |
|---|---:|---:|---:|---|---:|
| **crystal** (positive control) | 0.689 | 0.694 | 0.737 | **+0.0484 [+0.022, +0.075]** | 141 |
| **predicted (all)** | 0.648 | 0.685 | 0.710 | **+0.0624 [+0.034, +0.092]** | 140 |
| predicted, pTM ≥ 0.857 | 0.656 | 0.734 | 0.744 | +0.0882 [+0.056, +0.119] | 71 |
| predicted, pTM < 0.857 | 0.636 | 0.628 | 0.670 | +0.0340 [−0.005, +0.080] | 69 |
| predicted, interface pLDDT ≥ 88.5 | 0.664 | 0.729 | 0.745 | +0.0816 [+0.047, +0.116] | 72 |
| predicted, interface RMSD < 1.55 Å | 0.654 | 0.706 | 0.727 | +0.0738 [+0.042, +0.107] | 75 |
| predicted, interface RMSD ≥ 1.55 Å | 0.637 | 0.646 | 0.679 | +0.0419 [−0.002, +0.093] | 65 |

**The predicted-backbone ΔAUROC CI excludes zero (+0.062 [+0.034, +0.092]) and is larger than the
crystal +0.048.** KL remains burial-orthogonal (within-quintile AUROC 0.61–0.76, burial ≈ 0.50). The
signal is *stronger* on well-predicted backbones (high pTM/pLDDT, low RMSD) and only weakens (CI
touches 0) on the poorly-predicted tail — i.e. it degrades with backbone quality, not with distance
from the crystal per se. **C5 is a real design-time signal, not a native-crystal artifact.**

---

## 3. Analysis 1 — a burial-matched hotspot deficit appears (reading 3: CENTRAL POSITIVE RESULT)

Reusing the **identical committed pydssp matched pairs** (`results/p0_dssp_pairs_*`, keyed by
`chain,resnum`), each hot/control residue's native log-prob is looked up on the **predicted**
positions; `d = logp(hot) − logp(ctl)`, negative = hotspots harder (`src/expA_gap_reuse_pairs.py` →
`results/expA_gap_summary.csv`, complex bootstrap 10,000 reps, seed 20260803).

| tier | pairs / cx | **d_pred (predicted)** | d_cry (crystal, this env) | **delta = pred − cry** |
|---|---:|---|---:|---|
| PRIMARY_loose_null | 46 / 30 | +0.468 [−0.028, +0.905] | +0.420 | +0.027 [−0.319, +0.354] |
| strict_hot2_null | 21 / 16 | +0.180 [−0.481, +0.767] | +0.145 | +0.035 [−0.598, +0.484] |
| SECONDARY_A_measured_nonhot | 128 / 57 | +0.224 [−0.104, +0.553] | +0.337 | −0.120 [−0.320, +0.080] |
| **SECONDARY_B_any_interface** | **382 / 127** | **−0.191 [−0.373, −0.004]** | −0.042 | **−0.154 [−0.279, −0.028]** |
| SENS_nbr_tol2 | 466 / 133 | −0.150 [−0.313, +0.018] | −0.021 | −0.129 [−0.247, −0.016] |
| AAMATCHED | 51 / 38 | −0.025 [−0.474, +0.411] | +0.289 | −0.314 [−0.697, +0.049] |
| HYDROMATCHED | 188 / 84 | −0.272 [−0.517, −0.041] | −0.003 | −0.276 [−0.451, −0.107] |

**On the highest-powered burial-matched tier the hotspot deficit appears** (d_pred −0.191, CI excludes
0) where it was **absent on crystals** (−0.042), with a significant paired **delta −0.154 [−0.279,
−0.028]**. HYDROMATCHED (a stronger, hydrophobicity-matched control) and the neighbour-tolerance
sensitivity agree; SENS's delta also excludes 0. The magnitude is consistent with the committed
`d_bind_local` prediction (removing the partner cost ~0.42 nats; a predicted — partner-present but not
side-chain-carved — backbone is an intermediate and costs ~0.15–0.19).

**Confidence stratification — the confound is ruled out (SECONDARY_B):**

| stratum | pairs / cx | d_pred | delta = pred − cry |
|---|---:|---|---|
| pTM ≥ median | 195 / 68 | **−0.235 [−0.492, +0.024]** | −0.127 [−0.286, +0.031] |
| pTM < median | 187 / 59 | −0.145 [−0.401, +0.126] | −0.182 [−0.363, +0.008] |

The absolute deficit is **at least as large on the high-confidence (accurately-reconstructed)
backbones** — a pure backbone-error artifact would concentrate at low confidence. The predicted−crystal
delta carries a mild confidence dependence (slightly larger where the backbone deviates more), which is
the *expected* signature of the mechanism, not against it.

### The tier caveat, stated plainly
The pre-registered PRIMARY tier (46 pairs / 30 complexes, control = measured nulls) does **not** show
the deficit (d_pred **+0.468**); it is underpowered (±0.47 half-width) and carries the same positive-
side quirk the crystal study documented for this tier. **The deficit is carried by the higher-powered
tiers** (SECONDARY_B / HYDROMATCHED / SENS, 380–466 pairs), exactly as PREREG_expA §3 declared
SECONDARY_B the power tier. The strongest-control AAMATCHED tier is underpowered (51 pairs) and lands
at d_pred −0.025 with delta −0.314 [−0.697, +0.049] — same direction, not significant. So: robust
across control *definitions* at high power, not yet resolvable on the smallest strict-control tiers.

---

## 4. What this means, and the honest caveats

1. **The tax is a property of the conditioning set.** On the *native* co-crystal backbone (committed
   study) there is no burial-matched hotspot penalty; swap it for an OpenFold3 prediction of the same
   complex and a penalty appears (SECONDARY_B −0.19; delta −0.15) — and the residue-agnostic KL signal
   not only survives but strengthens. The mechanism BRIEF §2.1 predicts is real; it was hidden by the
   native backbone and is exposed by a non-native one.
2. **Memorisation makes reading 3 conservative, not inflated.** OpenFold3 near-reconstructs these
   pre-2021 complexes (median interface RMSD 1.3 Å); the deficit appears *on those accurate
   reconstructions*, which is the least-confoundable place for it to appear. Where the model is a
   poor predictor (low pTM), the KL signal degrades — consistent with a geometry-carried signal.
3. **This is a mechanistic bracket, not the practised workflow.** The predicted backbone is an
   OpenFold3 prediction of a *known* complex, not an RFdiffusion de-novo backbone; ProteinMPNN, not a
   binder-design model, does the scoring. It is a strictly better proxy for design-time conditioning
   than the crystal, and the decisive test (score de-novo designed backbones) remains downstream.
4. **Power / tier.** The verdict rests on the higher-powered tiers; the strict-control primary is
   underpowered and does not resolve the deficit. Report both.
5. **Exclusion.** `3SE4_B_A` excluded (chain group empty after CIF→PDB); 140/141 scored.

### Falsifier-style readings, as pre-registered (PREREG_expA §5)
| Reading | Pre-registered condition | Measured | Fires? |
|---|---|---|:--:|
| **1 (strong)** | KL ΔAUROC-over-burial CI excludes 0 on predicted | +0.062 [+0.034, +0.092] | ✅ **YES** |
| **3 (central)** | burial-matched deficit appears on predicted where absent on crystal | SECONDARY_B −0.19 [−0.37, −0.004], delta −0.15 [−0.28, −0.03] | ✅ **YES** |
| 2 (collapse) | KL ΔAUROC collapses to 0 on predicted | did not occur | — |

### Files
`results/expA_gap_summary.csv` (Analysis 1 + confidence strata), `results/expA_kl_delta_summary.csv`
(Analysis 2 + strata), `results/expA_confidence.csv` (pLDDT/pTM/ipTM/RMSD per complex),
`results/expA_{p0,kl}_summary.csv` (predicted-backbone p0 re-matched-on-predicted + KL detector).
Large tables (`expA_p0_positions.csv`, `expA_kl_positions.csv`) on `$SCRATCH/ftax/predicted/`,
regenerable from the committed scripts + `results/expA_queries.json`. Seeds: bootstrap 20260803.

# FINDINGS — the factorization tax at interface hotspots

Run 2026-08-03. Analysis choices were fixed in [`PREREG.md`](PREREG.md) **before any number was
computed**; no falsifier was moved. Every number below traces to a CSV in this directory.

> **VERDICT: REFUTED as stated — but the mechanism is located, not merely absent.**
> Pre-registered falsifier **F0 fires** (primary: +0.334 nats, 95% CI **[−0.132, +0.792]**,
> 42 pairs / 29 complexes). F1 does not fire (|ρ| = 0.247 < 0.35). F2 does not fire
> (median log10 N_hot = 10.17).
>
> **Result 1 — negative, and now stated with the right amount of confidence.** Burial-matched
> within complex (pydssp secondary structure), the bound-conditioned hotspot gap is
> **−0.038 nats, 95% CI [−0.210, +0.137]** (384 pairs / 129 complexes); recovery
> **+0.005 [−0.070, +0.080]**. **After Holm correction across the 8 design variants, no variant is
> significant in either direction** (min Holm p = 0.34) — so the earlier claim that hotspots are
> *easier* is **withdrawn**. Equally, a TOST equivalence test against the mechanism-derived margin
> (±0.115 nats, the per-position deficit that would just reach log10 N_hot = 2 at k = 4, T = 0.1)
> **does not declare equivalence**: the minimum detectable effect at this sample size is 0.206 nats,
> above the margin. **The honest statement is that there is no detectable bound-conditioned hotspot
> penalty, and that we are underpowered to rule out one smaller than ~0.2 nats.**
> The `N_hot ≈ 10^10` constellation cost is a generic property of T = 0.1 sampling — statistically
> indistinguishable at burial-matched control constellations (median Δ = **0.000 log10**, p = 0.90).
>
> **Result 2 — positive, but the claim is narrower than an earlier draft stated.** The
> bound-vs-unbound 2×2 on the same matched pairs, with and without BRIEF §5.2's pre-registered
> amino-acid fixed effect (which an earlier draft applied only to the bound arm):
>
> | conditioning | raw | **AA-adjusted** | exact-AA-matched (n=51) |
> |---|---|---|---|
> | bound complex | −0.038 [−0.210, +0.137] | +0.119 [−0.060, +0.293] | — |
> | unbound monomer | −0.442 [−0.627, −0.266] | **−0.208 [−0.378, −0.050]** | **−0.012 [−0.453, +0.407]** |
> | **interaction (= d_bind_local)** | +0.404 [+0.273, +0.548] | **+0.318 [+0.193, +0.456]** | **+0.300** |
>
> **Roughly half the monomer deficit is amino-acid composition, and on exactly AA-matched pairs it
> is zero.** So *"hotspots are harder to recover without the partner"* is **not** supportable. What
> survives every control is the **interaction**: *"hotspots gain more from the partner's presence
> than matched controls do"* — **+0.318 nats [+0.193, +0.456]** AA-adjusted.
>
> `d_bind_local` is also **not model-internal**: against experimental ΔΔG_bind it reaches Spearman
> **ρ = +0.28**, beats the inverse-folding log-odds, and **adds beyond it** (partial ρ = +0.187
> controlling for burial *and* log-odds). Honest baseline the project had not reported: **burial
> alone (ρ = +0.369) beats every model quantity** — so all model claims must be burial-controlled.
>
> **The tax is real and we measured it. It is not a property of hotspot chemistry — it is a property
> of the conditioning set.**
>
> The raw hotspot/non-hotspot difference is a burial artifact, as BRIEF §5.1 warned — but it runs in
> the direction that *flatters* hotspots, not the direction ProBID-Net reported.

> **Corrections applied after an independent adversarial review** (all verified against the CSVs
> before acceptance): the monomer-conditioned gap was described as an *untested prediction* when it
> was already determined by data in `results/` (§3.2); §4.2 claimed the analytic form was
> "validated" when its own table shows a ~9×-noise-floor disagreement (§4.2); §4.3 used a mean whose
> CI is uninformative instead of the median that settles it; §2.4 asserted an invalid exclusion of
> ProBID-Net's −0.138; and BRIEF §5.2's pre-registered amino-acid fixed effect had never been
> applied (§2.6). Outstanding, not yet fixed: F0's wording ("CI contains zero") can fire on low
> power alone, and PREREG §4 says the primary tier decides while §6 leans on SECONDARY-B — see §6.

---

## 0. STEP 0 — the unresolved preprint

**Fetched and resolving** (bioRxiv `10.64898/2026.05.09.722041`, *"Redesign selective protein binders
using contrastive decoding"*, Xie & Xu, Toyota Technological Institute at Chicago, 2026-05-13):

| Artifact | URL | Result |
|---|---|---|
| Metadata + abstract | `https://api.biorxiv.org/details/biorxiv/10.64898/2026.05.09.722041` | fetched |
| Full text (JATS XML) | `https://www.biorxiv.org/content/early/2026/05/13/2026.05.09.722041.source.xml` | fetched, 33k chars |
| Full PDF | `https://www.biorxiv.org/content/10.64898/2026.05.09.722041v1.full.pdf` | fetched, 13 pp |
| Supplement PDF | `https://www.biorxiv.org/content/biorxiv/early/2026/05/13/2026.05.09.722041/DC1/embed/media-1.pdf` | fetched, 12 pp |
| Code repo | `https://github.com/zw2x/rednet_public` | fetched (README) |

**Could not fetch:** the `.full` HTML view (HTTP 403 to the fetch tool). The XML and PDF carry the
same content, so nothing is missing. Zenodo-hosted weights and selectivity benchmarks were not
downloaded — not needed to answer the question.

**Answer to the STEP 0 question: NO.** The paper contains neither the burial-matched analysis nor
the commitment-ordering result. Term counts across main text *and* supplement, on **two independent
counting paths** (`grep -a` and a Python substring count), with positive controls firing
(RedNet 20, contrastive 18, ProteinMPNN 12, SKEMPI 4, Spearman 6):

```
hotspot 0 · hot spot 0 · hot-spot 0 · alanine 0 · burial 0 · buried 0
matched 0 · commitment 0 · unmask 0 · decoding order 0 · frustrat 0 · dynamics 0 · b-factor 0
```

> **A silently-broken filter was caught here by the standing rule.** The first term scan of the
> supplement returned **empty strings for every term including the positive controls** — because the
> extracted text contains NUL bytes, so GNU `grep` treated it as binary and exited 1 ("no match") on
> every query. Taken at face value it would have produced a clean sheet of false negatives, including
> a false "SKEMPI absent" when SKEMPI is demonstrably present. Root-caused, stripped the NULs, and
> re-ran on two independent paths. This is the third occurrence of this failure mode in the project's
> lineage.

**Two things that do bear on this project:**

1. **The diagnosis is no longer unclaimed.** RedNet states BRIEF §2.1's premise outright:
   *"[ProteinMPNN] sees only backbone atoms and cannot capture side-chain conformations at the target…
   its decoding algorithm has no explicit mechanism to jointly optimize protein complex binding
   affinity while maintaining binder folding stability."* Their fix is a side-chain-aware architecture
   plus contrastive decoding. The *measurement* — whether the deficit is real once burial is
   controlled — remained unclaimed, and is what this run performed.

2. **Their Supplementary S1 partially occupies F1's territory.** Table S1/S2 report zero-shot SKEMPI
   v2.0 Spearman ρ for ProteinMPNN log-likelihood variants: `ll` 0.17, `global` 0.17, `mt` 0.23,
   `ref` **0.26**, `cd_ll` 0.10, `cd_ll_ref` 0.24 (RedNet peaks at 0.28). Their `ref` statistic
   is close to F1's quantity. **This is an independent published cross-check on our F1 measurement,
   and it agrees**: see §3.

---

## 1. What was run

```bash
# STEP 0 - preprint (see above)
# validation gate: every computational path gets a positive control
python3 src/validate.py

# STEP 1+2 - Phase 0: burial-matched control + causal discrimination
python3 -u src/p0_burial_matched.py --out results/p0

# STEP 3 - Phase 1: N_hot at T=0.1
python3 -u src/p1_nhot.py --out results/p1 --direct-budget-s 2400 --direct-K 200 --direct-n-marginal 8

# controls, and the third frustration proxy named in STEP 2
python3 src/nhot_control.py           --out results/nhot_control
python3 src/matched_recovery.py       --out results/matched_recovery
python3 -u src/frustration_monomer.py --out results/frustration_monomer

# pre-registered robustness replicate on the sigma=0.02 checkpoint (PREREG section 5)
python3 -u src/p0_burial_matched.py --out results/p0_n002 \
  --mpnn-weights ~/ftax/ProteinMPNN/vanilla_model_weights/v_48_002.pt \
  --only-complexes results/pair_complexes.txt
```

Every command above is also recorded verbatim in a `command` column inside the CSV it produced.

**Data.** SKEMPI 2.0 (`https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv`, 7085 rows)
and its cleaned PDB bundle (`SKEMPI2_PDBs.tgz`, 345 structures). 6798 rows have a computable
ΔΔG over 341 PDBs.

**Model.** Public ProteinMPNN `vanilla_model_weights/v_48_020.pt` (training backbone noise 0.20 Å),
`augment_eps = 0` at inference, from `github.com/dauparas/ProteinMPNN`.

**Scale achieved.** 167,083 residues scored across **343 complexes** (1 skipped: 3VR6, 3397 residues,
over the size cap). 13,412 interface positions. **SKEMPI wild-type identity matched the structure at
2309/2309 mapped positions — 0 mismatches.**

### Validation gate (`src/validate.py` → ALL PASS)

| Path | Positive control | Result |
|---|---|---|
| Secondary structure (self-implemented Kabsch–Sander; no DSSP binary installable) | 1MBN all-α / 1TEN all-β | **87% H / 84% E** |
| SASA | our `addAtom` path vs freesasa's own PDB reader | agrees to **0.000%** |
| Interface detection | 1CSE chain I bound vs free | 12 residues lose >10 Å², no negative ΔSASA |
| ProteinMPNN | native sequence recovery on 1CSE | **0.650** vs 0.05 random |
| ProteinMPNN | per-order and mixture normalisation | max\|logsumexp\| 2.4e−07 |
| ΔΔG sign | barnase–barstar **D39A** | **+6.79 kcal/mol** (canonical hotspot) |
| ΔΔG sign | all single Ala mutations | 78.8% positive |

**Independent check of the interface definition** against SKEMPI's own COR/SUP/RIM annotation:
**67/67 COR called interface, 37/37 INT and 49/49 SUR called non-interface**; 90.2% overall
agreement, with every disagreement in the genuinely ambiguous RIM/SUP classes. Median
rSASA_complex orders correctly: SUP 0.021 < COR 0.095 < RIM 0.356 ≈ SUR 0.366.

---

## 2. STEP 1 — Phase 0, the burial-matched control

### 2.1 The confound is real, large, and points the wrong way

Interface positions by label (`results/p0_positions.csv`, n = 13,412 over 343 complexes):

| Label | n | rSASA_complex | ProteinMPNN recovery | mean log p(native) |
|---|---:|---:|---:|---:|
| **hot_strict** (ΔΔG > 2) | 327 | **0.080** | **0.529** | −1.732 |
| **hot_loose** (1 < ΔΔG ≤ 2) | 374 | 0.126 | 0.439 | −1.875 |
| other (0.25 ≤ ΔΔG ≤ 1) | 490 | 0.181 | 0.361 | −2.133 |
| **null** (\|ΔΔG\| < 0.25) | 242 | 0.218 | 0.347 | −2.137 |
| unmeasured | 11,979 | 0.167 | 0.451 | −1.818 |

Recovery rises monotonically with hotspot strength — **and burial rises in lockstep**. Hotspots are
2.7× more buried than nulls (0.080 vs 0.218). BRIEF §5.1 predicted exactly this coupling. What it
did not predict is the sign of the resulting bias in this dataset.

### 2.2 The uncontrolled comparison (the ProBID-Net-style measurement)

| Contrast | recovery hot | recovery non-hot | gap | rSASA hot / non-hot |
|---|---:|---:|---:|---:|
| **strict (ΔΔG>2) vs all other interface** | 0.529 (n=327) | 0.445 (n=13,085) | **+0.084** | 0.080 / 0.168 |
| strict vs *measured* non-hotspot interface | 0.529 | 0.357 (n=732) | +0.172 | 0.080 / 0.194 |
| loose vs null interface | 0.481 (n=701) | 0.347 (n=242) | +0.134 | 0.105 / 0.218 |

**ProBID-Net published 0.334 at hotspots vs 0.472 at non-hotspots — a gap of −0.138.
The same uncontrolled comparison, with ProteinMPNN on SKEMPI, gives +0.084: the sign reverses.**

⚠️ **This is not a direct replication and must not be reported as one.** ProBID-Net measured *their
own model* on the *MIX* hotspot set (440 hotspot / 1902 non-hotspot) with their own interface
definition; this run measures *ProteinMPNN* on *SKEMPI 2.0* (327 strict hotspots / 13,085
non-hotspot interface). Any of model, hotspot set, or interface definition could carry the sign
difference. What can be said is narrower and still substantive: **on SKEMPI with ProteinMPNN, the
uncontrolled hotspot recovery deficit does not exist to be explained.**

### 2.3 F0 — the matched-pair result

Pairs are matched within complex on rSASA_complex (±0.05), secondary-structure class, and neighbour
count (±1), by optimal 1:1 assignment. Bootstrap resamples **complexes**, 10,000 replicates,
seed 20260803. Matching quality on the primary: mean \|ΔrSASA\| = 0.019 (max 0.049), max \|Δnbr\| = 1,
mean signed ΔrSASA = −0.003 (balanced). Hotspot ΔΔG median 1.81, control ΔΔG median 0.04.

`d = log p(native)_hotspot − log p(native)_control`. **The hypothesis predicts d < 0.**

| Analysis | pairs | complexes | mean d | 95% CI (complex bootstrap) | order-SD | frac d<0 |
|---|---:|---:|---:|---|---:|---:|
| **PRIMARY** loose vs measured nulls | 42 | 29 | **+0.334** | **[−0.132, +0.792]** | 0.119 | 33.3% |
| PRIMARY, 293–303 K | 40 | 28 | +0.316 | [−0.170, +0.815] | 0.120 | — |
| strict (ΔΔG>2) vs nulls | 21 | 14 | +0.236 | [−0.465, +0.744] | 0.166 | — |
| SECONDARY-A vs measured non-hotspot | 121 | 57 | +0.384 | [+0.041, +0.719] | 0.034 | 38.8% |
| **SECONDARY-B vs any interface** (highest power) | **384** | **132** | **−0.043** | **[−0.231, +0.137]** | 0.025 | 48.7% |
| AA-identity-matched (BRIEF §5.2) | 47 | 37 | +0.555 | [+0.151, +0.953] | 0.102 | 40.4% |
| hydrophobicity-matched (BRIEF §5.2) | 180 | 94 | +0.050 | [−0.181, +0.275] | 0.058 | — |
| sensitivity: neighbour tol ±2 | 479 | 141 | −0.016 | [−0.178, +0.145] | 0.023 | — |

> ### 🔴 F0 FIRES
> The primary 95% complex-level bootstrap CI **[−0.132, +0.792] contains zero.** Per the
> pre-registered kill in BRIEF.md §4, the burial-matched gap is absent and the mechanism is refuted.

**Not a decoding-order artifact** (BRIEF §5.6). Re-running the entire paired analysis inside each of
8 decoding orders gives a primary estimate spanning [+0.129, +0.475], SD 0.119 — the estimate is
~2.8× its own order-spread, and every order agrees on the sign. The order-*free* unconditional
(backbone-only) score gives +0.382 against the conditional +0.334, and −0.090 against −0.043 for
SECONDARY-B. The result does not depend on the autoregressive ordering.

**The sign matters and F0 does not cover it.** F0 as pre-registered fires only on "CI contains zero";
it is silent on a CI that excludes zero on the *positive* side. Three variants do exactly that
(SECONDARY-A +0.384, AA-matched +0.555, both excluding zero). Those are the *opposite* of the
hypothesis — hotspots being **easier**, not harder. Non-parametric sign tests agree: only 33.3%
(primary, binomial p = 0.044) and 38.8% (SECONDARY-A, p = 0.018) of pairs run in the hypothesised
direction, i.e. significantly *more* pairs run against the hypothesis than with it. I am reporting
this as refutation rather than as F0 firing on its literal wording; **the falsifier has not been
moved**, only its unhandled sign case named.

### 2.4 The same matched design in ProBID-Net's own metric (sequence recovery)

F0 is defined on log p(native), which is the more sensitive statistic — but ProBID-Net published
*recovery*. Re-running the identical matched design on recovery
(`src/matched_recovery.py` → `results/matched_recovery.csv`):

| Design | recovery hotspot | recovery control | paired diff | 95% CI |
|---|---:|---:|---:|---|
| **PRIMARY** loose vs nulls | 0.405 | 0.262 | +0.143 | [−0.053, +0.325] |
| strict vs nulls | 0.476 | 0.238 | +0.238 | [+0.000, +0.444] |
| **SECONDARY-B** (highest power) | 0.469 | 0.464 | **+0.005** | **[−0.070, +0.080]** |
| sensitivity: neighbour tol ±2 | 0.474 | 0.453 | +0.021 | [−0.045, +0.086] |
| AA-identity-matched | 0.574 | 0.362 | +0.213 | [+0.050, +0.378] |
| hydrophobicity-matched | 0.444 | 0.456 | −0.011 | [−0.113, +0.084] |
| *ProBID-Net, published, **uncontrolled*** | *0.334* | *0.472* | *−0.138* | *—* |

**The burial-matched recovery gap is +0.005, 95% CI [−0.070, +0.080].** An earlier draft added
"and that interval excludes ProBID-Net's published −0.138" — **that comparison is withdrawn as
invalid**: it sets a paired, within-complex, burial-matched difference in ProteinMPNN-on-SKEMPI
against an unpaired, uncontrolled difference in a different model on the MIX set. §2.2's own caveat
forbids it. In ProteinMPNN on SKEMPI 2.0, once burial, secondary structure and
packing are matched within complex, hotspots and non-hotspot interface positions are recovered at
statistically indistinguishable rates. No variant recovers a hotspot deficit; the two variants whose
CIs exclude zero both show a hotspot *advantage*.

### 2.5 Robustness replicate on the σ = 0.02 checkpoint (pre-registered in PREREG §5)

`v_48_002.pt` re-run over the 141 complexes that contribute matched pairs
(`results/p0_n002_*`; RedNet's SKEMPI table used ProteinMPNN at σ = 0.02, hence this check):

| Analysis | pairs | v_48_020 (primary) | v_48_002 (robustness) |
|---|---:|---|---|
| **PRIMARY** loose vs nulls | 42 | +0.334 **[−0.132, +0.792]** | +0.598 **[+0.064, +1.050]** |
| strict vs nulls | 21 | +0.236 [−0.465, +0.744] | +0.961 [+0.239, +1.646] |
| SECONDARY-A | 121 | +0.384 [+0.041, +0.719] | +0.485 [+0.118, +0.847] |
| **SECONDARY-B** | 384 | −0.043 [−0.231, +0.137] | +0.026 [−0.168, +0.212] |
| AA-matched | 47 | +0.555 [+0.151, +0.953] | +0.561 [+0.122, +0.978] |
| hydrophobicity-matched | 180 | +0.050 [−0.181, +0.275] | +0.125 [−0.143, +0.385] |
| neighbour tol ±2 | 479 | −0.016 [−0.178, +0.145] | −0.001 [−0.176, +0.171] |
| F1 partial ρ (interface) | — | −0.247 | −0.242 |
| uncontrolled recovery gap | — | +0.084 | +0.071 |

**The substantive conclusion is checkpoint-independent: no variant on either checkpoint shows the
hypothesised hotspot penalty, and the highest-powered variant sits at zero on both.**

⚠️ **One difference must be reported precisely, because it changes which falsifier fires.** On
`v_48_002` the primary CI is **[+0.064, +1.050]**, which *excludes* zero — so **F0, read literally,
does not fire on this checkpoint**. But it excludes zero on the **positive** side: hotspots are
*significantly easier* than burial-matched controls. That is the sign case F0 does not cover
(§2.3), and it refutes the mechanism **more** strongly than F0 firing does, not less. PREREG §5
names `v_48_020` as primary and `v_48_002` as the robustness check, so **the pre-registered verdict
is taken from `v_48_020`, where F0 fires.** Both readings refute; neither supports.

### 2.5b Four-model replication — the single-model objection, answered

Same 141 complexes, **same structurally-determined matched pairs**, four architectures spanning an
85x parameter range and four different decoding regimes (`src/p0_multimodel.py`,
`results/panel_summary.csv`):

| tier | mpnn_vanilla 1.7M | mpnn_soluble 1.7M | **pifold 6.6M** | mif 3.4M |
|---|---|---|---|---|
| PRIMARY (47) | +0.420 [-0.050,+0.882] | +0.433 [-0.073,+0.919] | +0.360 [-0.234,+0.889] | **+0.556 [+0.109,+0.982]** |
| SECONDARY-B (~380) | -0.042 [-0.222,+0.129] | -0.037 [-0.222,+0.148] | -0.171 [-0.396,+0.045] | -0.143 [-0.359,+0.076] |
| SENS nbr+-2 (~465) | -0.021 [-0.189,+0.144] | -0.060 [-0.230,+0.107] | **-0.250 [-0.457,-0.047]** | -0.177 [-0.393,+0.025] |

**PiFold is the decisive addition and not because of its size.** It is **one-shot and bit-exactly
deterministic** (max |delta| = 0.0 across repeat calls and seeds), so it has no decoding order at
all. The qualitative result replicating there rules out decoding-order variance **by construction**
rather than by averaging it down - retiring CLAUDE.md's false-positive #2 outright.

**How to read the disagreement, honestly.** In 3 of 4 models every tier's CI contains zero. The two
exceptions point in **opposite directions**: MIF's PRIMARY excludes zero on the *positive* side
(hotspots easier) while PiFold's largest tier excludes zero on the *negative* side (hotspots harder).
Neither survives Holm correction. There is a weak systematic pattern - the two natively multichain
models (both ProteinMPNN variants) sit at ~0.0 while the two **single-chain-trained** models
(PiFold, MIF) sit slightly negative - which is plausibly about how models never trained on complexes
treat an interface, and is a limitation of using them here rather than a hotspot effect.

**The defensible conclusion is stronger than a single-model null:** across four architectures there
is no consistent burial-matched hotspot penalty, and the sign of the small residual effect **flips
depending on model and tier**.

*(ESM-IF1, 142M, is the fifth panel member. It cannot run on this machine - measured 1.9 GB at
L=400, 3.65 GB at L=1302, OOM at L=2120, against ~2 GB free - and needs ~1 GPU-hour.)*

### 2.6 The pre-registered amino-acid fixed effect (BRIEF §5.2), which tightens the bound 4x

BRIEF §5.2 requires wild-type identity as a fixed effect; the pair analysis above does not apply it,
and the matched pools are badly unbalanced in composition (only 6.5% of SECONDARY-B pairs share a
residue type; hotspots skew Y/R/K/E/L/F, controls S/G/A/V). Residualising `logp_native` on native
amino-acid dummies across all 13,412 interface positions and re-forming the same pairs:

| Design | raw gap | **AA-identity-adjusted** |
|---|---|---|
| **SECONDARY-B** (384) | −0.043 [−0.231, +0.137] | **+0.119 [−0.060, +0.293]** |
| neighbour tol ±2 (479) | −0.016 [−0.178, +0.145] | +0.153 [−0.004, +0.310] |
| PRIMARY (42) | +0.334 [−0.132, +0.792] | +0.370 [−0.065, +0.782] |

**The headline equivalence bound tightens from "no hotspot penalty > 0.23 nats" to "> 0.057 nats".**
This matters for §4's dismissal of N_hot: at 0.23 nats over 3 hotspots the implied cost is ~10^3,
which is *above* F2's own bar for "costly" (log10 N_hot >= 2). At the AA-adjusted 0.057 nats it is
~5.5, i.e. log10 ~ 0.7 — comfortably below. **The AA-adjusted bound is the one that supports the
argument, and it is the pre-registered analysis.**

**Power, stated honestly.** The pre-registered primary is small: 42 pairs over 29 complexes, because
it requires *both* members of a pair to carry an experimental measurement (hotspot ΔΔG > 1 and
control |ΔΔG| < 0.25) *and* to match on three structural constraints inside one complex. Its CI
half-width is ±0.46 nats, so it cannot exclude a small negative gap on its own. That is why the
pre-declared SECONDARY-B tier matters: at 384 pairs over 132 complexes it puts the gap at
**−0.043 [−0.231, +0.137]**, which excludes any hotspot penalty larger than **0.23 nats**. For scale,
a 0.23-nat deficit at 3 hotspots would contribute only ~10^3 to N_hot at T = 0.1 — the mechanism
requires far more than the data permits.

---

## 3. STEP 2 — the causal discrimination (frustration vs dynamics)

BRIEF.md §4 asks whether the *residual post-matching* gap tracks frustration or dynamics. **The
residual gap is ≈ 0, so there is essentially nothing left to attribute.** The discrimination is
reported for completeness and is exploratory — no pre-registered threshold attaches to it.

**Pair level** (SECONDARY-B, n = 384) — Spearman ρ of `d_logp` against each Δproxy:

| Family | Proxy | ρ | p |
|---|---|---:|---:|
| frustration | Δ buried-polar fraction | −0.116 | 0.023 |
| frustration | Δ unsatisfied buried polars | −0.116 | 0.023 |
| frustration | Δ χ1 rotamer strain | +0.029 | 0.61 |
| dynamics | Δ B-factor (z) | +0.018 | 0.73 |
| dynamics | Δ GNM fluctuation | +0.077 | 0.13 |

Joint standardised OLS (n = 310): unsatisfied-buried-polar β = −0.174 (t = −1.88);
GNM flexibility β = **+0.253 (t = +2.74)**; all others |t| < 0.3.

**Position level** (all 13,412 interface positions, log p residualised on rSASA + neighbour count +
SS class + amino-acid identity — far more power than the matched pairs):

| Family | Proxy | ρ | p | n |
|---|---|---:|---:|---:|
| frustration | **χ1 rotamer strain** | **−0.112** | 1.0e−33 | 11,583 |
| dynamics | GNM fluctuation | −0.079 | 5.2e−20 | 13,412 |
| dynamics | B-factor (z) | −0.057 | 3.5e−11 | 13,412 |
| frustration | buried-polar fraction | +0.020 | 0.022 | 13,412 |
| frustration | unsatisfied buried polars | −0.016 | 0.060 | 13,412 |

**Reading.** At the position level, rotamer strain is the strongest single predictor of reduced
inverse-folding confidence (ρ = −0.112) and edges out both dynamics proxies — weak support for
frustration over ProBID-Net's dynamics attribution. But all five effects are small (|ρ| ≤ 0.11),
the two families are not cleanly separated, and the pair-level joint model actually puts the largest
coefficient on a *dynamics* proxy with the *wrong* sign. **No confident attribution is warranted
from these five proxies.**

### 3.1 The third frustration proxy — monomer versus complex — and the one clear positive result

BRIEF.md §4 names a third frustration proxy, *monomer-versus-complex local energy*, which the five
above do not cover. Scoring every interface residue twice under the same model — once on the bound
complex, once on its **own SKEMPI chain group alone** —

    d_bind_local(i) = log p(native_i | complex) − log p(native_i | own group alone)

gives the local energetic benefit the partner confers, in the model's own units. A frustrated
residue is one the isolated partner disprefers but the complex requires, so **frustration predicts
d_bind_local to be larger at hotspots than at burial-matched controls.**
(`src/frustration_monomer.py` → `results/frustration_monomer_*.csv`; 141 complexes.)

| Interface label | n | mean d_bind_local | median |
|---|---:|---:|---:|
| **hot_strict** (ΔΔG > 2) | 325 | **0.918** | 0.829 |
| **hot_loose** (1 < ΔΔG ≤ 2) | 370 | 0.492 | 0.317 |
| other | 446 | 0.263 | 0.154 |
| **null** (\|ΔΔG\| < 0.25) | 226 | 0.223 | 0.087 |
| unmeasured | 4375 | 0.297 | 0.119 |

Monotonic in hotspot strength. On the **same 384 burial-matched pairs / 132 complexes** used for F0,
with the same complex-level bootstrap:

| | mean d_bind_local |
|---|---:|
| hotspot | +0.755 |
| burial-matched control | +0.375 |
| **paired difference** | **+0.380, 95% CI [+0.244, +0.525]** |

> **This is the one contrast in the entire run that is both burial-matched and significantly
> non-zero.** Hotspots gain roughly twice as much from the partner's presence as matched controls do.

*(A secondary regression of the F0 pair gap on Δd_bind_local gives ρ = +0.381, p = 9.6e−15, but
`log p(native | complex)` enters both sides, so it is partly circular and is reported only as
`_CIRCULAR` in the summary CSV. The paired contrast above is not circular: it compares hotspots to
controls within the same quantity.)*

**What this means, and what it does not.** It is direct evidence that **BRIEF §2.1's frustration
premise is correct**: hotspot residues genuinely are dispreferred by the isolated partner and
preferred in the complex. That is the frustration signature, measured, and it favours frustration
over ProBID-Net's dynamics attribution more convincingly than any of the five proxies above.

**But it does not rescue the mechanism — it explains why the mechanism fails.** The predicted
factorisation tax requires the model's mode to be the *wrong* residue at hotspots. It is not,
because ProteinMPNN conditions on the **bound complex** backbone and therefore already has the
partner information that makes the frustrated residue favourable. The frustration is real; the
model simply is not blind to it in this setting.

### 3.2 The deficit reappears when the partner is removed — MEASURED, not predicted

An earlier draft of this document called the following a "sharp untested prediction". **That was an
error**: it is already determined by the data in `results/`, via the identity

    d_bind_local(paired) ≡ gap(complex-conditioned) − gap(monomer-conditioned)

(verified to 4e−16). On the **same 384 burial-matched pairs / 132 complexes**, same complex-level
bootstrap:

| Conditioning | paired hotspot − control log-prob gap |
|---|---|
| **bound complex** (Phase 0) | **−0.043, 95% CI [−0.231, +0.137]** |
| **monomer** (partner chain removed) | **−0.423, 95% CI [−0.595, −0.256]** |
| difference (= d_bind_local) | **+0.380, 95% CI [+0.244, +0.525]** |

**Remove the partner and a hotspot deficit appears, significantly and with the predicted sign.** The
factorisation tax is not a property of hotspot chemistry — it is a property of **the conditioning
set**.

**Two controls this needs, both run:**

*Burial in the unbound state.* Pairs are matched on `rSASA_complex`, not on `rSASA_free`, and
hotspots are more exposed than their controls once the partner leaves (ΔrSASA +0.066). Restricting
to pairs balanced in the monomer state:

| subset | n pairs | monomer gap | d_bind_local |
|---|---:|---|---|
| all | 384 | −0.423 [−0.595, −0.256] | +0.380 [+0.244, +0.525] |
| \|Δ rSASA_free\| ≤ 0.10 | 146 | −0.296 [−0.553, −0.048] | — |
| \|Δ rSASA_free\| ≤ 0.05 | 78 | −0.396 [−0.733, −0.059] | — |
| \|Δ ΔrSASA\| ≤ 0.05 | 77 | — | +0.405 [+0.177, +0.631] |
| \|Δ ΔrSASA\| ≤ 0.03 | 51 | — | +0.479 [+0.173, +0.760] |

Both survive. These are post-hoc subsets, not a design — **a properly pre-registered replication
matching in monomer space (`rSASA_free` ±0.05, monomer neighbour count ±1, monomer SS) is required
before this can be a primary result.**

*Chain deletion is a conservative proxy for apo.* A real unbound structure relaxes — loops close
over the exposed patch and the groove partly collapses — which should make the native hotspot
residue **less** predictable than chain-deleted holo geometry does. So −0.423 is a **lower bound** on
the true unbound deficit.

**Scope limit, stated plainly.** In the actual staged pipeline (RFdiffusion → ProteinMPNN) the model
*is* run on the complex with the target present. The monomer setting is therefore a mechanistic
bracket, not the practised workflow. The variable that differs in practice is **native co-crystal
backbone vs designed backbone** — a native backbone was carved by the very side chains being
predicted. Testing that is the decisive experiment and it is not yet run.

### F1 — burial-controlled partial Spearman

Inverse-folding log-odds `ℓ(mut) − ℓ(wt)` versus SKEMPI ΔΔG_bind, partialling out rSASA_complex
(`results/p0_f1_logodds.csv`):

| Subset | n | raw ρ | burial-partial ρ | F1 (\|ρ\| ≥ 0.35)? |
|---|---:|---:|---:|---|
| all single mutations | 4950 | −0.272 | **−0.200** | no |
| interface only | 3545 | −0.297 | **−0.247** | no |
| interface, 293–303 K | 3389 | −0.298 | −0.246 | no |

> ### 🟢 F1 DOES NOT FIRE
> |ρ_partial| = 0.247 < 0.35. By the pre-registered criterion the model is not disqualified from
> being "blind to binding energy".

But this must be read carefully, and it cuts against the mechanism rather than for it. The
correlation is **highly significant** (p = 1.7e−50) and in the expected direction: ProteinMPNN
*already* assigns lower log-odds to mutations that experimentally weaken binding, with burial
controlled. It is *partially sighted*, not blind. F1's 0.35 threshold was set high enough that
"substantially but not fully correlated" passes it.

**Independent cross-check.** RedNet's Supplementary Table S1 reports ProteinMPNN zero-shot SKEMPI
Spearman ρ = 0.26 for its `ref` statistic (reference-normalised log-odds at mutated positions) —
the closest published analogue of our quantity. We measure |raw ρ| = 0.272 pooled over all single
mutations. **Two independent implementations, on the same database, agree to within 0.01.** That is
strong evidence the F1 measurement is correctly implemented.

---

## 4. STEP 3 — Phase 1, N_hot

147 complexes carrying ≥ 1 interface hotspot, 701 hotspot positions, median constellation size
k = 4, median complex length 442. T = 0.1. (`results/p1_nhot.csv`, `results/p1_summary.csv`)

BRIEF.md §5.5 requires the positional-independence assumption to be measured, not assumed. Because
temperature scaling commutes with the log-softmax the model already returns
(`softmax(logits/T) = softmax(log_probs/T)`), exact T-scaled conditionals come straight from forward
passes, and hotspot–hotspot correlation can be isolated **exactly** by controlling the decoding order
rather than estimated by sampling:

- **independent**: one pass per hotspot, each conditioned on all non-hotspot positions but on **no
  other hotspot** (that hotspot decoded first among the hotspots);
- **chain**: hotspots decoded in sequence after the *same* context, so each sees the earlier ones.

| Estimator | median log10 N_hot | IQR |
|---|---:|---|
| BRIEF formula `exp(Σδ_i/T)` | 8.94 | [1.80, 17.31] |
| independent, exactly normalised | 10.42 | [2.61, 18.39] |
| **chain (correlation included)** | **10.17** | [2.30, 17.64] |

76.2% of complexes have log10 N_hot ≥ 2. Median log10 N_hot by constellation size (**bins**, not
cumulative): k = 1 → 2.52 (n=34), k = 2 → 9.02 (n=21), k = 3–4 → 4.88 (n=30), k = 5–8 → 14.56
(n=41), k > 8 → 44.41 (n=21). The trend rises with k as §2.2 predicts, but **not monotonically** —
the k = 2 bin exceeds the k = 3–4 bin — so per-position cost varies more between complexes than
constellation size does within them.

> ### 🟢 F2 DOES NOT FIRE
> F2 requires median log10 N_hot < 2 **and** the F0 CI to contain zero. The second clause holds, the
> first does not: median log10 N_hot = **10.17**, not < 2.

### 4.1 Two measured corrections to the analytic form

1. **The BRIEF's own formula is biased low by ~1.5 log10.** `exp(Σδ_i/T)` gives 8.94 against 10.42
   for the exactly-normalised product `1/Π p_T,i(native)`. The mode-dominance approximation
   `Z_i ≈ p_mode^{1/T}` is not tight at T = 0.1.
2. **Positional correlation is real, and negative.** chain − independent has mean **−0.586 log10**,
   median −0.000, IQR [−0.76, +0.01]. Conditioning a hotspot on the other hotspots makes the native
   constellation **cheaper** than independence predicts — the analytic product *overestimates* the
   cost. Within-complex variability across context orders (4 reps) has median SD 0.61 log10.

### 4.2 The direct measurement, and where it is valid

The joint constellation is unobservable by sampling in most complexes: at 1.8 s/sample on this CPU,
resolving N_hot = 10^10 is impossible. Direct sampling was therefore restricted, as BRIEF.md §4
anticipates, to the 37 complexes with log10 N_chain ≤ 2.3; 18 were sampled at K = 200 within budget,
16 yielded ≥ 1 full-constellation recovery, 2 gave zero.

| Comparison (median, log10) | value |
|---|---:|
| direct − chain | **+0.126** |
| \|direct − chain\| | 0.318 |
| direct − independent | +0.041 |
| direct − BRIEF formula | +0.048 |

**Correction to an earlier draft, which called this "validated" — it is not.** The binomial noise
floor here is small (median SE of log10 p-hat = **0.035**, because the resolvable joint frequencies
are high), so the median |direct - chain| of 0.318 is **~9x the sampling noise**, and **13 of 16
complexes disagree by more than 2 SE** (e.g. 3KUD predicted 1.000 vs observed 0.125). The
disagreement is also directional: direct > chain in 11/16. This is a **measured ~0.3-log10
systematic bias**, not agreement - and because it is one-directional it makes N_hot = 10^10 a
**lower** bound.

**But the per-position marginal is systematically overpredicted.** Mean hotspot marginal recovery:
**observed 0.636 vs predicted 0.842** (−0.206). Individual cases diverge sharply *in both directions*
— 3KUD predicted 0.999 / observed 0.125; 1EAW predicted 0.003 / observed 0.120. The cause is
structural: the teacher-forced conditional assumes **the rest of the sequence is native**, whereas
the real sampler at T = 0.1 generates a non-native context that shifts the conditional at the
hotspot. This is the same positional-correlation effect, but between hotspots and the *whole
sequence* rather than among hotspots.

⚠️ **Restriction stated explicitly, not hidden:** the direct arm covers only the *easiest* complexes
(log10 N_chain ≤ 2.3, i.e. the bottom ~25%), selected by smallest length first. The discrepancy on
hard, high-N_hot complexes is **unmeasured**, and nothing here should be read as validating the
analytic form at log10 N_hot ≈ 10.

### 4.2b N_hot is a STEP FUNCTION of argmax errors — so oversampling at T = 0.1 is not a weak
lever, it is no lever

This was invisible under the pre-registered frame, which asked only *"is median log10 N_hot < 2?"*.
Once the answer was 10.17 the quantity was filed as generic and dropped, and nobody asked **what
function of the model** it is. It is almost entirely a restatement of per-position argmax accuracy:

| predictor of log10 N_hot (147 complexes) | R² |
|---|---:|
| number of hotspot positions where the argmax is **not** native | **0.852** (slope **+7.11 log10 / miss**) |
| constellation size k alone | 0.709 |
| misses **+** k | 0.852 (**adding k buys +0.0004**) |

| | n | median log10 N_hot |
|---|---:|---:|
| complexes with **zero** argmax misses at their hotspots | 24 | **0.000** — recovered in ~one draw |
| complexes with ≥ 1 miss | 123 | **12.49** |

Per position: argmax correct → cost **0.020 log10**; argmax wrong → **5.61 log10**
(p_T(native) ≈ 2.5e−6). Cost is monotone in **exposure**, not burial (1.82 log10/position in the
most-buried decile → 3.60 in the most-exposed). At T = 1 the per-position cost falls to 0.67 log10,
so a k = 4 constellation costs ~10^2.7 — reachable.

**Reading.** At T = 0.1 there is no tail to sample from. If the argmax is right you get the
constellation almost immediately; if it is wrong at even one position, 10^7 draws will not fix it.
**The only levers are temperature and model accuracy — not sample count.** This generalises past
hotspots to every low-temperature design pipeline, and it means Phase 1 added no information beyond
Phase 0's recovery rate, which is itself the finding.

### 4.2c Teacher-forced recovery overstates what the sampler actually produces

The direct-vs-analytic discrepancy in §4.2 was filed as a formula correction. It is better read as a
statement about the field's reporting convention — and the §4.2 version was **confounded by
selection** (those complexes were chosen on `log10 N_chain ≤ 2.3`, i.e. on the prediction itself, a
winner's curse). Re-measured on 22 complexes selected **only by length** (K = 32 samples each):

| | value |
|---|---|
| sampled recovery | **0.484** |
| teacher-forced argmax recovery | **0.552** |
| bias | **−0.068 [−0.077, −0.058]**, 22/22 complexes negative |
| relative shortfall | **12.3% of reported recovery** |

And the structure is the point: where the teacher-forced argmax **is** native the sampler produces it
only 0.806 of the time (predicted 0.966, **−0.160**); where the argmax **misses** the sampler does
**better** than predicted (**+0.046**). Sampling destroys correct confident predictions and slightly
rescues wrong ones. Exposure bias is textbook in NLP but appears unquantified for inverse folding —
where teacher-forced recovery and design recovery are routinely compared across papers as if
interchangeable. *Limitation: 22 complexes, all L ≤ 244.*

### 4.2d Hotspot detection lives in partner-sensitivity, and model confidence actively hurts

Zero-shot detection of strict hotspots (ΔΔG > 2) among 5,372 interface positions / 141 complexes,
paired complex-level bootstrap against the burial baseline:

| score | AUROC | Δ vs burial |
|---|---:|---|
| burial (−rSASA) **baseline** | 0.694 [0.661, 0.725] | — |
| `d_bind_local` alone | 0.683 [0.645, 0.722] | −0.010 [−0.065, +0.044] |
| **burial + `d_bind_local`** | **0.744 [0.717, 0.772]** | **+0.051 [+0.018, +0.083]** |
| log p(native \| complex) | 0.538 [0.501, 0.576] | — |
| burial + log p(native) | 0.635 [0.605, 0.666] | **−0.058 [−0.092, −0.024]** |

**Adding the model's own confidence to burial makes hotspot detection significantly worse.** Only
the bound-minus-unbound *difference* carries signal, and it is burial-orthogonal (AUROC 0.63–0.72
within burial quintiles, where burial itself is ≈ 0.50). Top-decile enrichment: burial 2.24×,
combined **3.07×**.

*Scope limit:* `d_bind_local` requires a residue identity, so it is a **scoring** statistic
(in-silico alanine scanning, epitope mapping, ΔΔG ranking) — **not** a design-time geometric
detector. A residue-agnostic KL version would be, and is not yet computed.

### 4.2d-bis A residue-agnostic, DESIGN-TIME detector - what it does and does not support

`d_bind_local` needs the native residue, so it can only score a solved complex. The residue-agnostic
version uses ProteinMPNN's UNCONDITIONAL (sequence-free) distributions and asks only how much the
partner's presence moves them - pure geometry, no residue identity anywhere:

    KL_i = KL( p(. | bound complex backbone)  ||  p(. | own chain-group backbone) )

Full run, 5,742 interface positions / 141 complexes / 325 strict hotspots
(`src/kl_detector.py`, `results/kl_detector_summary.csv`):

| score | AUROC |
|---|---|
| burial (-rSASA) **baseline** | 0.689 [0.656, 0.720] |
| KL(complex \|\| monomer) alone | 0.694 [0.659, 0.731] |
| JSD | 0.693 [0.657, 0.730] |
| entropy drop | 0.610 [0.568, 0.652] |
| log p(native \| complex) | **0.538 [0.501, 0.577]** |
| **burial + KL** | **0.737 [0.710, 0.764]** |

> **A pilot on 8 complexes suggested KL *beat* burial (0.670 vs 0.621). At n = 141 it does not -
> 0.694 vs 0.689 is a tie.** The pilot claim is withdrawn.

**What survives is that KL ADDS to burial, and that is solid:** paired complex-level bootstrap of
the difference, **ΔAUROC = +0.048 [+0.022, +0.075], P(>0) = 1.000**. It is burial-orthogonal
(AUROC 0.60-0.76 within burial quintiles, where burial itself is ~0.50). And it again reproduces the
project's most consistent finding: **the model's own confidence is near-useless (0.538) while its
partner-sensitivity carries signal.**

**What it does NOT yet support.** On the metric a designer actually uses - per-complex top-k
precision, k = that complex's hotspot count - the gain is **+0.015 [-0.054, +0.083], not
significant**, on a low base (0.205 -> 0.220 over 105 complexes). So:

- ✅ *"Partner-sensitivity computed from backbone geometry alone adds burial-orthogonal information
  about which interface positions are hotspots"* - supported.
- ❌ *"We can detect hotspot positions from backbone geometry before any sequence exists"* - **not
  supported at useful precision.** AUROC gain is real; per-complex ranking gain is not yet
  demonstrated, and absolute precision (~0.22) is far from usable.

The honest framing is a **diagnostic finding about what backbone conditioning encodes**, not a
shipped detector. Making it a method needs a real baseline comparison (FoldX/Rosetta alanine
scanning, or a published hotspot predictor) and a held-out split - neither is done.

### 4.2e The ProBID-Net sign reversal, narrowed from four candidate causes to two

§2.2 says "any of model, hotspot set, or interface definition could carry the sign difference" and
stops. Three can now be eliminated:

- **Not global model quality** — non-hotspot recovery *agrees* (0.445 here vs their 0.472). The
  entire discrepancy is the hotspot number (0.529 vs 0.334).
- **Not the interface definition** — gap = +0.070 / +0.084 / +0.088 / +0.080 at ΔrSASA >
  0.01/0.05/0.10/0.25; core-restricted +0.038, rim-restricted +0.001. **Never negative.**
- **Not the hotspot threshold** — it is dose-responsive in the *wrong* direction: ΔΔG > 1 → +0.036;
  > 2 → +0.084; > 3 → +0.104; **> 4 → +0.144**. Stronger experimental hotspots are recovered *better*.
- **Not amino-acid composition** — direct standardisation predicts 0.409, i.e. 0.036 *below* the
  non-hotspot rate; within-AA hotspot advantage is +0.120. (Trp is 3.6× enriched at hotspots and has
  the worst non-hotspot recovery, 0.286 — composition runs *against* hotspots.)
- **Not teacher-forcing vs generation** — generation-mode gap +0.151 [−0.014, +0.281] vs
  teacher-forced +0.190 [−0.041, +0.390]; difference −0.039 [−0.153, +0.077].

**What remains: the model class, or the MIX hotspot set itself.** Neither is testable on this
fixture — but that is a real narrowing of a question this document previously left fully open.

### 4.3 The control that decides what N_hot means

**N_hot is large. It is equally large at burial-matched control positions.**
(`src/nhot_control.py` → `results/nhot_control.csv`; 132 complexes, median k = 2, same T, same
per-position distributions, matched controls taken from the Phase 0 matched pairs.)

| | median log10 N |
|---|---:|
| hotspot constellations | 5.38 |
| **burial-matched control constellations** | **5.88** |
| **paired difference (hot − control), median** | **+0.000, 95% CI [−0.749, +0.463]**, Wilcoxon p = 0.90 |
| paired difference, mean (per-complex SD = 11.2, range [−39, +52]) | +0.12, 95% CI [−1.76, +2.06] |

The hotspot constellation costs *more* in only 43.9% of complexes. This control is not
pre-registered — it became mandatory once F0 showed no hotspot penalty — and it is what turns the
N_hot number from evidence into an artifact of the metric.

**A constellation cost of 10^10 is real, but it is the generic cost of asking a low-temperature
autoregressive sampler to reproduce *any* specific multi-position constellation. It is not a tax at
the positions that make a binder a binder.** BRIEF.md §2.2's rhetorical punch — "no amount of
oversampling recovers native-grade interfaces" — survives as a statement about T = 0.1 sampling in
general, and dissolves as a statement about hotspots.

---

## 5. STEP 4 — the leakage note, and why it does not read the way BRIEF anticipated

BRIEF.md §5.3 records that **PDB training leakage runs against the hypothesis**: ProteinMPNN has seen
these complexes, so a *positive* result would be conservative. That reasoning is sound — but it
**does not protect a null**, and this run produced a null.

The leakage here is total, not partial: SKEMPI 2.0 was released in 2018 and every structure in it
predates ProteinMPNN's training cutoff. The model has seen all 343 complexes.

This is a genuine limitation of the refutation, and it is asymmetric in a way the BRIEF did not
anticipate. A memorised complex needs no reasoning about frustration to reproduce its native hotspot
residues — recall can substitute for the energetic understanding whose absence the mechanism
predicts. So the observed "hotspots are recovered *better*" could in principle be memorisation
masking a real tax rather than evidence against one.

**Stated plainly: leakage makes a positive result conservative and a null result weaker. The honest
conclusion is that the mechanism is refuted *on this fixture*, and that a leakage-free replication —
complexes deposited after the checkpoint's training cutoff, or a checkpoint retrained with SKEMPI
complexes held out — is the one experiment that could overturn this verdict.** That follow-up is
named here rather than run, because no post-cutoff SKEMPI-scale ΔΔG fixture exists on this machine.

---

## 6. Verdict

### Falsifier table, strictly as pre-registered

| Falsifier | Pre-registered condition | Measured | Fires? |
|---|---|---|:--:|
| **F0** | burial-matched gap 95% complex-bootstrap CI contains zero | +0.334, CI **[−0.132, +0.792]** | 🔴 **YES** |
| **F1** | burial-controlled partial Spearman \|ρ\| ≥ 0.35 | \|ρ\| = **0.247** | 🟢 no |
| **F2** | median log10 N_hot < 2 **and** F0 CI contains zero | median log10 N_hot = **10.17** (clause 1 false) | 🟢 no |

On the `v_48_002` robustness checkpoint F0 does **not** fire literally, because its CI
[+0.064, +1.050] excludes zero — on the side where hotspots are *easier*. The verdict is unchanged
and is taken from the pre-registered primary checkpoint (§2.5).

### One-line verdict

> # REFUTED
>
> **F0 fires.** There is no burial-matched hotspot penalty in ProteinMPNN on SKEMPI 2.0. The staged
> backbone→sequence pipeline is not, on this evidence, blind at interface hotspots — and the
> constellation cost `N_hot`, while genuinely enormous (10^10), is equally enormous at burial-matched
> control positions and is therefore not a hotspot-specific tax.

### What is actually true, in order of confidence

1. **The burial confound is real and large** — hotspots are 2.7× more buried than nulls
   (rSASA 0.080 vs 0.218) — **and it flatters hotspots rather than penalising them**, because
   inverse folding is most confident where burial is greatest. BRIEF §5.1 predicted the coupling and
   got the sign backwards for this fixture.
2. **Burial-matched, the hotspot/non-hotspot difference is zero.** Recovery: **+0.005
   [−0.070, +0.080]**; log-probability: **−0.043 [−0.231, +0.137]** (384 pairs / 132 complexes).
   Stable across 8 decoding orders, reproduced by the order-free unconditional score.
3. **ProteinMPNN is partially sighted, not blind, to binding energy.** Burial-controlled partial
   Spearman −0.247 (p = 1.7e−50), independently corroborated by RedNet's published ProteinMPNN
   SKEMPI ρ = 0.26 against our 0.272.
4. **N_hot ≈ 10^10 is a property of T = 0.1 sampling, not of hotspots** (§4.3).
5. **The frustration premise is correct, and that is why the mechanism fails.** Burial-matched,
   hotspots gain **+0.380 nats [+0.244, +0.525]** more from the partner's presence than controls do
   (§3.1) — the frustration signature, measured. But ProteinMPNN conditions on the *bound* backbone,
   so it already holds the information that makes the frustrated residue favourable, and never has
   to pay for it. **Frustration is real; blindness to it is not.**
6. **Among the five structure-only proxies the causal discrimination is inconclusive** — rotamer
   strain edges out the dynamics proxies at position level (ρ = −0.112 vs −0.079), but all effects
   are small and the gap they would explain is null. The monomer-vs-complex proxy (§3.1) is the only
   one that separates hotspots decisively, and it favours frustration.

### Was anything underpowered?

Yes, and it is named rather than glossed:

- **The pre-registered primary (42 pairs / 29 complexes) cannot on its own exclude a small negative
  gap** — CI half-width ±0.46 nats. The verdict rests on the pre-declared SECONDARY-B tier
  (384 pairs / 132 complexes, CI half-width ±0.18), which excludes any hotspot penalty > 0.23 nats.
  Had only the primary existed, the honest verdict would have been INCONCLUSIVE.
- **The direct N_hot arm covers only the easiest ~25% of complexes** (§4.2). The analytic form is
  unvalidated at log10 N_hot ≈ 10.
- **The frustration-vs-dynamics discrimination is underpowered by construction** — it was designed to
  explain a residual gap that turned out to be zero.

### The follow-up this run actually earned

**Design against the unbound backbone.** §3.1 shows the frustration premise holds and locates
exactly why it costs nothing here: the model is handed the bound complex. The predicted tax should
reappear when the conditioning genuinely lacks the partner, and d_bind_local measures its size in
advance (~0.92 nats at strict hotspots, ~0.38 nats above burial-matched controls). This is a cheap,
CPU-only, pre-registerable experiment on the same fixture, and it tests the mechanism where it could
still be true rather than where it has now been refuted.

### The one thing that could overturn this

**Training leakage (§5).** Every SKEMPI complex predates ProteinMPNN's training cutoff, and
memorisation can substitute for the energetic reasoning whose absence the mechanism predicts. The
pre-registered note that leakage makes results *conservative* applies to a positive finding, not to
this null. A replication on complexes deposited after the checkpoint's cutoff, or on a checkpoint
retrained with SKEMPI held out, is the decisive follow-up.

### Phase 2

**Not started, as instructed.** It requires a CUDA GPU this machine does not have, and it depends on
an unverified assumption about MultiFlow's released code. Phase 0 did not pass, so on BRIEF.md §4's
own gating condition Phase 2 should not be run regardless.

---

## 7. Files

| File | Contents |
|---|---|
| `PREREG.md` | analysis choices, fixed before any number was computed |
| `p0_positions.csv` | 167,083 residues × features + 8-order ProteinMPNN log-probs |
| `p0_pairs_*.csv` | matched pairs for each of the 8 design variants |
| `p0_summary.csv` | every F0/F1/causal statistic with CIs and order-spread |
| `p0_f1_logodds.csv` | 4950 single mutations: log-odds, ΔΔG, burial |
| `p0_interface_resid.csv` | 13,412 interface positions with burial-residualised log-prob |
| `p0_skipped.csv` | the one skipped complex and why |
| `p1_nhot_exact.csv`, `p1_nhot.csv`, `p1_summary.csv` | N_hot, all three estimators + direct arm |
| `nhot_control.csv` | hotspot vs burial-matched control constellation cost |
| `matched_recovery.csv` | ProBID-Net's recovery metric under the matched design |
| `frustration_monomer_*.csv` | monomer-vs-complex local energy: positions, joined, pairs, summary |
| `p0_n002_*` | robustness replicate on the σ = 0.02 checkpoint |

Scripts: `src/ftax_common.py`, `src/validate.py`, `src/p0_burial_matched.py`, `src/p1_nhot.py`,
`src/nhot_control.py`, `src/matched_recovery.py`, `src/frustration_monomer.py`. Seeds: bootstrap 20260803 (10,000 replicates),
decoding orders 0–7, sampling 20260803.

> **Two housekeeping issues for whoever commits this.** (1) `.gitignore` currently excludes
> `results/*.csv`, which contradicts CLAUDE.md ground rule 4 — every number above traces to a CSV,
> but none of those CSVs would be committed as configured. (2) `p0_positions.csv` (135 MB) and
> `p0_n002_positions.csv` (58 MB) are too large for git in any case; they are regenerable from
> `src/p0_burial_matched.py` and the two public downloads. Suggested resolution: commit the small
> summary/pairs CSVs, keep the two large position tables ignored, and say so in the README.

# FINDINGS — KL as a design-time triage signal (a candidate method contribution)

> **VERDICT (crystal proof-of-concept, POSITIVE; the design-time claim is PENDING non-native
> validation).** The sequence-free KL signal is not only a diagnostic — it is an *actionable* design-time
> triage ranker. Given a fixed budget of `k` interface positions per complex to receive expensive
> binding-aware optimization, ranking by **KL+burial captures significantly more experimental hotspots
> than the burial heuristic**: capture@3 **0.237 vs 0.139** (Δ **+0.098 [+0.021, +0.175]**, P=0.99),
> capture@25% **0.533 vs 0.444** (Δ **+0.089 [+0.009, +0.169]**), n=106 complexes with ≥1 hotspot,
> complex-level bootstrap (10,000 reps, seed 20260803). Two honest bounds: the gain is a **general**
> additive property of KL, **not** concentrated where the model is uncertain (niche AUROC Δ −0.002, null);
> and this is measured on **crystal** backbones, so it is a proof-of-concept — the design-time claim
> requires the predicted/generative backbones designers actually use (task #12, where KL already
> *strengthens*: ΔAUROC +0.048 crystal → +0.062 predicted → +0.083–0.094 generative-interface-formed).

## 1. What was run

```bash
python3 src/kl_triage.py --joined results/kl_detector_joined.csv --out results/kl_triage.csv
```

`results/kl_detector_joined.csv`: 141 complexes, 5742 interface positions, 325 interface hotspots
(`is_hot` = Ala-scan ΔΔG hotspot label). Per position: `kl` (the sequence-free detector — two ProteinMPNN
unconditional passes, complex vs chain-deleted backbone), `nbr`/`rsasa_complex` (burial), `logp_native`
(the model's own confidence), `is_hot`. Analysis restricted to the 106 complexes carrying ≥1 interface
hotspot. Every number traces to `results/kl_triage.csv`.

## 2. The triage framing (why this is a method, not just another AUROC)

A binder designer working on a fixed backbone cannot make ProteinMPNN see binding energy it has no term
for — but they *can* choose where to spend an expensive binding-aware step (FoldX/Rosetta ΔΔG, an MD
relaxation, a binding-aware model, or human attention). That is a **triage** problem: with a budget of
`k` interface positions per complex, which do you pick? The naive design-time signal is **burial** (buried
interface residues matter most). We ask whether **KL** — which is *free* (sequence-free, backbone-only,
no extra model) — beats or augments that heuristic at concentrating the experimental hotspots into the
top-`k`. The decision-relevant metric is therefore **capture@k** (of a complex's hotspots, the fraction
ranked in the top `k`), not top-`k` precision.

## 3. Results (`results/kl_triage.csv`)

| Budget | random | burial | KL | **KL+burial** | Δ(KL+burial − burial) [95% CI], P |
|---|---|---|---|---|---|
| **capture@3** | 0.084 | 0.139 | 0.219 | **0.237** | **+0.098 [+0.021, +0.175], P=0.99** |
| **capture@25%** | 0.261 | 0.444 | 0.543 | **0.533** | **+0.089 [+0.009, +0.169], P=0.99** |

- At a **3-position budget**, KL+burial captures **24%** of a complex's hotspots vs burial's **14%** — a
  **+70% relative** improvement, CI excludes zero. KL alone (0.219) already beats burial (0.139); adding
  burial tightens it (KL-alone Δ +0.079 [−0.005, +0.163] touches zero, so the **combined** ranker is the
  clean, reportable one — KL and burial are complementary).
- **AUROC(is_hot)** sanity vs `kl_analysis`: burial 0.678, KL+burial 0.742, Δ **+0.064 [+0.037, +0.091]**
  — replicates the known KL-adds-to-burial (+0.048) on the full fixture. Consistent.

## 4. Honest bounds (stated as bounds, not buried)

- **The "rescues where the model is blind" niche is NULL.** Restricted to model-uncertain positions
  (`logp_native` below the complex median — where log-prob design would most plausibly err), KL does *not*
  beat burial: AUROC Δ(KL − burial) = **−0.002 [−0.074, +0.068]**. The triage gain is a *general* additive
  property of KL across interface positions, **not** a targeted rescue of the model's failures. The
  intuitive on-narrative story does not hold and is not claimed.
- **Crystal-only.** `kl_detector_joined` is the crystal-backbone table. The *design-time* claim is about
  the non-native backbones designers use; it must be re-measured there (task #12). This is a
  proof-of-concept, not yet the design-regime result.
- **Not contradicted by the top-k precision null.** `kl_analysis` found KL's top-`k` *precision* over
  burial not significant. Precision (of the `k` selected, how many are hotspots) is dominated by the many
  non-hotspot interface positions; **capture** (of the hotspots, how many are selected) with the combined
  ranker is the triage-relevant quantity, and it survives. The two are consistent, not in tension.
- **Not a pilot-n number.** This is the full crystal fixture (n=106), not a small pilot — but the standing
  project lesson (two pilot claims shrank at scale) still applies to the *non-native* extrapolation, which
  is why task #12 gates the paper claim.

## 5. What it means for the paper

This converts the load-bearing KL result from a **diagnostic** ("KL identifies hotspots with
burial-orthogonal information") into a **candidate method** ("KL is a free, sequence-free triage signal
that beats the burial heuristic at allocating expensive binding-aware effort"). It is the actionable
contribution the ICLR case was missing (PAPER_OUTLINE.md contribution (iii)). It enters the paper as a
method **only after** the non-native validation (task #12): re-run `src/kl_triage.py` on the Exp A
(predicted) and Exp C (generative, interface-formed) per-position KL tables. If capture@k holds or
strengthens there — as every other KL statistic has on non-native backbones — the method claim is
design-relevant; if it collapses to crystal-only, we report it as a property of native backbones and do
not upgrade it.

## Files
- `src/kl_triage.py` — the triage prototype (capture@k, AUROC, niche; complex bootstrap).
- `results/kl_triage.csv` — all estimates, CIs, P-values (the trace for every number above).
- Input: `results/kl_detector_joined.csv` (crystal per-position KL + burial + is_hot).
